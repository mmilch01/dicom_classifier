#!/usr/bin/env python3

import argparse
import ast
import csv
import re
import sys
from collections import defaultdict


ANATOMICAL_TYPES = {
    "MPRAGE", "T1HI", "T1LO", "T2FLAIR", "T2HI", "T2LO", "SWI"
}
T1W_TYPES = {"MPRAGE", "T1HI", "T1LO"}


def norm(value):
    """Case-insensitive comparison form with surrounding whitespace removed."""
    return (value or "").strip().upper()


def parse_list(value):
    """Parse CSV fields such as ['ORIGINAL', 'PRIMARY', ...] or [0.9, 0.9]."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        result = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(result, (list, tuple)):
        return None

    return list(result)


def parse_image_type(value):
    values = parse_list(value)
    if values is None:
        return None
    return [norm(v) for v in values]


def parse_pixel_spacing(value):
    values = parse_list(value)
    if values is None or len(values) < 2:
        return None

    try:
        return float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return None


def parse_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_series_number(scan_id):
    """
    Interpret the XNAT scan ID as a series number.

    Examples:
      601       -> 601
      100-CT1   -> 100

    Returns None if the numeric first component cannot be determined.
    """
    if scan_id is None:
        return None

    text = str(scan_id).strip()
    if not text:
        return None

    first = text.split("-", 1)[0].strip()
    if not re.fullmatch(r"\d+", first):
        return None

    return int(first)


def has_original_primary(image_type):
    return (
        image_type is not None
        and len(image_type) >= 2
        and image_type[0] == "ORIGINAL"
        and image_type[1] == "PRIMARY"
    )


def image_type_has(image_type, value):
    if image_type is None:
        return False
    return norm(value) in image_type


def classify_primary(row):
    """
    Apply the primary-type rules in order.

    The original labels1 value is used for all rules explicitly conditioned
    on labels1 == OT. A successful rule stops primary-rule processing.
    """
    labels1_raw = row.get("labels1", "")
    labels1 = norm(labels1_raw)
    primary = labels1_raw.strip() if labels1_raw is not None else ""

    image_type = parse_image_type(row.get("ImageType"))
    series_description = norm(row.get("SeriesDescription"))
    slice_thickness = parse_float(row.get("SliceThickness"))
    pixel_spacing = parse_pixel_spacing(row.get("PixelSpacing"))

    # Rule 1: anatomical scans must begin with ORIGINAL/PRIMARY.
    # If ImageType is malformed/missing, the rule is skipped.
    if labels1 in ANATOMICAL_TYPES and image_type is not None:
        if not has_original_primary(image_type):
            return "OT"

    # Rules below apply only when the ORIGINAL labels1 is OT.
    if labels1 != "OT":
        # Rule 10: correct an anatomical label when SeriesDescription says FLAIR.
        if (
            series_description
            and "FLAIR" in series_description
            and labels1 in ANATOMICAL_TYPES
            and labels1 != "T2FLAIR"
        ):
            return "T2FLAIR"
        return primary

    # Rule 2a: ADC -> MD
    if (
        ("ADC" in series_description if series_description else False)
        or image_type_has(image_type, "ADC")
    ):
        return "MD"

    # Rule 2b: TRACEW
    if image_type_has(image_type, "TRACEW"):
        return "TRACEW"

    # Rule 3: MPRAGE
    if (
        series_description
        and "MPRAGE" in series_description
        and has_original_primary(image_type)
        and slice_thickness is not None
        and 0.5 <= slice_thickness <= 2.0
        and pixel_spacing is not None
        and 0.5 <= pixel_spacing[0] <= 1.5
        and 0.5 <= pixel_spacing[1] <= 1.5
    ):
        return "MPRAGE"

    # Rule 4: PBP
    if image_type_has(image_type, "PBP"):
        return "PBP"

    # Rule 5: RCBV -> CBV
    if image_type_has(image_type, "RCBV"):
        return "CBV"

    # Rule 6: FMRI
    if image_type_has(image_type, "FMRI"):
        return "FMRI"

    # Rule 7: FLAIR
    if (
        series_description
        and "FLAIR" in series_description
        and has_original_primary(image_type)
    ):
        return "T2FLAIR"

    # Rule 8: DWI
    if (
        series_description
        and "DIFFUSION" in series_description
        and has_original_primary(image_type)
        and image_type_has(image_type, "DIFFUSION")
    ):
        return "DWI"

    # Rule 9: DSC
    if (
        series_description
        and "PERFUSION" in series_description
        and has_original_primary(image_type)
        and image_type_has(image_type, "PERFUSION")
    ):
        return "DSC"

    return primary


def description_contrast(series_description):
    """Return 'nc', 'ce', or None from SeriesDescription."""
    desc = norm(series_description)
    if not desc:
        return None

    if desc.endswith("_PRE") or "_PRE_" in desc:
        return "nc"

    if desc.endswith("_POST") or "_POST_" in desc or desc.endswith("+C"):
        return "ce"

    return None


def classify_experiment(rows):
    """
    Classify all scans in one experiment.

    Primary labels are assigned first. Contrast classification then uses:
      1. SeriesDescription PRE/POST/+C rules.
      2. Adjacency to a SeriesDescription-classified contrast scan.
      3. DSC ordering, which overrides the other contrast rules.
    """
    work = []

    for row in rows:
        primary = classify_primary(row)
        work.append({
            "row": row,
            "primary": primary,
            "series_number": get_series_number(row.get("scan")),
            "contrast": None,
        })

    # Step 1: SeriesDescription-based T1w contrast classification.
    pre_candidates = []
    post_candidates = []

    for item in work:
        if norm(item["primary"]) not in T1W_TYPES:
            continue

        # Per instructions, scans without a determinable SeriesNumber do not
        # participate in contrast classification.
        if item["series_number"] is None:
            continue

        contrast = description_contrast(item["row"].get("SeriesDescription"))
        if contrast == "nc":
            pre_candidates.append(item)
        elif contrast == "ce":
            post_candidates.append(item)

    # Sanity check: when both groups exist, every PRE must precede every POST.
    step1_valid = True
    if pre_candidates and post_candidates:
        max_pre = max(x["series_number"] for x in pre_candidates)
        min_post = min(x["series_number"] for x in post_candidates)
        step1_valid = max_pre < min_post

    if step1_valid:
        for item in pre_candidates:
            item["contrast"] = "nc"
        for item in post_candidates:
            item["contrast"] = "ce"

        # Step 3: infer contrast from adjacency to the first T1w scan whose
        # SeriesDescription explicitly identifies it as post-contrast.
        if post_candidates:
            first_contrast = min(
                item["series_number"] for item in post_candidates
            )
            for item in work:
                if norm(item["primary"]) not in T1W_TYPES:
                    continue
                if item["series_number"] is None:
                    continue

                if item["series_number"] < first_contrast:
                    item["contrast"] = "nc"
                elif item["series_number"] > first_contrast:
                    item["contrast"] = "ce"

    # Step 2: first DSC = DSC scan with the lowest valid SeriesNumber.
    dsc_numbers = [
        item["series_number"]
        for item in work
        if norm(item["primary"]) == "DSC" and item["series_number"] is not None
    ]

    if dsc_numbers:
        first_dsc = min(dsc_numbers)

        for item in work:
            if norm(item["primary"]) not in T1W_TYPES:
                continue
            if item["series_number"] is None:
                continue

            if item["series_number"] < first_dsc:
                item["contrast"] = "nc"
            elif item["series_number"] > first_dsc:
                item["contrast"] = "ce"

    # Step 4: final heuristic label.
    for item in work:
        primary = item["primary"]
        if norm(primary) in T1W_TYPES and item["contrast"]:
            heuristic_label = f"{primary}_{item['contrast']}"
        else:
            heuristic_label = primary

        item["row"]["heuristic_label"] = heuristic_label

    return [item["row"] for item in work]


def run(input_csv, output_csv):
    with open(input_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")

        required = {
            "labels1",
            "SeriesDescription",
            "experiment",
            "scan",
            "SliceThickness",
            "PixelSpacing",
            "ImageType",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        rows = list(reader)
        input_fieldnames = list(reader.fieldnames)

    # Preserve experiment order as first encountered and row order within each experiment.
    grouped = defaultdict(list)
    experiment_order = []

    for row in rows:
        experiment = row.get("experiment", "")
        if experiment not in grouped:
            experiment_order.append(experiment)
        grouped[experiment].append(row)

    output_rows = []
    for experiment in experiment_order:
        output_rows.extend(classify_experiment(grouped[experiment]))

    fieldnames = input_fieldnames + ["heuristic_label"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Apply heuristic MRI scan classification to an input CSV."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV file"
    )
    args = parser.parse_args()

    try:
        run(args.input, args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
