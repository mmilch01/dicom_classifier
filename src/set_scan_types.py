"""
Set XNAT scan type fields from a classifier CSV.

Type I  (run_classifier_xnat output):  one row per scan.
  Required columns: experiment, scan, <label column>
  Each row sets one scan's type to the value in <label column>.

Type II (run_heuristic_classifier output): one row per experiment.
  Required columns: experiment, T1nc, T1ce, T2, FLAIR (scan numbers)
  Each of those columns yields: scan_number -> type = column name.
"""

import argparse
import math
import sys

import pandas as pd
from pyxnat import Interface

TYPE2_COLS = {"T1nc", "T1ce", "T2", "FLAIR"}
TYPE2_LABELS = ["T1nc", "T1ce", "T2", "FLAIR"]


def detect_csv_type(df: pd.DataFrame) -> int:
    if TYPE2_COLS.issubset(df.columns):
        return 2
    if {"experiment", "scan"}.issubset(df.columns):
        return 1
    raise SystemExit(
        f"Cannot determine CSV type.\n"
        f"Type I requires columns: experiment, scan, <label>.\n"
        f"Type II requires columns: experiment, T1nc, T1ce, T2, FLAIR.\n"
        f"Found: {list(df.columns)}"
    )


def _is_valid_scan(v) -> bool:
    try:
        if v is None or pd.isna(v):
            return False
        if isinstance(v, str) and not v.strip():
            return False
        return True
    except Exception:
        return False


def _scan_id_to_str(v) -> str:
    if isinstance(v, float) and math.isfinite(v) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def filter_experiment(df: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    if not experiment_id:
        return df
    if "experiment" not in df.columns:
        raise SystemExit("Cannot filter by experiment: CSV has no 'experiment' column.")

    experiment_values = df["experiment"].astype(str).str.strip()
    filtered = df[experiment_values == experiment_id]
    if not filtered.empty:
        print(f"Filtered to experiment '{experiment_id}' ({len(filtered)} rows).")
        return filtered

    if experiment_values.isin(["", "NA", "N/A", "nan", "None"]).all():
        df = df.copy()
        df["experiment"] = experiment_id
        print(f"Using experiment '{experiment_id}' for all {len(df)} rows.")
        return df

    if filtered.empty:
        raise SystemExit(f"No rows found for experiment '{experiment_id}'.")


class XnatClient:
    def __init__(self, server: str, user: str, password: str):
        # pyxnat Interface expects server like "https://xnat.example.org"
        self.session = Interface(server=server, user=user, password=password)

    def set_scan_type(self, project_id: str, experiment_id: str, scan_id: str, label: str) -> None:
        scan = self.session.select.project(project_id).experiment(experiment_id).scan(scan_id)
        scan.attrs.set("type", label)


def apply_type1(
    df: pd.DataFrame,
    label_column: str,
    client: XnatClient,
    project_id: str,
    verbose: bool,
):
    n_done = n_fail = 0
    total = len(df)
    for _, row in df.iterrows():
        experiment_id = str(row["experiment"])
        scan_id = _scan_id_to_str(row["scan"])
        label = str(row[label_column])
        try:
            client.set_scan_type(project_id, experiment_id, scan_id, label)
            n_done += 1
            if verbose:
                print(f"  {experiment_id}/{scan_id} -> {label}")
            elif n_done % 100 == 0:
                print(f"  Updated {n_done}/{total} scans...")
        except Exception as e:
            print(f"WARNING: {experiment_id}/{scan_id}: {e}")
            n_fail += 1

    return n_done, n_fail


def apply_type2(
    df: pd.DataFrame,
    client: XnatClient,
    project_id: str,
    verbose: bool,
):
    n_done = n_fail = 0
    total = sum(
        1
        for _, row in df.iterrows()
        for col in TYPE2_LABELS
        if col in row.index and _is_valid_scan(row[col])
    )
    for _, row in df.iterrows():
        experiment_id = str(row["experiment"])
        for col_label in TYPE2_LABELS:
            if col_label not in row.index:
                continue
            raw = row[col_label]
            if not _is_valid_scan(raw):
                continue
            scan_id = _scan_id_to_str(raw)
            try:
                client.set_scan_type(project_id, experiment_id, scan_id, col_label)
                n_done += 1
                if verbose:
                    print(f"  {experiment_id}/{scan_id} -> {col_label}")
                elif n_done % 100 == 0:
                    print(f"  Updated {n_done}/{total} scans...")
            except Exception as e:
                print(f"WARNING: {experiment_id}/{scan_id}: {e}")
                n_fail += 1
    return n_done, n_fail


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Set XNAT scan types from a classifier CSV (type I or II)."
    )
    
    ap.add_argument("-i", "--input", required=True, help="Input CSV file")
    ap.add_argument("--server", required=True, help="XNAT server URL")
    ap.add_argument("--user", required=True, help="XNAT username")
    ap.add_argument("--password", required=True, help="XNAT password")
    ap.add_argument("--project", required=True, help="XNAT project ID")
    ap.add_argument("--experiment", default="", help="Only update rows for this XNAT experiment/session")
    ap.add_argument(
        "--label-column",
        default="labels1",
        help="Label column for type I CSV [labels1]",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = filter_experiment(df, args.experiment)
    csv_type = detect_csv_type(df)
    print(f"Detected CSV type {csv_type} ({len(df)} rows).")

    client = XnatClient(server=args.server, user=args.user, password=args.password)

    if csv_type == 1:
        if args.label_column not in df.columns:
            raise SystemExit(
                f"Label column '{args.label_column}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )
        print(f"Using label column: '{args.label_column}'")
        n_done, n_fail = apply_type1(
            df, args.label_column, client, args.project, args.verbose
        )
    else:
        n_done, n_fail = apply_type2(df, client, args.project, args.verbose)

    print(f"Done. Updated {n_done} scans, {n_fail} failures.")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
