#!/usr/bin/env python3
"""
Select (T1nc, T1ce, T2, FLAIR) per experiment from a scan-list CSV/XLSX and write 1 row/experiment.

Example:
  python select_mri4.py -i scans.xlsx -o selected_scans.csv
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Parsing helpers
# ----------------------------

def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def parse_pixel_spacing(v: Any) -> Optional[Tuple[float, float]]:
    """
    Accepts formats like:
      - [0.9, 0.9]
      - "0.9\\0.9"
      - "0.9,0.9"
      - "(0.9, 0.9)"
      - "['0.9','0.9']"
    """
    if _is_nan(v):
        return None

    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return None

    s = str(v).strip()
    if not s:
        return None

    # list-like
    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, (list, tuple)) and len(lst) >= 2:
                return float(lst[0]), float(lst[1])
        except Exception:
            pass

    # split by common separators
    parts = re.split(r"[\\,/;\s]+", s.strip("()[]"))
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1])
        except Exception:
            return None
    return None


def parse_float(v: Any) -> Optional[float]:
    if _is_nan(v):
        return None
    try:
        s = str(v).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def normalize_imagetype(v: Any) -> str:
    """
    Returns uppercase '/'-joined tokens, e.g. "ORIGINAL/PRIMARY/M_IR/M/IR".
    Accepts:
      - list/tuple
      - "['ORIGINAL', 'PRIMARY', ...]"
      - "ORIGINAL\\PRIMARY\\..."
      - "ORIGINAL/PRIMARY/..."
    """
    if _is_nan(v):
        return ""
    if isinstance(v, (list, tuple)):
        toks = [str(t).strip() for t in v if str(t).strip()]
        return "/".join(toks).upper()

    s = str(v).strip()
    if not s:
        return ""

    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, (list, tuple)):
                toks = [str(t).strip() for t in lst if str(t).strip()]
                return "/".join(toks).upper()
        except Exception:
            pass

    s = s.replace("\\", "/")
    toks = [t.strip() for t in s.split("/") if t.strip()]
    return "/".join(toks).upper()


def imtype_compatible(imtype_norm: str) -> bool:
    return imtype_norm.startswith("ORIGINAL/PRIMARY") and not imtype_norm.endswith("/SUB")

def is_dsc_perf(imtype_norm: str) -> bool:
    # rule: ORIGINAL/PRIMARY/PERFUSION (token-level)
    return imtype_norm.startswith("ORIGINAL/PRIMARY/PERFUSION")


def min_dimension(ps: Optional[Tuple[float, float]], st: Optional[float]) -> Optional[float]:
    if ps is None or st is None:
        return None
    return min(ps[0], ps[1], st)


def series_has(sd: str, needle: str) -> bool:
    return needle.upper() in (sd or "").upper()


def series_endswith(sd: str, suffix: str) -> bool:
    return (sd or "").upper().endswith(suffix.upper())


# ----------------------------
# Classification & scoring
# ----------------------------

def score_t1(label1: str, series: str) -> int:
    s = 0
    l1 = (label1 or "").upper()
    sd = series or ""
    if l1 in {"T1HI", "MPRAGE"}:
        s += 10
    if series_has(sd, "_T1_") or series_has(sd, "MPRAGE") or series_endswith(sd, "_T1"):
        s += 5
    return s


def score_t2(label1: str, series: str) -> int:
    s = 0
    l1 = (label1 or "").upper()
    sd = series or ""
    if l1 in {"T2HI", "T2LO"}:
        s += 10
    if series_has(sd, "_T2_") or series_endswith(sd, "_T2"):
        s += 5
    return s


def score_flair(label1: str, series: str) -> int:
    s = 0
    l1 = (label1 or "").upper()
    sd = series or ""
    if l1 == "T2FLAIR":
        s += 10
    if series_has(sd, "FLAIR"):
        s += 5
    return s


def is_pre(series: str) -> bool:
    sd = (series or "").upper()
    return sd.endswith("_PRE") or ("_PRE_" in sd)


def is_post(series: str) -> bool:
    sd = (series or "").upper()
    return sd.endswith("_POST") or ("_POST_" in sd)


def passes_t1t2_qc(ps: Optional[Tuple[float, float]], st: Optional[float], max_slice_thickness: float = 2.5) -> bool:
    if ps is None or st is None:
        return False
    if ps[0] > 1.5 or ps[1] > 1.5:
        return False
    if st > max_slice_thickness:
        return False
    return True


# ----------------------------
# Selection logic
# ----------------------------

def pick_best(
    candidates: List[Dict[str, Any]],
    likelihood_col: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Primary key:
      - if likelihood_col present: higher likelihood_col
      - else: higher computed_score
    Tie-break:
      - higher min_dim (as requested: "highest minimum dimension")
      - then smaller scan number
    """
    if not candidates:
        return None

    def key(c: Dict[str, Any]) -> Tuple[float, float, int]:
        # likelihood
        lk = None
        if likelihood_col and likelihood_col in c and c[likelihood_col] is not None:
            try:
                lk = float(c[likelihood_col])
            except Exception:
                lk = None
        lk_val = lk if lk is not None else float(c.get("computed_score", 0))

        md = c.get("min_dim")
        md_val = float(md) if md is not None else float("-inf")

        #scn = c.get("scan_int", 10**9)
        return (lk_val, md_val)  # -scn => prefer smaller scan if ties

    return max(candidates, key=key)


def autodetect_likelihood_col(df: pd.DataFrame) -> Optional[str]:
    # common names; prioritize explicit ones
    for col in ["likelihood", "score", "prob", "probs1", "confidence", "pred_margin_confidence"]:
        if col in df.columns:
            return col
    return None

# ----------------------------
# Main
# ----------------------------

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input CSV/XLSX sorted by experiment")
    ap.add_argument("-o", "--output", required=True, help="Output CSV (1 row per experiment)")
    ap.add_argument("--experiment-col", default="experiment")
    ap.add_argument("--scan-col", default="scan")
    ap.add_argument("--series-col", default="SeriesDescription")
    ap.add_argument("--imagetype-col", default="ImageType")
    ap.add_argument("--pixelspacing-col", default="PixelSpacing")
    ap.add_argument("--slicethickness-col", default="SliceThickness")
    ap.add_argument("--label1-col", default="labels1")
    ap.add_argument("--likelihood-col", default="", help="Optional: override likelihood column name")
    ap.add_argument("--t2w-min-slice-thickness", type=float, default=2.5, help="Maximum allowed T2w slice thickness in mm [2.5]")
    args = ap.parse_args()

    df = read_table(args.input)

    require_cols(df, [
        args.experiment_col,
        args.scan_col,
        args.series_col,
        args.imagetype_col,
        args.pixelspacing_col,
        args.slicethickness_col,
        args.label1_col,
    ])

    likelihood_col = None

    out_rows: List[Dict[str, Any]] = []

    # Group by experiment (csv assumed sorted, but groupby is safe)
    n,nfound=0,0
    for exp, g in df.groupby(args.experiment_col, sort=False):
        exp_str = str(exp)
        subject = exp_str[:9]
        n=n+1
        scans: List[Dict[str, Any]] = []
        for _, r in g.iterrows():
            scan_raw = r[args.scan_col]
            try:
                scan_int = int(scan_raw)
            except Exception:
                # skip if scan number not parseable
                continue

            ps = parse_pixel_spacing(r[args.pixelspacing_col])
            st = parse_float(r[args.slicethickness_col])
            sd = "" if _is_nan(r[args.series_col]) else str(r[args.series_col])
            l1 = "" if _is_nan(r[args.label1_col]) else str(r[args.label1_col])

            imtype = normalize_imagetype(r[args.imagetype_col])
            if not imtype_compatible(imtype):
                continue

            md = min_dimension(ps, st)

            item = {
                "experiment": exp_str,
                "subject": subject,
                "scan_raw": scan_raw,
                "scan_int": scan_int,
                "SeriesDescription": sd,
                "label1": l1,
                "PixelSpacing": ps,
                "SliceThickness": st,
                "ImageType_norm": imtype,
                "min_dim": md,
            }

            if likelihood_col and likelihood_col in r.index and not _is_nan(r[likelihood_col]):
                try:
                    item[likelihood_col] = float(r[likelihood_col])
                except Exception:
                    item[likelihood_col] = None

            scans.append(item)

        if not scans:
            continue

        # DSC detection (take the first by scan number)
        dsc_scans = sorted([s for s in scans if is_dsc_perf(s["ImageType_norm"])], key=lambda x: x["scan_int"])
        n_dsc = dsc_scans[0]["scan_int"] if dsc_scans else None

        # Candidate buckets
        t1w: List[Dict[str, Any]] = []
        t2w: List[Dict[str, Any]] = []
        flr: List[Dict[str, Any]] = []

        for s in scans:
            ps, st = s["PixelSpacing"], s["SliceThickness"]
            sd, l1 = s["SeriesDescription"], s["label1"]

            # FLAIR
            sf = score_flair(l1, sd)
            if sf > 0 and s["min_dim"] is not None:
                c = dict(s)
                c["computed_score"] = sf
                flr.append(c)
                continue

            st1 = score_t1(l1, sd)
            if st1 > 0 and passes_t1t2_qc(ps, st):
                c = dict(s)
                c["computed_score"] = st1
                t1w.append(c)

            st2 = score_t2(l1, sd)
            if st2 > 0 and passes_t1t2_qc(ps, st, max_slice_thickness=args.t2w_min_slice_thickness):
                c = dict(s)
                c["computed_score"] = st2
                t2w.append(c)

        # Need T2 and FLAIR candidates
        best_t2 = pick_best(t2w, likelihood_col)
        best_fl = pick_best(flr, likelihood_col)
        if best_t2 is None or best_fl is None:
            continue

        # Split T1w into nc/ce using DSC override, else PRE/POST, else fallback by scan order
        t1nc_cands: List[Dict[str, Any]] = []
        t1ce_cands: List[Dict[str, Any]] = []

        if n_dsc is not None:
            for c in t1w:
                if c["scan_int"] < n_dsc:
                    t1nc_cands.append(c)
                elif c["scan_int"] > n_dsc:
                    t1ce_cands.append(c)
                # if equal, ignore
        else:
            for c in t1w:
                sd = c["SeriesDescription"]
                if is_pre(sd) and not is_post(sd):
                    t1nc_cands.append(c)
                elif is_post(sd) and not is_pre(sd):
                    t1ce_cands.append(c)

            # If still ambiguous, fallback: earliest => nc, latest => ce
            if (not t1nc_cands or not t1ce_cands) and len(t1w) >= 2:
                t1_sorted = sorted(t1w, key=lambda x: x["scan_int"])
                # allow multiple nc/ce cands from ends to keep scoring working
                t1nc_cands = t1nc_cands or [t1_sorted[0]]
                t1ce_cands = t1ce_cands or [t1_sorted[-1]]

        best_t1nc = pick_best(t1nc_cands, likelihood_col)
        best_t1ce = pick_best(t1ce_cands, likelihood_col)
        if best_t1nc is None or best_t1ce is None:
            continue

        # Sanity check: T1nc scan number < T1ce scan number
        if not (best_t1nc["scan_int"] < best_t1ce["scan_int"]):
            continue

        def fmt_ps(ps: Optional[Tuple[float, float]]) -> str:
            if ps is None:
                return ""
            return f"{ps[0]:g}\\{ps[1]:g}"

        out_rows.append({
            "subject": subject,
            "experiment": exp_str,

            "T1nc": best_t1nc["scan_int"],
            "T1nc_PixelSpacing": fmt_ps(best_t1nc["PixelSpacing"]),
            "T1nc_SeriesDescription": best_t1nc["SeriesDescription"],
            "T1nc_SliceThickness": best_t1nc["SliceThickness"],

            "T1ce": best_t1ce["scan_int"],
            "T1ce_PixelSpacing": fmt_ps(best_t1ce["PixelSpacing"]),
            "T1ce_SeriesDescription": best_t1ce["SeriesDescription"],
            "T1ce_SliceThickness": best_t1ce["SliceThickness"],

            "T2": best_t2["scan_int"],
            "T2_PixelSpacing": fmt_ps(best_t2["PixelSpacing"]),
            "T2_SeriesDescription": best_t2["SeriesDescription"],
            "T2_SliceThickness": best_t2["SliceThickness"],

            "FLAIR": best_fl["scan_int"],
            "FLAIR_PixelSpacing": fmt_ps(best_fl["PixelSpacing"]),
            "FLAIR_SeriesDescription": best_fl["SeriesDescription"],
            "FLAIR_SliceThickness": best_fl["SliceThickness"],
        })
        nfound=nfound+1
        if nfound % 100 == 0: 
            print(f'found 100 more matching experiments, nfound={nfound}')
        #DEBUG: stop after 10 successful.
        #if n>9: break

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
