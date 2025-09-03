#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd




def fmt_float(x, digits=3):
    """Format a float with fixed digits; scientific for very small non-zero."""
    if pd.isna(x):
        return ""
    try:
        f = float(x)
    except Exception:
        return str(x)
    if (abs(f) < 1e-4) and (f != 0.0):
        return f"{f:.{digits}e}"
    return f"{f:.{digits}f}"


def fmt_pct(x, digits=3):
    """Format a proportion 0..1 as percentage with \%."""
    if pd.isna(x):
        return ""
    try:
        f = float(x) * 100.0
    except Exception:
        return str(x)
    return f"{f:.{digits}f}\\%"


def write_tabular(df: pd.DataFrame, outpath: str, col_format=None, escape=False):
    """
    Write LaTeX tabular only (no table/caption/label) using booktabs commands.
    Caller (main.tex) should own the table float.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if col_format is None:
        col_format = "l" + "r" * (len(df.columns) - 1)
    tex = df.to_latex(index=False,
                      escape=escape,         # allow math in headers ($\Delta$)
                      longtable=False,
                      bold_rows=False,
                      column_format=col_format)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(tex)




def build_shift_signature(shift_dir: str, outdir: str, digits: int):
    rows = []
    for split in sorted(os.listdir(shift_dir)):
        sigp = os.path.join(shift_dir, split, f"{split}_shift_signature.csv")
        if not os.path.exists(sigp):
            continue
        df = pd.read_csv(sigp)
        df.columns = [c.strip().lower() for c in df.columns]
        rows.append({
            "Split": split.capitalize(),
            "Mean JS": fmt_float(df.get("mean_js", [np.nan])[0], digits),
            "Mean KS (cont.)": fmt_float(df.get("mean_ks_cont", [np.nan])[0], digits),
            "Mean Wasserstein": fmt_float(df.get("mean_wasserstein", [np.nan])[0], digits),
            "Label p(src)": fmt_pct(df.get("label_p_src", [np.nan])[0], digits),
            "Label p(tgt)": fmt_pct(df.get("label_p_tgt", [np.nan])[0], digits),
            "$\\Delta$ label": fmt_pct(df.get("label_delta", [np.nan])[0], digits),
            "p-value": fmt_float(df.get("label_pval", [np.nan])[0], digits),
        })

    if not rows:
        return

    out = pd.DataFrame(rows)
    # keep a stable column order
    out = out[[
        "Split",
        "Mean JS",
        "Mean KS (cont.)",
        "Mean Wasserstein",
        "Label p(src)",
        "Label p(tgt)",
        "$\\Delta$ label",
        "p-value"
    ]]

    write_tabular(
        out,
        os.path.join(outdir, "shift_signature.tex"),
        col_format="lrrrrrrr",
        escape=False
    )


def build_topdrifting(shift_dir: str, outdir: str, top_n: int, digits: int):
    for split in sorted(os.listdir(shift_dir)):
        top = os.path.join(shift_dir, split, f"{split}_topdrifting.csv")
        if not os.path.exists(top):
            continue
        df = pd.read_csv(top)
        df.columns = [c.strip() for c in df.columns]

        # expected columns; keep concise subset
        keep_cols = []
        if "feature" in df.columns: keep_cols.append("feature")
        if "type" in df.columns: keep_cols.append("type")
        if "JS" in df.columns: keep_cols.append("JS")
        if "KS" in df.columns: keep_cols.append("KS")
        if "Wasserstein" in df.columns: keep_cols.append("Wasserstein")
        if "KS_sig_FDR" in df.columns: keep_cols.append("KS_sig_FDR")
        if "CHI2_sig_FDR" in df.columns: keep_cols.append("CHI2_sig_FDR")

        if keep_cols:
            df = df[keep_cols]
        # Sort by JS desc and trim
        if "JS" in df.columns:
            df = df.sort_values("JS", ascending=False)
        df = df.head(top_n).reset_index(drop=True)

        # Rename headers for publication
        rename = {
            "feature": "Feature",
            "type": "Type",
            "JS": "JS",
            "KS": "KS",
            "Wasserstein": "Wasserstein",
            "KS_sig_FDR": "KS (FDR)",
            "CHI2_sig_FDR": "$\\chi^2$ (FDR)"
        }
        df = df.rename(columns=rename)

        # Format numerics
        for c in ("JS", "KS", "Wasserstein"):
            if c in df.columns:
                df[c] = df[c].apply(lambda x: fmt_float(x, digits))

        # Bool → checkmark/blank
        for c in ("KS (FDR)", "$\\chi^2$ (FDR)"):
            if c in df.columns:
                df[c] = df[c].map(lambda v: "$\\checkmark$" if bool(v) else "")

        write_tabular(
            df,
            os.path.join(outdir, f"topdrifting_{split}.tex"),
            col_format="l" + "r" * (len(df.columns) - 1),
            escape=False
        )


def build_missingness(shift_dir: str, outdir: str, top_n: int, digits: int):
    for split in sorted(os.listdir(shift_dir)):
        mis = os.path.join(shift_dir, split, f"{split}_missingness_drift.csv")
        if not os.path.exists(mis):
            continue
        df = pd.read_csv(mis)
        # Sort by absolute delta_missing and trim
        if "delta_missing" not in df.columns:
            continue
        df["abs_delta"] = df["delta_missing"].abs()
        df = df.sort_values("abs_delta", ascending=False).head(top_n).reset_index(drop=True)

        # restrict to tidy columns
        cols = ["feature", "train_missing", "test_missing", "delta_missing"]
        df = df[[c for c in cols if c in df.columns]]

        df = df.rename(columns={
            "feature": "Feature",
            "train_missing": "Missing (src)",
            "test_missing": "Missing (tgt)",
            "delta_missing": "$\\Delta$ missing"
        })

        # Format all as percentages
        for c in ("Missing (src)", "Missing (tgt)", "$\\Delta$ missing"):
            if c in df.columns:
                df[c] = df[c].apply(lambda x: fmt_pct(x, digits))

        write_tabular(
            df,
            os.path.join(outdir, f"missingness_{split}.tex"),
            col_format="lrrr",
            escape=False
        )


def build_perf_link(perf_link_csv: str, outdir: str, digits: int):
    if not os.path.exists(perf_link_csv):
        return
    df = pd.read_csv(perf_link_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    # choose/keep a clean set if present
    keep = [c for c in (
        "split", "model", "calibrator",
        "mean_js", "mean_ks",
        "delta_auroc_random", "delta_ece_random",
        "delta_auroc_notuned", "delta_ece_notuned"
    ) if c in df.columns]
    if not keep:
        return
    df = df[keep].copy()

    # Pretty headers / rename
    rename_map = {
        "split": "Split",
        "model": "Model",
        "calibrator": "Cal.",
        "mean_js": "Mean JS",
        "mean_ks": "Mean KS",
        "delta_auroc_random": "$\\Delta$AUROC (rnd)",
        "delta_ece_random": "$\\Delta$ECE (rnd)",
        "delta_auroc_notuned": "$\\Delta$AUROC ($-$notuned)",
        "delta_ece_notuned": "$\\Delta$ECE ($-$notuned)"
    }
    df = df.rename(columns=rename_map)

    # Capitalize split
    if "Split" in df.columns:
        df["Split"] = df["Split"].astype(str).str.capitalize()

    # Format floats
    for c in ("Mean JS", "Mean KS",
              "$\\Delta$AUROC (rnd)", "$\\Delta$ECE (rnd)",
              "$\\Delta$AUROC ($-$notuned)", "$\\Delta$ECE ($-$notuned)"):
        if c in df.columns:
            df[c] = df[c].apply(lambda x: fmt_float(x, digits))

    # Use hyphenated filename to match your main.tex
    outpath = os.path.join(outdir, "perf-link.tex")
    # Column format: 3 left + rest right
    col_format = "lll" + "r" * (len(df.columns) - 3)
    write_tabular(df, outpath, col_format=col_format, escape=False)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift_dir", default="results/shift")
    ap.add_argument("--perf_link", default="results/shift/perf_link.csv")
    ap.add_argument("--outdir", default="results/tables")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--float_digits", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    build_shift_signature(args.shift_dir, args.outdir, args.float_digits)
    build_topdrifting(args.shift_dir, args.outdir, args.top_n, args.float_digits)
    build_missingness(args.shift_dir, args.outdir, args.top_n, args.float_digits)
    build_perf_link(args.perf_link, args.outdir, args.float_digits)

    print(f"[OK] LaTeX tabulars written to: {args.outdir}")


if __name__ == "__main__":
    main()
