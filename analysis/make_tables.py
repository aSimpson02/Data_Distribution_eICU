#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np

# ------------------------ Formatting helpers ------------------------

def fmt_float(x, digits=3):
    if pd.isna(x):
        return ""
    try:
        f = float(x)
    except Exception:
        return str(x)
    # show small p-values in scientific, otherwise fixed
    if (abs(f) < 1e-4) and (f != 0.0):
        return f"{f:.{digits}e}"
    return f"{f:.{digits}f}"

def fmt_pct(x, digits=1):
    if pd.isna(x):
        return ""
    try:
        f = float(x) * 100.0
    except Exception:
        return str(x)
    return f"{f:.{digits}f}\\%"

def write_tex(df: pd.DataFrame, outpath: str, caption: str, label: str, col_format=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # Default column format: left + right for rest
    if col_format is None:
        col_format = "l" + "r" * (len(df.columns) - 1)
    tex = df.to_latex(index=False,
                      escape=True,
                      longtable=False,
                      caption=caption,
                      label=label,
                      bold_rows=False,
                      column_format=col_format)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(tex)

# ------------------------ Builders ------------------------

def build_shift_signature(shift_dir: str, outdir: str, digits: int):
    rows = []
    for split in sorted(os.listdir(shift_dir)):
        sigp = os.path.join(shift_dir, split, f"{split}_shift_signature.csv")
        if not os.path.exists(sigp):
            continue
        df = pd.read_csv(sigp)
        df.columns = [c.strip().lower() for c in df.columns]
        # expected keys
        mean_js   = df.get("mean_js",   pd.Series([np.nan])).iloc[0]
        mean_ks   = df.get("mean_ks_cont", pd.Series([np.nan])).iloc[0]
        mean_w    = df.get("mean_wasserstein", pd.Series([np.nan])).iloc[0]
        p_src     = df.get("label_p_src", pd.Series([np.nan])).iloc[0]
        p_tgt     = df.get("label_p_tgt", pd.Series([np.nan])).iloc[0]
        p_delta   = df.get("label_delta", pd.Series([np.nan])).iloc[0]
        pval      = df.get("label_pval",  pd.Series([np.nan])).iloc[0]

        rows.append({
            "Split": split.capitalize(),
            "Mean JS": fmt_float(mean_js, digits),
            "Mean KS (cont.)": fmt_float(mean_ks, digits),
            "Mean Wasserstein": fmt_float(mean_w, digits),
            "Label p(src)": fmt_pct(p_src, digits),
            "Label p(tgt)": fmt_pct(p_tgt, digits),
            "Δ label": fmt_pct(p_delta, digits),
            "p-value": fmt_float(pval, digits),
        })

    if not rows:
        return

    out = pd.DataFrame(rows)
    out = out[["Split", "Mean JS", "Mean KS (cont.)", "Mean Wasserstein",
               "Label p(src)", "Label p(tgt)", "Δ label", "p-value"]]

    write_tex(
        out,
        os.path.join(outdir, "shift_signature.tex"),
        caption="Shift signatures across splits.",
        label="tab:shift-signature",
        col_format="lrrrrrrr"
    )

def build_topdrifting(shift_dir: str, outdir: str, top_n: int, digits: int):
    for split in sorted(os.listdir(shift_dir)):
        top = os.path.join(shift_dir, split, f"{split}_topdrifting.csv")
        if not os.path.exists(top):
            continue
        df = pd.read_csv(top)
        df.columns = [c.strip() for c in df.columns]
        # Standardise column names that might appear
        # Expected: feature, type, JS, KS, KS_p, chi2, chi2_p, Wasserstein, KS_sig_FDR, CHI2_sig_FDR
        # Keep a concise subset
        keep_cols = []
        if "feature" in df.columns: keep_cols.append("feature")
        if "type" in df.columns: keep_cols.append("type")
        if "JS" in df.columns: keep_cols.append("JS")
        if "KS" in df.columns: keep_cols.append("KS")
        if "chi2" in df.columns: keep_cols.append("chi2")
        if "Wasserstein" in df.columns: keep_cols.append("Wasserstein")
        if "KS_sig_FDR" in df.columns: keep_cols.append("KS_sig_FDR")
        if "CHI2_sig_FDR" in df.columns: keep_cols.append("CHI2_sig_FDR")

        df = df[keep_cols] if keep_cols else df
        # Sort by JS desc and trim
        if "JS" in df.columns:
            df = df.sort_values("JS", ascending=False)
        df = df.head(top_n).reset_index(drop=True)

        # Format
        rename = {
            "feature": "Feature",
            "type": "Type",
            "JS": "JS",
            "KS": "KS",
            "chi2": "$\\chi^2$",
            "Wasserstein": "Wasserstein",
            "KS_sig_FDR": "KS (FDR)",
            "CHI2_sig_FDR": "$\\chi^2$ (FDR)"
        }
        df = df.rename(columns=rename)

        for c in ("JS","KS","Wasserstein","$\\chi^2$"):
            if c in df.columns:
                df[c] = df[c].apply(lambda x: fmt_float(x, digits))

        # Bool → ✓/blank
        for c in ("KS (FDR)","$\\chi^2$ (FDR)"):
            if c in df.columns:
                df[c] = df[c].map(lambda v: "$\\checkmark$" if bool(v) else "")

        # Write table
        write_tex(
            df,
            os.path.join(outdir, f"topdrifting_{split}.tex"),
            caption=f"Top drifting features (by JS) on the {split.capitalize()} split.",
            label=f"tab:topdrifting-{split}",
            col_format="l" + "r"*(len(df.columns)-1)
        )

def build_missingness(shift_dir: str, outdir: str, top_n: int, digits: int):
    for split in sorted(os.listdir(shift_dir)):
        mis = os.path.join(shift_dir, split, f"{split}_missingness_drift.csv")
        if not os.path.exists(mis):
            continue
        df = pd.read_csv(mis)
        # Sort by absolute delta and trim
        df["abs_delta"] = df["delta_missing"].abs()
        df = df.sort_values("abs_delta", ascending=False).head(top_n).reset_index(drop=True)
        df = df[["feature","train_missing","test_missing","delta_missing"]]

        # Format %
        df = df.rename(columns={
            "feature": "Feature",
            "train_missing": "Missing (src)",
            "test_missing": "Missing (tgt)",
            "delta_missing": "$\\Delta$ missing"
        })
        for c in ("Missing (src)","Missing (tgt)","$\\Delta$ missing"):
            df[c] = df[c].apply(lambda x: fmt_pct(x, digits))

        write_tex(
            df,
            os.path.join(outdir, f"missingness_{split}.tex"),
            caption=f"Features with largest missingness drift on the {split.capitalize()} split.",
            label=f"tab:missingness-{split}",
            col_format="lrrr"
        )

def build_perf_link(perf_link_csv: str, outdir: str, digits: int):
    if not os.path.exists(perf_link_csv):
        return
    df = pd.read_csv(perf_link_csv)
    # Keep concise columns if present
    order = [c for c in (
        "split","model","calibrator",
        "mean_js","mean_ks",
        "delta_auroc_random","delta_ece_random",
        "delta_auroc_notuned","delta_ece_notuned"
    ) if c in df.columns]
    if not order:
        return
    df = df[order].copy()

    # Formatting
    if "split" in df.columns:
        df["split"] = df["split"].str.capitalize()
    for c in ("mean_js","mean_ks",
              "delta_auroc_random","delta_ece_random",
              "delta_auroc_notuned","delta_ece_notuned"):
        if c in df.columns:
            df[c] = df[c].apply(lambda x: fmt_float(x, digits))

    rename = {
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
    df = df.rename(columns=rename)

    write_tex(
        df,
        os.path.join(outdir, "perf_link.tex"),
        caption="Shift magnitude and performance deltas by split, model, and calibrator.",
        label="tab:perf-link",
        col_format="lll" + "r"*(len(df.columns)-3)
    )

# ------------------------ CLI ------------------------

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

    print(f"[OK] LaTeX tables written to: {args.outdir}")

if __name__ == "__main__":
    main()
