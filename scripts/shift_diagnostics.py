#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, numpy as np, pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from src.data.io import load_three, drop_leaky

def _is_cat(s: pd.Series) -> bool:
    return s.dtype.name in ("category","object","bool")

def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(float) + eps; q = q.astype(float) + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))

def _hist_kl(x_src, x_tgt, bins=30) -> float:
    mn = np.nanmin([np.nanmin(x_src), np.nanmin(x_tgt)])
    mx = np.nanmax([np.nanmax(x_src), np.nanmax(x_tgt)])
    if not np.isfinite(mn) or not np.isfinite(mx) or mn==mx:
        return 0.0
    hist_src, edges = np.histogram(x_src, bins=bins, range=(mn, mx))
    hist_tgt, _     = np.histogram(x_tgt, bins=edges)
    return _kl(hist_src, hist_tgt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["random","temporal","hospital"])
    ap.add_argument("--label_col", default="hospital_mortality")
    ap.add_argument("--splits_dir", default="data/csv_splits")
    ap.add_argument("--outdir", default="results/shift")
    args = ap.parse_args()

    tr, va, te = load_three(args.split, args.splits_dir)
    tr = drop_leaky(tr, args.label_col)
    te = drop_leaky(te, args.label_col)

    rows = []
    for c in tr.columns:
        if c == args.label_col: 
            continue
        s_src = tr[c].dropna(); s_tgt = te[c].dropna()
        if len(s_src)==0 or len(s_tgt)==0: 
            continue
        if _is_cat(s_src) or _is_cat(s_tgt):
            cats = sorted(set(s_src.astype(str)).union(set(s_tgt.astype(str))))
            freq_src = pd.Series(s_src.astype(str)).value_counts().reindex(cats, fill_value=0).values
            freq_tgt = pd.Series(s_tgt.astype(str)).value_counts().reindex(cats, fill_value=0).values
            stat, p, _, _ = chi2_contingency(np.vstack([freq_src, freq_tgt]))
            d_ks = np.nan
            d_kl = _kl(freq_src, freq_tgt)
        else:
            # numeric
            s_src = pd.to_numeric(s_src, errors="coerce").dropna()
            s_tgt = pd.to_numeric(s_tgt, errors="coerce").dropna()
            d_ks = ks_2samp(s_src, s_tgt).statistic
            d_kl = _hist_kl(s_src.values, s_tgt.values)
            p = np.nan
        rows.append({"feature": c, "KS": d_ks, "KL": d_kl, "p_chi2": p})
    df = pd.DataFrame(rows).sort_values(["KS","KL"], ascending=False)

    prev_src = float(tr[args.label_col].mean())
    prev_tgt = float(te[args.label_col].mean())

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{args.split}_feature_shift.csv", index=False)

    sig = pd.DataFrame([{
        "split": args.split,
        "n_features": len(df),
        "mean_KS": float(np.nanmean(df["KS"])) if "KS" in df else np.nan,
        "mean_KL": float(np.nanmean(df["KL"])) if "KL" in df else np.nan,
        "label_prev_src": prev_src,
        "label_prev_tgt": prev_tgt,
        "delta_prev": prev_tgt - prev_src
    }])
    sig.to_csv(outdir / f"{args.split}_summary.csv", index=False)
    print(f"[OK] wrote {outdir}/{args.split}_feature_shift.csv and {args.split}_summary.csv")

if __name__ == "__main__":
    main()
