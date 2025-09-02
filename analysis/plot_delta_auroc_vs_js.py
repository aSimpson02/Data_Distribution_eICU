#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import scipy.stats as st
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def pick_delta_auroc(df: pd.DataFrame) -> pd.Series:
    """
    Prefer delta_auroc_random. If NaN or missing, use delta_auroc_notuned.
    Returns a pandas Series with chosen deltas.
    """
    y = pd.Series(np.nan, index=df.index, dtype=float)
    if "delta_auroc_random" in df.columns:
        y = df["delta_auroc_random"].astype(float)
    if "delta_auroc_notuned" in df.columns:
        # fill NaNs in y with notuned deltas if available
        y = y.where(~y.isna(), df["delta_auroc_notuned"].astype(float))
    return y

def scatter_with_fit(x, y, labels, title, outfile):
    plt.figure(figsize=(6, 4.5))

    # Scatter
    plt.scatter(x, y, s=42, alpha=0.85)
    # Annotate each point with model-calibrator
    for (xi, yi, lbl) in zip(x, y, labels):
        if np.isfinite(xi) and np.isfinite(yi):
            plt.annotate(lbl, (xi, yi), fontsize=8, alpha=0.75,
                         textcoords="offset points", xytext=(4, 2))

    # Fit simple least-squares line if we have at least 2 points
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if finite_mask.sum() >= 2:
        coef = np.polyfit(x[finite_mask], y[finite_mask], deg=1)
        xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        yy = coef[0] * xx + coef[1]
        plt.plot(xx, yy, linestyle="--", linewidth=1.5)

        # Correlations
        try:
            r, rp = (np.corrcoef(x[finite_mask], y[finite_mask])[0, 1],
                     np.nan)  # Pearson p-value not from corrcoef
        except Exception:
            r, rp = (np.nan, np.nan)
        if HAS_SCIPY:
            pr, pp = st.pearsonr(x[finite_mask], y[finite_mask])
            sr, sp = st.spearmanr(x[finite_mask], y[finite_mask])
            corr_text = f"Pearson r={pr:.3f} (p={pp:.2g}) | Spearman ρ={sr:.3f} (p={sp:.2g})"
        else:
            corr_text = f"Pearson r={r:.3f}"

        # Put correlation in the title footer
        title_full = f"{title}\n{corr_text}"
    else:
        title_full = title

    plt.title(title_full)
    plt.xlabel("Mean JS divergence")
    plt.ylabel("ΔAUROC (random − target)  [fallback: − notuned]")
    plt.grid(alpha=0.2, linestyle=":")
    plt.tight_layout()
    ensure_dir(os.path.dirname(outfile))
    plt.savefig(outfile, dpi=220)
    plt.close()
    print(f"[OK] Wrote {outfile}")

def build_and_plot(perf_link_csv: str, out_dir: str, splits: list):
    if not os.path.exists(perf_link_csv):
        raise FileNotFoundError(f"Missing perf_link CSV: {perf_link_csv}")

    df = pd.read_csv(perf_link_csv)
    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    if "split" not in df.columns:
        raise ValueError("perf_link.csv must contain a 'split' column.")

    # Only keep columns we need
    need_cols = ["split", "model", "calibrator", "mean_js"]
    for nc in ("delta_auroc_random", "delta_auroc_notuned"):
        if nc in df.columns:
            need_cols.append(nc)
    df = df[[c for c in need_cols if c in df.columns]].copy()

    # Label for points
    cal = df["calibrator"] if "calibrator" in df.columns else "none"
    df["label"] = df["model"].astype(str) + "-" + cal.astype(str)

    # y target series (ΔAUROC)
    y = pick_delta_auroc(df)
    df["delta_auroc_plot"] = y

    # Combined (Hospital + Temporal) — ignore Random rows
    mask_combined = df["split"].str.lower().isin([s.lower() for s in splits])
    dsub = df[mask_combined].dropna(subset=["mean_js", "delta_auroc_plot"])
    if not dsub.empty:
        scatter_with_fit(
            x=dsub["mean_js"].values.astype(float),
            y=dsub["delta_auroc_plot"].values.astype(float),
            labels=dsub["label"].values,
            title="ΔAUROC vs Mean JS (Hospital + Temporal)",
            outfile=os.path.join(out_dir, "delta_auroc_vs_js.png"),
        )
    else:
        print("[WARN] No non-NaN rows for combined Hospital + Temporal plot.")

    # Per-split plots
    for split in splits:
        dsplit = df[df["split"].str.lower() == split.lower()].dropna(
            subset=["mean_js", "delta_auroc_plot"]
        )
        if dsplit.empty:
            print(f"[WARN] No rows for split '{split}'. Skipping.")
            continue
        scatter_with_fit(
            x=dsplit["mean_js"].values.astype(float),
            y=dsplit["delta_auroc_plot"].values.astype(float),
            labels=dsplit["label"].values,
            title=f"ΔAUROC vs Mean JS ({split.capitalize()})",
            outfile=os.path.join(out_dir, split.lower(), "delta_auroc_vs_js.png"),
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf_link", default="results/shift/perf_link.csv")
    ap.add_argument("--out_dir", default="results/shift")
    ap.add_argument("--splits", nargs="+", default=["hospital", "temporal"],
                    help="Which splits to plot (default: hospital temporal).")
    args = ap.parse_args()

    build_and_plot(args.perf_link, args.out_dir, args.splits)

if __name__ == "__main__":
    main()
