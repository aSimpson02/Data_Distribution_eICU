#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Where per-example predictions are written by eval_no_tune.py / tune_with_val.py
PREDS_DIR = Path("results/preds")
OUT_DIR   = Path("results/figs/calibration")

# What to try by default
SPLITS = ["random", "temporal", "hospital"]
MODELS = ["lr", "rf", "xgb", "lgbm"]
TAGS   = ["notuned", "none", "platt", "isotonic"]  # any subset may exist

def reliability_curve(probs, y, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    confs, accs = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            confs.append(probs[m].mean()); accs.append(y[m].mean())
    return np.array(confs), np.array(accs)

def find_pred_file(split, model, tag):
    # Prefer seed42, else first available seed
    cand = list(PREDS_DIR.glob(f"{split}_{model}_{tag}_seed42.csv"))
    if not cand:
        cand = sorted(PREDS_DIR.glob(f"{split}_{model}_{tag}_seed*.csv"))
    return cand[0] if cand else None

def plot_one(split, model, tag):
    f = find_pred_file(split, model, tag)
    if f is None:
        print(f"[SKIP] {split}/{model}/{tag} (no preds)"); return False
    df = pd.read_csv(f)
    if "y_true" not in df.columns or "p" not in df.columns:
        print(f"[SKIP] {f} missing columns y_true/p"); return False
    y = pd.to_numeric(df["y_true"], errors="coerce").values
    p = pd.to_numeric(df["p"], errors="coerce").values
    conf, acc = reliability_curve(p, y)

    out_dir = OUT_DIR / split / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"calibration_{split}_{model}_{tag}.png"

    plt.figure()
    plt.plot([0,1],[0,1], "--", label="perfect")
    plt.plot(conf, acc, "o-", label=f"{split}-{model}-{tag}")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Calibration: {split} / {model} ({tag})")
    plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=160); plt.close()
    print(f"[OK] {split}/{model}/{tag} -> {out} (from {f.name})")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
        help="Plot for all splits/models/tags discovered in results/preds/")
    ap.add_argument("--splits", nargs="*", default=["all"])
    ap.add_argument("--models", nargs="*", default=["all"])
    ap.add_argument("--tags",   nargs="*", default=["all"])
    args = ap.parse_args()

    splits = SPLITS if args.all or args.splits == ["all"] else args.splits
    models = MODELS if args.all or args.models == ["all"] else args.models
    tags   = TAGS   if args.all or args.tags   == ["all"] else args.tags

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    any_ok = False
    for s in splits:
        for m in models:
            for t in tags:
                ok = plot_one(s, m, t)
                any_ok = any_ok or ok

    if not any_ok:
        print("[WARN] No plots produced. Ensure per-example preds exist in results/preds/ "
              "(y_true,p). Re-run eval_no_tune/tune_with_val if needed.")

if __name__ == "__main__":
    main()
