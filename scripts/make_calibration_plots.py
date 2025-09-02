#!/usr/bin/env python3
from pathlib import Path
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx  = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    confs, accs = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any(): confs.append(probs[m].mean()); accs.append(y[m].mean())
    return np.array(confs), np.array(accs)

def find_pred_file(preds_dir: Path, split: str, model: str, tag: str):
    cand = list((preds_dir).glob(f"{split}_{model}_{tag}_seed42.csv"))
    if not cand:
        cand = sorted((preds_dir).glob(f"{split}_{model}_{tag}_seed*.csv"))
    return cand[0] if cand else None

def plot_one(preds_dir: Path, out_dir: Path, split: str, model: str, tag: str) -> bool:
    f = find_pred_file(preds_dir, split, model, tag)
    if f is None:
        print(f"[SKIP] {split}/{model}/{tag} (no preds)"); return False
    df = pd.read_csv(f)
    if "y_true" not in df.columns or "p" not in df.columns:
        print(f"[SKIP] {f} missing columns y_true/p"); return False
    y = pd.to_numeric(df["y_true"], errors="coerce").values
    p = pd.to_numeric(df["p"], errors="coerce").values
    conf, acc = reliability_curve(p, y)

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
    ap.add_argument("--all", action="store_true", help="Plot for all splits/models/tags.")
    ap.add_argument("--splits", nargs="*", default=["random","temporal","hospital"])
    ap.add_argument("--models", nargs="*", default=["lr","rf","xgb","lgbm","tabpfn","protoicl"])
    ap.add_argument("--tags",   nargs="*", default=["notuned","none","platt","isotonic","icl"])
    ap.add_argument("--preds_dir", default="results/preds")
    ap.add_argument("--outdir",    default="results/figs/calibration")
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    outdir = Path(args.outdir)
    any_ok=False
    for s in args.splits:
        for m in args.models:
            for t in args.tags:
                any_ok |= plot_one(preds_dir, outdir / s / m, s, m, t)
    if not any_ok:
        print(f"[WARN] No plots produced. Check {preds_dir}.")

if __name__ == "__main__":
    main()
