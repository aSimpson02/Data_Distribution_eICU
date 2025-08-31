#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

DATA = Path("data/splits")
PREDS = Path("results/preds_tuned")
FIGS = Path("results/figs"); FIGS.mkdir(parents=True, exist_ok=True)
TARGET = "hospital_mortality"

def bin_stats(y, p, n_bins=15):
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(p, bins)-1
    out=[]
    for b in range(n_bins):
        m = idx==b
        if not np.any(m): continue
        conf = p[m].mean(); acc = y[m].mean(); frac = m.mean()
        out.append((conf, acc, frac))
    return np.array(out)

def plot_rel(y, p, title, path):
    s = bin_stats(y, p, n_bins=15)
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(s[:,0], s[:,1], marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="hospital5src",
                    choices=["random","temporal","hospital","hospital5src"])
    ap.add_argument("--model", default="lgbm", choices=["lr","xgb","lgbm"])
    args = ap.parse_args()

    y = pd.read_parquet(DATA/f"{args.split}_test.parquet")[TARGET].astype(int).values
    p = np.load(PREDS/f"{args.split}_{args.model}_test.npy")
    out = FIGS/f"calibration_{args.split}_{args.model}.png"
    plot_rel(y, p, f"Calibration: {args.split} / {args.model}", out)
    print("[OK] Saved →", out)

if __name__ == "__main__":
    main()
