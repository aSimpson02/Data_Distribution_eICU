#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

PRED = Path("results/preds")
OUT = Path("results/figs/calibration"); OUT.mkdir(parents=True, exist_ok=True)

def reliability_curve(probs, y, n_bins=15):
    bins = np.linspace(0,1,n_bins+1)
    idx = np.clip(np.digitize(probs, bins)-1, 0, n_bins-1)
    confs, accs, counts = [], [], []
    for b in range(n_bins):
        m = idx==b
        if not m.any(): continue
        confs.append(probs[m].mean())
        accs.append(y[m].mean())
        counts.append(m.sum())
    return np.array(confs), np.array(accs), np.array(counts)

def plot_one(split, model):
    f = PRED / f"{split}_{model}.csv"
    if not f.exists(): 
        print(f"[SKIP] {f} missing"); return
    df = pd.read_csv(f)
    y, p = df["y_true"].values, df["p"].values
    conf, acc, n = reliability_curve(p, y)
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--", label="perfect")
    plt.plot(conf, acc, marker="o", label=f"{split}-{model}")
    plt.xlabel("predicted probability"); plt.ylabel("observed frequency")
    plt.title(f"Calibration: {split} / {model}")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / f"calib_{split}_{model}.png", dpi=160)
    plt.close()
    print(f"[OK] calib → {OUT/f'calib_{split}_{model}.png'}")

if __name__ == "__main__":
    for split in ["random","temporal","hospital","hospital5src"]:
        for model in ["lr","xgb","lgbm"]:
            plot_one(split, model)
