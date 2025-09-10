#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"  # must be set before import

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score

from tabpfn import TabPFNClassifier


def parse_args():
    p = argparse.ArgumentParser("ICL TabPFN (CPU-safe, val-calibrated)")
    p.add_argument("--scenario", required=True, choices=["random", "temporal", "hospital"])
    p.add_argument("--label_col", required=True)
    p.add_argument("--splits_dir", default="data/csv_splits")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--cap", type=int, default=600, help="Max support samples for CPU (<=1000)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


LEAK = {"apachescore", "predictedhospitalmortality"}
STEM = {"random": "random", "temporal": "temporal", "hospital": "hospital"}

def ece(y, p, bins=20):
    y = np.asarray(y).astype(int); p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        m = (p >= lo) & (p < (hi if i < bins-1 else hi + 1e-12))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def load_three(stem: str, splits_dir: str, label_col: str):
    d = Path(splits_dir)
    tr = pd.read_csv(d / f"{stem}_train.csv")
    va = pd.read_csv(d / f"{stem}_val.csv")
    te = pd.read_csv(d / f"{stem}_test.csv")
    # drop obvious leakage + IDs (kept simple and consistent with your pipeline)
    def prep(df: pd.DataFrame):
        df = df.copy()
        num = df.select_dtypes(include=[np.number])
        drop = {label_col} | LEAK | {c for c in num.columns if "id" in c.lower()}
        keep_num = [c for c in num.columns if c not in drop]
        # allow diagnosis_bucket_* engineered dummies (if present)
        diag = [c for c in df.columns if str(c).startswith("diagnosis_bucket_")]
        X = pd.concat([df[keep_num], df[diag]], axis=1)
        y = df[label_col].astype(int).to_numpy()
        return X, y
    Xtr, ytr = prep(tr); Xva, yva = prep(va); Xte, yte = prep(te)
    return Xtr, ytr, Xva, yva, Xte, yte

def cap_stratified(X, y, n_max=600, seed=42):
    if len(y) <= n_max:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_max, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X.iloc[idx], y[idx]


def main():
    a = parse_args()
    stem = STEM[a.scenario]

    # load
    Xtr, ytr, Xva, yva, Xte, yte = load_three(stem, a.splits_dir, a.label_col)

    # CPU-safe cap (<=1000) and float32 arrays
    Xtr_c, ytr_c = cap_stratified(Xtr, ytr, n_max=a.cap, seed=a.seed)
    Xtr_np = Xtr_c.to_numpy(dtype=np.float32, copy=False)
    Xva_np = Xva.to_numpy(dtype=np.float32, copy=False)
    Xte_np = Xte.to_numpy(dtype=np.float32, copy=False)

    # TabPFN (CPU) with limits disabled
    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    clf.fit(Xtr_np, ytr_c)

    # probs on VAL (for calibration) and TEST
    pv = clf.predict_proba(Xva_np)[:, 1]
    pt = clf.predict_proba(Xte_np)[:, 1]

    # Platt on VAL (no sklearn CalibratedCV to avoid warnings)
    pl = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=a.seed)
    pl.fit(pv.reshape(-1, 1), yva)
    pt_pl = pl.predict_proba(pt.reshape(-1, 1))[:, 1]

    # metrics (base)
    base = dict(
        auroc=float(roc_auc_score(yte, pt)),
        ece=float(ece(yte, pt)),
        brier=float(brier_score_loss(yte, pt)),
        nll=float(log_loss(yte, pt, labels=[0, 1])),
        acc=float(accuracy_score(yte, (pt >= 0.5).astype(int))),
    )
    # metrics (platt)
    pl_m = dict(
        auroc=float(roc_auc_score(yte, pt_pl)),
        ece=float(ece(yte, pt_pl)),
        brier=float(brier_score_loss(yte, pt_pl)),
        nll=float(log_loss(yte, pt_pl, labels=[0, 1])),
        acc=float(accuracy_score(yte, (pt_pl >= 0.5).astype(int))),
    )

    # write preds where your plotting expects them
    results_dir = Path(a.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = results_dir / "preds"; preds_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"y_true": yte, "p": pt}).to_csv(preds_dir / f"{stem}_tabpfn_none_seed{a.seed}.csv", index=False)
    pd.DataFrame({"y_true": yte, "p": pt_pl}).to_csv(preds_dir / f"{stem}_tabpfn_platt_seed{a.seed}.csv", index=False)

    print(f"[ICL][TabPFN][{a.scenario}] cap={len(ytr_c)}  "
          f"BASE AUC={base['auroc']:.4f} ECE={base['ece']:.4f} | "
          f"PLATT AUC={pl_m['auroc']:.4f} ECE={pl_m['ece']:.4f}")

if __name__ == "__main__":
    main()
