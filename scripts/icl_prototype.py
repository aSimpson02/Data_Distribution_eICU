#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    f1_score, balanced_accuracy_score
)

# local imports
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.data.io import load_three, drop_leaky
from src.feats.selector import FeaturePolicy


def strat_cap(X: pd.DataFrame, y: np.ndarray, n: int, seed: int = 42):
    if len(X) <= n: return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X.iloc[idx], y[idx]


class ProtoICL(BaseEstimator, ClassifierMixin):
    def __init__(self, k=50, metric="euclidean", weight="distance"):
        self.k = k
        self.metric = metric
        self.weight = weight  # 'uniform' or 'distance'

    def fit(self, X, y):
        self.X_ = X if isinstance(X, np.ndarray) else X.to_numpy()
        self.y_ = y.astype(np.float32)
        self.nn_ = NearestNeighbors(n_neighbors=self.k, metric=self.metric, n_jobs=-1).fit(self.X_)
        return self

    def _weights(self, dist):
        eps = 1e-8
        if self.weight == "uniform":
            w = np.ones_like(dist)
        else:
            w = 1.0 / np.maximum(dist, eps)
        return w / (w.sum(axis=1, keepdims=True) + eps)

    def predict_proba(self, X):
        Xv = X if isinstance(X, np.ndarray) else X.to_numpy()
        dist, idx = self.nn_.kneighbors(Xv, n_neighbors=self.k, return_distance=True)
        w = self._weights(dist)
        p1 = (w * self.y_[idx]).sum(axis=1)
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.vstack([1.0 - p1, p1]).T


def ece(y, p, bins=20) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < (hi if i < bins - 1 else hi))
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["random", "temporal", "hospital"])
    ap.add_argument("--label_col", default="hospital_mortality")
    ap.add_argument("--splits_dir", default="data/csv_splits")
    ap.add_argument("--results_dir", default="results/icl")
    ap.add_argument("--cap_train", type=int, default=20000)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan", "cosine"])
    ap.add_argument("--weight", default="distance", choices=["uniform", "distance"])
    ap.add_argument("--calib", nargs="+", default=["none", "platt"], choices=["none", "platt"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tr, va, te = load_three(args.scenario, args.splits_dir)
    tr = drop_leaky(tr, args.label_col)
    va = drop_leaky(va, args.label_col)
    te = drop_leaky(te, args.label_col)

    ytr = tr[args.label_col].astype(int).values
    yva = va[args.label_col].astype(int).values
    yte = te[args.label_col].astype(int).values

    start = [c for c in tr.columns if c != args.label_col]
    policy = FeaturePolicy(feat_select="none", missing_thresh=0.40).fit(
        tr[[args.label_col] + start], label_col=args.label_col
    )
    Xtr = policy.transform(tr)
    Xva = policy.transform(va)
    Xte = policy.transform(te)

    # cap and scale (TRAIN only)
    Xtr_c, ytr_c = strat_cap(Xtr, ytr, n=args.cap_train, seed=args.seed)
    scaler = StandardScaler().fit(Xtr_c)
    Xtr_s = scaler.transform(Xtr_c)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    base = ProtoICL(k=args.k, metric=args.metric, weight=args.weight).fit(Xtr_s, ytr_c)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = results_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    for calib in args.calib:
        if calib == "none":
            p = base.predict_proba(Xte_s)[:, 1]
        else:
            cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
            cal.fit(Xva_s, yva)
            p = cal.predict_proba(Xte_s)[:, 1]

        yhat = (p >= 0.5).astype(int)

        pred_path = preds_dir / f"{args.scenario}_protoicl_{calib}_seed{args.seed}.csv"
        pd.DataFrame({"y_true": yte, "p": p}).to_csv(pred_path, index=False)

        row = dict(
            scenario=args.scenario, model="protoicl", seed=args.seed,
            source="icl", calib=calib, calib_tag=calib,
            feat="none", feat_select="none", n_features=Xtr.shape[1],
            auroc=roc_auc_score(yte, p),
            auprc=average_precision_score(yte, p),
            brier=brier_score_loss(yte, p),
            nll=log_loss(yte, p, labels=[0, 1]),
            ece=ece(yte, p),
            tau=0.5,
            f1_at_tau=f1_score(yte, yhat),
            balacc_at_tau=balanced_accuracy_score(yte, yhat),
            file=pred_path.name
        )
        pd.DataFrame([row]).to_csv(
            results_dir / f"{args.scenario}_protoicl_{calib}_seed{args.seed}.csv",
            index=False
        )
        print(f"[OK][ICL-Proto] {args.scenario} | {calib} -> AUROC={row['auroc']:.4f}, ECE={row['ece']:.4f}")

if __name__ == "__main__":
    main()
