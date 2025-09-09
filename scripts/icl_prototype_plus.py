#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
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
    if len(X) <= n:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X.iloc[idx], y[idx]


def feature_weights(X: np.ndarray, y: np.ndarray, mode: str, seed: int) -> np.ndarray:
    p = X.shape[1]
    if mode == "none":
        w = np.ones(p, dtype=float)
    elif mode == "l1":
        try:
            clf = LogisticRegression(
                penalty="l1", solver="saga", C=1.0, max_iter=1000,
                random_state=seed, n_jobs=-1
            ).fit(X, y)
            w = np.abs(clf.coef_.ravel()) + 1e-6
        except Exception:
            w = mutual_info_classif(X, y, random_state=seed) + 1e-6
    elif mode == "mi":
        w = mutual_info_classif(X, y, random_state=seed) + 1e-6
    else:
        raise ValueError(f"Unknown feature_weight mode: {mode}")
    mu = w.mean()
    return w / (mu if mu > 0 else 1.0)


class ProtoICLPlus(BaseEstimator, ClassifierMixin):
    """
    Stronger prototype ICL:
      - cosine/euclidean metric
      - weighting: uniform | 1/dist | softmax(-dist/T)
    """
    _estimator_type = "classifier"   # make sklearn treat this as a classifier

    def __init__(self, k=64, metric="cosine", weight="softmax", temperature=0.1):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.temperature = float(temperature)

    def fit(self, X, y):
        self.X_ = X if isinstance(X, np.ndarray) else X.to_numpy()
        self.y_ = y.astype(np.float32)
        self.classes_ = np.unique(y)     # required for CalibratedClassifierCV(cv='prefit')
        algo = "brute" if self.metric in ("cosine", "euclidean") else "auto"
        self.nn_ = NearestNeighbors(
            n_neighbors=self.k, metric=self.metric, algorithm=algo, n_jobs=-1
        ).fit(self.X_)
        return self

    def _weights(self, dist):
        eps = 1e-8
        if self.weight == "uniform":
            w = np.ones_like(dist)
        elif self.weight == "distance":
            w = 1.0 / np.maximum(dist, eps)
        elif self.weight == "softmax":
            w = np.exp(-dist / max(self.temperature, eps))
        else:
            raise ValueError(f"Unknown weight mode: {self.weight}")
        return w / (w.sum(axis=1, keepdims=True) + eps)

    def predict_proba(self, X):
        Xv = X if isinstance(X, np.ndarray) else X.to_numpy()
        dist, idx = self.nn_.kneighbors(Xv, n_neighbors=self.k, return_distance=True)
        w = self._weights(dist)
        p1 = (w * self.y_[idx]).sum(axis=1)
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


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
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--cap_train", type=int, default=50000)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--weight", default="softmax", choices=["uniform", "distance", "softmax"])
    ap.add_argument("--temperature", type=float, default=0.1)

    ap.add_argument("--feature_weight", default="l1", choices=["none", "mi", "l1"])
    ap.add_argument("--repr", default="pca", choices=["raw", "pca"])
    ap.add_argument("--pca_components", type=int, default=64)

    ap.add_argument("--calib", nargs="+", default=["none", "platt"], choices=["none", "platt"])
    args = ap.parse_args()

    # load + drop leakage
    tr, va, te = load_three(args.scenario, args.splits_dir)
    tr = drop_leaky(tr, args.label_col); va = drop_leaky(va, args.label_col); te = drop_leaky(te, args.label_col)

    ytr = tr[args.label_col].astype(int).values
    yva = va[args.label_col].astype(int).values
    yte = te[args.label_col].astype(int).values

    # policy (same as baseline)
    start = [c for c in tr.columns if c != args.label_col]
    policy = FeaturePolicy(feat_select="none", missing_thresh=0.40).fit(
        tr[[args.label_col] + start], label_col=args.label_col
    )
    Xtr = policy.transform(tr); Xva = policy.transform(va); Xte = policy.transform(te)

    # cap train size for index build
    Xtr_c, ytr_c = strat_cap(Xtr, ytr, n=args.cap_train, seed=args.seed)

    # TRAIN-only scaling
    scaler = StandardScaler().fit(Xtr_c)
    Xtr_s = scaler.transform(Xtr_c); Xva_s = scaler.transform(Xva); Xte_s = scaler.transform(Xte)

    # optional PCA representation
    if args.repr == "pca":
        n_comp = min(args.pca_components, Xtr_s.shape[1])
        pca = PCA(n_components=n_comp, random_state=args.seed)
        Xtr_r = pca.fit_transform(Xtr_s); Xva_r = pca.transform(Xva_s); Xte_r = pca.transform(Xte_s)
    else:
        Xtr_r, Xva_r, Xte_r = Xtr_s, Xva_s, Xte_s

    # optional feature weights (in representation space)
    if args.feature_weight != "none":
        w = feature_weights(Xtr_r, ytr_c, args.feature_weight, args.seed)
        Xtr_w, Xva_w, Xte_w = Xtr_r * w, Xva_r * w, Xte_r * w
    else:
        Xtr_w, Xva_w, Xte_w = Xtr_r, Xva_r, Xte_r

    base = ProtoICLPlus(k=args.k, metric=args.metric, weight=args.weight, temperature=args.temperature).fit(Xtr_w, ytr_c)

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = results_dir / "preds"; preds_dir.mkdir(parents=True, exist_ok=True)

    for calib in args.calib:
        if calib == "none":
            p = base.predict_proba(Xte_w)[:, 1]
        else:
            # Try standard Platt, then fall back to manual Platt on probabilities
            try:
                cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
                cal.fit(Xva_w, yva)
                p = cal.predict_proba(Xte_w)[:, 1]
            except Exception as e:
                # Manual Platt on base probabilities (robust to estimator type quirks)
                p_va = base.predict_proba(Xva_w)[:, 1].reshape(-1, 1)
                p_te = base.predict_proba(Xte_w)[:, 1].reshape(-1, 1)
                lr = LogisticRegression(max_iter=1000, solver="lbfgs")
                lr.fit(p_va, yva)
                p = lr.predict_proba(p_te)[:, 1]

        yhat = (p >= 0.5).astype(int)
        pred_path = preds_dir / f"{args.scenario}_protoicl_plus_{calib}_seed{args.seed}.csv"
        pd.DataFrame({"y_true": yte, "p": p}).to_csv(pred_path, index=False)

        row = dict(
            scenario=args.scenario, model="protoicl_plus", seed=args.seed, source="icl",
            calib=calib, calib_tag=calib, feat="none", feat_select="none",
            n_features=Xtr.shape[1],
            auroc=roc_auc_score(yte, p), auprc=average_precision_score(yte, p),
            brier=brier_score_loss(yte, p), nll=log_loss(yte, p, labels=[0, 1]),
            ece=ece(yte, p), tau=0.5, f1_at_tau=f1_score(yte, yhat),
            balacc_at_tau=balanced_accuracy_score(yte, yhat),
            file=pred_path.name
        )
        pd.DataFrame([row]).to_csv(
            results_dir / f"{args.scenario}_protoicl_plus_{calib}_seed{args.seed}.csv",
            index=False
        )
        print(f"[OK][ICL-Proto++] {args.scenario} | {calib} -> AUROC={row['auroc']:.4f}, ECE={row['ece']:.4f}")


if __name__ == "__main__":
    main()
