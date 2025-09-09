#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    f1_score, balanced_accuracy_score
)


from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.data.io import load_three, drop_leaky
from src.feats.selector import FeaturePolicy


def strat_cap(X: pd.DataFrame, y: np.ndarray, n: int, seed: int = 42):
    """Stratified cap to at most n rows, preserving class balance."""
    if len(X) <= n:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X.iloc[idx], y[idx]

def chunked_predict_proba(model, X, batch=1024) -> np.ndarray:
    """Predict in batches to avoid RAM spikes."""
    Xv = X.values if hasattr(X, "values") else X
    outs = []
    for i in range(0, len(Xv), batch):
        outs.append(model.predict_proba(Xv[i:i + batch]))
    return np.vstack(outs)

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



def vif_filter(X: pd.DataFrame, thresh: float = 5.0, max_iter: int = 200):
    """Greedy VIF pruning on numeric matrix; returns X with cols removed if VIF>thresh."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    except Exception:
        return X  # statsmodels not available; skip
    cols = list(X.columns)
    Xn = X.to_numpy(dtype=float)
    Xn = Xn + 1e-12*np.random.default_rng(123).standard_normal(Xn.shape)  # tiny jitter
    for _ in range(max_iter):
        vifs = np.array([vif(Xn, i) for i in range(Xn.shape[1])])
        mx = float(np.nanmax(vifs))
        if not np.isfinite(mx) or mx <= thresh or Xn.shape[1] <= 2:
            break
        drop_idx = int(np.nanargmax(vifs))
        cols.pop(drop_idx)
        Xn = np.delete(Xn, drop_idx, axis=1)
    return X[cols]

def rfe_logit_topk(X: pd.DataFrame, y: np.ndarray, k: int = 64, seed: int = 42):
    """Rank by |coef| from L2-logit; keep top-k."""
    k = min(k, X.shape[1])
    try:
        lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=500, random_state=seed, n_jobs=-1)
        lr.fit(X, y)
        coef = np.abs(lr.coef_.ravel())
    except Exception:
        coef = np.var(X.to_numpy(dtype=float), axis=0)  # fallback: variance
    idx = np.argsort(coef)[::-1][:k]
    keep = X.columns[idx]
    return X[keep]

# wrapper to chunk isotonic during calibration
class _ChunkedPrefit:
    def __init__(self, base, bs=1024):
        self.base = base; self.bs = bs
    def fit(self, X, y): return self
    def predict_proba(self, X):
        Xv = X if isinstance(X, np.ndarray) else X.to_numpy()
        outs=[]
        for i in range(0, len(Xv), self.bs):
            outs.append(self.base.predict_proba(Xv[i:i+self.bs]))
        return np.vstack(outs)

# robust TabPFN builder across versions (v1 vs v2)
def build_tabpfn(n_ens: int, device: str):
    from tabpfn import TabPFNClassifier
    # Try v2 style, then v1 style, then minimal
    for kwargs in (
        dict(n_estimators=n_ens, device=device, ignore_pretraining_limits=True),
        dict(N_ensemble_configurations=n_ens, device=device, ignore_pretraining_limits=True),
        dict(device=device),
    ):
        try:
            return TabPFNClassifier(**kwargs)
        except TypeError:
            continue
    return TabPFNClassifier(device=device)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["random","temporal","hospital"])
    ap.add_argument("--label_col", default="hospital_mortality")
    ap.add_argument("--splits_dir", default="data/csv_splits")
    ap.add_argument("--results_dir", default="results/tabfm")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--cap_train", type=int, default=4000)
    ap.add_argument("--cap_val",   type=int, default=5000)
    ap.add_argument("--batch",     type=int, default=1024)
    ap.add_argument("--n_ensembles", type=int, default=8)

    ap.add_argument("--calib", nargs="+", default=["none","platt"], choices=["none","platt","isotonic"])

    ap.add_argument("--feat_select", default="none", choices=["none","vif","vif_rfe"])
    ap.add_argument("--vif_thresh", type=float, default=5.0)
    ap.add_argument("--rfe_keep",   type=int,   default=64)

    args = ap.parse_args()

    # If running on CPU, allow larger datasets for TabPFN v2 guard
    if args.device == "cpu":
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

    # load data
    tr, va, te = load_three(args.scenario, args.splits_dir)
    tr = drop_leaky(tr, args.label_col)
    va = drop_leaky(va, args.label_col)
    te = drop_leaky(te, args.label_col)

    ytr = tr[args.label_col].astype(int).values
    yva = va[args.label_col].astype(int).values
    yte = te[args.label_col].astype(int).values

    # baseline feature policy (same as rest of repo)
    start = [c for c in tr.columns if c != args.label_col]
    policy = FeaturePolicy(feat_select="none", missing_thresh=0.40).fit(
        tr[[args.label_col] + start], label_col=args.label_col
    )
    Xtr = policy.transform(tr)
    Xva = policy.transform(va)
    Xte = policy.transform(te)

    # optional compact feature model
    if args.feat_select != "none":
        Xtr = vif_filter(Xtr, thresh=args.vif_thresh) if "vif" in args.feat_select else Xtr
        keep_cols = list(Xtr.columns)
        Xva = Xva[keep_cols]; Xte = Xte[keep_cols]
        if args.feat_select == "vif_rfe":
            Xtr = rfe_logit_topk(Xtr, ytr, k=args.rfe_keep, seed=args.seed)
            keep_cols = list(Xtr.columns)
            Xva = Xva[keep_cols]; Xte = Xte[keep_cols]

    # cap TRAIN for TabPFN
    Xtr_c, ytr_c = strat_cap(Xtr, ytr, n=args.cap_train, seed=args.seed)

    # build + fit TabPFN
    clf = build_tabpfn(args.n_ensembles, args.device)
    clf.fit(Xtr_c.to_numpy(), ytr_c)

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = results_dir / "preds"; preds_dir.mkdir(parents=True, exist_ok=True)

    for calib in args.calib:
        # cap VAL for calibration stability
        Xva_c, yva_c = strat_cap(Xva, yva, n=args.cap_val, seed=args.seed)

        if calib == "none":
            p = chunked_predict_proba(clf, Xte, batch=args.batch)[:,1]
        elif calib == "platt":
            cal = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
            cal.fit(Xva_c.to_numpy(), yva_c)
            p = chunked_predict_proba(cal, Xte, batch=args.batch)[:,1]
        else:  # isotonic with chunked base
            chunked = _ChunkedPrefit(clf, bs=args.batch)
            cal = CalibratedClassifierCV(chunked, method="isotonic", cv="prefit")
            cal.fit(Xva_c, yva_c)  # pandas DF ok
            p = cal.predict_proba(Xte)[:,1]

        yhat = (p >= 0.5).astype(int)

        # per-example preds
        pred_path = preds_dir / f"{args.scenario}_tabfm_{calib}_seed{args.seed}.csv"
        pd.DataFrame({"y_true": yte, "p": p}).to_csv(pred_path, index=False)

        # one-row metrics
        row = dict(
            scenario=args.scenario, model="tabfm", seed=args.seed,
            source="icl", calib=calib, calib_tag=calib,
            feat=args.feat_select, feat_select=args.feat_select, n_features=Xtr.shape[1],
            auroc=roc_auc_score(yte, p),
            auprc=average_precision_score(yte, p),
            brier=brier_score_loss(yte, p),
            nll=log_loss(yte, p, labels=[0,1]),
            ece=ece(yte, p),
            tau=0.5,
            f1_at_tau=f1_score(yte, yhat),
            balacc_at_tau=balanced_accuracy_score(yte, yhat),
            file=pred_path.name
        )
        out = results_dir / f"{args.scenario}_tabfm_{calib}_seed{args.seed}.csv"
        pd.DataFrame([row]).to_csv(out, index=False)
        print(f"[OK][TabFM] {args.scenario} | {calib} -> AUROC={row['auroc']:.4f}, ECE={row['ece']:.4f}, n_feats={Xtr.shape[1]}")

if __name__ == "__main__":
    main()
