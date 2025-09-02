#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    f1_score, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedShuffleSplit

# local imports
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["random", "temporal", "hospital"])
    ap.add_argument("--label_col", default="hospital_mortality")
    ap.add_argument("--splits_dir", default="data/csv_splits")
    ap.add_argument("--results_dir", default="results/icl")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--cap_train", type=int, default=4000)
    ap.add_argument("--calib", nargs="+", default=["none", "platt"], choices=["none", "platt"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n_ensembles", type=int, default=8)
    args = ap.parse_args()

    # Import TabPFN and be robust to API differences (v1 vs v2)
    from tabpfn import TabPFNClassifier
    def build_tabpfn(n_ens: int, device: str):
        # v2 uses n_estimators; v1 used N_ensemble_configurations; some builds accept neither
        try:
            return TabPFNClassifier(n_estimators=n_ens, device=device)
        except TypeError:
            try:
                return TabPFNClassifier(N_ensemble_configurations=n_ens, device=device)
            except TypeError:
                return TabPFNClassifier(device=device)

    # Load data
    tr, va, te = load_three(args.scenario, args.splits_dir)
    tr = drop_leaky(tr, args.label_col)
    va = drop_leaky(va, args.label_col)
    te = drop_leaky(te, args.label_col)

    ytr = tr[args.label_col].astype(int).values
    yva = va[args.label_col].astype(int).values
    yte = te[args.label_col].astype(int).values

    # Feature policy (same as baseline pipeline)
    start = [c for c in tr.columns if c != args.label_col]
    policy = FeaturePolicy(feat_select="none", missing_thresh=0.40).fit(
        tr[[args.label_col] + start], label_col=args.label_col
    )
    Xtr = policy.transform(tr)
    Xva = policy.transform(va)
    Xte = policy.transform(te)

    # Cap training set for TabPFN
    Xtr_c, ytr_c = strat_cap(Xtr, ytr, n=args.cap_train, seed=args.seed)

    # Fit TabPFN
    clf = build_tabpfn(args.n_ensembles, args.device)
    clf.fit(Xtr_c.to_numpy(), ytr_c)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = results_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    for calib in args.calib:
        if calib == "none":
            p = chunked_predict_proba(clf, Xte, batch=args.batch)[:, 1]
        else:
            cal = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
            cal.fit(Xva.to_numpy(), yva)
            p = chunked_predict_proba(cal, Xte, batch=args.batch)[:, 1]

        yhat = (p >= 0.5).astype(int)

        # write preds
        pred_path = preds_dir / f"{args.scenario}_tabpfn_{calib}_seed{args.seed}.csv"
        pd.DataFrame({"y_true": yte, "p": p}).to_csv(pred_path, index=False)

        # metrics row
        row = dict(
            scenario=args.scenario, model="tabpfn", seed=args.seed,
            source="icl", calib=calib, calib_tag=calib,
            feat="none", feat_select="none", n_features=len(policy.selected_features_),
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
            results_dir / f"{args.scenario}_tabpfn_{calib}_seed{args.seed}.csv",
            index=False
        )
        print(f"[OK][ICL-TabPFN] {args.scenario} | {calib} -> AUROC={row['auroc']:.4f}, ECE={row['ece']:.4f}")

if __name__ == "__main__":
    main()
