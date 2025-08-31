#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    f1_score, balanced_accuracy_score
)

# Optional LightGBM 
try:
    import lightgbm as lgbm  # type: ignore
except Exception:
    lgbm = None  # LightGBM disabled or not installed

# Configuration
PREDICTOR_DROP = {
    "patientunitstayid", "hospitalid", "hospitaldischargeyear",
    "apachescore", "predictedhospitalmortality", "admissionoffset"
}
DEFAULT_RESULTS_DIR = "results"
DEFAULT_SPLITS_DIR = "data/csv_splits"

STEM_MAP = {
    "iid": "random",
    "random": "random",
    "hospital": "hospital",
    "hospital_ood": "hospital",
    "temporal": "temporal",
    "temporal_ood": "temporal",
}

# Helpers
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                   choices=list(STEM_MAP.keys()))
    p.add_argument("--models", nargs="+", default=["lr", "rf"])
    p.add_argument("--calib", nargs="+", default=["none", "platt", "isotonic"])
    p.add_argument("--label_col", required=True)
    p.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR)
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    return p.parse_args()

def load_three(stem: str, splits_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(splits_dir)
    tr = pd.read_csv(d / f"{stem}_train.csv")
    va = pd.read_csv(d / f"{stem}_val.csv")
    te = pd.read_csv(d / f"{stem}_test.csv")
    return tr, va, te

def choose_features(df: pd.DataFrame, label_col: str) -> list[str]:
    cols = []
    for c in df.columns:
        if c == label_col or c in PREDICTOR_DROP:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or c.startswith("diagnosis_bucket_"):
            cols.append(c)
    return cols

def expected_calibration_error(y_true, y_prob, n_bins: int = 20) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0., 1., n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

def best_tau_balacc(y_true, y_prob):
    taus = np.linspace(0, 1, 200)
    best_tau, best_ba = 0.5, -1.0
    for t in taus:
        yhat = (y_prob >= t).astype(int)
        sens = (y_true[yhat == 1].mean() if (yhat == 1).any() else 0.0)
        # balanced accuracy via sklearn (more robust):
        ba = balanced_accuracy_score(y_true, yhat)
        if ba > best_ba:
            best_ba, best_tau = ba, t
    return best_tau, best_ba

def add_calibration(model, X_val, y_val, method: str):
    if method == "none":
        return model
    if method == "platt":
        return CalibratedClassifierCV(model, method="sigmoid", cv="prefit").fit(X_val, y_val)
    if method == "isotonic":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CalibratedClassifierCV(model, method="isotonic", cv="prefit").fit(X_val, y_val)
    raise ValueError(f"Unknown calibration: {method}")

def build_model(name: str, seed: int):
    if name == "lr":
        return LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, class_weight="balanced_subsample", random_state=seed
        )
    if name == "lgbm":
        if lgbm is None:
            print("[skip] LightGBM requested but not installed/disabled.")
            return None
        return lgbm.LGBMClassifier(
            objective="binary",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed
        )
    raise ValueError(f"Unknown model: {name}")

# Main
def main():
    args = parse_args()
    stem = STEM_MAP[args.scenario]
    tr, va, te = load_three(stem, args.splits_dir)

    label = args.label_col
    feats = choose_features(tr, label)

    # train means for leak-safe imputation
    means = tr[feats].mean(numeric_only=True)

    def impute(df: pd.DataFrame) -> pd.DataFrame:
        X = df[feats].copy()
        for c in means.index:
            if c in X.columns:
                X[c] = X[c].fillna(means[c])
        for c in X.columns:
            if c not in means.index:
                X[c] = X[c].fillna(0)
        return X

    Xtr, ytr = impute(tr), tr[label].astype(int).values
    Xva, yva = impute(va), va[label].astype(int).values
    Xte, yte = impute(te), te[label].astype(int).values

    out_dir = Path(DEFAULT_RESULTS_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    for m in args.models:
        clf = build_model(m, seed=42)
        if clf is None:
            continue  # e.g., lgbm requested but not available
        clf.fit(Xtr, ytr)

        for cal in args.calib:
            clf_cal = add_calibration(clf, Xva, yva, cal)

            p_tr = clf_cal.predict_proba(Xtr)[:, 1]
            p_va = clf_cal.predict_proba(Xva)[:, 1]
            p_te = clf_cal.predict_proba(Xte)[:, 1]

            tau, _ = best_tau_balacc(yva, p_va)
            yhat_te = (p_te >= tau).astype(int)

            row = {
                "scenario": args.scenario,
                "Htgt": args.scenario,
                "model": m,
                "seed": 42,
                "calib": cal,
                "tau": float(tau),
                "auroc": roc_auc_score(yte, p_te),
                "auprc": average_precision_score(yte, p_te),
                "brier": brier_score_loss(yte, p_te),
                "nll":   log_loss(yte, np.vstack([1 - p_te, p_te]).T, labels=[0, 1]),
                "ece":   expected_calibration_error(yte, p_te),
                "f1_at_tau": f1_score(yte, yhat_te, zero_division=0),
                "balacc_at_tau": balanced_accuracy_score(yte, yhat_te),
            }

            out_file = out_dir / f"{args.scenario}_{m}_{cal}_seed42.csv"
            pd.DataFrame([row]).to_csv(out_file, index=False)
            print(f"[OK] {args.scenario} | {m} | calib={cal} → AUROC={row['auroc']:.4f}, ECE={row['ece']:.4f}")

if __name__ == "__main__":
    main()
