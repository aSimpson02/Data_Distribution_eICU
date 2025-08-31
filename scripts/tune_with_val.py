#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

# ensure repo root on path (for src/... and scripts/ imports)
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

from src.feats.selector import FeaturePolicy
from scripts.build_models import build_model, get_model_names  # model factory

PREDICTOR_DROP = {
    "patientunitstayid", "hospitalid", "hospitaldischargeyear",
    "apachescore", "predictedhospitalmortality", "admissionoffset"
}

STEM_MAP = {
    "iid": "random", "random": "random",
    "hospital": "hospital", "hospital_ood": "hospital",
    "temporal": "temporal", "temporal_ood": "temporal"
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=list(STEM_MAP.keys()))
    p.add_argument("--models", nargs="+", default=["lr","rf"], choices=get_model_names())
    p.add_argument("--calib", nargs="+", default=["none","platt","isotonic"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--label_col", required=True)
    p.add_argument("--splits_dir", default="data/csv_splits")
    p.add_argument("--results_dir", default="results")
    # feature-selection knobs
    p.add_argument("--feat_select", choices=["none","vif","rfe","vif_rfe"], default="none")
    p.add_argument("--rfe_keep", type=int, default=30)
    p.add_argument("--missing_thresh", type=float, default=0.40)
    p.add_argument("--vif_thresh", type=float, default=10.0)
    return p.parse_args()

def load_three(stem: str, splits_dir: str):
    d = Path(splits_dir)
    tr = pd.read_csv(d / f"{stem}_train.csv")
    va = pd.read_csv(d / f"{stem}_val.csv")
    te = pd.read_csv(d / f"{stem}_test.csv")
    return tr, va, te

def choose_starting_features(df: pd.DataFrame, label_col: str) -> list[str]:
    cols = []
    for c in df.columns:
        if c == label_col or c in PREDICTOR_DROP:
            continue
        # numeric + diagnosis buckets only
        if pd.api.types.is_numeric_dtype(df[c]) or str(c).startswith("diagnosis_bucket_"):
            cols.append(c)
    return cols

def add_calibration(model, X_val, y_val, method: str):
    if method == "none":
        return model
    if method == "platt":
        return CalibratedClassifierCV(model, method="sigmoid", cv="prefit").fit(X_val, y_val)
    if method == "isotonic":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CalibratedClassifierCV(model, method="isotonic", cv="prefit").fit(X_val, y_val)
    raise ValueError(method)

def expected_calibration_error(y_true, y_prob, n_bins: int = 20) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0., 1., n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins-1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(m):
            continue
        ece += (np.sum(m)/len(y_prob)) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)

def best_tau_balacc(y_true, y_prob):
    from sklearn.metrics import balanced_accuracy_score
    taus = np.linspace(0, 1, 200)
    best_tau, best_ba = 0.5, -1.0
    for t in taus:
        yhat = (y_prob >= t).astype(int)
        ba = balanced_accuracy_score(y_true, yhat)
        if ba > best_ba:
            best_ba, best_tau = ba, t
    return best_tau, best_ba

def run_once(args, model_name: str, calib: str, seed: int):
    stem = STEM_MAP[args.scenario]
    tr, va, te = load_three(stem, args.splits_dir)
    label = args.label_col

    # Feature policy
    start_cols = choose_starting_features(tr, label)
    tr_filt = tr[[label] + start_cols].copy()
    policy = FeaturePolicy(
        feat_select=args.feat_select,
        missing_thresh=args.missing_thresh,
        vif_thresh=args.vif_thresh,
        rfe_keep=args.rfe_keep,
    ).fit(tr_filt, label_col=label)

    # Transform splits
    Xtr = policy.transform(tr); ytr = tr[label].astype(int).values
    Xva = policy.transform(va); yva = va[label].astype(int).values
    Xte = policy.transform(te); yte = te[label].astype(int).values

    # Build model (skip gracefully if lib missing)
    try:
        clf = build_model(model_name, seed)
    except RuntimeError as e:
        print(f"[SKIP] {args.scenario} | {model_name} | {calib} | seed={seed} → {e}")
        return

    # === XGBoost name-safety: use NumPy so it ignores column names with [,],<,> ===
    is_xgb = (model_name == "xgb")
    Xtr_in = Xtr.to_numpy() if is_xgb else Xtr
    Xva_in = Xva.to_numpy() if is_xgb else Xva
    Xte_in = Xte.to_numpy() if is_xgb else Xte

    # Train + calibrate
    clf.fit(Xtr_in, ytr)
    clf_cal = add_calibration(clf, Xva_in, yva, calib)

    # Predict + metrics
    p_va = clf_cal.predict_proba(Xva_in)[:, 1]
    p_te = clf_cal.predict_proba(Xte_in)[:, 1]
    tau, _ = best_tau_balacc(yva, p_va)

    row = {
        "scenario": args.scenario,
        "Htgt": args.scenario,
        "model": model_name,
        "seed": seed,
        "calib": calib,
        "feat_select": args.feat_select,
        "rfe_keep": args.rfe_keep if args.feat_select in ("rfe","vif_rfe") else None,
        "missing_thresh": args.missing_thresh,
        "vif_thresh": args.vif_thresh if args.feat_select in ("vif","vif_rfe") else None,
        "n_features": len(policy.selected_features_),
        "auroc": roc_auc_score(yte, p_te),
        "auprc": average_precision_score(yte, p_te),
        "brier": brier_score_loss(yte, p_te),
        "nll": log_loss(yte, p_te, labels=[0,1]),
        "ece": expected_calibration_error(yte, p_te),
        "tau": float(tau),
    }

    out_dir = Path(args.results_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stemfile = f"{args.scenario}_{model_name}_{calib}_{args.feat_select}_seed{seed}"
    pd.DataFrame([row]).to_csv(out_dir / f"{stemfile}.csv", index=False)
    with open(out_dir / f"{stemfile}_features.txt","w") as f:
        f.write("\n".join(policy.selected_features_))

    print(f"[OK] {args.scenario} | {model_name} | {calib} | {args.feat_select} | seed={seed} "
          f"→ AUROC={row['auroc']:.4f}, ECE={row['ece']:.4f}, n_feats={row['n_features']}")

def main():
    args = parse_args()
    print(f"Tuning & evaluating → {args.scenario} | feat_select={args.feat_select}")
    for m in args.models:
        for c in args.calib:
            for s in args.seeds:
                run_once(args, m, c, s)

if __name__ == "__main__":
    main()
