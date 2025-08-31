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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

# Optional LightGBM **FOCUS ON LATER**
# try:
#     import lightgbm as lgbm  
# except Exception:
#     lgbm = None  

#Constants / config
PREDICTOR_DROP = {
    "patientunitstayid", "hospitalid", "hospitaldischargeyear",
    "apachescore", "predictedhospitalmortality", "admissionoffset"
}
STEM_MAP = {
    "iid": "random",
    "random": "random",
    "hospital": "hospital",
    "hospital_ood": "hospital",
    "temporal": "temporal",
    "temporal_ood": "temporal",
}
CSV_DIR = Path("data/csv_splits")
PRED_DIR = Path("results/preds_conformal")
PRED_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = Path("results/conformal_summary.csv")
DEFAULT_LABEL = "hospital_mortality"

# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=list(STEM_MAP.keys()))
    p.add_argument("--models", nargs="+", default=["lr", "rf", "lgbm"])
    p.add_argument("--calib", nargs="+", default=["none", "platt", "isotonic"])
    p.add_argument("--alpha", type=float, default=0.1)          # target error 10% = 90% coverage
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label_col", type=str, default=DEFAULT_LABEL)
    return p.parse_args()

# Data I/O 
def load_three(stem: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = CSV_DIR
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

def imputer_from_train(train_df: pd.DataFrame, feat_cols: list[str]):
    means = train_df[feat_cols].mean(numeric_only=True)
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        X = df[feat_cols].copy()
        # numeric - fill with TRAIN means
        for c in means.index:
            if c in X.columns:
                X[c] = X[c].fillna(means[c])
        # non-numeric (rare: bool) → fill zeros
        for c in X.columns:
            if c not in means.index:
                X[c] = X[c].fillna(0)
        return X
    return _impute

# Models / calibration
def build_model(name: str, seed: int):
    if name == "lr":
        return LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, class_weight="balanced_subsample", random_state=seed
        )
    # if name == "lgbm":
    #     if lgbm is None:
    #         print("[skip] LightGBM requested but not installed/disabled.")
    #         return None
    #     return lgbm.LGBMClassifier(
    #         objective="binary",
    #         n_estimators=600,
    #         learning_rate=0.03,
    #         num_leaves=63,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         reg_lambda=1.0,
    #         random_state=seed
        # )
    raise ValueError(f"Unknown model: {name}")

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

# Conformal utils
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Quantile with conformal correction:
      q = k-th order statistic, k = ceil((n+1)*(1-alpha))
    Using 'higher' to be conservative on ties.
    """
    scores = np.asarray(scores)
    n = len(scores)
    if n == 0:
        return 1.0  # degenerate; include everything
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])

def conformal_sets(p1: np.ndarray, q0: float, q1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given positive-class probabilities p1 and class-conditional quantiles q0, q1,
    include 1 if (1 - p1) <= q1  ⇒ p1 >= 1 - q1
    include 0 if (1 - (1-p1)) <= q0  ⇒ (1 - p1) >= 1 - q0  ⇒ p1 <= q0
    Return include0, include1, set_size.
    """
    p1 = np.asarray(p1)
    include1 = (p1 >= (1.0 - q1))
    include0 = (p1 <= q0)
    set_size = include0.astype(int) + include1.astype(int)
    return include0, include1, set_size

def expected_calibration_error(y_true, y_prob, n_bins: int = 20) -> float:
    y_true = np.asarray(y_true).astype(int)
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
        ece += (np.sum(mask) / len(y_prob)) * abs(acc - conf)
    return float(ece)

# Main run
def run_once(scenario: str, model_name: str, calib: str, alpha: float, seed: int, label_col: str):
    stem = STEM_MAP[scenario]
    tr, va, te = load_three(stem)
    feats = choose_features(tr, label_col)
    impute = imputer_from_train(tr, feats)

    Xtr, ytr = impute(tr), tr[label_col].astype(int).values
    Xva, yva = impute(va), va[label_col].astype(int).values
    Xte, yte = impute(te), te[label_col].astype(int).values

    # train base
    model = build_model(model_name, seed)
    if model is None:
        return None  # LGBM skipped
    model.fit(Xtr, ytr)

    # calibrate on VAL
    model_cal = add_calibration(model, Xva, yva, calib)

    # probs
    p1_tr = model_cal.predict_proba(Xtr)[:, 1]
    p1_va = model_cal.predict_proba(Xva)[:, 1]
    p1_te = model_cal.predict_proba(Xte)[:, 1]

    # conformity scores on VAL, class-conditional
    s_pos = 1.0 - p1_va[yva == 1]
    s_neg = 1.0 - (1.0 - p1_va[yva == 0])  # = p1_va for negatives? Careful: for y=0, s=1 - p0 = 1 - (1-p1) = p1
    # With our definition s_y = 1 - p_y:
    s1 = 1.0 - p1_va[yva == 1]     # y=1
    s0 = 1.0 - (1.0 - p1_va[yva == 0])  # y=0 → 1 - p0 = 1 - (1-p1) = p1
    q1 = conformal_quantile(s1, alpha)
    q0 = conformal_quantile(s0, alpha)

    # build prediction sets on TEST
    include0, include1, set_size = conformal_sets(p1_te, q0=q0, q1=q1)
    abstain = (set_size > 1)

    # empirical coverage (does the set contain true label???)
    cover = (include1 & (yte == 1)) | (include0 & (yte == 0))
    coverage = float(cover.mean())
    abst_rate = float(abstain.mean())

    # standard metrics on point probs
    auroc = roc_auc_score(yte, p1_te)
    auprc = average_precision_score(yte, p1_te)
    brier = brier_score_loss(yte, p1_te)
    nll   = log_loss(yte, np.vstack([1 - p1_te, p1_te]).T, labels=[0, 1])
    ece   = expected_calibration_error(yte, p1_te)

    # save per-stay predictions
    pid_col = "patientunitstayid" if "patientunitstayid" in te.columns else None
    out_cols = {}
    if pid_col:
        out_cols[pid_col] = te[pid_col].values
    out = pd.DataFrame({
        **out_cols,
        "y_true": yte,
        "p1": p1_te,
        "include0": include0.astype(int),
        "include1": include1.astype(int),
        "set_size": set_size.astype(int),
        "abstain": abstain.astype(int),
    })
    pred_path = PRED_DIR / f"{scenario}_{model_name}_{calib}_alpha{alpha}_seed{seed}.csv"
    out.to_csv(pred_path, index=False)

    # append summary
    summ_row = pd.DataFrame([{
        "scenario": scenario,
        "model": model_name,
        "calib": calib,
        "alpha": alpha,
        "seed": seed,
        "q0": q0,
        "q1": q1,
        "coverage": coverage,
        "abstain_rate": abst_rate,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "nll": nll,
        "ece": ece,
    }])
    if SUMMARY_PATH.exists():
        prev = pd.read_csv(SUMMARY_PATH)
        pd.concat([prev, summ_row], ignore_index=True).to_csv(SUMMARY_PATH, index=False)
    else:
        summ_row.to_csv(SUMMARY_PATH, index=False)

    print(f"[OK] {scenario} | {model_name} | {calib} | alpha={alpha:.2f} "
          f"→ coverage={coverage:.3f}, abstain={abst_rate:.3f}, AUROC={auroc:.3f}, ECE={ece:.3f}")
    return {
        "coverage": coverage,
        "abstain_rate": abst_rate,
        "auroc": auroc,
        "ece": ece,
        "q0": q0,
        "q1": q1
    }

def main():
    args = parse_args()
    for m in args.models:
        for c in args.calib:
            run_once(args.scenario, m, c, args.alpha, args.seed, args.label_col)

if __name__ == "__main__":
    main()
