#!/usr/bin/env python3

from __future__ import annotations
import argparse, inspect, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd

# Limit thread fan-out (helps memory on macOS/CPU)
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# make repo root importable
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedShuffleSplit

from src.feats.selector import FeaturePolicy

PREDICTOR_DROP = {
    "patientunitstayid","hospitalid","hospitaldischargeyear",
    "apachescore","predictedhospitalmortality","admissionoffset"
}
STEM_MAP = {
    "iid":"random","random":"random",
    "hospital":"hospital","hospital_ood":"hospital",
    "temporal":"temporal","temporal_ood":"temporal",
}

def parse_args():
    p = argparse.ArgumentParser("Evaluate TabPFN/TabFM-style classifier")
    p.add_argument("--scenario", required=True, choices=list(STEM_MAP.keys()))
    p.add_argument("--calib", nargs="+", default=["none","platt","isotonic"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--label_col", required=True)
    p.add_argument("--splits_dir", default="data/csv_splits")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    # feature selection
    p.add_argument("--feat_select", choices=["none","vif","rfe","vif_rfe"], default="none")
    p.add_argument("--rfe_keep", type=int, default=30)
    p.add_argument("--missing_thresh", type=float, default=0.40)
    p.add_argument("--vif_thresh", type=float, default=10.0)
    # performance / safety
    p.add_argument("--cap_train", type=int, default=10000,
                   help="Max training rows via stratified sampling (0 = no cap).")
    p.add_argument("--ensembles", type=int, default=4,
                   help="TabPFN ensemble configurations.")
    p.add_argument("--posterior", type=int, default=4,
                   help="Posterior samples at predict time.")
    p.add_argument("--save_probs", action="store_true",
                   help="Save test probabilities and labels to results/probs/*.npz")
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
        if c == label_col or c in PREDICTOR_DROP: continue
        if pd.api.types.is_numeric_dtype(df[c]) or c.startswith("diagnosis_bucket_"):
            cols.append(c)
    return cols

def add_calibration(model, X_val, y_val, method: str):
    if method == "none": return model
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
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(m): continue
        ece += (np.sum(m) / len(y_prob)) * abs(y_true[m].mean() - y_prob[m].mean())
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

def stratified_cap(X: np.ndarray, y: np.ndarray, n_max: int, seed: int):
    n = len(y)
    if n_max <= 0 or n <= n_max: return X, y
    test_size = n_max / n
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, idx_keep = next(splitter.split(X, y))
    return X[idx_keep], y[idx_keep]

# ---------------------- TabPFN (version-safe) ----------------------
def make_tabpfn(seed: int = 42, device: str = "cpu", ensembles: int = 4, posterior: int = 4):
    try:
        from tabpfn import TabPFNClassifier  # type: ignore[import]
    except Exception:
        try:
            from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier  # type: ignore[import]
        except Exception as e:
            raise RuntimeError(f"TabPFN not installed or import failed: {e}")

    sig = inspect.signature(TabPFNClassifier.__init__)
    allowed = set(sig.parameters.keys())
    params = {}
    if "device" in allowed: params["device"] = device
    if "seed" in allowed: params["seed"] = seed
    if "random_state" in allowed: params["random_state"] = seed
    if "ignore_pretraining_limits" in allowed: params["ignore_pretraining_limits"] = True
    for k in ["N_ensemble_configurations","n_ensemble_configurations","num_ensemble_configurations"]:
        if k in allowed: params[k] = max(1, int(ensembles)); break
    for k in ["predict_posterior_samples","n_posterior_samples"]:
        if k in allowed: params[k] = max(1, int(posterior)); break
    return TabPFNClassifier(**params)

def tabpfn_fit(clf, X, y):
    sig = inspect.signature(clf.fit)
    allowed = set(sig.parameters.keys())
    kwargs = {}
    if "overwrite_warning" in allowed: kwargs["overwrite_warning"] = True
    return clf.fit(X, y, **kwargs)

# ----------------------------- core run -----------------------------
def run_once(args, calib: str, seed: int):
    model_name = "tabpfn"
    stem = STEM_MAP[args.scenario]
    tr, va, te = load_three(stem, args.splits_dir)
    label = args.label_col

    # Feature selection policy (fit on TRAIN with a safe starting set)
    start_cols = choose_starting_features(tr, label)
    tr_filt = tr[[label] + start_cols].copy()
    policy = FeaturePolicy(
        feat_select=args.feat_select,
        missing_thresh=args.missing_thresh,
        vif_thresh=args.vif_thresh,
        rfe_keep=args.rfe_keep,
    ).fit(tr_filt, label_col=label)

    # Transform splits (avoid NaNs) + cast to float32
    Xtr = policy.transform(tr)
    Xva = policy.transform(va)
    Xte = policy.transform(te)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=1e12, neginf=-1e12).astype(np.float32, copy=False)
    Xva = np.nan_to_num(Xva, nan=0.0, posinf=1e12, neginf=-1e12).astype(np.float32, copy=False)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=1e12, neginf=-1e12).astype(np.float32, copy=False)

    ytr = tr[label].astype(int).values
    yva = va[label].astype(int).values
    yte = te[label].astype(int).values

    # Cap training rows for memory
    if args.cap_train and len(ytr) > args.cap_train:
        Xtr, ytr = stratified_cap(Xtr, ytr, args.cap_train, seed=seed)
        print(f"[info] Capped TRAIN to {len(ytr)} rows (stratified) for TabPFN.")

    # Build & fit TabPFN
    try:
        clf = make_tabpfn(seed=seed, device=args.device, ensembles=args.ensembles, posterior=args.posterior)
    except RuntimeError as e:
        print(f"[SKIP] {args.scenario} | {model_name} | {calib} | seed={seed} → {e}")
        return
    tabpfn_fit(clf, Xtr, ytr)

    # Calibrate on VAL
    clf_cal = add_calibration(clf, Xva, yva, calib)

    # Predict probs helper
    def prob1(est, X):
        p = est.predict_proba(X)
        if p.ndim == 1: return p.astype(float)
        if p.shape[1] == 2: return p[:, 1].astype(float)
        if hasattr(est, "classes_"):
            classes = list(est.classes_)
            if 1 in classes: return p[:, classes.index(1)].astype(float)
        return p[:, -1].astype(float)

    p_va = prob1(clf_cal, Xva)
    p_te = prob1(clf_cal, Xte)

    # Save probs for UQ/bagging if requested
    if args.save_probs:
        probs_dir = Path(args.results_dir) / "probs"
        probs_dir.mkdir(parents=True, exist_ok=True)
        stemfile = f"{args.scenario}_{model_name}_{calib}_{args.feat_select}_seed{seed}"
        np.savez(probs_dir / f"{stemfile}.npz", p=p_te, y=yte)

    # Pick tau on VAL (balanced accuracy) & metrics on TEST
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
        "nll": log_loss(yte, np.vstack([1 - p_te, p_te]).T, labels=[0, 1]),
        "ece": expected_calibration_error(yte, p_te),
        "tau": float(tau),
    }

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stemfile = f"{args.scenario}_{model_name}_{calib}_{args.feat_select}_seed{seed}"
    pd.DataFrame([row]).to_csv(out_dir / f"{stemfile}.csv", index=False)
    with open(out_dir / f"{stemfile}_features.txt", "w") as f:
        f.write("\n".join(policy.selected_features_))
    print(f"[OK] {args.scenario} | {model_name} | {calib} | {args.feat_select} | seed={seed} "
          f"→ AUROC={row['auroc']:.4f}, ECE={row['ece']:.4f}, n_feats={row['n_features']}")

def main():
    args = parse_args()
    print(f"TabPFN evaluation → {args.scenario} | device={args.device} | feat_select={args.feat_select}")
    for c in args.calib:
        for s in args.seeds:
            run_once(args, c, s)

if __name__ == "__main__":
    main()
