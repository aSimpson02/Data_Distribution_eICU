#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

# repo root for imports (so src/... works)
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    balanced_accuracy_score, f1_score
)
from src.feats.selector import FeaturePolicy
from scripts.build_models import build_model, get_model_names

PREDICTOR_DROP = {
    "patientunitstayid","hospitalid","hospitaldischargeyear",
    "apachescore","predictedhospitalmortality","admissionoffset"
}

STEM_MAP = {
    "iid":"random","random":"random",
    "hospital":"hospital","hospital_ood":"hospital",
    "temporal":"temporal","temporal_ood":"temporal"
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=list(STEM_MAP.keys()))
    p.add_argument("--models", nargs="+", default=["lr","rf"], choices=get_model_names())
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--label_col", required=True)
    p.add_argument("--splits_dir", default="data/csv_splits")
    p.add_argument("--results_dir", default="results")
    return p.parse_args()

def load_three(stem: str, d: Path):
    tr = pd.read_csv(d / f"{stem}_train.csv")
    va = pd.read_csv(d / f"{stem}_val.csv")   # not used, but keep for symmetry
    te = pd.read_csv(d / f"{stem}_test.csv")
    return tr, va, te

def choose_features(df: pd.DataFrame, label_col: str):
    keep=[]
    for c in df.columns:
        if c == label_col or c in PREDICTOR_DROP:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or str(c).startswith("diagnosis_bucket_"):
            keep.append(c)
    return keep

def ece(y, p, bins=20):
    y = np.asarray(y).astype(int); p = np.asarray(p)
    edges = np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        lo,hi = edges[i], edges[i+1]
        m = (p>=lo)&(p<(hi if i<bins-1 else hi))
        if m.any():
            out += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(out)

def main():
    a = parse_args()
    stem = STEM_MAP[a.scenario]
    splits = Path(a.splits_dir)

    tr, _, te = load_three(stem, splits)
    ytr = tr[a.label_col].astype(int).values
    yte = te[a.label_col].astype(int).values

    # simple feature “policy” (no selection; just cleaning & missing handling)
    start = choose_features(tr, a.label_col)
    policy = FeaturePolicy(feat_select="none", missing_thresh=0.40).fit(
        tr[[a.label_col] + start], label_col=a.label_col
    )
    Xtr = policy.transform(tr)
    Xte = policy.transform(te)

    out_dir = Path(a.results_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for m in a.models:
        for s in a.seeds:
            try:
                clf = build_model(m, s)
            except RuntimeError as e:
                print(f"[SKIP] {a.scenario} | {m} | notuned | seed={s} → {e}")
                continue

            # === XGBoost name-safety: use NumPy to drop column names with [,],<,> ===
            is_xgb = (m == "xgb")
            Xtr_in = Xtr.to_numpy() if is_xgb else Xtr
            Xte_in = Xte.to_numpy() if is_xgb else Xte

            clf.fit(Xtr_in, ytr)
            p = clf.predict_proba(Xte_in)[:, 1]
            yhat = (p >= 0.5).astype(int)

            row = dict(
                scenario=a.scenario, Htgt=a.scenario, model=m, seed=s,
                calib="none", feat="notuned", feat_select="none",
                n_features=len(policy.selected_features_),
                auroc=roc_auc_score(yte, p),
                auprc=average_precision_score(yte, p),
                brier=brier_score_loss(yte, p),
                nll=log_loss(yte, p, labels=[0,1]),
                ece=ece(yte, p),
                tau=0.5,
                f1_at_tau=f1_score(yte, yhat),
                balacc_at_tau=balanced_accuracy_score(yte, yhat),
            )
            out = out_dir / f"{a.scenario}_{m}_notuned_seed{s}.csv"
            pd.DataFrame([row]).to_csv(out, index=False)
            print(f"[OK][no-tune] {a.scenario} | {m} | seed={s} → AUROC={row['auroc']:.4f}, ECE={row['ece']:.4f}")

if __name__ == "__main__":
    main()
