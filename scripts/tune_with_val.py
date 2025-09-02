#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    balanced_accuracy_score,
    f1_score,
)

# repo root
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from src.data.io import load_three, drop_leaky
from src.feats.selector import FeaturePolicy
from scripts.build_models import build_model, get_model_names

STEM_MAP = {
    "iid": "random",
    "random": "random",
    "hospital": "hospital",
    "hospital_ood": "hospital",
    "temporal": "temporal",
    "temporal_ood": "temporal",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=list(STEM_MAP.keys()))
    p.add_argument("--models", nargs="+", default=["lr","rf","xgb","lgbm"], choices=get_model_names())
    p.add_argument("--calib", nargs="+", default=["none","platt"], choices=["none","platt","isotonic"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--label_col", required=True)
    p.add_argument("--splits_dir", default="data/csv_splits")
    p.add_argument("--results_dir", default="results")
    return p.parse_args()

def choose_features(df: pd.DataFrame, label_col: str):
    keep=[]
    for c in df.columns:
        if c == label_col: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or str(c).startswith("diagnosis_bucket_"):
            keep.append(c)
    return keep

def ece(y, p, bins=20):
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    edges = np.linspace(0,1,bins+1)
    out=0.0
    for i in range(bins):
        lo,hi = edges[i], edges[i+1]
        m = (p>=lo)&(p<(hi if i<bins-1 else hi))
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)

def _dump_preds(split: str, model: str, seed: int, calib_tag: str, y_true, p, results_dir: Path):
    preds_dir = Path(results_dir) / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    out = preds_dir / f"{split}_{model}_{calib_tag}_seed{seed}.csv"
    pd.DataFrame({"y_true": y_true, "p": p}).to_csv(out, index=False)
    print(f"[preds] wrote {out}")

def main():
    a = parse_args()
    stem = STEM_MAP[a.scenario]

    tr, va, te = load_three(stem, a.splits_dir)
    tr = drop_leaky(tr, a.label_col)
    va = drop_leaky(va, a.label_col)
    te = drop_leaky(te, a.label_col)

    ytr = tr[a.label_col].astype(int).values
    yva = va[a.label_col].astype(int).values
    yte = te[a.label_col].astype(int).values

    start = choose_features(tr, a.label_col)
    policy = FeaturePolicy(feat_select="none", missing_thresh=0.40).fit(
        tr[[a.label_col] + start], label_col=a.label_col
    )
    Xtr = policy.transform(tr)
    Xva = policy.transform(va)
    Xte = policy.transform(te)

    results_dir = Path(a.results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    for m in a.models:
        for s in a.seeds:
            try:
                base = build_model(m, s)
            except RuntimeError as e:
                print(f"[SKIP] {a.scenario} | {m} | seed={s} -> {e}")
                continue

            # XGBoost name-safety
            is_xgb = (m == "xgb")
            Xtr_in = Xtr.to_numpy() if is_xgb else Xtr
            Xva_in = Xva.to_numpy() if is_xgb else Xva
            Xte_in = Xte.to_numpy() if is_xgb else Xte

            base.fit(Xtr_in, ytr)

            for calib in a.calib:
                if calib == "none":
                    p = base.predict_proba(Xte_in)[:, 1]
                else:
                    method = "sigmoid" if calib == "platt" else "isotonic"
                    cal = CalibratedClassifierCV(base, method=method, cv="prefit")
                    cal.fit(Xva_in, yva)
                    p = cal.predict_proba(Xte_in)[:, 1]

                yhat = (p >= 0.5).astype(int)

                _dump_preds(a.scenario, m, s, calib, yte, p, results_dir)

                row = dict(
                    scenario=a.scenario,
                    Htgt=a.scenario,
                    model=m,
                    seed=s,
                    source="tuned",
                    calib=calib,
                    calib_tag=calib,
                    feat="none",
                    feat_select="none",
                    n_features=len(policy.selected_features_),
                    auroc=roc_auc_score(yte, p),
                    auprc=average_precision_score(yte, p),
                    brier=brier_score_loss(yte, p),
                    nll=log_loss(yte, p, labels=[0, 1]),
                    ece=ece(yte, p),
                    tau=0.5,
                    f1_at_tau=f1_score(yte, yhat),
                    balacc_at_tau=balanced_accuracy_score(yte, yhat),
                )
                out = results_dir / f"{a.scenario}_{m}_{calib}_none_seed{s}.csv"
                pd.DataFrame([row]).to_csv(out, index=False)
                print(f"[OK][tuned] {a.scenario} | {m} | {calib} | seed={s} -> AUROC={row['auroc']:.4f}, ECE={row['ece']:.4f}")

if __name__ == "__main__":
    main()
