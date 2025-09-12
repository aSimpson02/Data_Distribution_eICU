#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, accuracy_score
)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# columns we never let into X
STATIC_DROP = {
    "patientunitstayid","hospitalid","hospital_id","site_id","site",
    "hospitaldischargeyear","apachescore","predictedhospitalmortality","admissionoffset"
}

def find_hosp_col(df: pd.DataFrame) -> str:
    cands = ["hospitalid","hospital_id","site_id","site"]
    lower = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in lower:
            return lower[k]
    raise SystemExit("Could not find a hospital ID column (tried hospitalid/hospital_id/site_id/site).")

def ece(y, p, bins=15):
    y = np.asarray(y).astype(int); p = np.asarray(p)
    edges = np.linspace(0,1,bins+1); out = 0.0
    idx = np.digitize(p, edges) - 1
    n = len(y)
    for b in range(bins):
        m = idx==b
        if m.any():
            out += (m.sum()/n) * abs(y[m].mean() - p[m].mean())
    return float(out)

def load_pooled_from_random(splits_dir: Path, label_col: str) -> pd.DataFrame:
    d = splits_dir
    parts = [pd.read_csv(d/"random_train.csv"),
             pd.read_csv(d/"random_val.csv"),
             pd.read_csv(d/"random_test.csv")]
    df = pd.concat(parts, axis=0, ignore_index=True)
    if label_col not in df.columns:
        raise SystemExit(f"label_col '{label_col}' not in pooled df.")
    return df

def prepare_xy(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    # keep numeric + one-hot-ish “diagnosis_bucket_” columns; drop known leakage/IDs
    num = df.select_dtypes(include=[np.number]).copy()
    drop = {label_col} | {c for c in num.columns if c.lower() in STATIC_DROP}
    X0 = num.drop(columns=[c for c in drop if c in num.columns], errors="ignore")
    diag = [c for c in df.columns if str(c).startswith("diagnosis_bucket_")]
    X = pd.concat([X0, df[diag]], axis=1) if diag else X0
    y = df[label_col].astype(int).to_numpy()
    return X, y

def train_lgbm(Xtr, ytr, Xva, yva, seed=42):
    spw = float(max(1.0, (ytr==0).sum()/max(1,(ytr==1).sum())))
    clf = LGBMClassifier(
        objective="binary", n_estimators=2000, learning_rate=0.03,
        num_leaves=64, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, scale_pos_weight=spw
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva,yva)], eval_metric="auc",
            callbacks=[early_stopping(100, first_metric_only=True), log_evaluation(0)])
    best = getattr(clf, "best_iteration_", None)
    return clf, (int(best) if best else None)

def metrics(y, p):
    return dict(
        auroc=float(roc_auc_score(y,p)),
        ece=float(ece(y,p)),
        brier=float(brier_score_loss(y,p)),
        nll=float(log_loss(y,p,labels=[0,1])),
        acc=float(accuracy_score(y,(p>=0.5).astype(int))),
    )

def bootstrap_ci(vals, B=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    if len(v)==1:
        return float(v[0]), float(v[0])
    idx = rng.integers(0, len(v), size=(B, len(v)))
    boots = v[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser("LOHO-CV over hospitals (built from random_* splits)")
    ap.add_argument("--splits_dir", type=Path, default=Path("data/csv_splits"))
    ap.add_argument("--label_col", type=str, default="hospital_mortality")
    ap.add_argument("--min-per-hospital", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results/loho"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    pooled = load_pooled_from_random(args.splits_dir, args.label_col)
    hosp_col = find_hosp_col(pooled)
    # filter hospitals by size
    sizes = pooled.groupby(hosp_col).size()
    hospitals = [h for h,n in sizes.items() if n >= args.min_per_hospital]
    if len(hospitals) < 2:
        raise SystemExit("Not enough hospitals meeting min-per-hospital.")

    folds = []
    for h in hospitals:
        df_te = pooled[pooled[hosp_col]==h].copy()
        df_src = pooled[pooled[hosp_col]!=h].copy()
        # train/val from source (stratified 80/20)
        y_src = df_src[args.label_col].astype(int).to_numpy()
        sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=args.seed)
        idx_tr, idx_va = next(sss.split(np.zeros(len(y_src)), y_src))
        df_tr = df_src.iloc[idx_tr].copy()
        df_va = df_src.iloc[idx_va].copy()

        Xtr,ytr = prepare_xy(df_tr, args.label_col)
        Xva,yva = prepare_xy(df_va, args.label_col)
        Xte,yte = prepare_xy(df_te, args.label_col)

        clf, best = train_lgbm(Xtr,ytr,Xva,yva, seed=args.seed)
        # base
        pt = (clf.predict_proba(Xte, num_iteration=best)[:,1]
              if best else clf.predict_proba(Xte)[:,1])
        base = metrics(yte, pt)
        # Platt on VAL → TEST
        pv = (clf.predict_proba(Xva, num_iteration=best)[:,1]
              if best else clf.predict_proba(Xva)[:,1])
        pl = LogisticRegression(max_iter=1000, solver="lbfgs").fit(pv.reshape(-1,1), yva)
        pt_pl = pl.predict_proba(pt.reshape(-1,1))[:,1]
        pl_m = metrics(yte, pt_pl)

        fold = dict(heldout=str(h), n_test=int(len(yte)), base=base, platt=pl_m)
        folds.append(fold)
        (args.out / f"fold_{h}.json").write_text(json.dumps(fold, indent=2))
        print(f"[LOHO] held-out={h:>6}  AUC(base)={base['auroc']:.4f}  ECE(base)={base['ece']:.4f}  n={len(yte)}")

    # aggregate mean across hospitals and bootstrap CI
    agg = {}
    for key in ("base","platt"):
        for m in ("auroc","ece","brier","nll","acc"):
            vals = [f[key][m] for f in folds]
            mean = float(np.mean(vals))
            lo, hi = bootstrap_ci(vals, seed=args.seed)
            agg[f"{key}_{m}_mean"] = mean
            agg[f"{key}_{m}_ci95"] = [lo, hi]

    summary = dict(hospitals=[str(h) for h in hospitals], folds=folds, aggregate=agg,
                   label_col=args.label_col, min_per_hospital=args.min_per_hospital)
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[Aggregate]")
    print(f"  base : AUROC mean={agg['base_auroc_mean']:.4f}  95%CI={agg['base_auroc_ci95']}")
    print(f"  platt: AUROC mean={agg['platt_auroc_mean']:.4f}  95%CI={agg['platt_auroc_ci95']}")
    print(f"  base : ECE   mean={agg['base_ece_mean']:.4f}    95%CI={agg['base_ece_ci95']}")
    print(f"  platt: ECE   mean={agg['platt_ece_mean']:.4f}   95%CI={agg['platt_ece_ci95']}")

if __name__ == "__main__":
    main()
