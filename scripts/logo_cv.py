#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, math, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

LABEL_CANDS = ("label","hospital_mortality")
HOSP_CANDS  = ("hospital_id","hospitalid","site_id","site","hospital")

STATIC_BLOCK = {"hospital_mortality","apachescore","predictedhospitalmortality"}
SEED=42
rng = np.random.default_rng(SEED)

def find_col(df: pd.DataFrame, cands: Tuple[str,...]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in lower: return lower[k]
    return None

def ece(y,p,bins=15):
    y=np.asarray(y); p=np.asarray(p)
    edges=np.linspace(0,1,bins+1); idx=np.digitize(p,edges)-1
    n=len(y); e=0.0
    for b in range(bins):
        m=idx==b
        if m.any():
            e+=(m.sum()/n)*abs(y[m].mean()-p[m].mean())
    return float(e)

def load_pooled(path: Path) -> pd.DataFrame:
    if path.suffix.lower()==".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def split_loho(df: pd.DataFrame, hosp_col: str, heldout: str, val_frac=0.2) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    df_tr = df[df[hosp_col]!=heldout].copy()
    df_te = df[df[hosp_col]==heldout].copy()
    # build val from train (stratified by label)
    lab = find_col(df, LABEL_CANDS); assert lab
    df_tr = df_tr.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pos = df_tr[df_tr[lab]==1]; neg = df_tr[df_tr[lab]==0]
    n_val_pos = max(1, int(len(pos)*val_frac)); n_val_neg = max(1, int(len(neg)*val_frac))
    df_va = pd.concat([pos.iloc[:n_val_pos], neg.iloc[:n_val_neg]], axis=0).sample(frac=1.0, random_state=SEED)
    df_tr2= pd.concat([pos.iloc[n_val_pos:], neg.iloc[n_val_neg:]], axis=0)
    return df_tr2, df_va, df_te

def prepare(df: pd.DataFrame, lab: str, keep=None, med=None):
    df = df.copy()
    df.columns = df.columns.str.replace(r"\s+","_",regex=True)
    num = df.select_dtypes(include=[np.number]).copy()
    drop = {lab} | STATIC_BLOCK | {c for c in num.columns if "id" in c.lower()}
    X0 = num.drop(columns=[c for c in drop if c in num.columns], errors="ignore")
    if keep is None:
        keep = X0.columns.tolist()
        med  = X0.median(numeric_only=True)
    X = X0.reindex(columns=keep).fillna(med)
    y = df[lab].astype(int).to_numpy()
    return X,y,keep,med

def run_one_fold(df: pd.DataFrame, hosp_col: str, heldout: str, seed=SEED) -> Dict:
    lab = find_col(df, LABEL_CANDS); assert lab, "label column not found"
    tr, va, te = split_loho(df, hosp_col, heldout)
    Xtr,ytr,keep,med = prepare(tr, lab)
    Xva,yva,_,_      = prepare(va, lab, keep, med)
    Xte,yte,_,_      = prepare(te, lab, keep, med)

    spw = float(max(1.0, (ytr==0).sum()/max(1,(ytr==1).sum())))
    clf = LGBMClassifier(
        objective="binary", n_estimators=2000, learning_rate=0.03,
        num_leaves=64, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, scale_pos_weight=spw
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva,yva)], eval_metric="auc",
            callbacks=[early_stopping(100, first_metric_only=True), log_evaluation(0)])
    best = getattr(clf, "best_iteration_", None)
    pv   = clf.predict_proba(Xva, num_iteration=best)[:,1] if best else clf.predict_proba(Xva)[:,1]
    pt   = clf.predict_proba(Xte, num_iteration=best)[:,1] if best else clf.predict_proba(Xte)[:,1]

    # calibrate on val
    pl = LogisticRegression(solver="lbfgs", max_iter=1000)
    pl.fit(pv.reshape(-1,1), yva)
    pt_pl = pl.predict_proba(pt.reshape(-1,1))[:,1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(pv, yva)
    pt_iso = iso.transform(pt)

    def metr(p):
        return dict(
            auroc=float(roc_auc_score(yte, p)),
            ece=float(ece(yte, p)),
            brier=float(brier_score_loss(yte, p)),
            nll=float(log_loss(yte, p)),
            acc=float(accuracy_score(yte, (p>=0.5).astype(int))),
            mae=float(mean_absolute_error(yte, p)),
        )
    base, pl_m, iso_m = metr(pt), metr(pt_pl), metr(pt_iso)

    return dict(heldout=heldout, n_test=len(yte), base=base, platt=pl_m, isotonic=iso_m)

def bootstrap_ci(values: List[float], B=5000, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if len(vals)==1: return float(vals[0]), float(vals[0]), float(vals[0])
    boots = np.array([vals[rng.integers(0,len(vals), size=len(vals))].mean() for _ in range(B)])
    lo, mid, hi = np.quantile(boots, [alpha/2, 0.5, 1-alpha/2])
    return float(lo), float(mid), float(hi)

def aggregate(fold_metrics: List[Dict]) -> Dict:
    # mean across folds + bootstrap CI over folds (simple, transparent)
    agg = {}
    for key in ("base","platt","isotonic"):
        for m in ("auroc","ece","brier","nll","acc","mae"):
            vals = [fm[key][m] for fm in fold_metrics]
            lo, mid, hi = bootstrap_ci(vals)
            agg[f"{key}_{m}_mean"] = float(np.mean(vals))
            agg[f"{key}_{m}_ci95"] = [lo, hi]
    return agg

def main():
    ap = argparse.ArgumentParser("LOHO-CV with bootstrap CIs")
    ap.add_argument("--pooled", type=Path, required=True, help="Pooled dataset (CSV/Parquet) with hospital_id + label")
    ap.add_argument("--min-per-hospital", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("yaib_logs/loho"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    df = load_pooled(args.pooled)
    lab = find_col(df, LABEL_CANDS)
    hosp_col = find_col(df, HOSP_CANDS)
    if lab is None or hosp_col is None:
        raise SystemExit("Need a label column (label/hospital_mortality) and a hospital column (hospital_id/...)")

    # filter hospitals by size
    sizes = df.groupby(hosp_col).size()
    hospitals = [h for h,n in sizes.items() if n >= args.min_per_hospital]
    if len(hospitals) < 2:
        raise SystemExit("Not enough hospitals meeting min-per-hospital")

    outdir = args.out; outdir.mkdir(parents=True, exist_ok=True)
    folds = []
    for h in hospitals:
        fm = run_one_fold(df, hosp_col, str(h), seed=args.seed)
        (outdir / f"fold_{h}.json").write_text(json.dumps(fm, indent=2))
        print(f"[LOHO] held-out={h:>6s}  AUC={fm['base']['auroc']:.4f}  ECE={fm['base']['ece']:.4f}  n={fm['n_test']}")
        folds.append(fm)

    agg = aggregate(folds)
    (outdir/"summary.json").write_text(json.dumps({"folds":folds, "aggregate":agg}, indent=2))
    print("\n[Aggregate] AUROC (base) mean={:.4f}  95% CI=[{:.4f}, {:.4f}]".format(
        agg["base_auroc_mean"], agg["base_auroc_ci95"][0], agg["base_auroc_ci95"][1]
    ))

if __name__ == "__main__":
    main()
