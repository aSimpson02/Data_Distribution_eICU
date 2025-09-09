#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score, mean_absolute_error
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

STATIC_BLOCK = {"hospital_mortality","apachescore","predictedhospitalmortality"}

def ece(y,p,bins=15):
    y=np.asarray(y); p=np.asarray(p)
    edges=np.linspace(0,1,bins+1); idx=np.digitize(p,edges)-1
    n=len(y); e=0.0
    for b in range(bins):
        m=idx==b
        if m.any(): e+=(m.sum()/n)*abs(y[m].mean()-p[m].mean())
    return float(e)

def load(root: Path, split: str):
    d = root/split
    tr=pd.read_parquet(d/"train.parquet"); va=pd.read_parquet(d/"val.parquet"); te=pd.read_parquet(d/"test.parquet")
    lab = "label" if "label" in tr.columns else "hospital_mortality"
    return tr, va, te, lab

def prep(df, lab, keep=None, med=None):
    df=df.copy(); df.columns=df.columns.str.replace(r"\s+","_",regex=True)
    num=df.select_dtypes(include=[np.number])
    drop={lab}|STATIC_BLOCK|{c for c in num.columns if "id" in c.lower()}
    X0=num.drop(columns=[c for c in drop if c in num.columns], errors="ignore")
    if keep is None:
        keep=X0.columns.tolist(); med=X0.median(numeric_only=True)
    return X0.reindex(columns=keep).fillna(med), df[lab].astype(int).to_numpy(), keep, med

def metr(y,p): 
    return dict(auroc=float(roc_auc_score(y,p)), ece=float(ece(y,p)),
                brier=float(brier_score_loss(y,p)), nll=float(log_loss(y,p)),
                acc=float(accuracy_score(y,(p>=0.5).astype(int))), mae=float(mean_absolute_error(y,p)))

def run(root: Path, split: str, n: int, seed: int, out: Path):
    tr,va,te,lab = load(root,split)
    Xtr,ytr,keep,med = prep(tr,lab)
    Xva,yva,_,_      = prep(va,lab,keep,med)
    Xte,yte,_,_      = prep(te,lab,keep,med)

    proba_va = []; proba_te = []
    for i in range(n):
        spw = float(max(1.0, (ytr==0).sum()/max(1,(ytr==1).sum())))
        clf = LGBMClassifier(objective="binary", n_estimators=1200, learning_rate=0.05,
                             num_leaves=64, subsample=0.8, colsample_bytree=0.8,
                             random_state=seed+i, n_jobs=-1, scale_pos_weight=spw)
        clf.fit(Xtr,ytr, eval_set=[(Xva,yva)], eval_metric="auc",
                callbacks=[early_stopping(100, first_metric_only=True), log_evaluation(0)])
        best = getattr(clf,"best_iteration_", None)
        pv   = clf.predict_proba(Xva, num_iteration=best)[:,1] if best else clf.predict_proba(Xva)[:,1]
        pt   = clf.predict_proba(Xte, num_iteration=best)[:,1] if best else clf.predict_proba(Xte)[:,1]
        proba_va.append(pv); proba_te.append(pt)

    Pva = np.vstack(proba_va)      # [n, n_val]
    Pte = np.vstack(proba_te)      # [n, n_test]
    mu_va = Pva.mean(axis=0); mu_te = Pte.mean(axis=0)
    var_te = Pte.var(axis=0)       # epistemic variance

    # Calibrate ensemble mean on VAL
    pl = LogisticRegression(max_iter=1000).fit(mu_va.reshape(-1,1), yva)
    mu_te_pl = pl.predict_proba(mu_te.reshape(-1,1))[:,1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(mu_va, yva)
    mu_te_iso = iso.transform(mu_te)

    base, pl_m, iso_m = metr(yte, mu_te), metr(yte, mu_te_pl), metr(yte, mu_te_iso)

    out.mkdir(parents=True, exist_ok=True)
    payload = dict(split=split, model="lgbm_ensemble", members=n,
                   base=base, platt=pl_m, isotonic=iso_m,
                   var_stats=dict(mean=float(var_te.mean()), median=float(np.median(var_te))))
    (out/f"ensemble_{split}.json").write_text(json.dumps(payload, indent=2))
    print(f"{split.upper():9s} ENSEMBLE({n})  AUC={base['auroc']:.4f}  ECE={base['ece']:.4f}  var(mean)={payload['var_stats']['mean']:.6f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("LGBM Deep Ensemble")
    ap.add_argument("--root", type=Path, default=Path("yaib_data/mortality24/eicu"))
    ap.add_argument("--split", type=str, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("yaib_logs/ensemble"))
    args = ap.parse_args()
    run(args.root, args.split, args.n, args.seed, args.out)
