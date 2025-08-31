#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, entropy, chi2_contingency

try:
    from statsmodels.stats.multitest import multipletests
    HAVE_SM = True
except Exception:
    HAVE_SM = False

import matplotlib.pyplot as plt

DATA = Path("data/splits")
TBL = Path("results/tables"); TBL.mkdir(parents=True, exist_ok=True)
FIG = Path("results/figs/shift"); FIG.mkdir(parents=True, exist_ok=True)

TARGET = "hospital_mortality"

def sym_kl(a, b, bins=50, eps=1e-9):
    hist_a, edges = np.histogram(a[~np.isnan(a)], bins=bins, density=True)
    hist_b, _ = np.histogram(b[~np.isnan(b)], bins=edges, density=True)
    pa = hist_a + eps; pa /= pa.sum()
    pb = hist_b + eps; pb /= pb.sum()
    return float(entropy(pa, pb) + entropy(pb, pa))

def chi2_p(x_tr, x_te, is_numeric):
    # numeric - bin to 10 quantiles; binary/categorical → use as-is
    if is_numeric:
        x = np.concatenate([x_tr, x_te])
        try:
            qs = np.unique(np.nanquantile(x[~np.isnan(x)], np.linspace(0,1,11)))
        except Exception:
            qs = np.linspace(np.nanmin(x), np.nanmax(x), 11)
        def binv(v): 
            if np.isnan(v): return -1
            return int(np.digitize(v, qs, right=False)) - 1
        tr_bins = np.array([binv(v) for v in x_tr])
        te_bins = np.array([binv(v) for v in x_te])
        cats = np.unique(np.concatenate([tr_bins, te_bins]))
        table = np.vstack([(tr_bins==c).sum() for c in cats] + [(te_bins==c).sum() for c in cats])
        table = table.reshape(2, -1)
    else:
        # treat any nonzero as 1
        tr1, te1 = np.nansum(x_tr!=0), np.nansum(x_te!=0)
        tr0, te0 = np.sum(~np.isnan(x_tr)) - tr1, np.sum(~np.isnan(x_te)) - te1
        table = np.array([[tr0, tr1],[te0, te1]])
    try:
        _, p, _, _ = chi2_contingency(table)
    except Exception:
        p = 1.0
    return float(p)

def plot_feature(split, feat, x_tr, x_te, is_numeric):
    outdir = FIG / split; outdir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if is_numeric:
        plt.hist(x_tr[~np.isnan(x_tr)], bins=40, density=True, alpha=0.5, label="train")
        plt.hist(x_te[~np.isnan(x_te)], bins=40, density=True, alpha=0.5, label="test")
        plt.xlabel(feat); plt.ylabel("density"); plt.legend()
    else:
        a = np.nanmean((x_tr!=0).astype(float))
        b = np.nanmean((x_te!=0).astype(float))
        plt.bar(["train","test"], [a,b])
        plt.ylabel("prevalence"); plt.title(feat)
    plt.tight_layout()
    plt.savefig(outdir / f"{feat}.png", dpi=160)
    plt.close()

def per_feature_stats(split):
    df_tr = pd.read_parquet(DATA / f"{split}_train.parquet")
    df_te = pd.read_parquet(DATA / f"{split}_test.parquet")
    ytr, yte = df_tr[TARGET].values, df_te[TARGET].values
    Xtr, Xte = df_tr.drop(columns=[TARGET]), df_te.drop(columns=[TARGET])

    rows, pvals = [], []
    for c in Xtr.columns:
        # detect numeric vs binary
        s_tr, s_te = Xtr[c].values, Xte[c].values
        non_nan = np.concatenate([s_tr[~np.isnan(s_tr)], s_te[~np.isnan(s_te)]]) if np.issubdtype(s_tr.dtype, np.number) else np.array([])
        is_numeric = np.issubdtype(s_tr.dtype, np.number) and len(np.unique(non_nan)) > 5

        # KS only for numeric
        KS = ks_2samp(s_tr[~np.isnan(s_tr)], s_te[~np.isnan(s_te)]).statistic if is_numeric else np.nan
        KL = sym_kl(s_tr.astype(float), s_te.astype(float)) if is_numeric else np.nan
        p = chi2_p(s_tr.astype(float), s_te.astype(float), is_numeric)
        pvals.append(p)

        rows.append({"feature": c, "KS": KS, "symKL": KL, "chi2_p": p, "type": "num" if is_numeric else "cat"})

    df = pd.DataFrame(rows)
    # BH-FDR
    if HAVE_SM:
        _, q, _, _ = multipletests(df["chi2_p"].values, alpha=0.05, method="fdr_bh")
        df["q_bh"] = q
    else:
        # manual BH
        p = np.clip(df["chi2_p"].values, 0, 1)
        order = np.argsort(p)
        m = len(p); q = np.empty_like(p)
        prev = 1.0
        for rank, idx in enumerate(order[::-1], start=1):
            i = m - rank + 1
            prev = min(prev, p[idx] * m / i)
            q[idx] = prev
        df["q_bh"] = q

    # summary numbers
    prev_tr, prev_te = float(np.mean(ytr)), float(np.mean(yte))
    H = lambda p: -(p*np.log(p+1e-12) + (1-p)*np.log(1-p+1e-12))
    dH = H(prev_te) - H(prev_tr)

    df.to_csv(TBL / f"shift_per_feature_{split}.csv", index=False)
    topks = df.sort_values("KS", ascending=False).head(10)
    topsym = df.sort_values("symKL", ascending=False).head(10)

    # quick plots of top-shifted 6 features (numeric prioritized)
    plotted = 0
    for feat in pd.concat([topks["feature"], topsym["feature"]]).unique():
        s_tr, s_te = Xtr[feat].values, Xte[feat].values
        is_numeric = (df.loc[df["feature"]==feat, "type"].iloc[0] == "num")
        plot_feature(split, feat, s_tr, s_te, is_numeric)
        plotted += 1
        if plotted >= 6: break

    # train/test AUROC from saved LGBM preds if present
    auc_tr = auc_te = np.nan
    p_tr = TBL.parent / "preds" / f"{split}_lgbm.csv"
    if p_tr.exists():
        preds = pd.read_csv(p_tr)
        if len(preds) == len(yte):
            auc_te = float(pd.Series(preds["p"]).pipe(lambda s: roc_auc_score(yte, s)))
    try:
        from sklearn.metrics import roc_auc_score
        auc_tr = np.nan  # (we didn't save train preds)
    except Exception:
        pass

    meanKS = float(np.nanmean(df["KS"])) if "KS" in df else np.nan
    meanKL = float(np.nanmean(df["symKL"])) if "symKL" in df else np.nan

    summary = dict(split=split, mean_KS=meanKS, mean_symKL=meanKL, dH=dH,
                   prev_tr=prev_tr, prev_te=prev_te, AUC_tr=auc_tr, AUC_te=auc_te)
    pd.DataFrame([summary]).to_csv(TBL / ("shift_summary.csv"), index=False, mode="a", header=not (TBL / "shift_summary.csv").exists())
    print(f"[{split}] mean_KS={meanKS:.4f}, mean_symKL={meanKL:.4f}, ΔH={dH:.4f}, prev_tr={prev_tr:.3f}, prev_te={prev_te:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random","temporal","hospital","hospital5src","all"], default="all")
    args = ap.parse_args()
    splits = ["random","temporal","hospital","hospital5src"] if args.split=="all" else [args.split]
    for s in splits:
        per_feature_stats(s)

if __name__ == "__main__":
    main()
