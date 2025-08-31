#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt

# Try to import calibration_curve
try:
    from sklearn.calibration import calibration_curve
    HAS_SK = True
except Exception:
    HAS_SK = False

DATA = Path("data/splits")
OUT_TAB = Path("results/tables"); OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG = Path("results/figs"); OUT_FIG.mkdir(parents=True, exist_ok=True)

TARGET = "hospital_mortality"

def sym_kl(a, b, bins=50, eps=1e-9):
    a = np.asarray(a); b = np.asarray(b)
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]; b = b[finite]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    lo = np.nanmin([a.min(), b.min()])
    hi = np.nanmax([a.max(), b.max()])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hist_b, _     = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    pa = hist_a.astype(float) + eps
    pb = hist_b.astype(float) + eps
    pa /= pa.sum(); pb /= pb.sum()
    return float(np.sum(pa*np.log(pa/pb) + pb*np.log(pb/pa)))

def is_binary(s: pd.Series) -> bool:
    # treat {0,1} or {True,False} = binary!
    u = pd.unique(s.dropna())
    if len(u) <= 2:
        vals = set(map(lambda x: bool(x) if isinstance(x, (np.bool_, bool)) else x, u))
        return vals.issubset({0, 1, True, False})
    return False

def bh_fdr(pvals: np.ndarray, alpha=0.05):
    """Return adjusted p-values (Benjamini–Hochberg)."""
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty_like(pvals, dtype=float)
    cummin = 1.0
    for i, idx in enumerate(order, start=1):
        adj = pvals[idx] * m / i
        cummin = min(cummin, adj)
        ranked[idx] = cummin
    return np.clip(ranked, 0, 1)

def load_split(split: str):
    Xtr = pd.read_parquet(DATA / f"{split}_train.parquet")
    Xte = pd.read_parquet(DATA / f"{split}_test.parquet")
    # features only
    feat_cols = [c for c in Xtr.columns if c not in {TARGET, "patientunitstayid"}]
    return Xtr[feat_cols], Xte[feat_cols], feat_cols, Xte[TARGET].values if TARGET in Xte else None

def per_feature_stats(split: str) -> pd.DataFrame:
    Xtr, Xte, cols, _ = load_split(split)
    rows = []
    for c in cols:
        s_tr = Xtr[c]; s_te = Xte[c]
        # Decide type
        is_cat = (
            c.endswith("__missing") or
            c.startswith("unittype_") or
            c.startswith("diagnosis_bucket_") or
            is_binary(s_tr) and is_binary(s_te)
        )
        ks = symkl = chi2 = pval = np.nan

        if s_tr.dtype.kind in "fc" and s_te.dtype.kind in "fc" and not is_cat:
            # numeric
            try:
                ks = ks_2samp(s_tr.dropna(), s_te.dropna(), alternative="two-sided").statistic
            except Exception:
                ks = np.nan
            try:
                symkl = sym_kl(s_tr.values, s_te.values)
            except Exception:
                symkl = np.nan
        else:
            # categorical/binary: χ² on 2x2 (value in {0,1})
            try:
                a = np.array([np.sum(s_tr == 0), np.sum(s_tr == 1)], dtype=float)
                b = np.array([np.sum(s_te == 0), np.sum(s_te == 1)], dtype=float)
                table = np.vstack([a, b])
                chi2, pval, _, _ = chi2_contingency(table, correction=False)
            except Exception:
                chi2 = np.nan; pval = np.nan

        rows.append({"feature": c, "KS": ks, "symKL": symkl, "chi2": chi2, "pval": pval})
    df = pd.DataFrame(rows)
    # FDR on available p-values
    mask = df["pval"].notna()
    if mask.any():
        df.loc[mask, "pval_adj"] = bh_fdr(df.loc[mask, "pval"].values, alpha=0.05)
        df["significant_fdr_5pct"] = (df["pval_adj"] < 0.05).fillna(False)
    else:
        df["pval_adj"] = np.nan
        df["significant_fdr_5pct"] = False
    return df

def plot_numeric_hist(split: str, df: pd.DataFrame, top_n=8):
    Xtr, Xte, cols, _ = load_split(split)
    # choose top numeric by KS
    cand = df[df["KS"].notna()].sort_values("KS", ascending=False).head(top_n)["feature"].tolist()
    for c in cand:
        if Xtr[c].dtype.kind not in "fc":
            continue
        f = plt.figure(figsize=(5,4))
        tr = Xtr[c].to_numpy(); te = Xte[c].to_numpy()
        tr = tr[np.isfinite(tr)]; te = te[np.isfinite(te)]
        # subsample for speed
        if tr.size > 50000: tr = np.random.choice(tr, 50000, replace=False)
        if te.size > 50000: te = np.random.choice(te, 50000, replace=False)
        plt.hist(tr, bins=50, density=True, alpha=0.5, label="train")
        plt.hist(te, bins=50, density=True, alpha=0.5, label="test")
        plt.title(f"{split} — {c}\nKS={df.loc[df.feature==c,'KS'].values[0]:.3f}")
        plt.xlabel(c); plt.ylabel("density"); plt.legend()
        out = OUT_FIG / f"{split}_hist_{c}.png"
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(f)

def plot_categorical_bars(split: str, df: pd.DataFrame, top_n=8):
    Xtr, Xte, cols, _ = load_split(split)
    # pick top by pval_adj significance then chi2
    cat = df[df["pval"].notna()].copy()
    if cat.empty:
        return
    cat = cat.sort_values(["pval_adj","chi2"], ascending=[True, False]).head(top_n)
    for c in cat["feature"]:
        tr = Xtr[c]; te = Xte[c]
        p_tr = np.nanmean(tr.values.astype(float))
        p_te = np.nanmean(te.values.astype(float))
        f = plt.figure(figsize=(4,4))
        plt.bar(["train","test"], [p_tr, p_te])
        plt.ylim(0,1)
        pa = df.loc[df.feature==c, "pval_adj"].values[0]
        plt.title(f"{split} — {c}\np_adj={pa:.2e}")
        out = OUT_FIG / f"{split}_bar_{c}.png"
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(f)

def plot_calibration(split: str, model_pref=("lgbm","xgb","lr")):
    # find preds saved by tuned (prefer tuned) then baseline
    base_dirs = [Path("results/preds_tuned"), Path("results/preds")]
    yte = pd.read_parquet(DATA / f"{split}_test.parquet")[TARGET].values
    pred_path = None
    for d in base_dirs:
        for m in model_pref:
            cand = d / f"{split}_{m}_test.npy"
            if cand.exists():
                pred_path = cand; model = m; break
        if pred_path: break
    if pred_path is None or not HAS_SK:
        return
    p = np.load(pred_path)
    # clip for safety
    p = np.clip(p, 1e-7, 1-1e-7)
    prob_true, prob_pred = calibration_curve(yte, p, n_bins=15, strategy="uniform")
    f = plt.figure(figsize=(4.5,4.5))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"{split} — calibration ({model})")
    out = OUT_FIG / f"{split}_calibration_{model}.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random","temporal","hospital","hospital5src","all"], default="hospital5src")
    ap.add_argument("--topn", type=int, default=8)
    args = ap.parse_args()

    splits = ["random","temporal","hospital","hospital5src"] if args.split=="all" else [args.split]

    for sp in splits:
        print(f"[{sp}] computing χ²/KS/symKL + FDR …")
        df = per_feature_stats(sp)
        out_csv = OUT_TAB / f"shift_per_feature_{sp}_with_fdr.csv"
        df.to_csv(out_csv, index=False)
        print(f"  → table: {out_csv}")

        print(f"  plotting hists/bars/calibration …")
        plot_numeric_hist(sp, df, top_n=args.topn)
