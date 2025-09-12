#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, chi2_contingency
from statsmodels.stats.multitest import multipletests

EPS = 1e-8

def is_categorical(col: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(col):
        return True
    vals = col.dropna().unique()
    if len(vals) <= 10 and set(np.unique(vals)).issubset({0,1}):
        return True
    return False

def hist_divergences(x_src, x_tgt, bins=30):
    # common bins from pooled data, robust to outliers
    both = np.concatenate([x_src[~np.isnan(x_src)], x_tgt[~np.isnan(x_tgt)]])
    if both.size == 0:
        return np.nan, np.nan
    lo, hi = np.nanpercentile(both, [0.1, 99.9])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(both), np.nanmax(both)
        if lo == hi:  # constant
            return 0.0, 0.0
    edges = np.linspace(lo, hi, bins+1)
    ps, _ = np.histogram(x_src, bins=edges, density=False)
    pt, _ = np.histogram(x_tgt, bins=edges, density=False)
    ps = ps.astype(float); pt = pt.astype(float)
    ps = (ps + EPS) / (ps.sum() + EPS*bins)
    pt = (pt + EPS) / (pt.sum() + EPS*bins)
    # KL (both directions) + JS
    kl_st = float(np.sum(ps * np.log((ps + EPS)/(pt + EPS))))
    kl_ts = float(np.sum(pt * np.log((pt + EPS)/(ps + EPS))))
    skl   = 0.5*(kl_st + kl_ts)
    m     = 0.5*(ps + pt)
    js    = 0.5*np.sum(ps * np.log((ps + EPS)/(m + EPS))) + 0.5*np.sum(pt * np.log((pt + EPS)/(m + EPS)))
    return skl, float(js)

def cat_divergences(s_src: pd.Series, s_tgt: pd.Series):
    # KL/JS on category frequencies, plus χ² p-value
    vals = sorted(set(s_src.dropna().unique()).union(set(s_tgt.dropna().unique())))
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    ps = np.array([(s_src == v).sum() for v in vals], dtype=float)
    pt = np.array([(s_tgt == v).sum() for v in vals], dtype=float)
    ps = (ps + EPS) / (ps.sum() + EPS*len(vals))
    pt = (pt + EPS) / (pt.sum() + EPS*len(vals))
    kl_st = float(np.sum(ps * np.log((ps + EPS)/(pt + EPS))))
    kl_ts = float(np.sum(pt * np.log((pt + EPS)/(ps + EPS))))
    skl   = 0.5*(kl_st + kl_ts)
    m     = 0.5*(ps + pt)
    js    = 0.5*np.sum(ps * np.log((ps + EPS)/(m + EPS))) + 0.5*np.sum(pt * np.log((pt + EPS)/(m + EPS)))
    # χ² on contingency
    table = np.vstack([ps, pt])
    # rescale to counts for chi2 (multiply by total n)
    total = max((s_src.size + s_tgt.size), 1)
    chi2, p, _, _ = chi2_contingency(np.vstack([ps*total, pt*total]))
    return skl, float(js), float(p)

def load_split_csv(splits_dir: Path, split: str):
    tr = pd.read_csv(splits_dir / f"{split}_train.csv")
    va = pd.read_csv(splits_dir / f"{split}_val.csv")
    te = pd.read_csv(splits_dir / f"{split}_test.csv")
    return tr, va, te

def feature_drift_one_split(splits_dir: Path, split: str, label_col="hospital_mortality"):
    tr, va, te = load_split_csv(splits_dir, split)
    src = tr  # use TRAIN as source for covariate drift
    tgt = te  # use TEST as target

    rows = []
    pvals_cont = []
    pvals_cat  = []
    for c in src.columns:
        if c == label_col: 
            continue
        xs = src[c].values; xt = tgt[c].values
        if is_categorical(src[c]):
            skl, js, p = cat_divergences(src[c], tgt[c])
            rows.append(dict(feature=c, type="cat", ks=np.nan, w=np.nan, skl=skl, js=js, pval=p))
            pvals_cat.append(p)
        else:
            # KS and Wasserstein for continuous
            xsn = xs.astype(float); xtn = xt.astype(float)
            try:
                ks_stat = ks_2samp(xsn[~np.isnan(xsn)], xtn[~np.isnan(xtn)], alternative="two-sided", mode="auto").statistic
            except Exception:
                ks_stat = np.nan
            try:
                w = wasserstein_distance(xsn[~np.isnan(xsn)], xtn[~np.isnan(xtn)])
            except Exception:
                w = np.nan
            skl, js = hist_divergences(xsn, xtn, bins=30)
            rows.append(dict(feature=c, type="cont", ks=ks_stat, w=w, skl=skl, js=js, pval=np.nan))
            # approximate a p-value for KS (scipy returns it too, but we used statistic above)
            try:
                ks_res = ks_2samp(xsn[~np.isnan(xsn)], xtn[~np.isnan(xtn)], alternative="two-sided", mode="auto")
                pvals_cont.append(ks_res.pvalue)
                rows[-1]["pval"] = ks_res.pvalue
            except Exception:
                pass

    df = pd.DataFrame(rows)
    # BH-FDR
    if pvals_cont:
        mask = (df["type"]=="cont").values
        p = df.loc[mask, "pval"].fillna(1.0).values
        rej, _, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
        df.loc[mask, "fdr_sig"] = rej
    if pvals_cat:
        mask = (df["type"]=="cat").values
        p = df.loc[mask, "pval"].fillna(1.0).values
        rej, _, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
        df.loc[mask, "fdr_sig"] = rej
    df["fdr_sig"] = df["fdr_sig"].fillna(False)

    # split-level signature
    sig = dict(
        split=split,
        n_cont=int((df["type"]=="cont").sum()),
        n_cat =int((df["type"]=="cat").sum()),
        mean_ks=float(df.loc[df["type"]=="cont","ks"].mean(skipna=True)),
        mean_w =float(df.loc[df["type"]=="cont","w"].mean(skipna=True)),
        mean_js=float(df["js"].mean(skipna=True)),
        mean_skl=float(df["skl"].mean(skipna=True)),
        cont_sig_rate=float(df.loc[df["type"]=="cont","fdr_sig"].mean() if "fdr_sig" in df else 0.0),
        cat_sig_rate =float(df.loc[df["type"]=="cat","fdr_sig"].mean() if "fdr_sig" in df else 0.0),
    )
    return df, pd.DataFrame([sig])

def prediction_space_drift(preds_dir: Path, split: str, models=("lr","rf","xgb","lgbm"), calibs=("none","platt")):
    rows=[]
    for m in models:
        for c in calibs:
            f_test = preds_dir / f"{split}_{m}_{c}_seed42.csv"
            f_val  = preds_dir / f"{split}_{m}_{c}_seed42.csv"  # we don't have a separate val file saved; fallback: skip VAL vs TEST for random and use source vs target split semantics instead
            if not f_test.exists():
                continue
            te = pd.read_csv(f_test)
            p_tgt = te["p"].values
            if f_val.exists():
                p_src = pd.read_csv(f_val)["p"].values
            else:
                p_src = None

            # Wasserstein / JS on probs (binned)
            if p_src is not None:
                w = wasserstein_distance(p_src, p_tgt)
                # JS on binned probs
                edges = np.linspace(0,1,31)
                ps,_ = np.histogram(p_src, bins=edges, density=False); ps=(ps+EPS)/(ps.sum()+EPS*len(ps))
                pt,_ = np.histogram(p_tgt, bins=edges, density=False); pt=(pt+EPS)/(pt.sum()+EPS*len(pt))
                m_ = 0.5*(ps+pt)
                js = 0.5*np.sum(ps*np.log((ps+EPS)/(m_+EPS))) + 0.5*np.sum(pt*np.log((pt+EPS)/(m_+EPS)))
            else:
                w = np.nan; js = np.nan

            # entropy shift
            H = lambda p: -(p*np.log(p+EPS)+(1-p)*np.log(1-p+EPS))
            dH = float(np.nan) if p_src is None else float(np.mean(H(p_tgt)) - np.mean(H(p_src)))

            rows.append(dict(split=split, model=m, calib=c, w_pred=w, js_pred=js, delta_entropy=dH))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser("Compute KL/JS/Wasserstein/KS drift + prediction-space drift")
    ap.add_argument("--splits_dir", type=Path, default=Path("data/csv_splits"))
    ap.add_argument("--preds_dir",  type=Path, default=Path("results/preds"))
    ap.add_argument("--out",        type=Path, default=Path("results/shift_enhanced"))
    ap.add_argument("--label_col",  default="hospital_mortality")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sig_rows=[]

    for split in ("random","temporal","hospital"):
        print(f"[split] {split}")
        feat_df, sig_df = feature_drift_one_split(args.splits_dir, split, label_col=args.label_col)
        feat_df.to_csv(args.out / f"{split}_feature_drift.csv", index=False)
        sig_df.to_csv(args.out / f"{split}_signature.csv", index=False)
        sig_rows.append(sig_df)

        # prediction-space drift per model/calibration (if preds exist)
        ps = prediction_space_drift(args.preds_dir, split)
        ps.to_csv(args.out / f"{split}_prediction_space_drift.csv", index=False)

    sig_all = pd.concat(sig_rows, ignore_index=True)
    sig_all.to_csv(args.out / "shift_signatures_all.csv", index=False)

    # latex: split-level signature
    def fmt(x): 
        return "" if (pd.isna(x) or not np.isfinite(x)) else f"{x:.4f}"
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Shift signatures with KL/JS/Wasserstein/KS (continuous), and BH-FDR discovery rates.}",
        "\\label{tab:shift-signature-klw}",
        "\\begin{tabular}{l r r r r r r}",
        "\\toprule",
        "Split & Mean KS (cont) & Mean W (cont) & Mean JS (all) & Mean KL (all) & FDR sig (cont) & FDR sig (cat) \\\\",
        "\\midrule"
    ]
    order = {"random":0,"temporal":1,"hospital":2}
    sig_all["order"] = sig_all["split"].map(order)
    sig_all = sig_all.sort_values("order")
    for _,r in sig_all.iterrows():
        lines.append(f"{r['split'].title()} & {fmt(r['mean_ks'])} & {fmt(r['mean_w'])} & {fmt(r['mean_js'])} & {fmt(r['mean_skl'])} & {fmt(r['cont_sig_rate'])} & {fmt(r['cat_sig_rate'])} \\\\")
    lines += ["\\bottomrule","\\end{tabular}","\\end{table}"]
    (args.out/"shift_signature_klw.tex").write_text("\n".join(lines))

    print(f"[OK] wrote: {args.out}")

if __name__ == "__main__":
    main()
