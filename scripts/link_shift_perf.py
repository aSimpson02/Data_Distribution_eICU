#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st

CALIBRATORS = ["none","platt"]  # extend if you have others

def norm_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "split" not in df.columns and "scenario" in df.columns:
        df = df.rename(columns={"scenario":"split"})
    return df

def wide_to_long(df):
    rows=[]
    for _, r in df.iterrows():
        split = str(r["split"]).lower()
        model = r["model"]
        au_not = r.get("auroc_notuned", np.nan)
        ec_not = r.get("ece_notuned",   np.nan)
        for cal in CALIBRATORS:
            au_col = f"auroc_{cal}"
            ec_col = f"ece_{cal}"
            if au_col in r.index and ec_col in r.index and pd.notnull(r[au_col]) and pd.notnull(r[ec_col]):
                au = float(r[au_col]); ec = float(r[ec_col])
                dau_col = f"delta_auroc_{cal}_minus_notuned"
                dec_col = f"delta_ece_{cal}_minus_notuned"
                dau_val = r.get(dau_col, np.nan)
                dec_val = r.get(dec_col, np.nan)
                # derive from NOTUNED if delta_* is missing/NaN
                if pd.isna(dau_val) and pd.notnull(au_not):
                    dau = float(au_not) - au
                else:
                    dau = float(dau_val) if pd.notnull(dau_val) else np.nan
                if pd.isna(dec_val) and pd.notnull(ec_not):
                    dec = ec - float(ec_not)
                else:
                    dec = float(dec_val) if pd.notnull(dec_val) else np.nan
                rows.append({
                    "split": split,
                    "model": model,
                    "calibrator": cal,
                    "auroc": au,
                    "ece": ec,
                    "auroc_notuned": float(au_not) if pd.notnull(au_not) else np.nan,
                    "ece_notuned":   float(ec_not) if pd.notnull(ec_not) else np.nan,
                    "delta_auroc_notuned": dau,
                    "delta_ece_notuned":   dec
                })
    return pd.DataFrame(rows)

def attach_random_deltas(long_df):
    if not (long_df["split"].eq("random").any()):
        long_df["delta_auroc_random"] = np.nan
        long_df["delta_ece_random"]   = np.nan
        return long_df
    base = (long_df[long_df.split=="random"]
            .set_index(["model","calibrator"])[["auroc","ece"]]
            .rename(columns={"auroc":"auroc_random","ece":"ece_random"}))
    out = (long_df.set_index(["model","calibrator"])
                    .join(base, how="left")
                    .reset_index())
    mask = out["split"]!="random"
    out.loc[mask,"delta_auroc_random"] = out.loc[mask,"auroc_random"] - out.loc[mask,"auroc"]
    out.loc[mask,"delta_ece_random"]   = out.loc[mask,"ece"] - out.loc[mask,"ece_random"]
    return out

def add_shift_signatures(df, shift_dir):
    dfs = []
    for split in sorted(df["split"].unique()):
        sig_path = os.path.join(shift_dir, split, f"{split}_shift_signature.csv")
        if not os.path.exists(sig_path):
            tmp = df[df["split"]==split].copy()
            tmp["mean_js"] = np.nan; tmp["mean_ks"] = np.nan
            dfs.append(tmp); continue
        sig = pd.read_csv(sig_path)
        sig.columns = [c.strip().lower() for c in sig.columns]
        mean_js = float(sig.get("mean_js",[np.nan])[0])
        mean_ks = float(sig.get("mean_ks_cont",[np.nan])[0])
        tmp = df[df["split"]==split].copy()
        tmp["mean_js"] = mean_js; tmp["mean_ks"] = mean_ks
        dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True)

def run_reg_safe(df, yname, predictors):
    dd = df[predictors + [yname]].dropna()
    if dd[yname].nunique() <= 1:
        print(f"[INFO] {yname}: all values equal (no variance) → skip regression.")
        return
    if len(dd) < 3:
        print(f"[INFO] {yname}: not enough rows (n={len(dd)}) → skip regression.")
        return
    X = sm.add_constant(dd[predictors].fillna(0.0))
    y = dd[yname].values
    try:
        mod = sm.OLS(y, X).fit()
        print(f"\n=== Regression: {yname} ~ {' + '.join(predictors)} ===")
        print(mod.summary())
    except Exception as e:
        print(f"[WARN] Regression failed for {yname}: {e}")

def corr_report(df, yname):
    dd = df[["mean_js","mean_ks", yname]].dropna()
    if dd.empty:
        return
    print(f"\n=== Correlations with {yname} ===")
    for x in ("mean_js","mean_ks"):
        if dd[x].nunique() > 1 and dd[yname].nunique() > 1:
            r, p = st.pearsonr(dd[x], dd[yname])
            rho, pp = st.spearmanr(dd[x], dd[yname])
            print(f"{x}:  Pearson r={r:.3f} (p={p:.3g}) | Spearman ρ={rho:.3f} (p={pp:.3g})")
        else:
            print(f"{x}: insufficient variation to compute correlation.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/compare_all_scores.csv")
    ap.add_argument("--shift_dir", default="results/shift")
    ap.add_argument("--out", default="results/shift/perf_link.csv")
    args = ap.parse_args()

    if not os.path.exists(args.metrics):
        raise FileNotFoundError(f"Missing metrics: {args.metrics}")

    m = pd.read_csv(args.metrics)
    m = norm_cols(m)
    if "split" not in m.columns or "model" not in m.columns:
        raise ValueError(f"Metrics must have 'split' and 'model'. Got: {m.columns.tolist()}")

    long_df = wide_to_long(m)
    if long_df.empty:
        print("[WARN] No rows produced from wide metrics (check auroc_none/auroc_platt/ece_* columns).")
        return

    long_df = attach_random_deltas(long_df)
    long_df = add_shift_signatures(long_df, args.shift_dir)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    long_df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with {len(long_df)} rows")

    # Primary: Δ vs random
    for y in ("delta_auroc_random","delta_ece_random"):
        if y in long_df.columns:
            run_reg_safe(long_df, y, ["mean_js"])
            run_reg_safe(long_df, y, ["mean_ks"])
            run_reg_safe(long_df, y, ["mean_js","mean_ks"])
            corr_report(long_df, y)

    # Secondary: Δ vs notuned
    for y in ("delta_auroc_notuned","delta_ece_notuned"):
        if y in long_df.columns:
            run_reg_safe(long_df, y, ["mean_js"])
            run_reg_safe(long_df, y, ["mean_ks"])
            run_reg_safe(long_df, y, ["mean_js","mean_ks"])
            corr_report(long_df, y)

if __name__ == "__main__":
    main()
