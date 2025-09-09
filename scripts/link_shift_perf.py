#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd


def norm_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "split" not in df.columns and "scenario" in df.columns:
        df = df.rename(columns={"scenario":"split"})
    return df


def is_long_format(df):
    return {"split","model","calibrator","auroc","ece"}.issubset(df.columns)


def make_long_from_wide(df):
    """Convert wide metrics to long per-calibrator rows."""
    rows = []
    for _, r in df.iterrows():
        split = str(r["split"]).lower()
        model = r["model"]
        for cal in ("none","platt","notuned"):
            au_col = f"auroc_{cal}"
            ec_col = f"ece_{cal}"
            if au_col in r.index and ec_col in r.index and pd.notnull(r[au_col]) and pd.notnull(r[ec_col]):
                rows.append({
                    "split": split,
                    "model": model,
                    "calibrator": cal,
                    "auroc": float(r[au_col]),
                    "ece": float(r[ec_col]),
                })
    return pd.DataFrame(rows)


def attach_random_deltas(long_df):
    """
    Add delta_auroc_random, delta_ece_random:
      delta_auroc_random = AUROC_random - AUROC_split
      delta_ece_random   = ECE_split - ECE_random
    """
    out = long_df.copy()
    out["delta_auroc_random"] = np.nan
    out["delta_ece_random"] = np.nan

    # random baselines per (model,calibrator)
    base = (out[out["split"]=="random"]
              .set_index(["model","calibrator"])[["auroc","ece"]]
              .rename(columns={"auroc":"auroc_random","ece":"ece_random"}))

    idx = ["model","calibrator"]
    out = (out.set_index(idx)
               .join(base, how="left")
               .reset_index())

    mask = out["split"].ne("random") & out["auroc_random"].notna() & out["ece_random"].notna()
    out.loc[mask, "delta_auroc_random"] = out.loc[mask, "auroc_random"] - out.loc[mask, "auroc"]
    out.loc[mask, "delta_ece_random"]   = out.loc[mask, "ece"] - out.loc[mask, "ece_random"]
    return out


def attach_notuned_deltas(long_df):
    """
    If 'notuned' calibrator rows exist, compute deltas vs notuned:
      delta_auroc_notuned = AUROC_notuned - AUROC_cal
      delta_ece_notuned   = ECE_cal - ECE_notuned
    """
    out = long_df.copy()
    out["delta_auroc_notuned"] = np.nan
    out["delta_ece_notuned"] = np.nan

    # per (split,model)
    base = (out[out["calibrator"]=="notuned"]
              .set_index(["split","model"])[["auroc","ece"]]
              .rename(columns={"auroc":"auroc_notuned","ece":"ece_notuned"}))

    idx = ["split","model"]
    out = (out.set_index(idx)
              .join(base, how="left")
              .reset_index())

    mask = out["calibrator"].ne("notuned") & out["auroc_notuned"].notna() & out["ece_notuned"].notna()
    out.loc[mask, "delta_auroc_notuned"] = out.loc[mask, "auroc_notuned"] - out.loc[mask, "auroc"]
    out.loc[mask, "delta_ece_notuned"]   = out.loc[mask, "ece"] - out.loc[mask, "ece_notuned"]
    return out


def add_shift_signatures(df, shift_dir):
    df = df.copy()
    df["mean_js"] = np.nan
    df["mean_ks"] = np.nan
    splits = df["split"].unique()
    for split in splits:
        sig_path = os.path.join(shift_dir, split, f"{split}_shift_signature.csv")
        if not os.path.exists(sig_path):
            continue
        sig = pd.read_csv(sig_path)
        sig.columns = [c.strip().lower() for c in sig.columns]
        js = float(sig.get("mean_js", [np.nan])[0])
        ks = float(sig.get("mean_ks_cont", [np.nan])[0])
        df.loc[df["split"]==split, "mean_js"] = js
        df.loc[df["split"]==split, "mean_ks"] = ks
    return df


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

    # Determine format
    if is_long_format(m):
        long_df = m[["split","model","calibrator","auroc","ece"]].copy()
        long_df["split"] = long_df["split"].str.lower()
        long_df["calibrator"] = long_df["calibrator"].astype(str).str.lower()
    else:
        # expect wide; require split and model
        need = {"split","model"}
        if not need.issubset(m.columns):
            raise ValueError(f"Metrics must contain {need} or be long format with split/model/calibrator/auroc/ece.")
        long_df = make_long_from_wide(m)
        long_df["split"] = long_df["split"].str.lower()
        long_df["calibrator"] = long_df["calibrator"].astype(str).str.lower()

    # compute deltas
    long_df = attach_random_deltas(long_df)
    long_df = attach_notuned_deltas(long_df)

    # add shift signatures
    long_df = add_shift_signatures(long_df, args.shift_dir)

    # keep only splits of interest (skip random rows in final CSV unless you want them)
    keep = long_df["split"].isin(["temporal","hospital","random"])
    out = long_df.loc[keep, ["split","model","calibrator",
                             "mean_js","mean_ks",
                             "delta_auroc_random","delta_ece_random",
                             "delta_auroc_notuned","delta_ece_notuned"]].copy()

    # sort for readability
    out["split_ord"] = out["split"].map({"hospital":0,"temporal":1,"random":2}).fillna(99)
    out["model_ord"] = out["model"].map({"lr":0,"rf":1,"xgb":2,"lgbm":3}).fillna(99)
    out = out.sort_values(["split_ord","model_ord","calibrator"]).drop(columns=["split_ord","model_ord"])

    # write
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {len(out)} rows")


if __name__ == "__main__":
    main()
