#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import glob
import pandas as pd

OUT = "models/preds/preds_all.csv"

def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    ren = {}
    if "y" in df.columns and "y_true" not in df.columns: ren["y"] = "y_true"
    if "prob" in df.columns and "y_pred_proba" not in df.columns: ren["prob"] = "y_pred_proba"
    if "prediction" in df.columns and "y_pred_proba" not in df.columns: ren["prediction"] = "y_pred_proba"
    if "dataset" in df.columns and "set" not in df.columns: ren["dataset"] = "set"
    if "scenario" in df.columns and "split" not in df.columns: ren["scenario"] = "split"
    return df.rename(columns=ren)

def main():
    files = glob.glob("models/preds/**/*.csv", recursive=True)
    if not files:
        print("[WARN] No per-prediction CSVs found under models/preds/**")
        return

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df = normalize_cols(df)
            keep = [c for c in ("split","set","model","calibrator","y_true","y_pred_proba") if c in df.columns]
            if {"split","set","model","y_pred_proba"}.issubset(set(keep)):
                frames.append(df[keep].copy())
        except Exception as e:
            print(f"[WARN] skipping {fp}: {e}")

    if not frames:
        print("[WARN] No compatible prediction files found to combine.")
        return

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[OK] wrote {OUT} with {len(out):,} rows")

if __name__ == "__main__":
    main()
