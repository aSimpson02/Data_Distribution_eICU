#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# columns that should never be used as predictors!!
NON_PREDICTORS = {
    "patientunitstayid", "hospitalid", "hospitaldischargeyear",
    "apachescore", "predictedhospitalmortality", "admissionoffset"
}
DEFAULT_LABEL = "hospital_mortality"
CSV_DIR = Path("data/csv_splits")

STEM_MAP = {
    "iid": "random",
    "random": "random",
    "hospital": "hospital",
    "hospital_ood": "hospital",
    "temporal": "temporal",
    "temporal_ood": "temporal",
}

def load_split(stem: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr_p = CSV_DIR / f"{stem}_train.csv"
    va_p = CSV_DIR / f"{stem}_val.csv"
    te_p = CSV_DIR / f"{stem}_test.csv"
    if not (tr_p.exists() and va_p.exists() and te_p.exists()):
        raise FileNotFoundError(f"Could not find {stem}_train/val/test under {CSV_DIR}/")
    tr = pd.read_csv(tr_p)
    va = pd.read_csv(va_p)
    te = pd.read_csv(te_p)
    print("\nLoaded:")
    print(f"  - train: {tr_p}")
    print(f"  -  val : {va_p}")
    print(f"  - test : {te_p}\n")
    return tr, va, te

def describe_split(tag: str, df: pd.DataFrame, label: str) -> None:
    n = len(df)
    pos = int(df[label].sum()) if label in df.columns else 0
    pos_rate = (pos / n) if n > 0 else float("nan")
    print(f"[{tag}] rows={n:,}, features={df.shape[1]}")
    if label in df.columns:
        print(f"  label mean: {pos_rate:.4f} (pos={pos:,} / {n:,})")
    else:
        print(f"  label '{label}' not found.")
    # top missing
    miss = df.isna().mean().sort_values(ascending=False)
    print("  top missing:")
    print(miss.head(10).to_string())

def feature_overlap(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame) -> None:
    common = set(tr.columns) & set(va.columns) & set(te.columns)
    only_tr = set(tr.columns) - common
    only_va = set(va.columns) - common
    only_te = set(te.columns) - common
    print("\nFeature overlap:")
    print(f"  common: {len(common)}")
    print(f"  only in train: {len(only_tr)}")
    print(f"  only in val  : {len(only_va)}")
    print(f"  only in test : {len(only_te)}")
    if only_tr:
        print("  (train-only):", sorted(list(only_tr))[:10], "..." if len(only_tr) > 10 else "")
    if only_va:
        print("  (val-only)  :", sorted(list(only_va))[:10], "..." if len(only_va) > 10 else "")
    if only_te:
        print("  (test-only) :", sorted(list(only_te))[:10], "..." if len(only_te) > 10 else "")

def leakage_guard(df: pd.DataFrame) -> None:
    present = [c for c in NON_PREDICTORS if c in df.columns]
    if present:
        print("\n[leakage guard] Non-predictor columns present (ok in data, but do NOT use as features):")
        print(" ", ", ".join(sorted(present)))
    else:
        print("\n[leakage guard] Non-predictor columns not found in this dataframe (ok).")

def numeric_summary(df: pd.DataFrame, shared_cols: list[str], label: str) -> pd.DataFrame:
    num_cols = [c for c in shared_cols if pd.api.types.is_numeric_dtype(df[c]) and c != label]
    desc = df[num_cols].agg(["count","mean","std"]).T
    desc = desc.sort_index()
    return desc

def main():
    scen = sys.argv[1] if len(sys.argv) > 1 else "random"
    stem = STEM_MAP.get(scen, scen)

    tr, va, te = load_split(stem)
    label = DEFAULT_LABEL

    describe_split("train", tr, label); print()
    describe_split("val", va, label); print()
    describe_split("test", te, label)

    feature_overlap(tr, va, te)

    # report leakage-prone columns if exist exist 
    leakage_guard(tr)

    # shared-feature numeric summary 
    shared = list(set(tr.columns) & set(va.columns) & set(te.columns))
    stats = numeric_summary(tr, shared, label)
    out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"audit_{stem}_train_numeric_summary.csv"
    stats.to_csv(out_path)
    print(f"\nSaved numeric summary → {out_path}")

    # ching diagnosis buckets coverage
    diag_cols = [c for c in shared if c.startswith("diagnosis_bucket_")]
    if diag_cols:
        coverage = (tr[diag_cols].sum(axis=0) > 0).mean()
        print(f"\nDiagnosis bucket columns: {len(diag_cols)}  (nonzero coverage in train: {coverage:.2%})")

if __name__ == "__main__":
    main()
