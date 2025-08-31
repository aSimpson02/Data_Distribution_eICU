# scripts/build_splits.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

IN = Path("data/all.parquet")
OUT = Path("data/splits"); OUT.mkdir(parents=True, exist_ok=True)

TARGET = "hospital_mortality"
META_COLS = ["patientunitstayid", "hospitalid", "hospitaldischargeyear", "gender", "age", "unittype"]

# clinical clipping ranges (post temp-normalisation)
CLIP_RANGES = {
    "age": (0, 90),
    "apachescore": (0, 300),
    "predictedhospitalmortality": (0.0, 1.0),
    "heartrate": (20, 220),
    "systemicsystolic": (50, 260),
    "systemicdiastolic": (20, 160),
    "systemicmean": (30, 200),
    "respiration": (4, 80),
    "spo2": (50, 100),
    "temperature": (30, 43),  
    "glucose": (20, 600),
    "bun": (1, 200),
    "creatinine": (0.1, 20),
    "lactate": (0.1, 20),
    "albumin": (0.5, 6),
}

TOP_DIAG_K = 50  # top K diagnosisstring values to keep


def normalize_temperature_c(df: pd.DataFrame, col: str = "temperature") -> None:
    """Convert plausible Fahrenheit temps to Celsius, then clip later."""
    if col not in df.columns:
        return
    t = df[col].astype(float)
    # Fahrenheit-like values
    mask_f = (t > 46) & (t < 120)
    df.loc[mask_f, col] = (t[mask_f] - 32.0) * (5.0 / 9.0)
    # values >=120 are junk - leave to clip/NaN later


def clip_to_ranges(df: pd.DataFrame, ranges: dict) -> None:
    for c, (lo, hi) in ranges.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].clip(lower=lo, upper=hi)


def add_missing_flags(df: pd.DataFrame, cols: list[str]) -> list[str]:
    flags = []
    for c in cols:
        if c in df.columns:
            f = f"{c}__missing"
            df[f] = df[c].isna().astype(int)
            flags.append(f)
    return flags


def bucket_diagnosis(df: pd.DataFrame, col: str = "diagnosisstring", k: int = 50) -> str:
    if col not in df.columns:
        df["diagnosis_bucket"] = "UNKNOWN"
        return "diagnosis_bucket"
    s = df[col].astype(str).fillna("UNKNOWN").str.strip()
    top = s.value_counts().head(k).index
    df["diagnosis_bucket"] = np.where(s.isin(top), s, "OTHER")
    return "diagnosis_bucket"


def build_feature_frames(df: pd.DataFrame):
    # Drop missing columns
    miss = df.isna().mean()
    mostly_missing = miss[miss >= 0.999].index.tolist()
    df = df.drop(columns=mostly_missing)

    # Fix temperature (F->C if needed)
    normalize_temperature_c(df, "temperature")

    # Decide model features (exclude hospital/year)
    num_base = [c for c in [
        "age","apachescore","predictedhospitalmortality",
        "heartrate","systemicsystolic","systemicdiastolic","systemicmean",
        "respiration","temperature","spo2",
        "albumin","bun","glucose","lactate","creatinine",
    ] if c in df.columns]

    # Clip numeric to clinical ranges
    clip_to_ranges(df, CLIP_RANGES)

    # Missing flags BEFORE imputation
    miss_flags = add_missing_flags(df, num_base)

    # Categorical: gender, unittype, diagnosis bucket
    diag_col = bucket_diagnosis(df, "diagnosisstring", TOP_DIAG_K)
    cat_cols = [c for c in ["gender", "unittype", diag_col] if c in df.columns]

    # Meta we keep for reporting (not used as model features)
    meta = df[[c for c in META_COLS if c in df.columns]].copy()

    # X / y
    X_raw = df[num_base + miss_flags + cat_cols].copy()
    y = df[TARGET].astype(int)

    return X_raw, y, meta, num_base, miss_flags, cat_cols


def preprocess_fit_transform(X_raw: pd.DataFrame, num_base: list[str], miss_flags: list[str], cat_cols: list[str]):
    # numeric base - impute + scale// flags - passthrough; categoricals - ohe
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    ct = ColumnTransformer([
        ("num", num_pipe, num_base),
        ("flags", "passthrough", miss_flags),
        ("cat", cat_pipe, cat_cols),
    ])

    X_proc = ct.fit_transform(X_raw)

    # Build feature names
    feat_names = []
    feat_names += num_base
    feat_names += miss_flags
    if cat_cols:
        ohe = ct.named_transformers_["cat"]["ohe"]
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        cat_names = []
    feat_names += cat_names

    # Convert to dense DataFrame if sparse
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    X_proc = pd.DataFrame(X_proc, columns=feat_names, index=X_raw.index)
    return X_proc, ct, feat_names


def split_and_save(name: str, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                   train_idx: np.ndarray, test_idx: np.ndarray):
    tr = pd.concat([X.loc[train_idx], y.loc[train_idx]], axis=1)
    te = pd.concat([X.loc[test_idx],  y.loc[test_idx]],  axis=1)

    tr.to_parquet(OUT / f"{name}_train.parquet")
    te.to_parquet(OUT / f"{name}_test.parquet")
    meta.loc[train_idx].to_parquet(OUT / f"{name}_meta_train.parquet")
    meta.loc[test_idx].to_parquet(OUT / f"{name}_meta_test.parquet")


def main():
    print("Loading …")
    df = pd.read_parquet(IN)

    # Build features + meta
    X_raw, y, meta, num_base, miss_flags, cat_cols = build_feature_frames(df)

    # Fit preprocessing on ALL data (you can refit on train-only if you prefer)
    print("Preprocessing …")
    X_proc, ct, feat_names = preprocess_fit_transform(X_raw, num_base, miss_flags, cat_cols)

  
    # 1) Random 80/20 (stratified)
    
    tr_idx, te_idx = train_test_split(
        X_proc.index, test_size=0.2, random_state=42, stratify=y
    )
    split_and_save("random", X_proc, y, meta, tr_idx, te_idx)


    # 2) Temporal: 2014 -> 2015
   
    if "hospitaldischargeyear" in meta.columns:
        tr_mask = meta["hospitaldischargeyear"] == 2014
        te_mask = meta["hospitaldischargeyear"] == 2015
        split_and_save("temporal", X_proc, y, meta, tr_mask, te_mask)


    # 3) Inter-hospital: hold out largest hospital

    if "hospitalid" in meta.columns:
        holdout_h = meta["hospitalid"].value_counts().idxmax()
        tr_mask = meta["hospitalid"] != holdout_h
        te_mask = meta["hospitalid"] == holdout_h
        split_and_save("hospital", X_proc, y, meta, tr_mask, te_mask)

    print("Saved splits (features+target) and meta to data/splits/")
    print("   Files: *_train.parquet, *_test.parquet, *_meta_train.parquet, *_meta_test.parquet")


if __name__ == "__main__":
    main()
