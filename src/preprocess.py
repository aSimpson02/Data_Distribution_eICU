from __future__ import annotations
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .utils import age_to_bin

# flip to False if you intentionally want to keep it
LEAKY_COLS = {"predictedhospitalmortality": True}

def drop_leaks(df: pd.DataFrame) -> pd.DataFrame:
    for col, do_drop in LEAKY_COLS.items():
        if do_drop and col in df.columns:
            df = df.drop(columns=[col])
    return df

def build_preprocess_pipeline(categorical: List[str], numeric: List[str]) -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop"
    )

def prepare_features(df: pd.DataFrame, features_cfg: Dict[str, Any]):
    id_col = features_cfg["id_col"]
    target_col = features_cfg["target_col"]

    # meta
    meta_map = features_cfg["meta_cols"]
    for k, col in meta_map.items():
        if col not in df.columns:
            raise ValueError(f"meta col {k}:{col} missing in dataframe")
    meta = pd.DataFrame({
        "hospital": df[meta_map["hospital"]],
        "sex": df[meta_map["sex"]],
        "age": df[meta_map["age"]],
        "age_bin": df[meta_map["age"]].map(age_to_bin),
        "time": df[meta_map["time"]],
    })

    if target_col not in df.columns:
        raise ValueError(f"target col {target_col} missing")
    y = df[target_col].astype(int).to_numpy()

    df = drop_leaks(df)

    # keep only columns that are present
    categorical = [c for c in features_cfg.get("categorical", []) if c in df.columns and c != target_col]
    numeric     = [c for c in features_cfg.get("numeric", []) if c in df.columns and c != target_col]

    ct = build_preprocess_pipeline(categorical, numeric)
    X = ct.fit_transform(df[numeric + categorical])

    # feature names
    num_names = numeric
    try:
        ohe = ct.named_transformers_["cat"].named_steps["ohe"]
        cat_cols = ct.transformers_[1][2]
        cat_names = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        cat_names = [f"cat_{i}" for i in range(X.shape[1] - len(num_names))]

    feat_names = num_names + cat_names
    return X, y, meta, feat_names
