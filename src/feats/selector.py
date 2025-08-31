#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature selection utilities for eICU CSV matrices.

- Drops leakage/meta columns.
- Keeps numeric columns and one-hot diagnosis buckets (prefix: 'diagnosis_bucket_').
- Optional filters:
    * Missingness filter (default >40% missing dropped for numeric).
    * VIF (Variance Inflation Factor) to reduce collinearity on numeric features.
    * RFE (Recursive Feature Elimination) with Logistic Regression to prune features.
      We re-insert clinically critical features if they were dropped by RFE.

Usage pattern:
    policy = FeaturePolicy(feat_select="vif_rfe", rfe_keep=30)
    policy.fit(train_df, label_col="hospital_mortality")
    Xtr = policy.transform(train_df)
    Xte = policy.transform(test_df)
    feats = policy.selected_features_

This module is used by training/eval scripts via a --feat_select flag.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# statsmodels for VIF
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
DIAG_PREFIX = "diagnosis_bucket_"
NON_PREDICTORS = {
    "patientunitstayid", "hospitalid", "hospitaldischargeyear",
    "apachescore", "predictedhospitalmortality", "admissionoffset",
}

# Clinically important features to re-include if dropped by RFE (when present)
CLINICALLY_CRITICAL = {
    # vitals
    "heartrate", "respiration", "temperature", "spo2", "systemicmean",
    "systemicsystolic", "systemicdiastolic",
    # labs
    "lactate", "creatinine", "bun", "glucose", "bilirubin", "albumin",
    # demographics
    "age",
}

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def _split_feature_types(df: pd.DataFrame, label_col: str) -> tuple[list[str], list[str]]:
    diag_cols = [c for c in df.columns if c.startswith(DIAG_PREFIX)]
    numeric = [
        c for c in df.columns
        if c not in NON_PREDICTORS
        and c != label_col
        and c not in diag_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric, diag_cols

def _missingness_filter(train_df: pd.DataFrame, numeric: list[str], thresh: float) -> list[str]:
    if not numeric:
        return numeric
    miss = train_df[numeric].isna().mean()
    keep = [c for c in numeric if float(miss[c]) <= thresh]
    return keep

def _vif_prune(train_df: pd.DataFrame, numeric: list[str], vif_thresh: float) -> list[str]:
    if not numeric or len(numeric) < 2 or not _HAS_STATSMODELS:
        return numeric
    X = train_df[numeric].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].mean())
    keep = numeric.copy()
    # iteratively drop highest-VIF until all <= threshold
    while len(keep) >= 2:
        Xk = X[keep].to_numpy()
        vifs = []
        for i in range(Xk.shape[1]):
            try:
                vifs.append(variance_inflation_factor(Xk, i))
            except Exception:
                vifs.append(np.nan)
        vifs = np.asarray(vifs, dtype=float)
        if not np.isfinite(vifs).any():
            break
        max_i = int(np.nanargmax(vifs))
        max_v = float(vifs[max_i])
        if not np.isfinite(max_v) or max_v <= vif_thresh:
            break
        # drop the feature with the highest VIF
        keep.pop(max_i)
    return keep

def _rfe_select(train_df: pd.DataFrame, label_col: str, candidates: list[str], n_keep: int) -> list[str]:
    if not candidates or n_keep <= 0:
        return candidates
    n_keep = min(n_keep, len(candidates))
    # Simple logistic regression for RFE
    lr = LogisticRegression(max_iter=4000, solver="liblinear", class_weight="balanced")
    # Prepare X (impute with train means)
    X = train_df[candidates].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].mean())
    y = train_df[label_col].astype(int).values
    rfe = RFE(lr, n_features_to_select=n_keep)
    X_new = rfe.fit_transform(X, y)
    selected = [c for c, flag in zip(candidates, rfe.support_) if flag]
    return selected

# ----------------------------------------------------
# Main class
# ----------------------------------------------------
@dataclass
class FeaturePolicy:
    feat_select: str = "none"            # "none", "vif", "rfe", "vif_rfe"
    missing_thresh: float = 0.40         # drop numeric features missing > 40% on TRAIN
    vif_thresh: float = 10.0
    rfe_keep: int = 30                   # target number after RFE (on numeric+diag)
    selected_features_: List[str] = field(default_factory=list)
    train_means_: Optional[pd.Series] = None

    def fit(self, train_df: pd.DataFrame, label_col: str) -> "FeaturePolicy":
        numeric, diag_cols = _split_feature_types(train_df, label_col)

        # 1) missingness on numeric
        numeric = _missingness_filter(train_df, numeric, self.missing_thresh)

        # 2) VIF (optional)
        if self.feat_select in ("vif", "vif_rfe"):
            numeric = _vif_prune(train_df, numeric, self.vif_thresh)

        # 3) Candidate set to pass to RFE (numeric + diagnosis one-hots)
        candidates = numeric + diag_cols

        # 4) RFE (optional)
        if self.feat_select in ("rfe", "vif_rfe") and len(candidates) > 0:
            kept = _rfe_select(train_df, label_col, candidates, self.rfe_keep)
        else:
            kept = candidates

        # 5) Clinical reinsertion: add back critical features if present in data
        reinserts = [f for f in CLINICALLY_CRITICAL if f in train_df.columns]
        for f in reinserts:
            if f not in kept and f not in NON_PREDICTORS and f != label_col:
                kept.append(f)

        # 6) Finalize and compute train means for imputation
        self.selected_features_ = kept
        # Means only for numeric columns among selected
        num_sel = [c for c in self.selected_features_ if pd.api.types.is_numeric_dtype(train_df[c])]
        self.train_means_ = train_df[num_sel].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise RuntimeError("FeaturePolicy not fitted. Call .fit(train_df, label_col) first.")
        X = df[self.selected_features_].copy()
        # Impute numeric with train means; any non-numeric (rare) fill 0
        if self.train_means_ is not None:
            for c in self.train_means_.index:
                if c in X.columns:
                    X[c] = pd.to_numeric(X[c], errors="coerce")
                    X[c] = X[c].fillna(self.train_means_[c])
        for c in X.columns:
            if c not in (self.train_means_.index if self.train_means_ is not None else []):
                X[c] = X[c].fillna(0)
        return X
