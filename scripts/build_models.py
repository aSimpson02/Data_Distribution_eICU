#!/usr/bin/env python3
"""
Central model factory for tabular mortality experiments.
Returns unfitted sklearn/xgboost/lightgbm classifiers with sensible defaults.
Safe-imports boosters so callers can request them without crashing if missing.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Callable

# core sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# optional boosters (safe imports)
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    import lightgbm as lgbm  # type: ignore
except Exception:
    lgbm = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[..., object]     # function(seed:int) -> estimator
    available: bool                    # whether deps are present


def _make_lr(seed: int):
    return LogisticRegression(
        max_iter=4000, class_weight="balanced", solver="liblinear", random_state=seed
    )


def _make_rf(seed: int):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=seed,
    )


def _make_xgb(seed: int):
    if xgb is None:
        raise RuntimeError("XGBoost not installed.")
    return xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
        tree_method="hist",
    )


def _make_lgbm(seed: int):
    if lgbm is None:
        raise RuntimeError("LightGBM not installed.")
    return lgbm.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=seed,
    )


# registry
REGISTRY: Dict[str, ModelSpec] = {
    "lr":   ModelSpec("lr",   _make_lr,   True),
    "rf":   ModelSpec("rf",   _make_rf,   True),
    "xgb":  ModelSpec("xgb",  _make_xgb,  xgb is not None),
    "lgbm": ModelSpec("lgbm", _make_lgbm, lgbm is not None),
}


def get_model_names() -> list[str]:
    """All valid model keys (even if not installed)."""
    return list(REGISTRY.keys())


def is_available(name: str) -> bool:
    spec = REGISTRY.get(name)
    if spec is None:
        return False
    return spec.available


def build_model(name: str, seed: int):
    """Return an unfitted estimator or raise ValueError if name unknown."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown model '{name}'. Options: {', '.join(get_model_names())}")
    # If unavailable, raise RuntimeError so caller can SKIP cleanly.
    if not spec.available:
        raise RuntimeError(f"{name} not installed.")
    return spec.builder(seed)
