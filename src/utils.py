# --- add to src/utils.py ---

import json
from pathlib import Path
import pandas as pd
import numpy as np

# Where to persist canonical feature order (so val/test align to train)
_FEATURES_PATH = Path("artifacts") / "feature_names.json"

def save_feature_names(cols):
    """Persist the exact feature column order used at training."""
    _FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_FEATURES_PATH, "w") as f:
        json.dump(list(cols), f)

def load_feature_names():
    with open(_FEATURES_PATH) as f:
        return json.load(f)

def to_feature_df(X, feature_names):
    if isinstance(X, pd.Series):
        X = pd.DataFrame([X])
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Add any missing columns
    missing = [c for c in feature_names if c not in X.columns]
    for c in missing:
        X[c] = 0

    # Drop extras & reorder
    X = X.reindex(columns=feature_names)
    return X

def assert_feature_match(X, feature_names):
    cols = list(X.columns)
    assert cols == list(feature_names), (
        f"Feature mismatch!\nExpected first5: {list(feature_names)[:5]}\nGot first5: {cols[:5]}"
    )
