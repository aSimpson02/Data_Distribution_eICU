from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import pandas as pd

@dataclass
class FeaturePolicy:
    """
    Minimal, train-only feature policy used by scripts:
      - keep numeric + 'diagnosis_bucket_*' columns
      - drop columns with missing fraction > missing_thresh
      - drop zero-variance columns
    """
    feat_select: str = "none"                # kept for CLI compatibility
    missing_thresh: Optional[float] = 0.40
    keep_prefixes: Tuple[str, ...] = ("diagnosis_bucket_",)
    selected_features_: List[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame, label_col: str):
        if label_col not in df.columns:
            raise ValueError(f"label_col='{label_col}' not found")

        def _is_prefixed(c: str) -> bool:
            s = str(c)
            return any(s.startswith(p) for p in self.keep_prefixes)

        # candidates: numeric + prefixed, excluding label
        cand = [c for c in df.columns if c != label_col and
                (pd.api.types.is_numeric_dtype(df[c]) or _is_prefixed(c))]

        # missingness filter
        if self.missing_thresh is not None:
            miss = df[cand].isna().mean(numeric_only=False)
            cand = [c for c in cand if float(miss.get(c, 0.0)) <= self.missing_thresh]

        # zero-variance drop
        if cand:
            stds = df[cand].std(numeric_only=True)
            zero_var = set(stds[stds.fillna(0.0) == 0.0].index)
            if zero_var:
                cand = [c for c in cand if c not in zero_var]

        # preserve order
        self.selected_features_ = list(dict.fromkeys(cand))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise RuntimeError("FeaturePolicy not fitted")
        keep = [c for c in self.selected_features_ if c in df.columns]
        return df[keep]

    def get_feature_names_out(self) -> List[str]:
        return list(self.selected_features_)
