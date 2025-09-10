import numpy as np
import pandas as pd

def _ensure_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.select_dtypes(include=[np.number]).copy()
    if Xn.empty:
        raise ValueError("No numeric columns found for feature selection.")
    return Xn

def apply_vif(X: pd.DataFrame, thresh: float = 10.0, max_iter: int = 100, impute: bool = True):
    """
    Iterative VIF filter.
    Returns (X_reduced, kept_cols, last_vifs)
    """
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    except Exception:
        # statsmodels not installed → no-op
        return X, list(X.columns), pd.Series(dtype=float)

    Xn = _ensure_numeric_df(X)
    if impute:
        med = Xn.median(numeric_only=True)
        Xn = Xn.fillna(med)

    cols = list(Xn.columns)
    last_vifs = None
    it = 0
    while it < max_iter and len(cols) > 2:
        it += 1
        try:
            Xc = sm.add_constant(Xn[cols], has_constant="add")
            vifs = pd.Series([vif(Xc.values, i+1) for i in range(len(cols))], index=cols)
        except Exception:
            break
        last_vifs = vifs
        worst = vifs.idxmax()
        if vifs.max() < thresh:
            break
        cols.remove(worst)

    return X[cols], cols, (last_vifs if last_vifs is not None else pd.Series(dtype=float))

def apply_rfe(X: pd.DataFrame, y, n_out: int = 64, standardize: bool = True, random_state: int = 42):
    """
    Logistic-Regression-based RFE. Returns (X_reduced, kept_cols, support_mask).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFE
    from sklearn.preprocessing import StandardScaler

    Xn = _ensure_numeric_df(X)
    n_feats = Xn.shape[1]
    if n_out >= n_feats:
        return Xn, list(Xn.columns), np.ones(n_feats, dtype=bool)

    Xz = Xn
    if standardize:
        sc = StandardScaler().fit(Xn)
        Xz = pd.DataFrame(sc.transform(Xn), index=Xn.index, columns=Xn.columns)

    est = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state, n_jobs=None)
    rfe = RFE(estimator=est, n_features_to_select=n_out, step=0.1)
    rfe.fit(Xz, y)
    mask = rfe.support_
    kept = Xn.columns[mask].tolist()
    return X[kept], kept, mask
