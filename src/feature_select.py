import pandas as pd

def apply_vif(X: pd.DataFrame, thresh: float = 10.0) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception:
        return X
    cols = list(X.columns)
    while True:
        Xc = sm.add_constant(X[cols], has_constant='add')
        vifs = pd.Series([variance_inflation_factor(Xc.values, i+1) for i in range(len(cols))], index=cols)
        worst = vifs.idxmax()
        if vifs.max() < thresh or len(cols)<=2: break
        cols.remove(worst)
    return X[cols]

def apply_rfe(X: pd.DataFrame, y, n_out: int = 64):
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFE
    est = LogisticRegression(max_iter=2000, class_weight="balanced")
    rfe = RFE(estimator=est, n_features_to_select=n_out, step=0.1)
    rfe.fit(X, y)
    mask = rfe.support_
    return X.loc[:, mask]
