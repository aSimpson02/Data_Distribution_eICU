class _ChunkedPrefit:
    def __init__(self, base, bs=512):
        self.base = base; self.bs = bs
    def fit(self, X, y): return self
    def predict_proba(self, X):
        import numpy as np
        Xv = X.values if hasattr(X, "values") else X
        outs=[]
        for i in range(0, len(Xv), self.bs):
            outs.append(self.base.predict_proba(Xv[i:i+self.bs]))
        return np.vstack(outs)
