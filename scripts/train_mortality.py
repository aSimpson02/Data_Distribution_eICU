# scripts/train_mortality.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from src.feats.selector import FeatureSelector, TARGET, ID_COL, SPLIT_ONLY

# 0) load your already-aggregated first-24h dataframe (one row per ICU stay)
df = pd.read_parquet("data/eicu_first24.parquet")  # replace with your path

# 1) make splits (example: 5 hospitals train, 1 held-out H6)
train_hospitals = [101, 102, 103, 104, 105]
test_hospital = 106
train_df = df[df.hospitalid.isin(train_hospitals)].reset_index(drop=True)
test_df  = df[df.hospitalid.eq(test_hospital)].reset_index(drop=True)

# 2) fit selector on TRAIN ONLY (prevents leakage)
selector = FeatureSelector(rfe_keep=15).fit(train_df)

# 3) transform train/test
Xtr, tr_names = selector.transform(train_df)
ytr = train_df[TARGET].astype(int).values

Xte, te_names = selector.transform(test_df)
yte = test_df[TARGET].astype(int).values

# 4) train a simple LR baseline (you can swap for LightGBM later)
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Xtr, ytr)

# 5) evaluate
p_tr = clf.predict_proba(Xtr)[:,1]
p_te = clf.predict_proba(Xte)[:,1]

print("TRAIN AUROC:", roc_auc_score(ytr, p_tr))
print("TRAIN AUPRC:", average_precision_score(ytr, p_tr))
print("TRAIN Brier:", brier_score_loss(ytr, p_tr))

print("TEST  AUROC:", roc_auc_score(yte, p_te))
print("TEST  AUPRC:", average_precision_score(yte, p_te))
print("TEST  Brier:", brier_score_loss(yte, p_te))
