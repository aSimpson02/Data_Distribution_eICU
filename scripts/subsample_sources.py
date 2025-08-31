# scripts/subsample_sources.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

BASE = Path("data/splits")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sources", type=int, default=5)
    ap.add_argument("--min-stays", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    Xtr = pd.read_parquet(BASE/"hospital_train.parquet")
    Mtr = pd.read_parquet(BASE/"hospital_meta_train.parquet")
    Xte = pd.read_parquet(BASE/"hospital_test.parquet")
    Mte = pd.read_parquet(BASE/"hospital_meta_test.parquet")

    # eligible sources
    counts = Mtr["hospitalid"].value_counts()
    eligible = counts[counts >= args.min_stays].index.tolist()
    if len(eligible) < args.n_sources:
        raise SystemExit(f"Not enough hospitals ≥{args.min_stays} stays (found {len(eligible)}).")

    src_ids = rng.choice(eligible, size=args.n_sources, replace=False)
    Mtr_sub = Mtr[Mtr["hospitalid"].isin(src_ids)].copy()

    # match rows in features by patientunitstayid (safe join)
    if "patientunitstayid" in Xtr.columns:
        Xtr_sub = Xtr.merge(Mtr_sub[["patientunitstayid"]], on="patientunitstayid", how="inner")
    else:
        # fallback: join via an index column if needed
        Xtr_sub = Xtr.loc[Mtr_sub.index]

    # make a validation split inside the 5-source training subset
    y = Xtr_sub["hospital_mortality"].astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, val_idx = next(sss.split(Xtr_sub, y))
    Xtr_final, Xval_final = Xtr_sub.iloc[tr_idx].copy(), Xtr_sub.iloc[val_idx].copy()
    Mtr_final, Mval_final = Mtr_sub.iloc[tr_idx].copy(), Mtr_sub.iloc[val_idx].copy()

    # write new split files
    prefix = "hospital5src"
    (BASE/f"{prefix}_train.parquet").write_bytes(Xtr_final.to_parquet(index=False))
    (BASE/f"{prefix}_val.parquet").write_bytes(Xval_final.to_parquet(index=False))
    (BASE/f"{prefix}_test.parquet").write_bytes(Xte.to_parquet(index=False))

    (BASE/f"{prefix}_meta_train.parquet").write_bytes(Mtr_final.to_parquet(index=False))
    (BASE/f"{prefix}_meta_val.parquet").write_bytes(Mval_final.to_parquet(index=False))
    (BASE/f"{prefix}_meta_test.parquet").write_bytes(Mte.to_parquet(index=False))

    print(f"[OK] 5-source split:")
    print(f"  train {Xtr_final.shape} over hospitals {sorted(src_ids.tolist())}")
    print(f"  val   {Xval_final.shape}")
    print(f"  test  {Xte.shape} (target hospital stays from existing hospital split)")

if __name__ == "__main__":
    main()
