# scripts/make_val_splits.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

SPLITS = ["random","temporal","hospital"]
BASE = Path("data/splits")

def make_val(name: str, seed: int = 42, frac_val: float = 0.2):
    Xtr_p = BASE / f"{name}_train.parquet"
    Mtr_p = BASE / f"{name}_meta_train.parquet"
    if not Xtr_p.exists() or not Mtr_p.exists():
        print(f"[SKIP] {name}: train/meta not found")
        return
    Xtr = pd.read_parquet(Xtr_p)
    Mtr = pd.read_parquet(Mtr_p)
    y = Xtr["hospital_mortality"].astype(int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac_val, random_state=seed)
    tr_idx, val_idx = next(sss.split(Xtr, y))
    Xval, Mval = Xtr.iloc[val_idx].copy(), Mtr.iloc[val_idx].copy()

    Xval.to_parquet(BASE / f"{name}_val.parquet", index=False)
    Mval.to_parquet(BASE / f"{name}_meta_val.parquet", index=False)
    print(f"[OK] {name}: wrote {Xval.shape} -> {BASE}/{name}_val.parquet")

def main():
    for s in SPLITS:
        make_val(s)

if __name__ == "__main__":
    main()
