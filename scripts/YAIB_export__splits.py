#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import argparse, sys, re
import pandas as pd

LABEL_CANDIDATES = [
    "hospital_mortality", "mortality", "mortality_24h", "icu_mortality",
    "y", "label", "target", "death", "in_hospital_death"
]

def find_split_files(root: Path):
    root = Path(root)
    files = list(root.rglob("*.parquet")) + list(root.rglob("*.pq"))
    def pick(name: str, keys: list[str]) -> bool:
        n = name.lower()
        return any(k in n for k in keys)

    train = [p for p in files if pick(p.name, ["train", "trn"])]
    val   = [p for p in files if pick(p.name, ["val", "valid"])]
    test  = [p for p in files if pick(p.name, ["test", "tst"])]
    return train, val, test

def guess_label(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    for c in LABEL_CANDIDATES:
        if c in cols:
            return df.columns[cols.index(c)]
    # heuristic: binary column with only {0,1}
    for orig, low in zip(df.columns, cols):
        s = pd.to_numeric(df[orig], errors="coerce").dropna()
        if not s.empty and set(s.unique()).issubset({0,1}):
            return orig
    raise ValueError("Could not infer label column; pass a dir with known label fields.")

def read_any_one(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No candidate files found.")
    p = sorted(paths, key=lambda x: len(x.name))[0]
    return pd.read_parquet(p), p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaib_dir", required=True)
    ap.add_argument("--stem", default="yaib")  
    ap.add_argument("--label_out", default="hospital_mortality")
    ap.add_argument("--outdir", default="data/csv_splits")
    args = ap.parse_args()

    train, val, test = find_split_files(Path(args.yaib_dir))
    dtrain, ptrain = read_any_one(train)
    dval,   pval   = read_any_one(val)
    dtest,  ptest  = read_any_one(test)

    # sanitize
    for df in (dtrain, dval, dtest):
        df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

    # infer label and rename
    label = guess_label(dtrain)
    for name, df in [("train", dtrain), ("val", dval), ("test", dtest)]:
        if label not in df.columns:
            # try same-name in other splits
            for alt in LABEL_CANDIDATES:
                if alt in df.columns:
                    label = alt
                    break
        df.rename(columns={label: args.label_out}, inplace=True)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    dtrain.to_csv(outdir / f"{args.stem}_train.csv", index=False)
    dval.to_csv(outdir   / f"{args.stem}_val.csv", index=False)
    dtest.to_csv(outdir  / f"{args.stem}_test.csv", index=False)
    print(f"[OK] wrote {outdir}/{args.stem}_{{train,val,test}}.csv")
    print(f"     (read: {ptrain.name}, {pval.name}, {ptest.name})")

if __name__ == "__main__":
    main()
