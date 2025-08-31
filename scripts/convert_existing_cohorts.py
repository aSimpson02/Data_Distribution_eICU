from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.preprocess import prepare_features
import yaml, os

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def persist_processed(prefix: str, X, y, meta, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).to_parquet(out_dir / f"{prefix}_X.parquet", index=False)
    pd.DataFrame({"y": y}).to_parquet(out_dir / f"{prefix}_y.parquet", index=False)
    meta.to_parquet(out_dir / f"{prefix}_meta.parquet", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="configs/features.yaml")
    ap.add_argument("--cohorts_dir", default="data/cohorts")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="name=filename (under cohorts_dir). Example: ih_train=train.csv ih_test=test.csv")
    args = ap.parse_args()

    feat_cfg = load_yaml(args.features)
    cohorts_dir = Path(args.cohorts_dir)
    out_dir = Path(args.out_dir)

    for pair in args.pairs:
        name, fname = pair.split("=", 1)
        df = read_any(cohorts_dir / fname)
        X, y, meta, feat_names = prepare_features(df, feat_cfg)
        persist_processed(name, X, y, meta, out_dir)
        (out_dir / f"{name}_features.txt").write_text("\n".join(feat_names))
        print(f"[ok] {name}: X={X.shape} y={y.shape} meta={meta.shape}")
    print("Saved to:", out_dir)
