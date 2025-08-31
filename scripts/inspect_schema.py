import argparse, pandas as pd
from pathlib import Path

def read_any(path: Path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, nrows=2000)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for f in args.files:
        df = read_any(Path(f))
        print("\n===", f, "===")
        print(df.dtypes)
        print("cols:", list(df.columns))
