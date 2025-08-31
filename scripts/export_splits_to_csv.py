#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

DEFAULT_SPLITS = ["random", "temporal", "hospital", "hospital5src"]
PARTS = ["train", "val", "test", "meta_train", "meta_val", "meta_test"]


def export_one(indir: Path, outdir: Path, stem: str, gzip: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    wrote_any = False

    for part in PARTS:
        pin = indir / f"{stem}_{part}.parquet"
        if not pin.exists():
            continue

        df = pd.read_parquet(pin)
        suffix = ".csv.gz" if gzip else ".csv"
        pout = outdir / f"{stem}_{part}{suffix}"

        if gzip:
            df.to_csv(pout, index=False, compression="gzip")
        else:
            df.to_csv(pout, index=False)

        print(f"★ {stem}_{part}: {df.shape} -> {pout}")
        wrote_any = True

    if not wrote_any:
        print(f"[WARN] No files found for stem '{stem}' in {indir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export Parquet splits to CSV")
    p.add_argument("--indir", type=Path, default=Path("data/splits"),
                   help="Directory containing *_{train,val,test,meta_*}.parquet")
    p.add_argument("--outdir", type=Path, default=Path("data/csv_splits"),
                   help="Directory to write CSV files")
    p.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS,
                   help=f"Split stems to export (default: {', '.join(DEFAULT_SPLITS)})")
    p.add_argument("--gzip", action="store_true", help="Write CSVs compressed as .csv.gz")
    args = p.parse_args()

    for stem in args.splits:
        export_one(args.indir, args.outdir, stem, gzip=args.gzip)


if __name__ == "__main__":
    main()
