import argparse
import glob
import os
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import PREPROCESSING


def main():
    p = argparse.ArgumentParser(
        description="Ingest CSV trace files into Parquet in a streaming, memory-safe way."
    )
    p.add_argument("--raw_dir", required=True, help="Directory with CSV / CSV.GZ files.")
    p.add_argument("--out_dir", required=True, help="Output directory for Parquet files.")
    p.add_argument(
        "--keep_raw",
        action="store_true",
        help="If set, do NOT delete CSV files after Parquet is written.",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(args.raw_dir, "*.csv")) +
        glob.glob(os.path.join(args.raw_dir, "*.csv.gz"))
    )
    if not files:
        raise SystemExit(f"No CSV/CSV.GZ files found under {args.raw_dir}")

    needed = {"timestamp", PREPROCESSING.TARGET_COL, *PREPROCESSING.ID_COLS}
    scan_kwargs = dict(
        low_memory=True,
        try_parse_dates=False,
        infer_schema_length=50_000,
    )

    part_idx = 0

    for f in files:
        print(f"Reading {f} ...")
        df = pl.read_csv(f, **scan_kwargs)

        missing = needed - set(df.columns)
        if missing:
            print(f"[WARN] {f} missing columns {missing}; skipping.")
            continue

        df = df.drop_nulls(subset=list(needed))

        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp") / 1000, time_unit="s").alias(PREPROCESSING.TIME_COL)
        )

        df = df.select(
            "timestamp",
            PREPROCESSING.TIME_COL,
            *list(PREPROCESSING.ID_COLS),
            PREPROCESSING.TARGET_COL,
        )

        out_path = os.path.join(args.out_dir, f"part-{part_idx:05d}.parquet")
        print(f"  Writing {df.height} rows -> {out_path}")
        df.write_parquet(out_path, compression="zstd")
        part_idx += 1

        if not args.keep_raw:
            try:
                os.remove(f)
            except OSError as e:
                print(f"[WARN] Failed to remove {f}: {e}")

    print(f"Wrote {part_idx} parquet parts to {args.out_dir}")


if __name__ == "__main__":
    main()
