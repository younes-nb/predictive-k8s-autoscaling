import glob
import os
import sys
import argparse

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import PREPROCESSING


def main():
    p = argparse.ArgumentParser(
        description="Ingest CSV trace files into Parquet (CPU-only baseline)."
    )
    p.add_argument("--raw_dir", required=True, help="Directory with CSV files.")
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for Parquet files.",
    )
    p.add_argument(
        "--repartition",
        type=int,
        default=PREPROCESSING.REPARTITION,
        help=f"Number of Parquet parts (default {PREPROCESSING.REPARTITION}).",
    )
    p.add_argument(
        "--keep_raw",
        action="store_true",
        help="If set, do NOT delete CSV files after Parquet is written.",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(args.raw_dir, "*.csv"))
        + glob.glob(os.path.join(args.raw_dir, "*.csv.gz"))
    )

    if not files:
        raise SystemExit(f"No CSV/CSV.GZ files found under {args.raw_dir}")

    batches = []
    scan_kwargs = dict(low_memory=True, try_parse_dates=False, infer_schema_length=50000)
    needed = {"timestamp", "msname", "msinstanceid", "cpu_utilization"}

    for f in files:
        print(f"Reading {f} ...")
        df = pl.read_csv(f, **scan_kwargs)

        missing = needed - set(df.columns)
        if missing:
            print(f"[WARN] {f} missing columns {missing}; skipping.")
            continue

        df = df.drop_nulls(subset=list(needed))

        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp") / 1000, time_unit="s").alias("timestamp_dt")
        )

        df = df.select(
            "timestamp", "timestamp_dt", "msname", "msinstanceid", "cpu_utilization"
        )

        batches.append(df)

    if not batches:
        raise SystemExit("No usable CSV batches after validation")

    big = pl.concat(batches, rechunk=True)
    print(f"Total rows after concat: {big.height}")

    big = big.sort(["timestamp_dt", "msname", "msinstanceid"])
    big = big.rechunk()

    npart = max(1, args.repartition)
    rows = big.height
    rows_per = (rows + npart - 1) // npart

    for i in range(npart):
        start = i * rows_per
        if start >= rows:
            break
        end = min(rows, start + rows_per)
        out_path = os.path.join(args.out_dir, f"part-{i:05d}.parquet")
        print(f"Writing rows {start}:{end} -> {out_path}")
        big.slice(start, end - start).write_parquet(out_path, compression="zstd")

    print("Wrote parquet parts to", args.out_dir)

    if not args.keep_raw:
        print("Deleting raw CSV files to save space...")
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print(f"[WARN] Failed to remove {f}: {e}")


if __name__ == "__main__":
    main()
