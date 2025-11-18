import glob
import os
import argparse
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import PREPROCESSING


def main():
    p = argparse.ArgumentParser(
        description="Ingest CSV trace files into Parquet (CPU-only baseline, streaming)."
    )
    p.add_argument("--raw_dir", required=True, help="Directory with CSV files.")
    p.add_argument("--out_dir", required=True, help="Output directory for Parquet files.")
    p.add_argument(
        "--repartition",
        type=int,
        default=PREPROCESSING.REPARTITION,
        help="Approximate number of Parquet parts to write (default from config).",
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

    n_files = len(files)
    npart = max(1, min(args.repartition, n_files))

    print(f"Found {n_files} CSV files in {args.raw_dir}")
    print(f"Will create ~{npart} Parquet parts (streaming groups of CSVs).")

    needed = {"timestamp", "msname", "msinstanceid", "cpu_utilization"}
    scan_kwargs = dict(
        low_memory=True,
        try_parse_dates=False,
        infer_schema_length=50000,
    )

    part_idx = 0

    for i in range(npart):
        start = (n_files * i) // npart
        end = (n_files * (i + 1)) // npart
        batch_files = files[start:end]
        if not batch_files:
            continue

        print(
            f"\n[Part {part_idx}] Reading files {start}..{end - 1} "
            f"({len(batch_files)} CSVs)"
        )

        batches = []
        for f in batch_files:
            print(f"  Reading {f} ...")
            df = pl.read_csv(f, **scan_kwargs)

            missing = needed - set(df.columns)
            if missing:
                print(f"  [WARN] {f} missing columns {missing}; skipping.")
                continue

            df = df.drop_nulls(subset=list(needed))

            df = df.with_columns(
                pl.from_epoch(pl.col("timestamp") / 1000, time_unit="s").alias(
                    "timestamp_dt"
                )
            )

            df = df.select(
                "timestamp",
                "timestamp_dt",
                "msname",
                "msinstanceid",
                "cpu_utilization",
            )

            batches.append(df)

        if not batches:
            print(f"  [Part {part_idx}] No usable rows in this batch, skipping.")
            continue

        big = pl.concat(batches, rechunk=True)
        print(f"  [Part {part_idx}] Rows after concat: {big.height}")

        big = big.sort(["timestamp_dt", "msname", "msinstanceid"]).rechunk()

        out_path = os.path.join(args.out_dir, f"part-{part_idx:05d}.parquet")
        print(f"  [Part {part_idx}] Writing -> {out_path}")
        big.write_parquet(out_path, compression="zstd")

        part_idx += 1

    print(f"\nWrote {part_idx} parquet parts to {args.out_dir}")

    if not args.keep_raw:
        print("Deleting raw CSV files to save space...")
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print(f"[WARN] Failed to remove {f}: {e}")


if __name__ == "__main__":
    main()
