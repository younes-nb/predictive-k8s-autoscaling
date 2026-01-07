import glob
import os
import argparse
import sys

import polars as pl
import pyarrow.parquet as pq

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import (
    PREPROCESSING,
    FEATURE_SETS,
    DATASET_TABLES,
    table_to_raw_columns,
)


def main():
    p = argparse.ArgumentParser(
        description="Ingest CSV trace files into Parquet (feature-set aware, per-table)."
    )
    p.add_argument(
        "--table", required=True, help="Dataset table name, e.g. msresource, node"
    )
    p.add_argument(
        "--feature_set",
        type=str,
        default=PREPROCESSING.FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
        help="Controls which feature columns are retained for this table.",
    )
    p.add_argument("--raw_dir", required=True, help="Directory with CSV/CSV.GZ files.")
    p.add_argument(
        "--out_dir", required=True, help="Output directory for Parquet files."
    )
    p.add_argument(
        "--repartition",
        type=int,
        default=PREPROCESSING.REPARTITION,
        help="Approx number of Parquet parts to write.",
    )
    p.add_argument(
        "--delete_raw",
        action="store_false",
        dest="keep_raw",
        help="Delete raw CSV files after successful ingestion (default: keep them).",
    )
    args = p.parse_args()

    if args.table not in DATASET_TABLES:
        raise SystemExit(f"Unknown table '{args.table}'. Add it to DATASET_TABLES.")

    cfg = DATASET_TABLES[args.table]
    key_cols = list(cfg.get("key_cols", []))
    if not key_cols:
        raise SystemExit(
            f"DATASET_TABLES['{args.table}'] must define key_cols (e.g., ['msname','msinstanceid'] or ['nodeid'])."
        )

    needed_by_table = table_to_raw_columns(args.feature_set)
    feature_cols = needed_by_table.get(args.table, [])
    if not feature_cols:
        raise SystemExit(
            f"feature_set='{args.feature_set}' does not require any columns from table='{args.table}'."
        )

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(args.raw_dir, "*.csv"))
        + glob.glob(os.path.join(args.raw_dir, "*.csv.gz"))
    )
    if not files:
        raise SystemExit(f"No CSV/CSV.GZ files found under {args.raw_dir}")

    n_files = len(files)
    npart = max(1, min(int(args.repartition), n_files))

    base_needed = {"timestamp", *key_cols}
    needed = base_needed | set(feature_cols)

    print(f"Table: {args.table}")
    print(f"Feature set: {args.feature_set}")
    print(f"Key columns: {key_cols}")
    print(f"Keeping feature columns: {feature_cols}")
    print(f"Found {n_files} files, writing ~{npart} parquet parts")

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
            f"\n[Part {part_idx}] files {start}..{end - 1} ({len(batch_files)} files)"
        )
        writer = None
        total_rows = 0
        out_path = os.path.join(args.out_dir, f"part-{part_idx:05d}.parquet")

        for f in batch_files:
            print(f"  Reading {f} ...")
            df = pl.read_csv(f, **scan_kwargs)

            missing = needed - set(df.columns)
            if missing:
                print(f"  [WARN] missing columns {sorted(list(missing))}; skipping.")
                continue

            df = df.drop_nulls(subset=list(needed))
            if df.height == 0:
                continue

            df = df.with_columns(
                pl.from_epoch(pl.col("timestamp") / 1000, time_unit="s").alias(
                    "timestamp_dt"
                )
            )

            select_cols = ["timestamp", "timestamp_dt", *key_cols, *feature_cols]
            select_cols = list(dict.fromkeys(select_cols))

            sort_cols = ["timestamp_dt", *key_cols]
            df = df.select(select_cols).sort(sort_cols)

            if df.height == 0:
                continue

            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += table.num_rows

        if writer is not None:
            writer.close()
            print(f"  Wrote {total_rows} rows -> {out_path}")
            part_idx += 1
        else:
            print("  No usable rows; no parquet written.")

    print(f"\nWrote {part_idx} parquet parts to {args.out_dir}")

    if not args.keep_raw:
        print("Deleting raw CSV files...")
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print(f"[WARN] Failed to remove {f}: {e}")


if __name__ == "__main__":
    main()
