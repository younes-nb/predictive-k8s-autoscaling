import argparse
import os
import sys
import time
import logging

import pyarrow.compute as pc
import pyarrow.dataset as ds

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths


def find_mcr_columns(schema):
    return [name for name in schema.names if name.endswith("_mcr")]


def compute_max_per_column(dataset, columns, batch_size, logger):
    max_values = {col: None for col in columns}
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)

    batch_idx = 0
    total_rows = 0
    total_time_s = 0.0

    for batch in scanner.to_batches():
        batch_idx += 1
        nrows = batch.num_rows
        total_rows += nrows

        t0 = time.perf_counter()
        for col in columns:
            value = pc.max(batch.column(col))
            if not value.is_valid:
                continue
            value_py = value.as_py()
            current = max_values[col]
            max_values[col] = value_py if current is None else max(current, value_py)
        dt = time.perf_counter() - t0
        total_time_s += dt

        logger.info(
            "Batch %d: rows=%d, duration=%.3fs, rows_per_s=%.0f",
            batch_idx,
            nrows,
            dt,
            (nrows / dt) if dt > 0 else float("inf"),
        )

    logger.info(
        "Completed scan: batches=%d, total_rows=%d, total_duration=%.3fs, avg_rows_per_s=%.0f",
        batch_idx,
        total_rows,
        total_time_s,
        (total_rows / total_time_s) if total_time_s > 0 else float("inf"),
    )

    return max_values


def main():
    parser = argparse.ArgumentParser(
        description="Compute max values for all *_mcr columns in msrtmcre parquet files."
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default=Paths.PARQUET_MSRTMCRE,
        help="Directory containing msrtmcre parquet parts.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100_000,
        help="Number of rows per batch to scan (lower to reduce memory use).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Python logging level (e.g. DEBUG, INFO, WARNING).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.parquet_dir):
        raise SystemExit(f"Parquet directory does not exist: {args.parquet_dir}")

    dataset = ds.dataset(args.parquet_dir, format="parquet")
    if len(dataset.files) == 0:
        raise SystemExit(f"No parquet files found under {args.parquet_dir}")

    mcr_columns = find_mcr_columns(dataset.schema)
    if not mcr_columns:
        raise SystemExit("No *_mcr columns found in the parquet schema.")

    logger.info(
        "Starting scan: parquet_dir=%s, files=%d, columns=%d, batch_size=%d",
        args.parquet_dir,
        len(dataset.files),
        len(mcr_columns),
        args.batch_size,
    )

    max_values = compute_max_per_column(dataset, mcr_columns, args.batch_size, logger)

    print(f"Parquet dir: {args.parquet_dir}")
    print("Max values per MCR column:")
    for col in sorted(mcr_columns):
        value = max_values.get(col)
        print(f"  {col}: {value}")


if __name__ == "__main__":
    main()
