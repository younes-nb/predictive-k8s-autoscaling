import argparse
import os
import sys

import pyarrow.compute as pc
import pyarrow.dataset as ds

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths


def find_mcr_columns(schema):
    return [name for name in schema.names if name.endswith("_mcr")]


def compute_max_per_column(dataset, columns, batch_size):
    max_values = {col: None for col in columns}
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        for col in columns:
            value = pc.max(batch.column(col))
            if not value.is_valid:
                continue
            value_py = value.as_py()
            current = max_values[col]
            max_values[col] = value_py if current is None else max(current, value_py)
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
    args = parser.parse_args()

    if not os.path.exists(args.parquet_dir):
        raise SystemExit(f"Parquet directory does not exist: {args.parquet_dir}")

    dataset = ds.dataset(args.parquet_dir, format="parquet")
    if len(dataset.files) == 0:
        raise SystemExit(f"No parquet files found under {args.parquet_dir}")

    mcr_columns = find_mcr_columns(dataset.schema)
    if not mcr_columns:
        raise SystemExit("No *_mcr columns found in the parquet schema.")

    max_values = compute_max_per_column(dataset, mcr_columns, args.batch_size)

    print(f"Parquet dir: {args.parquet_dir}")
    print("Max values per MCR column:")
    for col in sorted(mcr_columns):
        value = max_values.get(col)
        print(f"  {col}: {value}")


if __name__ == "__main__":
    main()
