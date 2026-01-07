import os
import sys
import time
import argparse
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = THIS_DIR
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import (
    PATHS,
    PREPROCESSING,
    FEATURE_SETS,
    DATASET_TABLES,
    tables_for_feature_set,
)


def run(cmd, title: str):
    print(f"\n=== {title} ===")
    print("Running:", " ".join(cmd))
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    print(f"=== {title} completed in {elapsed:.2f}s ===")
    return elapsed


def main():
    ap = argparse.ArgumentParser(
        description="Preprocessing: fetch -> ingest(all needed tables) -> build_windows(join)"
    )
    ap.add_argument("--start_date", default="0d0")
    ap.add_argument("--end_date", default="7d0")

    ap.add_argument(
        "--feature_set",
        default=PREPROCESSING.FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
    )
    ap.add_argument("--windows_dir", default=PATHS.WINDOWS_DIR)

    ap.add_argument("--skip_fetch", action="store_true")
    ap.add_argument("--skip_ingest", action="store_true")
    ap.add_argument("--skip_windows", action="store_true")
    ap.add_argument(
        "--delete_raw",
        action="store_false",
        dest="keep_raw",
        help="Delete raw files after ingestion.",
    )

    args = ap.parse_args()

    needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))
    print(f"feature_set={args.feature_set} => tables={needed_tables}")

    fetch_script = os.path.join(REPO_ROOT, "preprocessing", "fetch_traces.py")
    ingest_script = os.path.join(REPO_ROOT, "preprocessing", "ingest_traces_parquet.py")
    windows_script = os.path.join(REPO_ROOT, "preprocessing", "build_windows.py")

    if not args.skip_fetch:
        cmd = [
            sys.executable,
            fetch_script,
            "--start_date",
            args.start_date,
            "--end_date",
            args.end_date,
            "--feature_set",
            args.feature_set,
        ]
        run(cmd, "Step 1: Fetch")
    else:
        print("\n=== Skipping fetch ===")

    if not args.skip_ingest:
        for t in needed_tables:
            cfg = DATASET_TABLES[t]
            cmd = [
                sys.executable,
                ingest_script,
                "--table",
                t,
                "--feature_set",
                args.feature_set,
                "--raw_dir",
                cfg["raw_dir"],
                "--out_dir",
                cfg["parquet_dir"],
            ]
            if not args.keep_raw:
                cmd.append("--delete_raw")
            run(cmd, f"Step 2: Ingest table={t}")
    else:
        print("\n=== Skipping ingest ===")

    if not args.skip_windows:
        cmd = [
            sys.executable,
            windows_script,
            "--out_dir",
            args.windows_dir,
            "--feature_set",
            args.feature_set,
        ]
        run(cmd, "Step 3: Build windows (join tables)")
    else:
        print("\n=== Skipping windows ===")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
