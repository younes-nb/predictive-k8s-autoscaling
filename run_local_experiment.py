import os
import sys
import argparse
import subprocess
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = THIS_DIR

os.environ["PIPELINE_ENV"] = "local"

from config.defaults import PATHS, PREPROCESSING, tables_for_feature_set


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
        description="Run lightweight local pipeline with Domain Shift Evaluation"
    )
    ap.add_argument(
        "--csv_path", default="hpa_historical_logs.csv", help="Path to load test CSV"
    )
    ap.add_argument("--start_date", default="0d0")
    ap.add_argument("--end_date", default="0d1")

    ap.add_argument("--skip_fetch", action="store_true")
    ap.add_argument("--skip_ingest", action="store_true")
    ap.add_argument("--skip_windows", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    os.makedirs(PATHS.RAW_MSRESOURCE, exist_ok=True)
    os.makedirs(PATHS.RAW_MSRTMCRE, exist_ok=True)
    os.makedirs(PATHS.WINDOWS_DIR, exist_ok=True)
    os.makedirs(PATHS.MODELS_DIR, exist_ok=True)

    needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))

    print(f"\n🚀 STARTING LOCAL LIGHTWEIGHT PIPELINE")
    print(f"Local Data Root: {os.path.abspath(os.path.join(THIS_DIR, 'local_data'))}")

    fetch_script = os.path.join(REPO_ROOT, "preprocessing", "fetch_traces_azure.py")
    ingest_script = os.path.join(REPO_ROOT, "preprocessing", "ingest_traces_parquet.py")
    windows_local_script = os.path.join(
        REPO_ROOT, "preprocessing", "build_windows_local.py"
    )
    train_script = os.path.join(REPO_ROOT, "training", "train.py")
    test_script = os.path.join(REPO_ROOT, "training", "evaluate.py")
    checkpoint_path = os.path.join(PATHS.MODELS_DIR, "model_local.pt")

    if not args.skip_fetch:
        for table in needed_tables:
            raw_dir = getattr(PATHS, f"RAW_{table.upper()}")
            os.makedirs(raw_dir, exist_ok=True)
            run(
                [
                    sys.executable,
                    fetch_script,
                    "--table",
                    table,
                    "--start_date",
                    args.start_date,
                    "--end_date",
                    args.end_date,
                    "--out_dir",
                    raw_dir,
                ],
                f"Fetching {table} Traces",
            )

    if not args.skip_ingest:
        for table in needed_tables:
            raw_dir = getattr(PATHS, f"RAW_{table.upper()}")
            pq_dir = getattr(PATHS, f"PARQUET_{table.upper()}")
            os.makedirs(pq_dir, exist_ok=True)
            run(
                [
                    sys.executable,
                    ingest_script,
                    "--table",
                    table,
                    "--feature_set",
                    args.feature_set,
                    "--raw_dir",
                    raw_dir,
                    "--out_dir",
                    pq_dir,
                ],
                f"Ingesting {table}",
            )

    if not args.skip_windows:
        run(
            [
                sys.executable,
                windows_local_script,
                "--out_dir",
                PATHS.WINDOWS_DIR,
                "--csv_path",
                args.csv_path,
                "--feature_set",
                args.feature_set,
            ],
            "Building Windows",
        )

    if not args.skip_train:
        run(
            [
                sys.executable,
                train_script,
                "--windows_dir",
                PATHS.WINDOWS_DIR,
                "--checkpoint_path",
                checkpoint_path,
                "--feature_set",
                PREPROCESSING.FEATURE_SET,
            ],
            "Training Local Model",
        )

    if not args.skip_testing:
        run(
            [
                sys.executable,
                test_script,
                "--windows_dir",
                PATHS.WINDOWS_DIR,
                "--checkpoint_path",
                checkpoint_path,
            ],
            "Evaluation",
        )


if __name__ == "__main__":
    main()
