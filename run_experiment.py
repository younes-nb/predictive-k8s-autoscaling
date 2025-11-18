import os
import sys
import time
import argparse
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = THIS_DIR

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import PATHS, DEFAULT_CHECKPOINT_PATH


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
        description="Run full pipeline: preprocessing -> train LSTM -> test LSTM"
    )
    ap.add_argument("--start_date", required=True, help="e.g. 0d0")
    ap.add_argument("--end_date", required=True, help="e.g. 1d0")

    ap.add_argument(
        "--windows_dir",
        default=PATHS.WINDOWS_DIR,
        help=f"Directory with windows .npy files (default: {PATHS.WINDOWS_DIR})",
    )
    ap.add_argument(
        "--checkpoint_path",
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to save/load LSTM checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU for train/test even if CUDA is available.",
    )

    ap.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip run_preprocessing.py (assume windows already built).",
    )
    ap.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training step (assume checkpoint already exists).",
    )
    ap.add_argument(
        "--skip_testing",
        action="store_true",
        help="Skip testing step.",
    )

    ap.add_argument(
        "--skip_fetch",
        action="store_true",
        help="Within preprocessing, skip fetch_traces.py.",
    )
    ap.add_argument(
        "--skip_ingest",
        action="store_true",
        help="Within preprocessing, skip ingest_traces_parquet.py.",
    )
    ap.add_argument(
        "--skip_windows",
        action="store_true",
        help="Within preprocessing, skip build_windows.py.",
    )
    ap.add_argument(
        "--keep_raw",
        action="store_true",
        help="Forward to preprocessing so ingest_traces_parquet.py keeps raw CSVs.",
    )

    args = ap.parse_args()

    preprocess_script = os.path.join(REPO_ROOT, "run_preprocessing.py")
    train_script = os.path.join(REPO_ROOT, "training", "train_lstm.py")
    test_script = os.path.join(REPO_ROOT, "training", "test_lstm.py")

    total_times = {}

    if not args.skip_preprocessing:
        cmd_pre = [
            sys.executable,
            preprocess_script,
            "--start_date",
            args.start_date,
            "--end_date",
            args.end_date,
        ]

        if args.skip_fetch:
            cmd_pre.append("--skip_fetch")
        if args.skip_ingest:
            cmd_pre.append("--skip_ingest")
        if args.skip_windows:
            cmd_pre.append("--skip_windows")
        if args.keep_raw:
            cmd_pre.append("--keep_raw")

        total_times["preprocessing"] = run(cmd_pre, "Step 1: Preprocessing")
    else:
        print("\n=== Skipping preprocessing step (per --skip_preprocessing) ===")

    if not args.skip_training:
        cmd_train = [
            sys.executable,
            train_script,
            "--windows_dir",
            args.windows_dir,
            "--checkpoint_path",
            args.checkpoint_path,
        ]
        if args.cpu:
            cmd_train.append("--cpu")

        total_times["training"] = run(cmd_train, "Step 2: Train LSTM")
    else:
        print("\n=== Skipping training step (per --skip_training) ===")

    if not args.skip_testing:
        cmd_test = [
            sys.executable,
            test_script,
            "--windows_dir",
            args.windows_dir,
            "--checkpoint_path",
            args.checkpoint_path,
        ]
        if args.cpu:
            cmd_test.append("--cpu")

        total_times["testing"] = run(cmd_test, "Step 3: Test LSTM")
    else:
        print("\n=== Skipping testing step (per --skip_testing) ===")

    print("\n========== PIPELINE SUMMARY ==========")
    for stage, t in total_times.items():
        print(f"{stage:>12}: {t:.2f}s")
    print("=======================================")


if __name__ == "__main__":
    main()
