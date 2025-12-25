import os
import sys
import time
import argparse
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = THIS_DIR

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, DEFAULT_CHECKPOINT_PATH, PREPROCESSING, FEATURE_SETS


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
        description="Run full pipeline: preprocessing -> train model -> test model"
    )

    ap.add_argument("--start_date", default="0d0", help="e.g. 0d0 (default: 0d0)")
    ap.add_argument("--end_date", default="7d0", help="e.g. 7d0 (default: 7d0)")

    ap.add_argument(
        "--feature_set",
        default=PREPROCESSING.FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
        help="Predefined feature set to use for preprocessing (default from config).",
    )

    ap.add_argument(
        "--windows_dir",
        default=PATHS.WINDOWS_DIR,
        help=f"Directory with windows .npy files (default: {PATHS.WINDOWS_DIR})",
    )
    ap.add_argument(
        "--checkpoint_path",
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to save/load model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU for train step even if CUDA is available.",
    )
    ap.add_argument(
        "--rnn_type",
        choices=["lstm", "gru"],
        default="lstm",
        help="RNN cell type to use in train_rnn.py/test_rnn.py (default: lstm).",
    )

    ap.add_argument("--input_len", type=int, help="Override input sequence length passed to train_rnn.py")
    ap.add_argument("--pred_horizon", type=int, help="Override prediction horizon passed to train_rnn.py")
    ap.add_argument("--hidden_size", type=int, help="Override hidden size passed to train_rnn.py")
    ap.add_argument("--num_layers", type=int, help="Override number of RNN layers passed to train_rnn.py")
    ap.add_argument("--dropout", type=float, help="Override dropout rate passed to train_rnn.py")

    ap.add_argument("--skip_preprocessing", action="store_true")
    ap.add_argument("--skip_training", action="store_true")
    ap.add_argument("--skip_testing", action="store_true")

    ap.add_argument("--skip_fetch", action="store_true")
    ap.add_argument("--skip_ingest", action="store_true")
    ap.add_argument("--skip_windows", action="store_true")
    ap.add_argument("--keep_raw", action="store_true")

    args = ap.parse_args()

    preprocess_script = os.path.join(REPO_ROOT, "run_preprocessing.py")
    train_script = os.path.join(REPO_ROOT, "training", "train_rnn.py")
    test_script = os.path.join(REPO_ROOT, "training", "test_rnn.py")

    total_times = {}

    if not args.skip_preprocessing:
        cmd_pre = [
            sys.executable,
            preprocess_script,
            "--start_date",
            args.start_date,
            "--end_date",
            args.end_date,
            "--feature_set",
            args.feature_set,
            "--windows_dir",
            args.windows_dir,
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
            "--rnn_type",
            args.rnn_type,
        ]
        if args.cpu:
            cmd_train.append("--cpu")

        if args.input_len is not None:
            cmd_train.extend(["--input_len", str(args.input_len)])
        if args.pred_horizon is not None:
            cmd_train.extend(["--pred_horizon", str(args.pred_horizon)])
        if args.hidden_size is not None:
            cmd_train.extend(["--hidden_size", str(args.hidden_size)])
        if args.num_layers is not None:
            cmd_train.extend(["--num_layers", str(args.num_layers)])
        if args.dropout is not None:
            cmd_train.extend(["--dropout", str(args.dropout)])

        total_times["training"] = run(cmd_train, "Step 2: Train Model")
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
            "--rnn_type",
            args.rnn_type,
        ]
        total_times["testing"] = run(cmd_test, "Step 3: Test Model")
    else:
        print("\n=== Skipping testing step (per --skip_testing) ===")

    print("\n========== PIPELINE SUMMARY ==========")
    for stage, t in total_times.items():
        print(f"{stage:>12}: {t:.2f}s")
    print("=======================================")


if __name__ == "__main__":
    main()
