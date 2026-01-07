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
    DEFAULT_CHECKPOINT_PATH,
    PREPROCESSING,
    FEATURE_SETS,
    TRAINING,
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
        description="Run full pipeline: preprocessing -> compute weights -> train model -> test model"
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
        help="RNN cell type (default: lstm).",
    )

    ap.add_argument("--input_len", type=int, help="Override input sequence length")
    ap.add_argument("--pred_horizon", type=int, help="Override prediction horizon")
    ap.add_argument("--hidden_size", type=int, help="Override hidden size")
    ap.add_argument("--num_layers", type=int, help="Override number of RNN layers")
    ap.add_argument("--dropout", type=float, help="Override dropout rate")

    ap.add_argument("--skip_preprocessing", action="store_true")
    ap.add_argument("--skip_training", action="store_true")
    ap.add_argument("--skip_testing", action="store_true")

    ap.add_argument("--skip_fetch", action="store_true")
    ap.add_argument("--skip_ingest", action="store_true")
    ap.add_argument("--skip_windows", action="store_true")
    ap.add_argument(
        "--delete_raw",
        action="store_false",
        dest="keep_raw",
        help="Delete raw files during preprocessing.",
    )

    ap.add_argument(
        "--use_weights",
        action="store_true",
        default=TRAINING.USE_WEIGHTS,
        help="Use adaptive boundary weights for training (requires a pre-trained checkpoint or 2-stage training)",
    )

    args = ap.parse_args()

    preprocess_script = os.path.join(REPO_ROOT, "run_preprocessing.py")
    train_script = os.path.join(REPO_ROOT, "training", "train.py")
    test_script = os.path.join(REPO_ROOT, "training", "evaluate.py")

    compute_weights_script = os.path.join(
        REPO_ROOT, "tools", "compute_boundary_weights.py"
    )

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
        if not args.keep_raw:
            cmd_pre.append("--delete_raw")

        total_times["preprocessing"] = run(cmd_pre, "Step 1: Preprocessing")
    else:
        print("\n=== Skipping preprocessing step (per --skip_preprocessing) ===")

    if not args.skip_training and args.use_weights:
        if not os.path.exists(args.checkpoint_path):
            print(
                f"[WARN] --use_weights requested but {args.checkpoint_path} not found."
            )
            print(
                "       Skipping weight computation (cannot compute uncertainty without a model)."
            )
        else:
            print("\n=== Computing boundary weights ===")
            cmd_weights = [
                sys.executable,
                compute_weights_script,
                "--windows_dir",
                args.windows_dir,
                "--checkpoint_path",
                args.checkpoint_path,
                "--rnn_type",
                args.rnn_type,
            ]

            if args.input_len:
                cmd_weights.extend(["--input_len", str(args.input_len)])
            if args.hidden_size:
                cmd_weights.extend(["--hidden_size", str(args.hidden_size)])
            if args.num_layers:
                cmd_weights.extend(["--num_layers", str(args.num_layers)])
            if args.dropout:
                cmd_weights.extend(["--dropout", str(args.dropout)])
            if args.pred_horizon:
                cmd_weights.extend(["--horizon", str(args.pred_horizon)])

            total_times["compute_weights"] = run(
                cmd_weights, "Step 2: Compute Boundary Weights"
            )

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
            "--feature_set",
            args.feature_set,
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
        if args.use_weights:
            cmd_train.append("--use_weights")

        total_times["training"] = run(cmd_train, "Step 3: Train Model")
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

        total_times["testing"] = run(cmd_test, "Step 4: Test Model")
    else:
        print("\n=== Skipping testing step (per --skip_testing) ===")

    print("\n========== PIPELINE SUMMARY ==========")
    for stage, t in total_times.items():
        print(f"{stage:>15}: {t:.2f}s")
    print("=======================================")


if __name__ == "__main__":
    main()
