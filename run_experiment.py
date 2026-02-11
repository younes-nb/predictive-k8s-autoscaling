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
        description="Run full pipeline: preprocessing -> train base -> compute weights -> train final -> test"
    )

    ap.add_argument("--start_date", default="0d0")
    ap.add_argument("--end_date", default="7d0")
    ap.add_argument(
        "--feature_set",
        default=PREPROCESSING.FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
    )
    ap.add_argument("--windows_dir", default=PATHS.WINDOWS_DIR)
    ap.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)

    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--input_len", type=int)
    ap.add_argument("--pred_horizon", type=int)
    ap.add_argument("--hidden_size", type=int)
    ap.add_argument("--num_layers", type=int)
    ap.add_argument("--dropout", type=float)

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
        "--no_weights",
        action="store_false",
        dest="use_weights",
        help="Disable adaptive boundary weights (force standard training).",
    )
    ap.add_argument(
        "--global_threshold",
        action="store_true",
        help="Use a single global adaptive threshold for all microservices in a batch.",
    )
    ap.set_defaults(use_weights=TRAINING.USE_WEIGHTS)

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
        print("\n=== Skipping preprocessing step ===")

    if not args.skip_training:

        def get_train_cmd(checkpoint_target, use_weights_flag):
            cmd = [
                sys.executable,
                train_script,
                "--windows_dir",
                args.windows_dir,
                "--checkpoint_path",
                checkpoint_target,
                "--rnn_type",
                args.rnn_type,
                "--feature_set",
                args.feature_set,
            ]
            if args.cpu:
                cmd.append("--cpu")
            if args.input_len:
                cmd.extend(["--input_len", str(args.input_len)])
            if args.pred_horizon:
                cmd.extend(["--pred_horizon", str(args.pred_horizon)])
            if args.hidden_size:
                cmd.extend(["--hidden_size", str(args.hidden_size)])
            if args.num_layers:
                cmd.extend(["--num_layers", str(args.num_layers)])
            if args.dropout:
                cmd.extend(["--dropout", str(args.dropout)])

            if use_weights_flag:
                cmd.append("--use_weights")

            return cmd

        if args.use_weights:
            print("\n>>> Adaptive Weights Pipeline Triggered <<<")

            base_checkpoint_path = args.checkpoint_path.replace(".pt", "_base.pt")
            if base_checkpoint_path == args.checkpoint_path:
                base_checkpoint_path += ".base"

            print(
                f"--- Phase 2a: Training Base Model (target: {os.path.basename(base_checkpoint_path)}) ---"
            )
            cmd_train_base = get_train_cmd(base_checkpoint_path, use_weights_flag=False)
            total_times["train_base"] = run(
                cmd_train_base, "Phase 2a: Train Base Model"
            )

            print(f"--- Phase 2b: Computing Weights using Base Model ---")
            cmd_weights = [
                sys.executable,
                compute_weights_script,
                "--windows_dir",
                args.windows_dir,
                "--checkpoint_path",
                base_checkpoint_path,
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
            if args.global_threshold:
                cmd_weights.append("--global_threshold")

            total_times["compute_weights"] = run(
                cmd_weights, "Phase 2b: Compute Boundary Weights"
            )

            print(
                f"--- Phase 2c: Training Final Model (target: {os.path.basename(args.checkpoint_path)}) ---"
            )
            cmd_train_final = get_train_cmd(args.checkpoint_path, use_weights_flag=True)
            total_times["train_final"] = run(
                cmd_train_final, "Phase 2c: Train Final Model"
            )

        else:
            print("\n>>> Standard Training Pipeline (No Weights) <<<")
            cmd_train = get_train_cmd(args.checkpoint_path, use_weights_flag=False)
            total_times["training"] = run(cmd_train, "Step 2: Train Model (Standard)")

    else:
        print("\n=== Skipping training step ===")

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

        if args.use_weights:
            cmd_test.append("--adaptive_threshold")
        else:
            cmd_test.append("--static_threshold")

        if args.global_threshold:
            cmd_test.append("--global_threshold")

        total_times["testing"] = run(cmd_test, "Step 4: Test Model")
    else:
        print("\n=== Skipping testing step ===")

    print("\n========== PIPELINE SUMMARY ==========")
    for stage, t in total_times.items():
        print(f"{stage:>25}: {t:.2f}s")
    print("=======================================")


if __name__ == "__main__":
    main()
