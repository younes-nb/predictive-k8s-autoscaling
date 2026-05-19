import os
import sys
import time
import argparse
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import (
    PATHS,
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
    ap = argparse.ArgumentParser(description="Run full pipeline")
    ap.add_argument("--start_date", default="0d0")
    ap.add_argument("--end_date", default="7d0")
    ap.add_argument(
        "--feature_set",
        default=PREPROCESSING.FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
    )
    ap.add_argument("--skip_fetch", action="store_true")
    ap.add_argument("--skip_ingest", action="store_true")
    ap.add_argument("--skip_windows", action="store_true")
    ap.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--windows_dir", default=PATHS.WINDOWS_DIR)
    ap.add_argument("--skip_preprocessing", action="store_true")
    ap.add_argument("--skip_training", action="store_true")
    ap.add_argument("--skip_testing", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--bidirectional", action="store_true", default=TRAINING.BIDIRECTIONAL
    )
    ap.add_argument(
        "--max_services",
        type=int,
        default=PREPROCESSING.MAX_SERVICES,
        help="Limit number of services for faster testing.",
    )

    args = ap.parse_args()

    preprocess_script = os.path.join(
        REPO_ROOT, "pipelines", "preprocessing_pipeline.py"
    )
    train_script = os.path.join(REPO_ROOT, "training", "train.py")
    test_script = os.path.join(REPO_ROOT, "training", "evaluate.py")
    compute_weights_script = os.path.join(
        REPO_ROOT, "tooling", "compute_boundary_weights.py"
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
        if args.max_services is not None:
            cmd_pre.extend(["--max_services", str(args.max_services)])
        if args.skip_fetch:
            cmd_pre.append("--skip_fetch")
        if args.skip_ingest:
            cmd_pre.append("--skip_ingest")
        if args.skip_windows:
            cmd_pre.append("--skip_windows")

        total_times["preprocessing"] = run(cmd_pre, "Step 1: Preprocessing")

    if TRAINING.USE_WEIGHTS and not args.skip_training:
        base_cmd_w = [
            sys.executable,
            compute_weights_script,
            "--windows_dir",
            args.windows_dir,
            "--theta_mode",
            TRAINING.THETA_MODE,
        ]
        run(base_cmd_w + ["--split", "train"], "Weights Generation (Train)")
        run(base_cmd_w + ["--split", "val"], "Weights Generation (Val)")

    current_checkpoint = PATHS.CHECKPOINT_PATH

    if not args.skip_training:
        cmd_train = [
            sys.executable,
            train_script,
            "--windows_dir",
            args.windows_dir,
            "--checkpoint_path",
            current_checkpoint,
            "--rnn_type",
            args.rnn_type,
            "--feature_set",
            args.feature_set,
            "--batch_size",
            str(TRAINING.BATCH_SIZE),
            "--epochs",
            str(TRAINING.EPOCHS),
        ]
        if TRAINING.USE_WEIGHTS:
            cmd_train.append("--use_weights")
        if args.cpu:
            cmd_train.append("--cpu")

        total_times["training"] = run(
            cmd_train, "Step 2: Training"
        )

    if not args.skip_testing:
        cmd_test = [
            sys.executable,
            test_script,
            "--windows_dir",
            args.windows_dir,
            "--checkpoint_path",
            current_checkpoint,
            "--batch_size",
            str(TRAINING.BATCH_SIZE),
        ]
        if args.cpu:
            cmd_test.append("--cpu")

        total_times["testing"] = run(cmd_test, "Step 3: Evaluation & Diagnostics")

    print("\n========== PIPELINE COMPLETE ==========")
    for stage, t in total_times.items():
        print(f"{stage:>20}: {t:.2f}s")
    print("=======================================")


if __name__ == "__main__":
    main()
