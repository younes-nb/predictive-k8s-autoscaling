import os
import sys
import time
import argparse
import subprocess
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = THIS_DIR

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import (
    PATHS,
    PREPROCESSING,
    FEATURE_SETS,
    TRAINING,
    get_checkpoint_path,
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
    ap.add_argument(
        "--delete_raw",
        action="store_false",
        dest="keep_raw",
        help="Delete raw files during preprocessing.",
    )
    ap.add_argument("--windows_dir", default=PATHS.WINDOWS_DIR)
    ap.add_argument(
        "--archetype_mode",
        action="store_true",
        default=TRAINING.ARCHETYPE_MODE,
        help="Enable loop over all detected cluster archetypes.",
    )
    ap.add_argument(
        "--archetype_id",
        type=int,
        default=None,
        help="Run pipeline for a specific cluster archetype only.",
    )
    ap.add_argument("--skip_preprocessing", action="store_true")
    ap.add_argument("--skip_clustering", action="store_true")
    ap.add_argument("--skip_training", action="store_true")
    ap.add_argument("--skip_testing", action="store_true")
    ap.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--bidirectional", action="store_true", default=TRAINING.BIDIRECTIONAL
    )
    ap.add_argument("--residual", action="store_true", default=TRAINING.RESIDUAL)
    ap.add_argument(
        "--max_services",
        type=int,
        default=PREPROCESSING.MAX_SERVICES,
        help="Number of microservices to process. None uses all microservices.",
    )

    args = ap.parse_args()

    preprocess_script = os.path.join(REPO_ROOT, "run_preprocessing.py")
    cluster_script = os.path.join(REPO_ROOT, "tools", "cluster_workloads.py")
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
        if args.max_services is not None:
            cmd_pre.extend(["--max_services", str(args.max_services)])
        if args.skip_fetch:
            cmd_pre.append("--skip_fetch")
        if args.skip_ingest:
            cmd_pre.append("--skip_ingest")
        if args.skip_windows:
            cmd_pre.append("--skip_windows")
        if not args.keep_raw:
            cmd_pre.append("--delete_raw")
        total_times["preprocessing"] = run(cmd_pre, "Step 1: Preprocessing")

    if args.archetype_mode and not args.skip_clustering:
        cmd_cluster = [
            sys.executable,
            cluster_script,
        ]
        if args.max_services is not None:
            cmd_cluster.extend(["--max_services", str(args.max_services)])
        total_times["clustering"] = run(cmd_cluster, "Step 1.5: Clustering Workloads")

    ids_to_run = [None]
    if args.archetype_id is not None:
        ids_to_run = [args.archetype_id]
    elif args.archetype_mode:
        if os.path.exists(PATHS.ARCHETYPE_MAPPING):
            with open(PATHS.ARCHETYPE_MAPPING, "r") as f:
                mapping = json.load(f)
                ids_to_run = sorted(list(set(mapping.values())))
        else:
            print(
                f"[WARN] Archetype mapping not found at {PATHS.ARCHETYPE_MAPPING}. Using global model."
            )

    for arch_id in ids_to_run:
        arch_label = f"ARCH {arch_id}" if arch_id is not None else "GLOBAL"
        print(f"\n--- Processing {arch_label} ---")

        current_checkpoint = get_checkpoint_path(arch_id)

        if TRAINING.USE_WEIGHTS and not args.skip_training:
            base_cmd_w = [
                sys.executable,
                compute_weights_script,
                "--windows_dir",
                args.windows_dir,
                "--theta_mode",
                TRAINING.THETA_MODE,
            ]
            if arch_id is not None:
                base_cmd_w.extend(["--archetype_id", str(arch_id)])

            run(base_cmd_w + ["--split", "train"], f"Weights Train ({arch_label})")
            run(base_cmd_w + ["--split", "val"], f"Weights Val ({arch_label})")

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
            ]
            if arch_id is not None:
                cmd_train.extend(["--archetype_id", str(arch_id)])
            if TRAINING.USE_WEIGHTS:
                cmd_train.append("--use_weights")

            total_times[f"train_{arch_id}"] = run(cmd_train, f"Training ({arch_label})")

        if not args.skip_testing:
            cmd_test = [
                sys.executable,
                test_script,
                "--windows_dir",
                args.windows_dir,
                "--checkpoint_path",
                current_checkpoint,
            ]
            if arch_id is not None:
                cmd_test.extend(["--archetype_id", str(arch_id)])

            total_times[f"test_{arch_id}"] = run(cmd_test, f"Testing ({arch_label})")

    print("\n========== PIPELINE COMPLETE ==========")
    for stage, t in total_times.items():
        print(f"{stage:>25}: {t:.2f}s")
    print("=======================================")


if __name__ == "__main__":
    main()
