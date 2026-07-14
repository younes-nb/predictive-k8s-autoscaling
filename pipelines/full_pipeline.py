import os
import sys
import argparse
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, PREPROCESSING, TRAINING
from shared.features import FEATURE_SETS
from shared.subprocess_utils import run


def main():
    ap = argparse.ArgumentParser(description="Run full pipeline")
    ap.add_argument("--start_date", default="0d0", help="Trace start date offset (e.g. '0d0' or '7d0')")
    ap.add_argument("--end_date", default="7d0", help="Trace end date offset (e.g. '7d0' or '14d0')")
    ap.add_argument(
        "--feature_set",
        default=PREPROCESSING.FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
        help="Which features to use for training (default: %(default)s)",
    )
    ap.add_argument("--skip_fetch", action="store_true", help="Skip downloading raw trace data from OSS")
    ap.add_argument("--skip_ingest", action="store_true", help="Skip converting raw traces to parquet")
    ap.add_argument("--skip_windows", action="store_true", help="Skip building sliding window datasets")
    ap.add_argument("--windows_dir", default=PATHS.WINDOWS_DIR, help="Output directory for window datasets")
    ap.add_argument("--skip_preprocessing", action="store_true", help="Skip the entire preprocessing pipeline (fetch+ingest+windows)")
    ap.add_argument("--skip_weights", action="store_true", help="Skip boundary weight computation")
    ap.add_argument("--skip_training", action="store_true", help="Skip the training step")
    ap.add_argument("--skip_testing", action="store_true", help="Skip the evaluation step")
    ap.add_argument("--cpu", action="store_true", help="Force CPU execution (no GPU/DDP)")
    ap.add_argument(
        "--resume_training",
        action="store_true",
        help="Resume training from the last saved state if available.",
    )
    ap.add_argument(
        "--hyperparam_optimizer",
        default=TRAINING.HYPERPARAM_OPTIMIZER,
        choices=["sfoa", "none"],
        help="Hyperparameter optimizer to use during training ('none' uses TrainingDefaults directly).",
    )
    ap.add_argument(
        "--sfoa_train_pct", type=float, default=TRAINING.SFOA_TRAIN_PCT,
        help="Percentage of training samples used during SFOA search (default: %(default)s%%)",
    )
    ap.add_argument("--sfoa_val_pct", type=float, default=TRAINING.SFOA_VAL_PCT,
        help="Percentage of validation samples used during SFOA search (default: %(default)s%%)")
    ap.add_argument("--sfoa_num_workers", type=int, default=TRAINING.SFOA_NUM_WORKERS,
        help="Dataloader workers for SFOA search (default: %(default)s)")
    ap.add_argument(
        "--max_services",
        type=int,
        default=PREPROCESSING.MAX_SERVICES,
        help="Limit number of services for faster testing (0 = all)",
    )
    ap.add_argument(
        "--preprocess_approach",
        default="none",
        choices=["none", "smoothing", "sv", "cskv"],
        help="Post-processing: none (raw windows), smoothing (moving avg), sv (SWT+VMD), cskv (CEEMDAN+SE+K-means+VMD)",
    )
    ap.add_argument(
        "--model_type",
        default="lstm",
        choices=["lstm", "gru", "bilstm", "bigrue", "cnn_bilstm", "tcn_bigru"],
        help="Model architecture: lstm/gru (unidirectional), bilstm/bigrue (bidirectional), cnn_bilstm, tcn_bigru",
    )
    ap.add_argument("--smooth_window", type=int, default=5, help="Moving average window size for 'smoothing' approach (default: %(default)s)")
    ap.add_argument("--dataset_workers", type=int, default=0, help="Dataloader workers for sv/cskv datasets (default: %(default)s)")
    ap.add_argument(
        "--train_pct",
        type=float,
        default=TRAINING.TRAIN_PCT,
        help="Percentage of training samples for main training; 25 means 25%%, not 0.25 (100 uses all; <=0 uses all).",
    )
    ap.add_argument(
        "--val_pct",
        type=float,
        default=TRAINING.VAL_PCT,
        help="Percentage of validation samples for main training; 25 means 25%%, not 0.25 (100 uses all; <=0 uses all).",
    )
    ap.add_argument(
        "--test_pct",
        type=float,
        default=TRAINING.TEST_PCT,
        help="Percentage of test samples for evaluation; 25 means 25%%, not 0.25 (100 uses all; <=0 uses all).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=TRAINING.SEED,
        help="Random seed for reproducibility (default: %(default)s)",
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
        cmd_pre.extend(["--preprocess_approach", args.preprocess_approach])
        cmd_pre.extend(["--smooth_window", str(args.smooth_window)])
        cmd_pre.extend(["--subset_seed", str(args.seed)])

        total_times["preprocessing"] = run(cmd_pre, "Step 1: Preprocessing")

    if TRAINING.USE_WEIGHTS and not args.skip_training and not args.skip_weights:
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

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.cpu or gpu_count <= 1:
        launcher_cmd = [sys.executable]
        if gpu_count == 1 and not args.cpu:
            print(
                f"\n[INFO] Detected 1 GPU. Running with standard python script invocation."
            )
        elif gpu_count == 0 or args.cpu:
            print(f"\n[INFO] Running strictly on CPU.")
    else:
        launcher_cmd = ["accelerate", "launch"]
        print(
            f"\n[INFO] Detected {gpu_count} GPUs! Accelerating via DistributedDataParallel (DDP)..."
        )

    if not args.skip_training:
        cmd_train = launcher_cmd + [
            train_script,
            "--windows_dir",
            args.windows_dir,
            "--checkpoint_path",
            current_checkpoint,
            "--feature_set",
            args.feature_set,
            "--batch_size",
            str(TRAINING.BATCH_SIZE),
            "--epochs",
            str(TRAINING.EPOCHS),
            "--model_type",
            args.model_type,
            "--preprocess_approach",
            args.preprocess_approach,
        ]
        if args.preprocess_approach in ("sv", "cskv"):
            cmd_train.extend(["--preprocess_dir", args.windows_dir])
            cmd_train.extend(["--dataset_workers", str(args.dataset_workers)])
        if TRAINING.USE_WEIGHTS:
            cmd_train.append("--use_weights")
        cmd_train.extend(["--hyperparam_optimizer", args.hyperparam_optimizer])
        cmd_train.extend(["--sfoa_train_pct", str(args.sfoa_train_pct)])
        cmd_train.extend(["--sfoa_val_pct", str(args.sfoa_val_pct)])
        cmd_train.extend(["--sfoa_num_workers", str(args.sfoa_num_workers)])
        cmd_train.extend(["--train_pct", str(args.train_pct)])
        cmd_train.extend(["--val_pct", str(args.val_pct)])
        if args.resume_training:
            cmd_train.append("--resume_training")
        if args.cpu:
            cmd_train.append("--cpu")
        cmd_train.extend(["--seed", str(args.seed)])

        total_times["training"] = run(cmd_train, "Step 2: Training")

    if not args.skip_testing:
        cmd_test = launcher_cmd + [
            test_script,
            "--windows_dir",
            args.windows_dir,
            "--checkpoint_path",
            current_checkpoint,
            "--batch_size",
            str(TRAINING.BATCH_SIZE),
        ]
        if args.preprocess_approach in ("sv", "cskv"):
            cmd_test.extend(["--preprocess_dir", args.windows_dir])
        if args.cpu:
            cmd_test.append("--cpu")
        cmd_test.extend(["--test_pct", str(args.test_pct)])
        cmd_test.extend(["--seed", str(args.seed)])

        total_times["testing"] = run(cmd_test, "Step 3: Evaluation & Diagnostics")

    print("\n========== PIPELINE COMPLETE ==========")
    for stage, t in total_times.items():
        print(f"{stage:>20}: {t:.2f}s")
    print("=======================================")


if __name__ == "__main__":
    main()
