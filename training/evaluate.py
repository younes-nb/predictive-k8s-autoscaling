import os
import sys
import random
import argparse
import logging
import time
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

from shared.config_paths import PATHS
from shared.config_training_defaults import TRAINING
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.logging_utils import setup_logging
from shared.features import target_features_for_feature_set, feature_names_for_feature_set
from core.dataset import ShardedWindowsDataset

from training.metrics import compute_metrics, find_max_inference_batch_size
from training.train_helpers import head_slice_dataset_by_pct
from training.sfoa_configs import get_config


MODEL_TYPES = ("lstm", "gru", "bilstm", "bigrue", "cnn_bilstm", "tcn_bigru")
PREPROCESS_APPROACHES = ("none", "smoothing", "sv", "cskv")


def _build_model_from_checkpoint(checkpoint, input_size, device):
    ckpt_args = checkpoint.get("args", {})
    model_type = checkpoint.get("model_type", "lstm")
    num_targets = len(target_features_for_feature_set(ckpt_args.get("feature_set", PREPROCESSING.FEATURE_SET)))

    cfg = get_config(model_type)
    hyperparams = checkpoint.get("hyperparams", cfg.DEFAULTS)
    return cfg.build_model(hyperparams, input_size, ckpt_args, num_targets, device)


def _load_test_dataset(args, ckpt_args, device, log_info, feature_set_name="cpu"):
    input_len = ckpt_args.get("input_len", PREPROCESSING.INPUT_LEN)
    horizon = ckpt_args.get("pred_horizon", PREPROCESSING.PRED_HORIZON)
    model_type = ckpt_args.get("model_type", "lstm")
    preprocess_approach = ckpt_args.get("preprocess_approach", "none")

    if preprocess_approach in ("none", "smoothing"):
        test_ds = ShardedWindowsDataset(
            args.windows_dir, "test", input_len, horizon, use_weights=False
        )
        total_test_samples = len(test_ds)
        test_ds = head_slice_dataset_by_pct(test_ds, args.test_pct)
        log_info(f"Test samples (Total): {total_test_samples}")
        log_info(
            f"Test samples (Used):  {len(test_ds)}/{total_test_samples} "
            f"({float(args.test_pct):g}%)"
        )
        if len(test_ds) > 0:
            first_x, *_ = test_ds[0]
            input_size = first_x.shape[-1]
        else:
            input_size = 1
        return test_ds, input_size
    elif preprocess_approach == "sv":
        preprocess_dir = getattr(args, "preprocess_dir", None)
        if not preprocess_dir:
            raise RuntimeError("--preprocess_dir required for sv evaluate")
        from preprocessing.sv.dataset import SvDataset
        from preprocessing.sv.config import CFG as SV_CFG
        test_ds = SvDataset(
            preprocess_dir, "test",
            input_len=SV_CFG.INPUT_LEN, pred_horizon=SV_CFG.PRED_HORIZON,
            stride=SV_CFG.STRIDE,
            train_frac=SV_CFG.TRAIN_FRAC, val_frac=SV_CFG.VAL_FRAC,
            feature_set=feature_set_name,
        )
        input_size = test_ds.n_channels
        return test_ds, input_size
    elif preprocess_approach == "cskv":
        preprocess_dir = getattr(args, "preprocess_dir", None)
        if not preprocess_dir:
            raise RuntimeError("--preprocess_dir required for cskv evaluate")
        from preprocessing.cskv.dataset import CskvDataset
        from preprocessing.cskv.config import CFG as CSKV_CFG
        test_ds = CskvDataset(
            preprocess_dir, "test",
            input_len=CSKV_CFG.INPUT_LEN, pred_horizon=CSKV_CFG.PRED_HORIZON,
            stride=CSKV_CFG.STRIDE,
            train_frac=CSKV_CFG.TRAIN_FRAC, val_frac=CSKV_CFG.VAL_FRAC,
        )
        input_size = 1
        return test_ds, input_size
    else:
        raise ValueError(f"Unknown preprocess_approach: {preprocess_approach}")


def evaluate(args):
    accelerator = Accelerator(cpu=args.cpu)
    device = accelerator.device

    seed = getattr(args, "seed", TRAINING.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_info = lambda msg: (
        logging.info(msg) if accelerator.is_local_main_process else None
    )

    log_path = None
    if accelerator.is_local_main_process:
        log_path = setup_logging("test")

    log_info("\n--- Configuration Inputs ---")
    for key, value in vars(args).items():
        log_info(f"{key:<20}: {value}")
    log_info("-" * 30)
    log_info(f"Device: {device} | Distributed Processes: {accelerator.num_processes}")

    if not os.path.exists(args.checkpoint_path):
        if accelerator.is_local_main_process:
            logging.error(f"Checkpoint not found at {args.checkpoint_path}")
        return

    log_info(f"Loading checkpoint: {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    model_type = checkpoint.get("model_type", "lstm")
    preprocess_approach = checkpoint.get("preprocess_approach", "none")
    horizon = ckpt_args.get("pred_horizon", PREPROCESSING.PRED_HORIZON)
    feature_set_name = ckpt_args.get("feature_set", PREPROCESSING.FEATURE_SET)
    target_features = target_features_for_feature_set(feature_set_name)
    feature_names = feature_names_for_feature_set(feature_set_name)
    num_targets = len(target_features)
    target_idxs_in_features = [feature_names.index(f) for f in target_features]

    log_info(f"Model Type:         {model_type}")
    log_info(f"Preprocess Approach:{preprocess_approach}")
    log_info(f"Target Feature(s):  {target_features}")

    log_info("\n--- Loading Test Dataset ---")
    test_ds, input_size = _load_test_dataset(args, ckpt_args, device, log_info, feature_set_name)

    model = _build_model_from_checkpoint(checkpoint, input_size, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device.type != "cpu":
        log_info("Tuning inference batch size to hardware limits...")
        max_batch = find_max_inference_batch_size(model, input_size, args, device)
        safe_batch_size = int(max_batch * 0.9)
        safe_batch_size = 2 ** int(math.log2(max(1, safe_batch_size)))
        log_info(f"Auto-selected per-GPU Inference Batch Size: {safe_batch_size}")
        args.batch_size = safe_batch_size

    system_cores = os.cpu_count() or 1
    gpu_count = torch.cuda.device_count() or 1
    optimal_workers = min(system_cores, 4 * gpu_count)
    log_info(f"Dynamically set num_workers to {optimal_workers}")

    def _worker_init_fn(worker_id):
        wseed = seed + worker_id
        random.seed(wseed)
        np.random.seed(wseed)
        torch.manual_seed(wseed)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=(device.type != "cpu"),
        worker_init_fn=_worker_init_fn,
    )

    model, test_loader = accelerator.prepare(model, test_loader)

    log_info("\n--- Starting Inference ---")

    all_preds = []
    all_trues = []
    all_lasts = []

    model.eval()
    start_time = time.time()
    total_batches = len(test_loader)

    for i, batch in enumerate(test_loader):
        x, y = batch[0], batch[1]

        with torch.no_grad():
            mu = model(x)

        if preprocess_approach == "sv":
            batch_last = batch[2]
            gathered_mu, gathered_y, gathered_last = accelerator.gather_for_metrics((mu, y, batch_last))
        else:
            gathered_mu, gathered_y, gathered_x = accelerator.gather_for_metrics((mu, y, x))

        if accelerator.is_local_main_process:
            if preprocess_approach == "sv":
                y_last = gathered_last.cpu().numpy()
            else:
                y_last = gathered_x[:, -1, :].cpu().numpy()
            all_preds.append(gathered_mu.cpu().numpy())
            all_trues.append(gathered_y.cpu().numpy())
            all_lasts.append(y_last)
            print(f"Batch {i+1}/{total_batches} processed...", end="\r", flush=True)

    if not accelerator.is_local_main_process:
        return

    print(" " * 50, end="\r")
    inference_time = time.time() - start_time

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)
    y_last_all = np.concatenate(all_lasts, axis=0)
    if y_last_all.ndim == 1:
        y_last_all = y_last_all[:, np.newaxis]

    total_samples = y_pred.shape[0]

    log_info("\n=== Inference Summary ===")
    log_info(f"Model: {model_type}")

    for t_idx, t_name in zip(target_idxs_in_features, target_features):
        y_last_t = y_last_all[:, t_idx]
        if num_targets > 1:
            y_pred_t = y_pred[:, :, t_idx]
            y_true_t = y_true[:, :, t_idx]
        else:
            y_pred_t = y_pred
            y_true_t = y_true
        compute_metrics(y_pred_t, y_true_t, y_last_t, horizon, total_samples, log_info, target_name=t_name)

    avg_inference_time_ms = (inference_time / max(1, total_samples)) * 1000.0
    log_info(f"Total Inference Time:  {inference_time:.2f}s")
    log_info(f"Avg Latency per Sample:{avg_inference_time_ms:.4f} ms")
    log_info("-" * 30)
    log_info(f"Log Saved to: {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument(
        "--test_pct",
        type=float,
        default=TRAINING.TEST_PCT,
        help="Percentage of test samples for evaluation; 25 means 25%, not 0.25 (100 uses all; <=0 uses all).",
    )
    p.add_argument("--preprocess_dir", default=None, help="Decomposition output dir (for sv/cskv)")
    p.add_argument("--seed", type=int, default=TRAINING.SEED, help="Random seed for reproducibility")

    try:
        evaluate(p.parse_args())
    except Exception:
        logging.error("Fatal Error during evaluation", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
