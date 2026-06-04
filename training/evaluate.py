import os
import sys
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
from core.dataset import ShardedWindowsDataset
from core.models import RNNForecaster

from training.metrics import compute_metrics, find_max_inference_batch_size


def evaluate(args):
    accelerator = Accelerator(cpu=args.cpu)
    device = accelerator.device

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

    hidden_size = ckpt_args.get("hidden_size", TRAINING.HIDDEN_SIZE)
    num_layers = ckpt_args.get("num_layers", TRAINING.NUM_LAYERS)
    dropout = ckpt_args.get("dropout", TRAINING.DROPOUT)
    horizon = ckpt_args.get("pred_horizon", PREPROCESSING.PRED_HORIZON)
    rnn_type = ckpt_args.get("rnn_type", "lstm")
    input_len = ckpt_args.get("input_len", PREPROCESSING.INPUT_LEN)
    bidirectional = ckpt_args.get("bidirectional", TRAINING.BIDIRECTIONAL)
    probabilistic = ckpt_args.get(
        "probabilistic", TRAINING.PROBABILISTIC_TRAINING
    )
    quantiles = TRAINING.QUANTILES if probabilistic else None

    log_info(
        f"RNN Architecture:   {rnn_type} (Layers: {num_layers}, Hidden: {hidden_size})"
    )

    log_info("\n--- Loading Test Dataset ---")

    test_ds = ShardedWindowsDataset(
        args.windows_dir, "test", input_len, horizon, use_weights=False
    )
    log_info(f"Test samples (Total): {len(test_ds)}")

    if len(test_ds) > 0:
        first_x, *_ = test_ds[0]
        input_size = first_x.shape[-1]
    else:
        input_size = checkpoint.get("input_size", 1)

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        quantiles=quantiles,
    ).to(device)

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

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=(device.type != "cpu"),
    )

    model, test_loader = accelerator.prepare(model, test_loader)

    log_info("\n--- Starting Inference ---")

    all_preds = []
    all_trues = []
    all_lasts = []

    model.eval()
    start_time = time.time()
    total_batches = len(test_loader)

    if probabilistic and quantiles:
        median_idx = min(
            range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5)
        )
    else:
        median_idx = None

    for i, batch in enumerate(test_loader):
        x, y = batch[0], batch[1]

        with torch.no_grad():
            mu = model(x)
            if median_idx is not None:
                mu = mu[..., median_idx]

        gathered_mu, gathered_y, gathered_x = accelerator.gather_for_metrics((mu, y, x))

        if accelerator.is_local_main_process:
            y_last = gathered_x[:, -1, 0].cpu().numpy()

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
    y_last = np.concatenate(all_lasts, axis=0)

    total_samples = y_pred.shape[0]

    log_info("\n=== Inference Summary ===")
    log_info(f"Model: {rnn_type}")
    compute_metrics(y_pred, y_true, y_last, horizon, total_samples, log_info)

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

    try:
        evaluate(p.parse_args())
    except Exception:
        logging.error("Fatal Error during evaluation", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
