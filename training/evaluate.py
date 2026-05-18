import os
import sys
import argparse
import logging
import time
import numpy as np
from datetime import datetime
from scipy.stats import pearsonr

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, TRAINING, PREPROCESSING
from core.dataset import ShardedWindowsDataset
from core.models import RNNForecaster


def setup_logging(mode="test"):
    os.makedirs(PATHS.LOGS_DIR, exist_ok=True)

    try:
        tehran_tz = ZoneInfo("Asia/Tehran")
    except Exception:
        print("[WARN] Could not find 'Asia/Tehran' timezone. Using system time.")
        tehran_tz = None

    now = datetime.now(tehran_tz)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    log_filename = f"{mode}_{timestamp}.log"
    log_path = os.path.join(PATHS.LOGS_DIR, log_filename)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    logging.info(f"=== {mode.upper()} SESSION STARTED ===")
    logging.info(f"Log file: {log_path}")
    logging.info(f"Timestamp (Tehran): {now}")
    return log_path


@torch.no_grad()
def forward_predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(x)


def evaluate(args):
    log_path = setup_logging("test")

    logging.info("\n--- Configuration Inputs ---")
    for key, value in vars(args).items():
        logging.info(f"{key:<20}: {value}")
    logging.info("-" * 30)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    logging.info(f"Device: {device}")

    if not os.path.exists(args.checkpoint_path):
        logging.error(f"Checkpoint not found at {args.checkpoint_path}")
        return

    logging.info(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    hidden_size = ckpt_args.get("hidden_size", TRAINING.HIDDEN_SIZE)
    num_layers = ckpt_args.get("num_layers", TRAINING.NUM_LAYERS)
    dropout = ckpt_args.get("dropout", TRAINING.DROPOUT)
    horizon = ckpt_args.get("pred_horizon", PREPROCESSING.PRED_HORIZON)
    rnn_type = ckpt_args.get("rnn_type", "lstm")
    input_len = ckpt_args.get("input_len", PREPROCESSING.INPUT_LEN)
    feature_set = ckpt_args.get("feature_set", PREPROCESSING.FEATURE_SET)
    bidirectional = ckpt_args.get("bidirectional", TRAINING.BIDIRECTIONAL)

    logging.info(
        f"RNN Architecture:   {rnn_type} (Layers: {num_layers}, Hidden: {hidden_size})"
    )

    logging.info("\n--- Loading Test Dataset ---")

    test_ds = ShardedWindowsDataset(
        args.windows_dir, "test", input_len, horizon, use_weights=False
    )
    logging.info(f"Test samples: {len(test_ds)}")

    if len(test_ds) > 0:
        first_x, *_ = test_ds[0]
        input_size = first_x.shape[-1]
    else:
        input_size = checkpoint.get("input_size", 1)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    logging.info("\n--- Starting Inference ---")

    all_preds = []
    all_trues = []
    all_lasts = []

    start_time = time.time()
    total_batches = len(test_loader)

    for i, batch in enumerate(test_loader):
        x, y = batch[0].to(device), batch[1].to(device)

        mu = forward_predict(model, x)

        y_last = x[:, -1, 0].cpu().numpy()

        all_preds.append(mu.cpu().numpy())
        all_trues.append(y.cpu().numpy())
        all_lasts.append(y_last)

        print(f"Batch {i+1}/{total_batches} processed...", end="\r", flush=True)

    print(" " * 50, end="\r")
    inference_time = time.time() - start_time

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)
    y_last = np.concatenate(all_lasts, axis=0)

    total_samples = y_pred.shape[0]
    if total_samples == 0:
        logging.warning("No samples found in test set.")
        return

    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))

    y_true_s1 = y_true[:, 0]
    y_pred_s1 = y_pred[:, 0]

    mse_naive = np.mean((y_true_s1 - y_last) ** 2)
    skill_score = 1.0 - (mse / mse_naive)

    actual_dir = np.sign(y_true_s1 - y_last)
    pred_dir = np.sign(y_pred_s1 - y_last)
    mda = np.mean(actual_dir == pred_dir)

    corr_0, _ = pearsonr(y_true_s1, y_pred_s1)
    corr_1, _ = pearsonr(y_last, y_pred_s1)

    is_shadowing = (skill_score < 0.05) or (corr_1 > corr_0) or (mda < 0.55)

    avg_inference_time_ms = (inference_time / total_samples) * 1000.0

    logging.info("\n=== Performance Metrics ===")
    logging.info(f"Model: {rnn_type}")
    logging.info("-" * 30)
    logging.info(f"MSE:                   {mse:.4f}")
    logging.info(f"MAE:                   {mae:.4f}")
    logging.info("-" * 30)
    logging.info(">>> SHADOWING DIAGNOSTICS <<<")
    logging.info(f"Skill Score (vs Naive): {skill_score:.4f}  (Ideal: > 0.1)")
    logging.info(f"Directional Acc (MDA):  {mda:.2%} (Ideal: > 60%)")
    logging.info(f"Correlation (Lag 0):    {corr_0:.4f}")
    logging.info(f"Correlation (Lag -1):   {corr_1:.4f}")

    if is_shadowing:
        logging.warning(
            "!! WARNING: Model shows signs of SHADOWING (overfitting to last step) !!"
        )
    else:
        logging.info("PASSED: Model appears to have learned temporal dynamics.")

    logging.info("-" * 30)
    logging.info(f"Total Inference Time:  {inference_time:.2f}s")
    logging.info(f"Avg Latency per Sample:{avg_inference_time_ms:.4f} ms")
    logging.info("-" * 30)
    logging.info(f"Log Saved to: {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=TRAINING.NUM_WORKERS)
    p.add_argument("--cpu", action="store_true", default=False)

    try:
        evaluate(p.parse_args())
    except Exception:
        logging.error("Fatal Error during evaluation", exc_info=True)
        sys.exit(1)
