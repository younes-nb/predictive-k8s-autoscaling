import os
import sys
import argparse
import logging
import time
from datetime import datetime

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
from common.dataset import ShardedWindowsDataset
from common.models import RNNForecaster


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
    bidirectional = ckpt_args.get("bidirectional", args.bidirectional)
    residual_mode = ckpt_args.get("residual", args.residual)

    logging.info(f"Model trained on feature_set: {feature_set}")
    logging.info(f"RNN Type: {rnn_type}")

    logging.info("\n--- Loading Test Dataset ---")
    test_ds = ShardedWindowsDataset(args.windows_dir, "test", input_len, horizon)
    logging.info(f"Test samples: {len(test_ds)}")

    if len(test_ds) > 0:
        first_x, *_ = test_ds[0]
        input_size = first_x.shape[-1] if first_x.ndim > 1 else 1
        logging.info(f"Inferred input_size={input_size} from dataset.")
    else:
        input_size = checkpoint.get("input_size", ckpt_args.get("input_size", 1))
        logging.warning(f"Test dataset empty. Fallback input_size={input_size}.")

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
        residual=residual_mode,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logging.info("\n--- Starting Deterministic Inference ---")

    mse_sum = 0.0
    mae_sum = 0.0
    total_samples = 0

    start_time = time.time()
    total_batches = len(test_loader)

    for i, (x, y, *_) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

        mu = forward_predict(model, x)

        diff = mu - y
        mse_sum += (diff**2).sum().item()
        mae_sum += diff.abs().sum().item()

        total_samples += x.size(0)

        print(f"Batch {i+1}/{total_batches} processed...", end="\r", flush=True)

    print(" " * 50, end="\r")

    inference_time = time.time() - start_time

    if total_samples == 0:
        logging.warning("No samples found in test set.")
        return

    mse = mse_sum / (total_samples * horizon)
    mae = mae_sum / (total_samples * horizon)
    avg_inference_time_ms = (inference_time / total_samples) * 1000.0

    logging.info("\n=== Test Results (Error & Latency) ===")
    logging.info(f"feature_set: {feature_set}")
    logging.info(f"RNN Type:    {rnn_type}")
    logging.info("-" * 30)
    logging.info(f"MSE:                   {mse:.6f}")
    logging.info(f"MAE:                   {mae:.6f}")
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
    p.add_argument(
        "--bidirectional", action="store_true", default=TRAINING.BIDIRECTIONAL
    )
    p.add_argument("--residual", action="store_true", default=TRAINING.RESIDUAL)
    try:
        evaluate(p.parse_args())
    except Exception:
        logging.error("Fatal Error during evaluation", exc_info=True)
        sys.exit(1)
