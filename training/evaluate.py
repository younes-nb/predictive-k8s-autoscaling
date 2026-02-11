import os
import sys
import argparse
import logging
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

from config.defaults import PATHS, TRAINING
from common.dataset import ShardedWindowsDataset
from common.models import RNNForecaster
from common.uncertainty import mc_dropout_predict, compute_adaptive_thresholds


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
    """Deterministic single forward pass (no MC Dropout)."""
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
    horizon = ckpt_args.get("pred_horizon", 5)
    rnn_type = ckpt_args.get("rnn_type", "lstm")
    input_len = ckpt_args.get("input_len", 60)
    feature_set = ckpt_args.get("feature_set", "unknown")

    logging.info(f"Model trained on feature_set: {feature_set}")
    logging.info(f"RNN Type: {rnn_type}")

    if args.static_threshold:
        use_adaptive = False
    elif args.adaptive_threshold:
        use_adaptive = True
    else:
        use_adaptive = True

    logging.info(
        f"Threshold mode: {'adaptive (mc-dropout)' if use_adaptive else 'static (base_threshold)'}"
    )
    logging.info(f"base_threshold: {args.base_threshold}")
    if use_adaptive:
        logging.info(
            f"adaptive params: repeats={args.inference_repeats}, k={args.k}, "
            f"theta_min={args.theta_min}, theta_max={args.theta_max}"
        )
        if args.global_threshold:
            logging.info(
                ">> Global Threshold Mode: ENABLED (Averaging sigma per batch)"
            )

    logging.info("\n--- Loading Test Dataset ---")
    test_ds = ShardedWindowsDataset(args.windows_dir, "test", input_len, horizon)
    logging.info(f"Test samples: {len(test_ds)}")

    if len(test_ds) > 0:
        first_x, _ = test_ds[0]
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
        bidirectional=args.bidirectional
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logging.info("\n--- Starting Inference ---")

    mse_sum = 0.0
    mae_sum = 0.0
    tp = fp = tn = fn = 0
    total_samples = 0
    sigma_sum = 0.0

    start_time = datetime.now()

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

        if use_adaptive:
            mu, sigma = mc_dropout_predict(model, x, repeats=args.inference_repeats)

            if args.global_threshold:
                sigma_global = sigma.mean()
                sigma = torch.full_like(sigma, sigma_global)

            sigma_sum += sigma.mean().item()

            theta_base = torch.full(mu.shape, args.base_threshold, device=device)
            thr = compute_adaptive_thresholds(
                theta_base,
                sigma,
                k=args.k,
                theta_min=args.theta_min,
                theta_max=args.theta_max,
            )
        else:
            mu = forward_predict(model, x)
            thr = torch.full(mu.shape, args.base_threshold, device=device)

        diff = mu - y
        mse_sum += (diff**2).sum().item()
        mae_sum += diff.abs().sum().item()

        y_true_cls = y.max(dim=1).values >= thr.max(dim=1).values
        y_pred_cls = mu.max(dim=1).values >= thr.max(dim=1).values

        tp += (y_pred_cls & y_true_cls).sum().item()
        fp += (y_pred_cls & ~y_true_cls).sum().item()
        tn += (~y_pred_cls & ~y_true_cls).sum().item()
        fn += (~y_pred_cls & y_true_cls).sum().item()

        total_samples += x.size(0)

        if i % 100 == 0:
            logging.info(f"Batch {i}/{len(test_loader)} processed...")

    inference_time = (datetime.now() - start_time).total_seconds()

    if total_samples == 0:
        logging.warning("No samples found in test set.")
        return

    mse = mse_sum / (total_samples * horizon)
    mae = mae_sum / (total_samples * horizon)
    avg_sigma = sigma_sum / len(test_loader) if use_adaptive else 0.0

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    avg_inference_time_ms = (inference_time / total_samples) * 1000.0

    logging.info("\n=== Test Results ===")
    logging.info(f"feature_set (from ckpt): {feature_set}")
    logging.info(f"RNN Type: {rnn_type}")
    logging.info(
        f"Threshold mode: {'adaptive (mc-dropout)' if use_adaptive else 'static (base_threshold)'}"
    )
    logging.info(f"base_threshold: {args.base_threshold}")
    if use_adaptive:
        logging.info(
            f"adaptive params: repeats={args.inference_repeats}, k={args.k}, "
            f"theta_min={args.theta_min}, theta_max={args.theta_max}"
        )
        if args.global_threshold:
            logging.info(">> Global Threshold Mode: ENABLED")
        logging.info(f"Avg Sigma (Uncertainty): {avg_sigma:.6f} <--- DIAGNOSTIC")

    logging.info(f"MSE:       {mse:.6f}")
    logging.info(f"MAE:       {mae:.6f}")
    logging.info("-" * 20)
    logging.info("Classification:")
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1 Score:  {f1:.4f}")
    logging.info("-" * 20)
    logging.info("Confusion Matrix:")
    logging.info(f"TP: {int(tp)}, FP: {int(fp)}")
    logging.info(f"FN: {int(fn)}, TN: {int(tn)}")
    logging.info("-" * 20)
    logging.info(f"Total Inference Time: {inference_time:.2f}s")
    logging.info(f"Latency per Sample:   {avg_inference_time_ms:.2f}ms")
    logging.info(f"Log Saved to: {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=TRAINING.NUM_WORKERS)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--adaptive_threshold", action="store_true")
    p.add_argument("--static_threshold", action="store_true")
    p.add_argument("--inference_repeats", type=int, default=TRAINING.INFERENCE_REPEATS)
    p.add_argument("--base_threshold", type=float, default=TRAINING.THETA_BASE)
    p.add_argument("--k", type=float, default=TRAINING.K_UNCERTAINTY)
    p.add_argument("--theta_min", type=float, default=TRAINING.THETA_MIN)
    p.add_argument("--theta_max", type=float, default=TRAINING.THETA_MAX)
    p.add_argument(
        "--global_threshold", action="store_true", default=TRAINING.GLOBAL_THRESHOLD
    )
    p.add_argument("--bidirectional", action="store_true", default=TRAINING.BIDIRECTIONAL)

    try:
        evaluate(p.parse_args())
    except Exception:
        logging.error("Fatal Error during evaluation", exc_info=True)
        sys.exit(1)
