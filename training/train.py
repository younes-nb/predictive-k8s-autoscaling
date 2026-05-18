import os
import sys
import argparse
import logging
import math
import random
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, PREPROCESSING, TRAINING, DEFAULT_CHECKPOINT_PATH
from core.dataset import ShardedWindowsDataset

from core.models import RNNForecaster


def setup_logging(mode="train", log_path=None):
    os.makedirs(PATHS.LOGS_DIR, exist_ok=True)

    try:
        tehran_tz = ZoneInfo("Asia/Tehran")
    except Exception:
        print("[WARN] Could not find 'Asia/Tehran' timezone. Using system time.")
        tehran_tz = None

    now = datetime.now(tehran_tz)
    if log_path is None:
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        log_filename = f"{mode}_{timestamp}.log"
        log_path = os.path.join(PATHS.LOGS_DIR, log_filename)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    file_handler = logging.FileHandler(log_path, mode="a")
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


def weighted_mse(preds, target, w=None, under_penalty=5.0):
    diff = preds - target
    sq_err = diff**2
    under_mask = (preds < target).float()
    asym_weight = 1.0 + (under_mask * (under_penalty - 1.0))
    value_loss = (sq_err * asym_weight).mean(dim=1)

    if w is None:
        return value_loss.mean()

    w = w.clamp(min=0.1, max=15.0)
    return (w * value_loss).sum() / w.sum().clamp_min(1e-6)


HIDDEN_SIZE_OPTIONS = [64, 128, 256]
NUM_LAYERS_OPTIONS = [1, 2, 3, 4]
DROPOUT_RANGE = (0.1, 0.5)
LR_RANGE = (5e-4, 5e-3)
HYPERPARAM_SAMPLE_ATTEMPTS = 5000
HYPERPARAM_CHECK_INTERVAL = 50
LOSS_CHANGE_THRESHOLD = 1e-4


def hyperparam_key(hyperparams):
    return (
        int(hyperparams["hidden_size"]),
        int(hyperparams["num_layers"]),
        round(float(hyperparams["dropout"]), 4),
        round(float(hyperparams["lr"]), 8),
    )


def sample_hyperparams(rng, used_keys):
    log_min = math.log10(LR_RANGE[0])
    log_max = math.log10(LR_RANGE[1])
    for _ in range(HYPERPARAM_SAMPLE_ATTEMPTS):
        candidate = {
            "hidden_size": rng.choice(HIDDEN_SIZE_OPTIONS),
            "num_layers": rng.choice(NUM_LAYERS_OPTIONS),
            "dropout": round(rng.uniform(*DROPOUT_RANGE), 4),
            "lr": round(10 ** rng.uniform(log_min, log_max), 8),
        }
        key = hyperparam_key(candidate)
        if key not in used_keys:
            used_keys.add(key)
            return candidate
    return None


def load_resume_state(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        sys.stderr.write(f"[WARN] Failed to load resume state: {exc}\n")
        return None


def save_resume_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def apply_hyperparams(args, hyperparams):
    args.hidden_size = hyperparams["hidden_size"]
    args.num_layers = hyperparams["num_layers"]
    args.dropout = hyperparams["dropout"]
    args.lr = hyperparams["lr"]


def train(args):
    resume_state = load_resume_state(PATHS.RESUME_STATE_FILE)
    if resume_state and "args" in resume_state:
        args = argparse.Namespace(**resume_state["args"])

    rng = random.Random(args.seed)
    log_path = setup_logging(
        "train", log_path=resume_state.get("log_path") if resume_state else None
    )

    used_keys = set(resume_state.get("used_hyperparams", [])) if resume_state else set()
    current_hyperparams = resume_state.get("hyperparams") if resume_state else None
    if current_hyperparams is not None:
        used_keys.add(hyperparam_key(current_hyperparams))

    if current_hyperparams is None:
        current_hyperparams = sample_hyperparams(rng, used_keys)

    if current_hyperparams is None:
        raise RuntimeError("Unable to select a unique hyperparameter set.")

    apply_hyperparams(args, current_hyperparams)

    start_epoch = int(resume_state.get("epoch", 0)) + 1 if resume_state else 1
    best_score = float(resume_state.get("best_score", float("inf"))) if resume_state else float("inf")
    window_start_epoch = resume_state.get("window_start_epoch") if resume_state else None
    window_start_loss = resume_state.get("window_start_loss") if resume_state else None

    if resume_state:
        logging.info(
            f"Resuming training from epoch {start_epoch} using {PATHS.RESUME_STATE_FILE}"
        )

    logging.info("\n--- Configuration Inputs ---")
    for key, value in vars(args).items():
        logging.info(f"{key:<20}: {value}")
    logging.info("-" * 30)

    if start_epoch > args.epochs:
        logging.info("Resume state indicates training has already completed.")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    logging.info(f"Device: {device}")
    torch.manual_seed(args.seed)

    logging.info("\n--- Loading Datasets ---")

    train_ds = ShardedWindowsDataset(
        args.windows_dir, "train", args.input_len, args.pred_horizon, args.use_weights
    )

    val_ds = ShardedWindowsDataset(
        args.windows_dir, "val", args.input_len, args.pred_horizon, args.use_weights
    )

    logging.info(f"Train samples: {len(train_ds)}")
    logging.info(f"Val samples:   {len(val_ds)}")

    if len(train_ds) > 0:
        first_x, _, *_ = train_ds[0]
        input_size = first_x.shape[-1]
        logging.info(f"Inferred Input Size: {input_size}")
    else:
        raise RuntimeError("Train dataset is empty.")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=args.pred_horizon,
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if resume_state:
        model_state = resume_state.get("model_state_dict")
        if model_state:
            model.load_state_dict(model_state)
        optimizer_state = resume_state.get("optimizer_state_dict")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

    logging.info("\n--- Starting Training Loop ---")

    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = datetime.now()
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            if args.use_weights:
                x, y, w, _ = batch
                w = w.to(device)
            else:
                x, y, _ = batch
                w = None

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = weighted_mse(preds, y, w, under_penalty=args.under_penalty)

            loss.backward()

            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            train_loss_accum += loss.item() * x.size(0)

        avg_train_loss = train_loss_accum / len(train_ds)

        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if args.use_weights:
                    x, y, w, _ = batch
                    w = w.to(device)
                else:
                    x, y, _ = batch
                    w = None

                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = weighted_mse(preds, y, w, under_penalty=args.under_penalty)

                val_loss_accum += loss.item() * x.size(0)

        avg_val_loss = val_loss_accum / len(val_ds) if len(val_ds) > 0 else 0.0
        epoch_duration = (datetime.now() - start_time).total_seconds()

        log_msg = (
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        if avg_val_loss < best_score:
            best_score = avg_val_loss
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "input_size": input_size,
                    "best_val_loss": best_score,
                },
                args.checkpoint_path,
            )
            log_msg += " [Checkpoint Saved]"

        logging.info(log_msg)
        history.append(
            {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        )

        if window_start_loss is None and epoch % HYPERPARAM_CHECK_INTERVAL == 1:
            window_start_epoch = epoch
            window_start_loss = avg_train_loss

        if (
            epoch % HYPERPARAM_CHECK_INTERVAL == 0
            and window_start_loss is not None
            and epoch < args.epochs
        ):
            delta = abs(avg_train_loss - window_start_loss)
            if delta < LOSS_CHANGE_THRESHOLD:
                new_hyperparams = sample_hyperparams(rng, used_keys)
                if new_hyperparams is not None:
                    logging.info(
                        "No train loss change between epochs "
                        f"{window_start_epoch} and {epoch} (Δ={delta:.4f}). "
                        "Switching hyperparameters."
                    )
                    apply_hyperparams(args, new_hyperparams)
                    model = RNNForecaster(
                        input_size=input_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        horizon=args.pred_horizon,
                        rnn_type=args.rnn_type,
                        bidirectional=args.bidirectional,
                    ).to(device)
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                    )
                    current_hyperparams = new_hyperparams
                    window_start_epoch = epoch + 1
                    window_start_loss = None
                else:
                    logging.info(
                        "No unused hyperparameter combinations remain; keeping current settings."
                    )

        resume_payload = {
            "epoch": epoch,
            "args": vars(args),
            "hyperparams": current_hyperparams,
            "used_hyperparams": list(used_keys),
            "best_score": best_score,
            "window_start_epoch": window_start_epoch,
            "window_start_loss": window_start_loss,
            "log_path": log_path,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        save_resume_state(PATHS.RESUME_STATE_FILE, resume_payload)

    logging.info("\n--- Training Completed ---")
    logging.info(f"Best Validation Loss: {best_score:.4f}")
    logging.info(f"Final Model Saved to: {args.checkpoint_path}")
    logging.info(f"Full Log Saved to:    {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--use_weights", action="store_true")
    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--pred_horizon", type=int, default=PREPROCESSING.PRED_HORIZON)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=TRAINING.EPOCHS)
    p.add_argument("--grad_clip", type=float, default=TRAINING.GRAD_CLIP)
    p.add_argument("--weight_decay", type=float, default=TRAINING.WEIGHT_DECAY)
    p.add_argument("--under_penalty", type=float, default=TRAINING.UNDER_PENALTY)
    p.add_argument("--num_workers", type=int, default=TRAINING.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=TRAINING.SEED)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--rnn_type", default="lstm")
    p.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET)
    p.add_argument("--bidirectional", action="store_true", default=TRAINING.BIDIRECTIONAL)
    
    try:
        train(p.parse_args())
    except Exception as e:
        logging.error("Fatal Error during training", exc_info=True)
        sys.exit(1)
