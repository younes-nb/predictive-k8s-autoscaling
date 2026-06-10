import os
import sys
import argparse
import logging
import math as _math
import random
from datetime import datetime, timedelta

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator, InitProcessGroupKwargs

from config.defaults import DEFAULT_CHECKPOINT_PATH
from shared.config_paths import PATHS
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.config_training_defaults import TRAINING
from shared.logging_utils import setup_logging
from core.dataset import ShardedWindowsDataset
from core.models import RNNForecaster

from training.loss import PinballLoss, weighted_mse
from training.train_helpers import (
    find_max_batch_size,
    hyperparam_key,
    sample_hyperparams,
    apply_hyperparams,
    load_resume_state,
    save_resume_state,
)
from training.sfoa_search import run_sfoa_search


def train(args):
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=14400))
    accelerator = Accelerator(cpu=args.cpu, kwargs_handlers=[timeout_kwargs])
    device = accelerator.device

    log_info = lambda msg: (
        logging.info(msg) if accelerator.is_local_main_process else None
    )

    if not hasattr(args, "resume_training"):
        args.resume_training = False

    resume_state = (
        load_resume_state(PATHS.RESUME_STATE_FILE) if args.resume_training else None
    )
    if resume_state and "args" in resume_state:
        args = argparse.Namespace(**resume_state["args"])
        args.resume_training = True
    if not hasattr(args, "probabilistic"):
        args.probabilistic = TRAINING.PROBABILISTIC_TRAINING
    hyperparam_optimizer = getattr(
        args, "hyperparam_optimizer", TRAINING.HYPERPARAM_OPTIMIZER
    )

    sfoa_done = False
    if resume_state:
        sfoa_done = resume_state.get("sfoa_done") is True
        if "hyperparam_optimizer" in resume_state.get("args", {}):
            args.hyperparam_optimizer = resume_state["args"]["hyperparam_optimizer"]

    rng = random.Random(args.seed)

    log_path = None
    if accelerator.is_local_main_process:
        log_path = setup_logging(
            "train", log_path=resume_state.get("log_path") if resume_state else None
        )

    used_keys = (
        {tuple(k) for k in resume_state.get("used_hyperparams", [])}
        if resume_state
        else set()
    )
    start_epoch = resume_state.get("epoch", 0) + 1 if resume_state else 1
    best_score = (
        resume_state.get("best_score", float("inf")) if resume_state else float("inf")
    )
    last_train_loss = resume_state.get("last_train_loss") if resume_state else None
    no_change_streak = (
        resume_state.get("no_change_streak", 0)
        if resume_state and last_train_loss is not None
        else 0
    )

    if resume_state:
        log_info("=== RESUMED TRAINING SESSION ===")
        log_info(
            f"Resuming training from epoch {start_epoch} using {PATHS.RESUME_STATE_FILE}"
        )

    if start_epoch > args.epochs:
        log_info("\n--- Configuration Inputs ---")
        for key, value in vars(args).items():
            log_info(f"{key:<20}: {value}")
        log_info("-" * 30)
        log_info("Resume state indicates training has already completed.")
        return

    log_info("\n--- Loading Datasets ---")

    train_ds = ShardedWindowsDataset(
        args.windows_dir, "train", args.input_len, args.pred_horizon, args.use_weights
    )

    val_ds = ShardedWindowsDataset(
        args.windows_dir, "val", args.input_len, args.pred_horizon, args.use_weights
    )

    log_info(f"Train samples (Total): {len(train_ds)}")
    log_info(f"Val samples (Total):   {len(val_ds)}")

    if len(train_ds) > 0:
        first_x, _, *_ = train_ds[0]
        input_size = first_x.shape[-1]
        log_info(f"Inferred Input Size: {input_size}")
    else:
        raise RuntimeError("Train dataset is empty.")

    current_hyperparams = resume_state.get("hyperparams") if resume_state else None
    if current_hyperparams is not None:
        used_keys.add(hyperparam_key(current_hyperparams))

    if current_hyperparams is None and not sfoa_done and hyperparam_optimizer == "sfoa":
        if accelerator.num_processes > 1:
            _sync = accelerator.wait_for_everyone
        else:
            _sync = None

        log_info("Running SFOA hyperparameter search before main training...")
        if _sync is not None:
            _sync()
        current_hyperparams = run_sfoa_search(
            args, train_ds, val_ds, rank_seed=args.seed, accelerator=accelerator
        )
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()
        if current_hyperparams is None:
            logging.warning(
                "[SFOA] Search did not return hyperparameters; falling back to random sampling."
            )
            current_hyperparams = sample_hyperparams(rng, used_keys)

    elif current_hyperparams is None:
        if sfoa_done:
            log_info(
                "[SFOA] Previously completed but best hyperparams missing from resume state. "
                "Falling back to random sampling."
            )
        current_hyperparams = sample_hyperparams(rng, used_keys)

    if current_hyperparams is not None:
        used_keys.add(hyperparam_key(current_hyperparams))

    if current_hyperparams is None:
        raise RuntimeError(
            "Unable to select a unique hyperparameter set after "
            f"{TRAINING.HYPERPARAM_SAMPLE_ATTEMPTS} attempts."
        )

    apply_hyperparams(args, current_hyperparams)

    log_info("\n--- Configuration Inputs ---")
    for key, value in vars(args).items():
        log_info(f"{key:<20}: {value}")
    log_info("-" * 30)

    log_info(f"Device: {device} | Distributed Processes: {accelerator.num_processes}")
    torch.manual_seed(args.seed)

    quantiles = TRAINING.QUANTILES
    pinball_loss = PinballLoss(quantiles).to(device) if args.probabilistic else None

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=args.pred_horizon,
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional,
        quantiles=quantiles if args.probabilistic else None,
    ).to(device)

    if device.type != "cpu":
        log_info("Tuning per-GPU batch size to hardware limits...")
        max_batch = find_max_batch_size(model, input_size, args, device, pinball_loss)

        safe_batch_size = int(max_batch * 0.8)
        safe_batch_size = 2 ** int(_math.log2(max(1, safe_batch_size)))

        log_info(
            f"Auto-selected per-GPU Batch Size: {safe_batch_size} (Global Batch Size: {safe_batch_size * accelerator.num_processes})"
        )
        args.batch_size = safe_batch_size

    system_cores = os.cpu_count() or 1
    gpu_count = torch.cuda.device_count() or 1
    optimal_workers = min(system_cores, 4 * gpu_count)
    log_info(
        f"Dynamically set num_workers to {optimal_workers} (Cores: {system_cores}, GPUs: {gpu_count})"
    )

    pin_memory = device.type != "cpu"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=pin_memory,
    )

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

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    log_info("\n--- Starting Training Loop ---")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = datetime.now()
        model.train()
        train_loss_accum = 0.0
        train_samples_seen = 0

        for batch in train_loader:
            if args.use_weights:
                x, y, w, _ = batch
            else:
                x, y, _ = batch
                w = None

            optimizer.zero_grad()
            preds = model(x)

            if args.probabilistic:
                loss = pinball_loss(preds, y, w)
            else:
                loss = weighted_mse(preds, y, w, under_penalty=args.under_penalty)

            accelerator.backward(loss)

            if args.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            train_loss_accum += loss.item() * x.size(0)
            train_samples_seen += x.size(0)

        local_avg_train = train_loss_accum / max(1, train_samples_seen)
        sync_train_tensor = torch.tensor(local_avg_train, device=device)
        avg_train_loss = accelerator.reduce(sync_train_tensor, reduction="mean").item()

        model.eval()
        val_loss_accum = 0.0
        val_samples_seen = 0

        with torch.no_grad():
            for batch in val_loader:
                if args.use_weights:
                    x, y, w, _ = batch
                else:
                    x, y, _ = batch
                    w = None

                preds = model(x)

                if args.probabilistic:
                    loss = pinball_loss(preds, y, w)
                else:
                    loss = weighted_mse(preds, y, w, under_penalty=args.under_penalty)

                val_loss_accum += loss.item() * x.size(0)
                val_samples_seen += x.size(0)

        local_avg_val = val_loss_accum / max(1, val_samples_seen)
        sync_val_tensor = torch.tensor(local_avg_val, device=device)
        avg_val_loss = accelerator.reduce(sync_val_tensor, reduction="mean").item()

        epoch_duration = (datetime.now() - start_time).total_seconds()

        log_msg = (
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        if avg_val_loss < best_score:
            best_score = avg_val_loss
            if accelerator.is_local_main_process:
                os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "model_state_dict": unwrapped_model.state_dict(),
                        "args": vars(args),
                        "input_size": input_size,
                        "best_val_loss": best_score,
                    },
                    args.checkpoint_path,
                )
            log_msg += " [Checkpoint Saved]"

        log_info(log_msg)

        delta = None
        if last_train_loss is not None:
            delta = abs(avg_train_loss - last_train_loss)
            if delta < TRAINING.LOSS_CHANGE_THRESHOLD:
                no_change_streak += 1
            else:
                no_change_streak = 0

        if hyperparam_optimizer != "sfoa":
            if (
                no_change_streak >= TRAINING.HYPERPARAM_CHECK_INTERVAL
                and epoch < args.epochs
            ):
                new_hyperparams = sample_hyperparams(rng, used_keys)
                if new_hyperparams is not None:
                    delta_display = f"{delta:.4f}" if delta is not None else "N/A"
                    log_info(
                        "Train loss change below threshold for "
                        f"{no_change_streak} consecutive epochs (Δ={delta_display}). "
                        "Switching hyperparameters."
                    )
                    log_info(f"New hyperparameters: {new_hyperparams}")

                    apply_hyperparams(args, new_hyperparams)

                    new_model = RNNForecaster(
                        input_size=input_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        horizon=args.pred_horizon,
                        rnn_type=args.rnn_type,
                        bidirectional=args.bidirectional,
                        quantiles=quantiles if args.probabilistic else None,
                    ).to(device)

                    new_optimizer = torch.optim.Adam(
                        new_model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                    )

                    model, optimizer = accelerator.prepare(new_model, new_optimizer)

                    current_hyperparams = new_hyperparams
                    last_train_loss = None
                else:
                    log_info(
                        "No unused hyperparameter combinations remain; keeping current settings."
                    )
                    last_train_loss = avg_train_loss
                no_change_streak = 0
            else:
                last_train_loss = avg_train_loss
        else:
            last_train_loss = avg_train_loss
            delta_display = f"{delta:.4f}" if delta is not None else "N/A"
            if no_change_streak > 0 and (delta is None or delta >= TRAINING.LOSS_CHANGE_THRESHOLD):
                log_info(
                    f"Train loss change above threshold for epoch {epoch} (Δ={delta_display}). "
                    f"No-change streak reset ({no_change_streak}/{TRAINING.NO_CHANGE_EPOCHS_LIMIT})."
                )
            elif no_change_streak >= TRAINING.NO_CHANGE_EPOCHS_LIMIT:
                log_info(
                    "\n=== Early Stopping ===\n"
                    f"Train loss has not changed by at least {TRAINING.LOSS_CHANGE_THRESHOLD} "
                    f"for {no_change_streak} consecutive epochs."
                )
                break

        if accelerator.is_local_main_process:
            resume_payload = {
                "epoch": epoch,
                "args": vars(args),
                "hyperparams": current_hyperparams,
                "sfoa_hyperparams": (
                    current_hyperparams if hyperparam_optimizer == "sfoa" else None
                ),
                "sfoa_done": sfoa_done,
                "used_hyperparams": list(used_keys),
                "best_score": best_score,
                "last_train_loss": last_train_loss,
                "no_change_streak": no_change_streak,
                "log_path": log_path,
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_resume_state(PATHS.RESUME_STATE_FILE, resume_payload)

    log_info("\n--- Training Completed ---")
    log_info(f"Best Validation Loss: {best_score:.4f}")
    log_info(f"Final Model Saved to: {args.checkpoint_path}")
    if log_path:
        log_info(f"Full Log Saved to:    {log_path}")


def main():
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
    p.add_argument("--seed", type=int, default=TRAINING.SEED)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--rnn_type", default="lstm")
    p.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET)
    p.add_argument(
        "--resume_training",
        action="store_true",
    )
    p.add_argument(
        "--bidirectional", action="store_true", default=TRAINING.BIDIRECTIONAL
    )
    p.add_argument(
        "--probabilistic",
        action="store_true",
        default=TRAINING.PROBABILISTIC_TRAINING,
    )
    p.add_argument(
        "--hyperparam_optimizer",
        default=TRAINING.HYPERPARAM_OPTIMIZER,
        choices=["random", "sfoa"],
    )

    try:
        train(p.parse_args())
    except Exception as e:
        logging.error("Fatal Error during training", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
