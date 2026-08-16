#!/usr/bin/env python
"""
EnbPI Training Pipeline

Trains a quantile ensemble forecaster and calibrates it using EnbPI (Ensemble Batch Prediction Intervals)
from tsbootstrap for distribution-free prediction intervals on time series.

Usage:
    python training/enbpi_train.py --windows_dir /proj/k8sautoscaledl-PG0/windows \
        --checkpoint_path /proj/k8sautoscaledl-PG0/models/enbpi_model.pt \
        --feature_set cpu_mem_http_rpc_replicas --preprocess_approach swt \
        --input_len 128 --pred_horizon 5 --ensemble_size 5 --n_bootstraps 999 \
        --cpu --epochs 100 --batch_size 4096
"""

import os
import sys
import random
import argparse
import logging
import warnings
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator, InitProcessGroupKwargs

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.config_paths import PATHS
from shared.config_training_defaults import TRAINING
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.logging_utils import setup_logging
from shared.features import target_features_for_feature_set
from core.dataset import ShardedWindowsDataset
from core.architectures.ensemble import QuantileEnsembleForecaster
from training.train_helpers import (
    head_slice_dataset_by_pct,
    load_resume_state,
    save_resume_state,
)
from preprocessing.swt.dataset import SwtDataset
from preprocessing.swt.config import CFG as SWT_CFG

MODEL_TYPES = ("quantile_ensemble",)
PREPROCESS_APPROACHES = ("none", "smoothing", "swt", "cskv")


def _load_datasets(args, preprocess_approach):
    if preprocess_approach == "none":
        train_ds = ShardedWindowsDataset(
            args.windows_dir, "train", args.input_len, args.pred_horizon
        )
        val_ds = ShardedWindowsDataset(
            args.windows_dir, "val", args.input_len, args.pred_horizon
        )
        return train_ds, val_ds
    elif preprocess_approach == "smoothing":
        smooth_dir = getattr(args, "preprocess_dir", None) or args.windows_dir
        train_ds = ShardedWindowsDataset(
            smooth_dir, "train", args.input_len, args.pred_horizon
        )
        val_ds = ShardedWindowsDataset(
            smooth_dir, "val", args.input_len, args.pred_horizon
        )
        return train_ds, val_ds
    elif preprocess_approach == "swt":
        swt_kw = dict(
            input_len=args.input_len, pred_horizon=args.pred_horizon,
            feature_set=args.feature_set,
        )
        for attr, cli_arg in [
            ("swt_level", "swt_level"), ("mem_swt_level", "mem_swt_level"),
        ]:
            val = getattr(args, cli_arg, None)
            if val is not None:
                swt_kw[attr] = val
        train_ds = SwtDataset(args.preprocess_dir, "train", **swt_kw)
        val_ds = SwtDataset(args.preprocess_dir, "val", **swt_kw)
        return train_ds, val_ds
    elif preprocess_approach == "cskv":
        from preprocessing.cskv.dataset import CskvDataset
        train_ds = CskvDataset(
            args.preprocess_dir, "train",
            input_len=args.input_len, pred_horizon=args.pred_horizon,
        )
        val_ds = CskvDataset(
            args.preprocess_dir, "val",
            input_len=args.input_len, pred_horizon=args.pred_horizon,
        )
        return train_ds, val_ds
    else:
        raise ValueError(f"Unknown preprocess_approach: {preprocess_approach}")


def _build_model(model_type, input_size, args, num_targets, hyperparams, device):
    quantiles = hyperparams.get("quantiles", [0.05, 0.5, 0.95])
    model = QuantileEnsembleForecaster(
        input_size=input_size,
        hidden_size=hyperparams.get("hidden_size", 128),
        num_layers=hyperparams.get("num_layers", 3),
        dropout=hyperparams.get("dropout", 0.2),
        horizon=args.pred_horizon,
        num_targets=num_targets,
        quantiles=quantiles,
        ensemble_size=hyperparams.get("ensemble_size", 5),
    ).to(device)
    return model


def _compute_quantile_loss(model, preds, y, quantile_weights=None):
    """Weighted pinball loss for quantile regression."""
    # preds: (B, H, num_targets, num_quantiles)
    # y: (B, H, num_targets)
    if preds.dim() == 3:
        # Single quantile case
        return F.mse_loss(preds, y)
    
    quantiles = torch.tensor(model.quantiles, device=preds.device, dtype=preds.dtype)
    errors = y.unsqueeze(-1) - preds  # (B, H, T, Q)
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
    
    if quantile_weights is not None:
        weights = torch.tensor(quantile_weights, device=preds.device, dtype=preds.dtype)
        loss = loss * weights.view(1, 1, 1, -1)
    
    return loss.mean()


def train(args):
    model_type = args.model_type
    preprocess_approach = args.preprocess_approach

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=14400))
    mixed_precision = "fp16" if (not args.cpu and torch.cuda.is_available()) else "no"
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=mixed_precision, kwargs_handlers=[timeout_kwargs])
    device = accelerator.device

    log_info = lambda msg: (
        logging.info(msg) if accelerator.is_local_main_process else None
    )

    log_path = None
    if accelerator.is_local_main_process:
        log_path = setup_logging("enbpi_train", log_dir=args.logs_dir)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    log_info(f"Seed set: {seed}")

    log_info("\n--- Loading Datasets ---")
    train_ds, val_ds = _load_datasets(args, preprocess_approach)

    total_train_samples = len(train_ds)
    total_val_samples = len(val_ds)
    log_info(f"Train samples (Total): {total_train_samples}")
    log_info(f"Val samples (Total):   {total_val_samples}")

    if len(train_ds) > 0:
        first_x, _, *_ = train_ds[0]
        input_size = first_x.shape[-1]
        log_info(f"Inferred Input Size: {input_size}")
    else:
        raise RuntimeError("Train dataset is empty.")

    train_ds = head_slice_dataset_by_pct(train_ds, args.train_pct)
    val_ds = head_slice_dataset_by_pct(val_ds, args.val_pct)
    log_info(f"Train samples (Used):  {len(train_ds)}/{total_train_samples} ({float(args.train_pct):g}%)")
    log_info(f"Val samples (Used):    {len(val_ds)}/{total_val_samples} ({float(args.val_pct):g}%)")

    num_targets = len(target_features_for_feature_set(args.feature_set))
    hyperparams = {
        "dropout": args.dropout,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "ensemble_size": args.ensemble_size,
        "quantiles": [0.10, 0.50, 0.95],
        "quantile_weights": [1.0, 1.0, 2.0],
    }

    model = _build_model(model_type, input_size, args, num_targets, hyperparams, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"Model parameters: {n_params:,}")

    batch_size = args.batch_size
    log_info(f"Using fixed per-GPU batch size: {batch_size} (Global: {batch_size * accelerator.num_processes})")

    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    pin_memory = device.type != "cpu"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0 if os.name == "nt" else min(12, os.cpu_count() or 1),
        pin_memory=pin_memory, worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0 if os.name == "nt" else min(12, os.cpu_count() or 1),
        pin_memory=pin_memory, worker_init_fn=_worker_init_fn,
    )

    lr = hyperparams.get("lr", 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    log_info("\n--- Starting Training Loop ---")

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_samples_seen = 0

        for batch in train_loader:
            x, y, _ = batch
            if accelerator.mixed_precision != "fp16":
                x = x.float()
                y = y.float()

            optimizer.zero_grad()

            with accelerator.autocast():
                preds = model(x)  # (B, H, T, Q)
                loss = _compute_quantile_loss(model, preds, y, hyperparams.get("quantile_weights"))

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_accum += loss.item() * x.size(0)
            train_samples_seen += x.size(0)

        avg_train_loss = train_loss_accum / max(1, train_samples_seen)
        avg_train_loss = accelerator.reduce(torch.tensor(avg_train_loss, device=device), reduction="mean").item()

        model.eval()
        val_loss_accum = 0.0
        val_samples_seen = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y, _ = batch
                if accelerator.mixed_precision != "fp16":
                    x = x.float()
                    y = y.float()

                with accelerator.autocast():
                    preds = model(x)
                    loss = _compute_quantile_loss(model, preds, y)

                val_loss_accum += loss.item() * x.size(0)
                val_samples_seen += x.size(0)

        avg_val_loss = val_loss_accum / max(1, val_samples_seen)
        avg_val_loss = accelerator.reduce(torch.tensor(avg_val_loss, device=device), reduction="mean").item()

        current_lr = optimizer.param_groups[0]["lr"]
        log_msg = (
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Patience: {patience_counter}/{args.patience} | "
        )

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if accelerator.is_local_main_process:
                os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
                best_model_state = {
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                    "hyperparams": hyperparams,
                    "input_size": input_size,
                    "model_type": model_type,
                    "preprocess_approach": preprocess_approach,
                }
                torch.save(best_model_state, args.checkpoint_path)
            log_msg += " [Checkpoint Saved]"
        else:
            patience_counter += 1

        log_info(log_msg)
        scheduler.step(avg_val_loss)

        if patience_counter >= args.patience:
            log_info(f"\nEarly Stopping: No improvement for {patience_counter} epochs.")
            break

    log_info(f"\nTraining Completed. Best Val Loss: {best_val_loss:.6f}")
    log_info(f"Model Saved to: {args.checkpoint_path}")

    # Fit CQR calibrator on validation residuals (t+5 only)
    if accelerator.is_local_main_process:
        log_info("\n--- Fitting CQR Calibrator (t+5 horizon) ---")
        cqr_calibrators = _fit_cqr_calibrator(
            args, accelerator, best_model_state, val_ds, device, log_info
        )
        
        # Save calibrators in checkpoint
        if cqr_calibrators:
            best_model_state["cqr_calibrators"] = cqr_calibrators
            best_model_state["cqr_alpha"] = 0.1
            torch.save(best_model_state, args.checkpoint_path)
            log_info(f"CQR calibrators saved to checkpoint")


def _fit_cqr_calibrator(args, accelerator, best_model_state, val_ds, device, log_info):
    """Fit CQR (Conformalized Quantile Regression) calibrator on validation residuals for t+5 horizon."""
    if best_model_state is None:
        log_info("No best model state found, skipping CQR calibration")
        return None

    # Load best model
    hyperparams = best_model_state["hyperparams"]
    num_targets = len(target_features_for_feature_set(args.feature_set))
    model = QuantileEnsembleForecaster(
        input_size=best_model_state["input_size"],
        hidden_size=hyperparams.get("hidden_size", 128),
        num_layers=hyperparams.get("num_layers", 3),
        dropout=hyperparams.get("dropout", 0.2),
        horizon=args.pred_horizon,
        num_targets=num_targets,
        quantiles=hyperparams.get("quantiles", [0.10, 0.50, 0.95]),
        ensemble_size=hyperparams.get("ensemble_size", 5),
    ).to(device)
    model.load_state_dict(best_model_state["model_state_dict"])
    model.eval()

    # Get validation predictions and targets
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0 if os.name == "nt" else min(12, os.cpu_count() or 1),
    )

    all_q10, all_q50, all_q95, all_y = [], [], [], []

    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device).float()
            preds = model(x)  # (B, H, T, Q)
            all_q10.append(preds[:, :, :, 0].cpu())   # q0.10
            all_q50.append(preds[:, :, :, 1].cpu())   # q0.50
            all_q95.append(preds[:, :, :, 2].cpu())   # q0.95
            all_y.append(y)

    q10 = torch.cat(all_q10).numpy()   # (N, H, T)
    q50 = torch.cat(all_q50).numpy()
    q95 = torch.cat(all_q95).numpy()
    y   = torch.cat(all_y).numpy()

    log_info(f"Calibration data: {q10.shape[0]} samples")

    # CQR calibration for t+5 only (horizon index 4, 0-indexed)
    horizon_idx = args.pred_horizon - 1  # t+5 is index 4
    alpha = 0.1  # 90% coverage

    calibrators = {}
    for t_idx in range(num_targets):
        target_name = target_features_for_feature_set(args.feature_set)[t_idx]
        log_info(f"Fitting CQR for {target_name} at horizon t+{args.pred_horizon}...")

        # CQR conformity scores: max(0, q10 - y, y - q95)
        scores = np.maximum(0, np.maximum(
            q10[:, horizon_idx, t_idx] - y[:, horizon_idx, t_idx],
            y[:, horizon_idx, t_idx] - q95[:, horizon_idx, t_idx]
        ))
        
        # Conformal quantile (using 'higher' for conservative coverage)
        q_conf = float(np.quantile(scores, 1 - alpha, method='higher'))
        
        calibrators[t_idx] = {
            'q_conf': q_conf,
            'horizon': args.pred_horizon,
            'alpha': alpha,
        }
        
        log_info(f"  {target_name}: q_conf = {q_conf:.4f}")

    return calibrators


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--logs_dir", default=PATHS.LOGS_DIR)
    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--pred_horizon", type=int, default=PREPROCESSING.PRED_HORIZON)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=TRAINING.EPOCHS)
    p.add_argument("--patience", type=int, default=TRAINING.EARLY_STOP_PATIENCE)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--ensemble_size", type=int, default=5)
    p.add_argument("--seed", type=int, default=TRAINING.SEED)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--train_pct", type=float, default=TRAINING.TRAIN_PCT)
    p.add_argument("--val_pct", type=float, default=TRAINING.VAL_PCT)
    p.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET)
    p.add_argument("--preprocess_approach", default="swt", choices=PREPROCESS_APPROACHES)
    p.add_argument("--preprocess_dir", default=None)
    p.add_argument("--swt_level", type=int, default=None)
    p.add_argument("--mem_swt_level", type=int, default=None)
    p.add_argument("--model_type", default="quantile_ensemble", choices=MODEL_TYPES)

    try:
        train(p.parse_args())
    except Exception:
        logging.error("Fatal Error during training", exc_info=True)
        sys.exit(1)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()