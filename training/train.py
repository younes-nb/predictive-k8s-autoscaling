import os
import sys
import random
import argparse
import logging
import warnings
import numpy as np
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
from shared.features import target_features_for_feature_set
from core.dataset import ShardedWindowsDataset

from training.loss import per_target_loss, per_target_huber_loss
from training.train_helpers import (
    head_slice_dataset_by_pct,
    load_resume_state,
    save_resume_state,
)
from training.sfoa_search import run_sfoa_search
from training.sfoa_configs import get_config


MODEL_TYPES = ("lstm", "gru", "bilstm", "bigrue", "cnn_bilstm", "dlinear", "dpam")
PREPROCESS_APPROACHES = ("none", "smoothing", "swt", "cskv")


def _build_model(model_type, input_size, args, num_targets, hyperparams, device):
    cfg = get_config(model_type)
    return cfg.build_model(hyperparams, input_size, args, num_targets, device)


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
        from preprocessing.swt.dataset import SwtDataset
        from preprocessing.swt.config import CFG as SWT_CFG
        swt_kw = dict(
            input_len=args.input_len, pred_horizon=PREPROCESSING.PRED_HORIZON,
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
        from preprocessing.cskv.config import CFG as CSKV_CFG
        train_ds = CskvDataset(
            args.preprocess_dir, "train",
            input_len=args.input_len, pred_horizon=PREPROCESSING.PRED_HORIZON,
        )
        val_ds = CskvDataset(
            args.preprocess_dir, "val",
            input_len=args.input_len, pred_horizon=PREPROCESSING.PRED_HORIZON,
        )
        return train_ds, val_ds
    else:
        raise ValueError(f"Unknown preprocess_approach: {preprocess_approach}")


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

    if not hasattr(args, "resume_training"):
        args.resume_training = False

    resume_state = None
    if args.resume_training:
        resume_state = load_resume_state(PATHS.RESUME_STATE_FILE)
        if resume_state and "args" in resume_state:
            _cli_overrides = {
                k: getattr(args, k) for k in (
                    "sfoa_train_pct", "sfoa_val_pct", "sfoa_num_workers",
                    "train_pct", "val_pct", "batch_size", "num_workers",
                    "epochs", "seed",
                    "swt_level", "mem_swt_level",
                    "max_services",
                ) if hasattr(args, k)
            }
            args = argparse.Namespace(**resume_state["args"])
            args.resume_training = True
            args.model_type = model_type
            args.preprocess_approach = preprocess_approach
            if hasattr(args, "preprocess_dir"):
                args.preprocess_dir = getattr(args, "preprocess_dir", None)
            for _k, _v in _cli_overrides.items():
                setattr(args, _k, _v)

    sfoa_defaults = {
        "sfoa_train_pct": TRAINING.SFOA_TRAIN_PCT,
        "sfoa_val_pct": TRAINING.SFOA_VAL_PCT,
        "sfoa_num_workers": TRAINING.SFOA_NUM_WORKERS,
    }
    for attr, default in sfoa_defaults.items():
        if not hasattr(args, attr):
            setattr(args, attr, default)

    dataset_pct_defaults = {
        "train_pct": TRAINING.TRAIN_PCT,
        "val_pct": TRAINING.VAL_PCT,
    }
    for attr, default in dataset_pct_defaults.items():
        if not hasattr(args, attr):
            setattr(args, attr, default)

    from preprocessing.swt.config import CFG as SWT_CFG
    swt_defaults = {
        "swt_level": SWT_CFG.SWT_LEVEL,
        "mem_swt_level": SWT_CFG.MEM_SWT_LEVEL,
    }
    for attr, default in swt_defaults.items():
        if not hasattr(args, attr) or getattr(args, attr) is None:
            setattr(args, attr, default)

    if not hasattr(args, "loss_mode"):
        setattr(args, "loss_mode", "per_target_mse")

    hyperparam_optimizer = getattr(
        args, "hyperparam_optimizer", TRAINING.HYPERPARAM_OPTIMIZER
    )

    sfoa_done = False
    if resume_state:
        sfoa_done = resume_state.get("sfoa_done") is True
        if "hyperparam_optimizer" in resume_state.get("args", {}):
            hyperparam_optimizer = resume_state["args"]["hyperparam_optimizer"]

    log_path = None
    if accelerator.is_local_main_process:
        log_path = setup_logging("train")

    start_epoch = resume_state.get("epoch", 0) + 1 if resume_state else 1
    best_val_loss = (
        resume_state.get("best_val_loss", float("inf"))
        if resume_state
        else float("inf")
    )
    patience_counter = (
        resume_state.get("patience_counter", 0) if resume_state else 0
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

    if preprocess_approach == "swt":
        from preprocessing.swt.config import CFG as SWT_CFG
        log_info(f"SWT Config: {SWT_CFG}")
        input_size = train_ds.n_channels
    elif preprocess_approach == "cskv":
        from preprocessing.cskv.config import CFG as CSKV_CFG
        log_info(f"CSKV Config: {CSKV_CFG}")

    cfg = get_config(model_type)
    log_info(f"Model Type: {model_type} | Search Space: {[s['name'] for s in cfg.SEARCH_SPACE]}")

    num_targets = len(target_features_for_feature_set(args.feature_set))

    current_hyperparams = resume_state.get("hyperparams") if resume_state else None

    if hyperparam_optimizer == "none":
        current_hyperparams = cfg.DEFAULTS.copy()
        sfoa_done = True

    if current_hyperparams is None and not sfoa_done and hyperparam_optimizer == "sfoa":
        if accelerator.num_processes > 1:
            _sync = accelerator.wait_for_everyone
        else:
            _sync = None

        log_info("Running SFOA hyperparameter search before main training...")
        if _sync is not None:
            _sync()
        current_hyperparams = run_sfoa_search(
            args, train_ds, val_ds,
            rank_seed=args.seed,
            accelerator=accelerator,
            resume=getattr(args, "resume_training", False),
            config=cfg,
        )
        sfoa_done = True
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()
        if current_hyperparams is None:
            logging.warning(
                "[SFOA] Search did not return hyperparameters; using fixed defaults."
            )
            current_hyperparams = cfg.DEFAULTS.copy()

    train_ds = head_slice_dataset_by_pct(train_ds, args.train_pct)
    val_ds = head_slice_dataset_by_pct(val_ds, args.val_pct)
    log_info(
        f"Train samples (Used):  {len(train_ds)}/{total_train_samples} "
        f"({float(args.train_pct):g}%)"
    )
    log_info(
        f"Val samples (Used):    {len(val_ds)}/{total_val_samples} "
        f"({float(args.val_pct):g}%)"
    )

    log_info("\n--- Configuration Inputs ---")
    for key, value in vars(args).items():
        log_info(f"{key:<20}: {value}")
    log_info("-" * 30)

    log_info(f"Device: {device} | Distributed Processes: {accelerator.num_processes}")
    log_info(f"Model Type: {model_type} | Preprocess Approach: {preprocess_approach}")

    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    warnings.filterwarnings(
        "ignore",
        message=r"Grad strides do not match bucket view strides.*",
    )
    log_info(f"Seed set: {seed}")

    model = _build_model(model_type, input_size, args, num_targets, current_hyperparams, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"Model parameters: {n_params:,}")

    batch_size = args.batch_size
    log_info(
        f"Using fixed per-GPU batch size: {batch_size} "
        f"(Global: {batch_size * accelerator.num_processes})"
    )
    optimal_workers = getattr(args, "num_workers", TRAINING.NUM_WORKERS)
    log_info(f"num_workers: {optimal_workers}")

    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed)

    pin_memory = device.type != "cpu"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        generator=val_generator,
    )

    lr = current_hyperparams.get("lr", TRAINING.LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if resume_state:
        model_state = resume_state.get("model_state_dict")
        if model_state:
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            if missing:
                raise RuntimeError(f"Resume state missing keys: {sorted(missing)[:10]}")
            if unexpected:
                log_info(f"Note: dropping {len(unexpected)} stale resume keys: {sorted(unexpected)[:5]}")
        optimizer_state = resume_state.get("optimizer_state_dict")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=TRAINING.LR_REDUCE_FACTOR,
        patience=TRAINING.LR_REDUCE_PATIENCE,
        min_lr=TRAINING.LR_MIN,
    )
    if resume_state and "scheduler_state_dict" in resume_state:
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])

    if accelerator.mixed_precision == "fp16":
        log_info("AMP (FP16 mixed precision) enabled via Accelerate")

    def _compute_loss(model, preds, y):
        if args.last_step_only:
            preds = preds[..., -1:, :]
            y = y[..., -1:, :]
        if args.loss_mode == "joint_mse":
            loss = nn.functional.mse_loss(preds, y)
        elif args.loss_mode == "per_target_huber":
            loss = per_target_huber_loss(
                preds, y, cpu_beta=args.huber_beta_cpu, mem_beta=args.huber_beta_mem,
                lambda_cpu=args.loss_w_cpu, lambda_mem=args.loss_w_mem,
            )
        else:
            mem_mode = "l1" if args.loss_mode == "per_target_mae" else "mse"
            loss = per_target_loss(preds, y, mem_mode=mem_mode)
        return loss

    log_info("\n--- Starting Training Loop ---")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = datetime.now()
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
                preds = model(x)
                loss = _compute_loss(model, preds, y)

            accelerator.backward(loss)

            grad_clip = getattr(args, "grad_clip", 0) or TRAINING.GRAD_CLIP
            if grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)

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
                x, y, _ = batch
                if accelerator.mixed_precision != "fp16":
                    x = x.float()
                    y = y.float()

                with accelerator.autocast():
                    preds = model(x)
                    loss = _compute_loss(model, preds, y)

                val_loss_accum += loss.item() * x.size(0)
                val_samples_seen += x.size(0)

        local_avg_val = val_loss_accum / max(1, val_samples_seen)
        sync_val_tensor = torch.tensor(local_avg_val, device=device)
        avg_val_loss = accelerator.reduce(sync_val_tensor, reduction="mean").item()

        epoch_duration = (datetime.now() - start_time).total_seconds()

        current_lr = optimizer.param_groups[0]["lr"]
        log_msg = (
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Patience: {patience_counter}/{TRAINING.EARLY_STOP_PATIENCE} | "
            f"Time: {epoch_duration:.1f}s"
        )

        if avg_val_loss < best_val_loss - TRAINING.EARLY_STOP_MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if accelerator.is_local_main_process:
                os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

                unwrapped_model = accelerator.unwrap_model(model)
                ckpt_payload = {
                    "model_state_dict": unwrapped_model.state_dict(),
                    "args": vars(args),
                    "input_size": input_size,
                    "best_val_loss": best_val_loss,
                    "model_type": model_type,
                    "preprocess_approach": preprocess_approach,
                    "hyperparams": current_hyperparams,
                }

                torch.save(ckpt_payload, args.checkpoint_path)
            log_msg += " [Checkpoint Saved]"
        else:
            patience_counter += 1

        log_info(log_msg)

        scheduler.step(avg_val_loss)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if patience_counter >= TRAINING.EARLY_STOP_PATIENCE:
            log_info(
                "\n=== Early Stopping ===\n"
                f"Val loss has not improved by more than {TRAINING.EARLY_STOP_MIN_DELTA} "
                f"for {patience_counter} consecutive epochs (patience={TRAINING.EARLY_STOP_PATIENCE})."
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
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            save_resume_state(PATHS.RESUME_STATE_FILE, resume_payload)

    log_info("\n--- Training Completed ---")
    log_info(f"Best Validation Loss: {best_val_loss:.6f}")
    log_info(f"Final Model Saved to: {args.checkpoint_path}")
    if log_path:
        log_info(f"Full Log Saved to:    {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--pred_horizon", type=int, default=PREPROCESSING.PRED_HORIZON)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=TRAINING.EPOCHS)
    p.add_argument("--grad_clip", type=float, default=TRAINING.GRAD_CLIP)
    p.add_argument("--weight_decay", type=float, default=TRAINING.WEIGHT_DECAY)
    p.add_argument("--huber_beta_cpu", type=float, default=0.01,
                   help="Huber beta for CPU branch in per_target_huber loss (default 0.01)")
    p.add_argument("--huber_beta_mem", type=float, default=0.002,
                   help="Huber beta for memory branch in per_target_huber loss (default 0.002)")
    p.add_argument("--loss_w_cpu", type=float, default=0.5,
                   help="Weight on CPU branch loss in per_target_huber (default 0.5)")
    p.add_argument("--loss_w_mem", type=float, default=0.5,
                   help="Weight on memory branch loss in per_target_huber (default 0.5)")
    p.add_argument(
        "--loss_mode",
        default="per_target_mse",
        choices=["joint_mse", "per_target_mse", "per_target_mae", "per_target_huber"],
        help="joint_mse: MSE over all targets. per_target_*: equal-weight per target; "
             "per_target_mae uses L1 for the memory target.",
    )
    p.add_argument("--last_step_only", action=argparse.BooleanOptionalAction, default=True,
                   help="Compute loss only on the final horizon step (H-1); use --no-last_step_only "
                        "to average over all steps (default: on).")
    p.add_argument("--seed", type=int, default=TRAINING.SEED)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num_workers", type=int, default=TRAINING.NUM_WORKERS)
    p.add_argument("--rnn_type", default="lstm")
    p.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET)
    p.add_argument("--resume_training", action="store_true")
    p.add_argument(
        "--hyperparam_optimizer",
        default=TRAINING.HYPERPARAM_OPTIMIZER,
        choices=["sfoa", "none"],
    )
    p.add_argument("--sfoa_train_pct", type=float, default=TRAINING.SFOA_TRAIN_PCT)
    p.add_argument("--sfoa_val_pct", type=float, default=TRAINING.SFOA_VAL_PCT)
    p.add_argument("--sfoa_num_workers", type=int, default=TRAINING.SFOA_NUM_WORKERS)
    p.add_argument("--train_pct", type=float, default=TRAINING.TRAIN_PCT)
    p.add_argument("--val_pct", type=float, default=TRAINING.VAL_PCT)
    p.add_argument(
        "--model_type",
        default="lstm",
        choices=list(MODEL_TYPES),
        help="Model architecture: lstm, gru, bilstm, bigrue, cnn_bilstm",
    )
    p.add_argument(
        "--preprocess_approach",
        default="swt",
        choices=list(PREPROCESS_APPROACHES),
        help="Preprocessing approach used: none, smoothing, swt, cskv",
    )
    p.add_argument("--preprocess_dir", default=None, help="Preprocessing output dir (for smoothing/swt/cskv)")
    p.add_argument("--dataset_workers", type=int, default=0, help="Workers for swt/cskv dataset loading")
    p.add_argument("--swt_level", type=int, default=None, help="SWT level for CPU (swt only, default: config)")
    p.add_argument("--mem_swt_level", type=int, default=None, help="SWT level for memory (swt only, default: config)")
    p.add_argument("--max_services", type=int, default=0, help="Max services for swt/cskv")
    p.add_argument("--dpam_cnn_kernels", type=int, nargs="+", default=None,
                   help="MultiKernelConv1D kernel sizes for dpam (ablation; default (3,5))")
    p.add_argument("--dpam_group_blocks", type=int, default=None,
                   help="Group MixerBlocks per group for dpam (ablation; default 2)")
    p.add_argument("--dpam_d_group", type=int, default=None,
                   help="CPU group hidden dim for dpam (ablation; default 64)")
    p.add_argument("--dpam_mem_d_group", type=int, default=None,
                   help="Memory group hidden dim for dpam (ablation; default 24)")
    p.add_argument("--dpam_pool_head_dim", type=int, default=None,
                   help="Pooled-MLP head dim for dpam (ablation; default 128)")
    p.add_argument("--dpam_cpu_recon", action=argparse.BooleanOptionalAction, default=True,
                   help="Level-correction MLP on CPU branch (ablation; default on)")

    try:
        train(p.parse_args())
    except Exception as e:
        logging.error("Fatal Error during training", exc_info=True)
        sys.exit(1)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
