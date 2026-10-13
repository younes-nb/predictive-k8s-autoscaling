"""Train Madu workload predictor(s) (§4.1, §5.1).

One individual model per microservice (paper §6.2.1). Select MSs with
--msname (repeatable) or --top_k (top-K by total CPM, UNKNOWN skipped).
Checkpoints store model weights + data/authoring config for evaluate.py.
"""
import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from madu.data import prepare_ms, top_ms_names
from madu.loss import gaussian_nll
from madu.model import MaduSeq2Seq
from shared.config_training_defaults import TRAINING
from shared.logging_utils import setup_logging


def train_one_ms(msname, args, device, log_info):
    log_info(f"\n=== Microservice: {msname} ===")
    prep = prepare_ms(msname, args.n, args.m, args.parquet_dir,
        args.train_frac, args.val_frac, args.max_minutes, args.transform,
        args.feat, args.resample, args.resample_how,
    )
    log_info(f"train/val/test windows: {len(prep['train'])}/{len(prep['val'])}/{len(prep['test'])} "
             f"| norm={prep['norm']}")
    train_loader = DataLoader(prep["train"], batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(prep["val"], batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = MaduSeq2Seq(
        input_size=prep["train"][0][0].shape[-1],
        hidden_size=args.hidden, num_layers=args.layers,
        horizon=args.m, attention=args.attention,
        uncertainty=not args.no_uncertainty, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log_info(f"params={n_params:,} attention={args.attention} "
             f"uncertainty={not args.no_uncertainty}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=TRAINING.LR_REDUCE_FACTOR,
            patience=TRAINING.LR_REDUCE_PATIENCE, min_lr=TRAINING.LR_MIN)

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.float().to(device), yb.float().to(device)
            opt.zero_grad()
            mu, sig = model(xb, y_teacher=yb)
            loss = (gaussian_nll(mu, sig, yb) if model.uncertainty
                    else torch.nn.functional.mse_loss(mu, yb))
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item() * len(xb)
            tr_n += len(xb)
        model.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.float().to(device), yb.float().to(device)
                mu, sig = model(xb)
                loss = (gaussian_nll(mu, sig, yb) if model.uncertainty
                        else torch.nn.functional.mse_loss(mu, yb))
                va_loss += loss.item() * len(xb)
                va_n += len(xb)
        va_loss /= max(1, va_n)
        msg = (f"[{msname}] epoch {epoch}/{args.epochs} "
               f"train={tr_loss / max(1, tr_n):.6f} val={va_loss:.6f}")
        if va_loss < best_val - args.min_delta:
            best_val, patience = va_loss, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            msg += " [saved]"
        else:
            patience += 1
        log_info(msg)
        if scheduler is not None:
            scheduler.step(va_loss)
        if patience >= args.patience:
            log_info(f"[{msname}] early stopping at epoch {epoch}")
            break

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"madu_{msname}.pt")
    torch.save({
        "model_state_dict": best_state,
        "config": {"n": args.n, "m": args.m, "hidden": args.hidden,
                   "layers": args.layers, "attention": args.attention,
                   "uncertainty": not args.no_uncertainty,
                   "dropout": args.dropout, "lr": args.lr,
                   "weight_decay": args.weight_decay,
                   "lr_scheduler": args.lr_scheduler,
                   "batch_size": args.batch_size, "epochs": args.epochs,
                   "patience": args.patience, "min_delta": args.min_delta,
                   "seed": args.seed},
        "norm": {**prep["norm"], "train_frac": args.train_frac,
                 "val_frac": args.val_frac, "max_minutes": args.max_minutes,
                 "transform": args.transform, "feat": args.feat,
                 "resample": args.resample, "resample_how": args.resample_how},
        "msname": msname,
        "parquet_dir": args.parquet_dir,
        "best_val_loss": best_val,
    }, ckpt_path)
    log_info(f"[{msname}] checkpoint -> {ckpt_path} (best val {best_val:.6f})")
    return ckpt_path


def main():
    p = argparse.ArgumentParser(description="Train Madu CPM predictors (SoCC'22 §4.1).")
    p.add_argument("--msname", action="append", default=[],
                   help="Microservice to train (repeatable).")
    p.add_argument("--top_k", type=int, default=0,
                   help="Train top-K MSs by total CPM instead.")
    p.add_argument("--parquet_dir", default="/dataset/ms_call_counts/per_minute")
    p.add_argument("--checkpoint_dir", default="/proj/k8sautoscaledl-PG0/madu")
    p.add_argument("--logs_dir", default=None)
    p.add_argument("--n", type=int, default=60, help="Encoder length (paper default 60).")
    p.add_argument("--m", type=int, default=5, help="Decoder/prediction length (paper default 5).")
    # Training hyperparameters default to shared.config_training_defaults
    # (env-overridable); --min_delta default 1e-4 per user instruction
    # (overrides the config's 1e-6 early-stop threshold).
    p.add_argument("--hidden", type=int, default=TRAINING.HIDDEN_SIZE)
    p.add_argument("--layers", type=int, default=TRAINING.NUM_LAYERS)
    p.add_argument("--attention", default="stochastic",
                   choices=["stochastic", "deterministic", "none"])
    p.add_argument("--no_uncertainty", action="store_true",
                   help="Deterministic Seq2Seq baseline (MSE, mu head only).")
    p.add_argument("--dropout", type=float, default=TRAINING.DROPOUT)
    p.add_argument("--epochs", type=int, default=TRAINING.EPOCHS)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=TRAINING.LR)
    p.add_argument("--weight_decay", type=float, default=TRAINING.WEIGHT_DECAY)
    p.add_argument("--lr_scheduler", default=TRAINING.LR_SCHEDULER,
                   choices=["ReduceLROnPlateau", "none"])
    p.add_argument("--grad_clip", type=float, default=TRAINING.GRAD_CLIP)
    p.add_argument("--patience", type=int, default=TRAINING.EARLY_STOP_PATIENCE)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--max_minutes", type=int, default=0,
                   help="Cap series length (most recent minutes) for quick tests.")
    p.add_argument("--transform", default="max", choices=["max", "log1p"],
                   help="CPM normalization: max (paper-style) or log1p+z "
                        "(for burst spikes spanning orders of magnitude).")
    p.add_argument("--feat", default="paper", choices=["paper", "full"],
                   help="Feature set: paper {CPM,tm,th,tw} or full (+cyclic "
                        "time + rolling mean/std of transformed CPM).")
    p.add_argument("--resample", type=int, default=1,
                   help="Bin minutes into N-minute bins (mean/max) to tame "
                        "isolated single-minute spikes (default 1 = off).")
    p.add_argument("--resample_how", default="mean", choices=["mean", "max"],
                   help="Bin aggregation: mean or max (peak).")
    p.add_argument("--num_workers", type=int, default=TRAINING.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=TRAINING.SEED)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip MSs whose checkpoint already exists (resume).")
    args = p.parse_args()

    setup_logging("madu_train", log_dir=args.logs_dir)
    log_info = logging.info
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    msnames = list(args.msname)
    if args.top_k:
        msnames += top_ms_names(args.top_k, args.parquet_dir)
    if not msnames:
        msnames = top_ms_names(1, args.parquet_dir)
        log_info(f"No --msname given; defaulting to top MS {msnames}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    log_info(f"device={device} msnames={msnames}")
    done, failed, skipped = [], [], []
    for ms in msnames:
        ckpt_path = os.path.join(args.checkpoint_dir, f"madu_{ms}.pt")
        if args.skip_existing and os.path.isfile(ckpt_path):
            log_info(f"[{ms}] checkpoint exists, skipping")
            skipped.append(ms)
            continue
        try:
            train_one_ms(ms, args, device, log_info)
            done.append(ms)
        except Exception as e:
            log_info(f"[{ms}] FAILED: {e}")
            failed.append(ms)
    log_info(f"trained {len(done)}/{len(msnames)}; skipped {len(skipped)}; "
             f"failed: {failed}")


if __name__ == "__main__":
    main()
