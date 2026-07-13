import argparse
import logging
import os
import random
import sys
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator, InitProcessGroupKwargs

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.tsdp.config import CFG as GLOBAL_CFG
from experiments.tsdp.tcn_bigru.config import CFG as ARCH_CFG
from experiments.tsdp.dataset import TsdpDataset, N_CHANNELS
from experiments.tsdp.tcn_bigru.model import TcnBiGru

ARCH_NAME = "TCN-BiGRU"


class _TehranFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            tz = ZoneInfo("Asia/Tehran")
        except Exception:
            tz = None
        ts = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"{ts} [{record.levelname}] {record.getMessage()}"


def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    try:
        tz = ZoneInfo("Asia/Tehran")
    except Exception:
        tz = None
    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{ts}.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = _TehranFormatter()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return log_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=f"Train TSDP model ({ARCH_NAME})."
    )
    ap.add_argument("--preprocess_dir", default="/dataset/tsdp_preprocess")
    ap.add_argument("--out_dir", default="/proj/k8sautoscaledl-PG0/models/tsdp")
    ap.add_argument("--log_dir", default="/proj/k8sautoscaledl-PG0/logs/tsdp")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--dataset_workers", type=int, default=max(1, int(os.cpu_count() * 0.7)))
    ap.add_argument("--max_services", type=int, default=0,
                    help="Limit to first N services (0 = all from index)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.out_dir, exist_ok=True)

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=14400))
    mixed_precision = "fp16" if not args.cpu and torch.cuda.is_available() else "no"
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=mixed_precision, kwargs_handlers=[timeout_kwargs])
    device = accelerator.device

    log_info = lambda msg, *a: (
        logging.info(msg, *a) if accelerator.is_local_main_process else None
    )

    if accelerator.is_local_main_process:
        log_path = setup_logging(args.log_dir)
    else:
        log_path = None

    epochs = args.epochs if args.epochs is not None else GLOBAL_CFG.EPOCHS

    log_info("=" * 60)
    log_info("TSDP — %s — Training on %s", ARCH_NAME, device)
    log_info("Shared config: %s", GLOBAL_CFG)
    log_info("Architecture config: %s", ARCH_CFG)
    if log_path:
        log_info("Log file: %s", log_path)
    log_info("=" * 60)

    train_ds = TsdpDataset(
        args.preprocess_dir, "train",
        input_len=GLOBAL_CFG.INPUT_LEN, pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        stride=GLOBAL_CFG.STRIDE,
        train_frac=GLOBAL_CFG.TRAIN_FRAC, val_frac=GLOBAL_CFG.VAL_FRAC,
        num_workers=args.dataset_workers,
        max_services=args.max_services,
    )
    val_ds = TsdpDataset(
        args.preprocess_dir, "val",
        input_len=GLOBAL_CFG.INPUT_LEN, pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        stride=GLOBAL_CFG.STRIDE,
        train_frac=GLOBAL_CFG.TRAIN_FRAC, val_frac=GLOBAL_CFG.VAL_FRAC,
        num_workers=args.dataset_workers,
        max_services=args.max_services,
    )

    if len(train_ds) == 0:
        logging.error("Empty training dataset. Aborting.")
        return

    n_train = len(train_ds)
    n_val = len(val_ds)

    model = TcnBiGru(
        in_channels=N_CHANNELS,
        input_len=GLOBAL_CFG.INPUT_LEN,
        pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        tcn_kernel_size=ARCH_CFG.TCN_KERNEL_SIZE,
        tcn_filters=ARCH_CFG.TCN_FILTERS,
        tcn_dilations=ARCH_CFG.TCN_DILATIONS,
        tcn_dropout=ARCH_CFG.TCN_DROPOUT,
        bigru_hidden=ARCH_CFG.BIGRU_HIDDEN,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=GLOBAL_CFG.LEARNING_RATE, weight_decay=GLOBAL_CFG.WEIGHT_DECAY)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info("Model parameters: %d", n_params)
    log_info("Train windows: %d | Val windows: %d", n_train, n_val)
    log_info("Epochs: %d", epochs)

    pin_memory = device.type != "cpu"
    train_loader = DataLoader(
        train_ds, batch_size=GLOBAL_CFG.BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=GLOBAL_CFG.BATCH_SIZE * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    del train_ds, val_ds

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    ckpt_path = os.path.join(args.out_dir, "tsdp.pt")
    best_val_loss = float("inf")
    train_loss_window = deque(maxlen=GLOBAL_CFG.NO_CHANGE_EPOCHS_LIMIT)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_accum, n_train_count = 0.0, 0
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            with accelerator.autocast():
                pred = model(x)
                loss = criterion(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                if accelerator.is_local_main_process:
                    logging.warning("NaN/Inf loss detected; skipping batch.")
                continue
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=GLOBAL_CFG.GRAD_CLIP_NORM)
            optimizer.step()
            train_accum += loss.item() * x.size(0)
            n_train_count += x.size(0)

        local_avg_train = train_accum / max(n_train_count, 1)
        sync_train_tensor = torch.tensor(local_avg_train, device=device)
        avg_train = accelerator.reduce(sync_train_tensor, reduction="mean").item()

        model.eval()
        val_accum, n_val_count = 0.0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                with accelerator.autocast():
                    pred = model(x)
                    loss = criterion(pred, y)
                val_accum += loss.item() * x.size(0)
                n_val_count += x.size(0)

        local_avg_val = val_accum / max(n_val_count, 1)
        sync_val_tensor = torch.tensor(local_avg_val, device=device)
        avg_val = accelerator.reduce(sync_val_tensor, reduction="mean").item()

        dt = time.time() - t0
        suffix = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "model_state_dict": unwrapped.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "arch": ARCH_NAME,
                        "cfg": {
                            "input_len": GLOBAL_CFG.INPUT_LEN,
                            "pred_horizon": GLOBAL_CFG.PRED_HORIZON,
                            "n_channels": N_CHANNELS,
                            "tcn_kernel_size": ARCH_CFG.TCN_KERNEL_SIZE,
                            "tcn_filters": ARCH_CFG.TCN_FILTERS,
                            "tcn_dilations": ARCH_CFG.TCN_DILATIONS,
                            "tcn_dropout": ARCH_CFG.TCN_DROPOUT,
                            "bigru_hidden": ARCH_CFG.BIGRU_HIDDEN,
                        },
                    },
                    ckpt_path,
                )
            suffix = " [Checkpoint Saved]"

        log_info(
            "Epoch %d/%d | train=%.8f | val=%.8f | %.1fs%s",
            epoch, epochs, avg_train, avg_val, dt, suffix,
        )

        train_loss_window.append(avg_train)
        if len(train_loss_window) == GLOBAL_CFG.NO_CHANGE_EPOCHS_LIMIT:
            cumulative_delta = abs(train_loss_window[-1] - train_loss_window[0])
            if cumulative_delta < GLOBAL_CFG.LOSS_CHANGE_THRESHOLD:
                log_info(
                    "\n=== Early Stopping ===\n"
                    f"Train loss has not changed by at least {GLOBAL_CFG.LOSS_CHANGE_THRESHOLD} "
                    f"over the last {GLOBAL_CFG.NO_CHANGE_EPOCHS_LIMIT} epochs "
                    f"(%.8f -> %.8f, Δ=%.8f).",
                    train_loss_window[0], train_loss_window[-1], cumulative_delta,
                )
                break

    log_info(
        "=== TSDP [%s] training complete. Best val loss: %.8f | Saved: %s ===",
        ARCH_NAME, best_val_loss, ckpt_path,
    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
