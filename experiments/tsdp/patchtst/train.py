import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

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
from experiments.tsdp.patchtst.config import CFG as ARCH_CFG
from experiments.tsdp.dataset import TsdpDataset, N_CHANNELS
from experiments.tsdp.patchtst.model import PatchTST

ARCH_NAME = "PatchTST"


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
    args = ap.parse_args()

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
    )
    val_ds = TsdpDataset(
        args.preprocess_dir, "val",
        input_len=GLOBAL_CFG.INPUT_LEN, pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        stride=GLOBAL_CFG.STRIDE,
        train_frac=GLOBAL_CFG.TRAIN_FRAC, val_frac=GLOBAL_CFG.VAL_FRAC,
        num_workers=args.dataset_workers,
    )

    if len(train_ds) == 0:
        logging.error("Empty training dataset. Aborting.")
        return

    n_train = len(train_ds)
    n_val = len(val_ds)

    model = PatchTST(
        input_len=GLOBAL_CFG.INPUT_LEN,
        pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        n_channels=N_CHANNELS,
        patch_len=ARCH_CFG.PATCH_LEN,
        stride=ARCH_CFG.PATCH_STRIDE,
        d_model=ARCH_CFG.D_MODEL,
        n_heads=ARCH_CFG.N_HEADS,
        n_layers=ARCH_CFG.N_LAYERS,
        d_ff=ARCH_CFG.D_FF,
        dropout=ARCH_CFG.DROPOUT,
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
    prev_train_loss = None
    no_improve_count = 0

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
                            "n_channels": GLOBAL_CFG.TOTAL_CHANNELS,
                            "patch_len": ARCH_CFG.PATCH_LEN,
                            "patch_stride": ARCH_CFG.PATCH_STRIDE,
                            "d_model": ARCH_CFG.D_MODEL,
                            "n_heads": ARCH_CFG.N_HEADS,
                            "n_layers": ARCH_CFG.N_LAYERS,
                            "d_ff": ARCH_CFG.D_FF,
                            "dropout": ARCH_CFG.DROPOUT,
                        },
                    },
                    ckpt_path,
                )
            suffix = " [Checkpoint Saved]"

        log_info(
            "Epoch %d/%d | train=%.8f | val=%.8f | %.1fs%s",
            epoch, epochs, avg_train, avg_val, dt, suffix,
        )

        if prev_train_loss is not None:
            delta = abs(avg_train - prev_train_loss)
            if delta < GLOBAL_CFG.EARLY_STOP_DELTA:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count >= GLOBAL_CFG.EARLY_STOP_PATIENCE:
                log_info(
                    "Early stopping: train loss unchanged (Δ<%.7f) for %d consecutive epochs at epoch %d.",
                    GLOBAL_CFG.EARLY_STOP_DELTA, GLOBAL_CFG.EARLY_STOP_PATIENCE, epoch,
                )
                break

        prev_train_loss = avg_train

    log_info(
        "=== TSDP [%s] training complete. Best val loss: %.8f | Saved: %s ===",
        ARCH_NAME, best_val_loss, ckpt_path,
    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
