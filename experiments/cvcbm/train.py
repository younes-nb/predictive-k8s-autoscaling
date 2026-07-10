
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
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.cvcbm.config import CFG, set_seed
from experiments.cvcbm.dataset import CvcbmDataset
from experiments.cvcbm.model import CvcbmModel


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
        description="Train CVCBM model (all Co-IMFs stacked as channels)."
    )
    ap.add_argument("--preprocess_dir", default="/dataset/cvcbm_preprocess",
                    help="Directory with Co-IMF numpy files (default: /dataset/cvcbm_preprocess)")
    ap.add_argument("--out_dir", default="/proj/k8sautoscaledl-PG0/models/cvcbm",
                     help="Directory for model checkpoint (default: /proj/k8sautoscaledl-PG0/models/cvcbm)")
    ap.add_argument("--log_dir", default="/proj/k8sautoscaledl-PG0/logs/cvcbm",
                     help="Directory for training logs (default: /proj/k8sautoscaledl-PG0/logs/cvcbm)")
    ap.add_argument("--epochs", type=int, default=None,
                     help="Override paper EPOCHS for smoke tests without editing config.py")
    ap.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = ap.parse_args()

    set_seed(CFG.SEED)

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=14400))
    mixed_precision = "fp16" if not args.cpu and torch.cuda.is_available() else "no"
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=mixed_precision, kwargs_handlers=[timeout_kwargs])
    device = accelerator.device

    log_info = lambda msg, *args: (
        logging.info(msg, *args) if accelerator.is_local_main_process else None
    )

    os.makedirs(args.out_dir, exist_ok=True)
    if accelerator.is_local_main_process:
        log_path = setup_logging(args.log_dir)
    else:
        log_path = None

    epochs = args.epochs if args.epochs is not None else CFG.EPOCHS

    log_info("=" * 60)
    log_info("CVCBM — Model Training on %s", device)
    log_info("Distributed processes: %d", accelerator.num_processes)
    if log_path:
        log_info("Log file: %s", log_path)
    log_info("=" * 60)

    train_ds = CvcbmDataset(
        args.preprocess_dir, "train",
        input_len=CFG.INPUT_LEN, pred_horizon=CFG.PRED_HORIZON,
        stride=CFG.STRIDE,
        train_frac=CFG.TRAIN_FRAC, val_frac=CFG.VAL_FRAC,
    )
    val_ds = CvcbmDataset(
        args.preprocess_dir, "val",
        input_len=CFG.INPUT_LEN, pred_horizon=CFG.PRED_HORIZON,
        stride=CFG.STRIDE,
        train_frac=CFG.TRAIN_FRAC, val_frac=CFG.VAL_FRAC,
    )

    if len(train_ds) == 0:
        log_info("Empty training dataset. Aborting.")
        return

    n_train = len(train_ds)
    n_val = len(val_ds)
    total_channels = train_ds.total_channels

    log_info("Total input channels: %d", total_channels)

    pin_memory = device.type != "cpu"
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    del train_ds, val_ds

    model = CvcbmModel(
        in_channels=total_channels,
        input_len=CFG.INPUT_LEN,
        pred_horizon=CFG.PRED_HORIZON,
        kernel_sizes=CFG.KERNEL_SIZES,
        conv1_out_ch=CFG.CONV1_OUT_CH,
        conv2_out_ch=CFG.CONV2_OUT_CH,
        bilstm_hidden=CFG.BILSTM_HIDDEN,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info("Model parameters: %d", n_params)
    log_info("Train windows: %d | Val windows: %d", n_train, n_val)
    log_info("Epochs: %d", epochs)
    log_info(
        "Using fixed per-GPU batch size: %d (Global: %d)",
        CFG.BATCH_SIZE, CFG.BATCH_SIZE * accelerator.num_processes,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    if accelerator.mixed_precision == "fp16":
        log_info("AMP (FP16 mixed precision) enabled via Accelerate")

    ckpt_path = os.path.join(args.out_dir, "cvcbm.pt")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_accum, n_train_count = 0.0, 0
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                log_info("NaN/Inf loss detected; skipping batch.")
                continue
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_accum += loss.item() * x.size(0)
            n_train_count += x.size(0)

        gathered_train = accelerator.gather(torch.tensor([train_accum, n_train_count], device=device))
        if gathered_train.dim() == 1:
            gathered_train = gathered_train.unsqueeze(0)
        train_accum = gathered_train[:, 0].sum().item()
        n_train_count = int(gathered_train[:, 1].sum().item())
        avg_train = train_accum / max(n_train_count, 1)

        model.eval()
        val_accum, n_val_count = 0.0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                pred = model(x)
                loss = criterion(pred, y)
                val_accum += loss.item() * x.size(0)
                n_val_count += x.size(0)

        gathered_val = accelerator.gather(torch.tensor([val_accum, n_val_count], device=device))
        if gathered_val.dim() == 1:
            gathered_val = gathered_val.unsqueeze(0)
        val_accum = gathered_val[:, 0].sum().item()
        n_val_count = int(gathered_val[:, 1].sum().item())
        avg_val = val_accum / max(n_val_count, 1)

        dt = time.time() - t0
        suffix = ""
        if accelerator.is_local_main_process and avg_val < best_val_loss:
            best_val_loss = avg_val
            unwrapped = accelerator.unwrap_model(model)
            torch.save(
                {
                    "model_state_dict": unwrapped.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "total_channels": total_channels,
                    "cfg": {
                        "input_len": CFG.INPUT_LEN,
                        "pred_horizon": CFG.PRED_HORIZON,
                        "kernel_sizes": CFG.KERNEL_SIZES,
                        "conv1_out_ch": CFG.CONV1_OUT_CH,
                        "conv2_out_ch": CFG.CONV2_OUT_CH,
                        "bilstm_hidden": CFG.BILSTM_HIDDEN,
                    },
                },
                ckpt_path,
            )
            suffix = " [Checkpoint Saved]"

        log_info(
            "Epoch %d/%d | train=%.8f | val=%.8f | %.1fs%s",
            epoch, epochs, avg_train, avg_val, dt, suffix,
        )

    log_info(
        "=== CVCBM training complete. Best val loss: %.8f | Saved: %s ===",
        best_val_loss, ckpt_path,
    )


if __name__ == "__main__":
    main()
