import argparse
import logging
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


class FastTensorDataLoader:
    def __init__(self, dataset, batch_size=64, shuffle=False, device=None):
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_len = len(dataset)
        self._cpu_fallback = False
        if device is not None:
            try:
                self.X = dataset.X.to(device)
                self.y = dataset.y.to(device)
                self.last = dataset.last.to(device)
            except Exception:
                logging.warning(
                    "Failed to move dataset to GPU; falling back to CPU storage "
                    "with per-batch transfer."
                )
                self._cpu_fallback = True
                self.X = dataset.X
                self.y = dataset.y
                self.last = dataset.last
        else:
            self.X = dataset.X
            self.y = dataset.y
            self.last = dataset.last

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = torch.arange(self.dataset_len)
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.dataset_len:
            raise StopIteration
        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        self.idx += self.batch_size
        if self._cpu_fallback:
            x = self.X[batch_indices].to(self.device, non_blocking=True)
            y = self.y[batch_indices].to(self.device, non_blocking=True)
            last = self.last[batch_indices].to(self.device, non_blocking=True)
        else:
            x = self.X[batch_indices]
            y = self.y[batch_indices]
            last = self.last[batch_indices]
        return x, y, last

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.tsdp.config import CFG as GLOBAL_CFG
from experiments.tsdp.cnn_bilstm.config import CFG as ARCH_CFG
from experiments.tsdp.dataset import TsdpDataset, N_CHANNELS
from experiments.tsdp.cnn_bilstm.model import CnnBiLSTM

ARCH_NAME = "CNN-BiLSTM"


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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = setup_logging(args.log_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    epochs = args.epochs if args.epochs is not None else GLOBAL_CFG.EPOCHS

    logging.info("=" * 60)
    logging.info("TSDP — %s — Training on %s", ARCH_NAME, device)
    logging.info("Architecture config: %s", ARCH_CFG)
    logging.info("Log file: %s", log_path)
    logging.info("=" * 60)

    train_ds = TsdpDataset(
        args.preprocess_dir, "train",
        input_len=GLOBAL_CFG.INPUT_LEN, pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        stride=GLOBAL_CFG.STRIDE,
        train_frac=GLOBAL_CFG.TRAIN_FRAC, val_frac=GLOBAL_CFG.VAL_FRAC,
    )
    val_ds = TsdpDataset(
        args.preprocess_dir, "val",
        input_len=GLOBAL_CFG.INPUT_LEN, pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        stride=GLOBAL_CFG.STRIDE,
        train_frac=GLOBAL_CFG.TRAIN_FRAC, val_frac=GLOBAL_CFG.VAL_FRAC,
    )

    if len(train_ds) == 0:
        logging.error("Empty training dataset. Aborting.")
        return

    n_train = len(train_ds)
    n_val = len(val_ds)

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    model = CnnBiLSTM(
        in_channels=N_CHANNELS,
        input_len=GLOBAL_CFG.INPUT_LEN,
        pred_horizon=GLOBAL_CFG.PRED_HORIZON,
        kernel_sizes=ARCH_CFG.KERNEL_SIZES,
        conv1_out_ch=ARCH_CFG.CONV1_OUT_CH,
        conv2_out_ch=ARCH_CFG.CONV2_OUT_CH,
        bilstm_hidden=ARCH_CFG.BILSTM_HIDDEN,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=GLOBAL_CFG.LEARNING_RATE)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model parameters: %d", n_params)
    logging.info("Train windows: %d | Val windows: %d", n_train, n_val)
    logging.info("Epochs: %d", epochs)

    train_loader = FastTensorDataLoader(
        train_ds, batch_size=GLOBAL_CFG.BATCH_SIZE, shuffle=True, device=device,
    )
    val_loader = FastTensorDataLoader(
        val_ds, batch_size=GLOBAL_CFG.BATCH_SIZE * 2, shuffle=False, device=device,
    )
    if not val_loader._cpu_fallback:
        logging.info("Moving val data to CPU fallback to free GPU memory.")
        val_loader.X = val_loader.X.cpu()
        val_loader.y = val_loader.y.cpu()
        val_loader.last = val_loader.last.cpu()
        val_loader._cpu_fallback = True
    del train_ds, val_ds

    ckpt_path = os.path.join(args.out_dir, "tsdp.pt")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_accum, n_train_count = 0.0, 0
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            with autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
                loss = criterion(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("NaN/Inf loss detected; skipping batch.")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GLOBAL_CFG.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            train_accum += loss.item() * x.size(0)
            n_train_count += x.size(0)
        avg_train = train_accum / max(n_train_count, 1)

        model.eval()
        val_accum, n_val_count = 0.0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                with autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(x)
                    loss = criterion(pred, y)
                val_accum += loss.item() * x.size(0)
                n_val_count += x.size(0)
        avg_val = val_accum / max(n_val_count, 1)

        dt = time.time() - t0
        suffix = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "arch": ARCH_NAME,
                    "cfg": {
                        "input_len": GLOBAL_CFG.INPUT_LEN,
                        "pred_horizon": GLOBAL_CFG.PRED_HORIZON,
                        "n_channels": N_CHANNELS,
                        "kernel_sizes": ARCH_CFG.KERNEL_SIZES,
                        "conv1_out_ch": ARCH_CFG.CONV1_OUT_CH,
                        "conv2_out_ch": ARCH_CFG.CONV2_OUT_CH,
                        "bilstm_hidden": ARCH_CFG.BILSTM_HIDDEN,
                    },
                },
                ckpt_path,
            )
            suffix = " [Checkpoint Saved]"

        logging.info(
            "Epoch %d/%d | train=%.8f | val=%.8f | %.1fs%s",
            epoch, epochs, avg_train, avg_val, dt, suffix,
        )

    logging.info(
        "=== TSDP [%s] training complete. Best val loss: %.8f | Saved: %s ===",
        ARCH_NAME, best_val_loss, ckpt_path,
    )


if __name__ == "__main__":
    main()
