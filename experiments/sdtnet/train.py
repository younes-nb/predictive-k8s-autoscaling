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
            except torch.cuda.OutOfMemoryError:
                logging.warning(
                    "CUDA OOM moving dataset to GPU; falling back to CPU storage "
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
        if self._cpu_fallback:
            self.X = self.X.to(self.device)
            self.y = self.y.to(self.device)
            self.last = self.last.to(self.device)
            self._cpu_fallback = False

        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.device)
        else:
            self.indices = torch.arange(self.dataset_len, device=self.device)
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.dataset_len:
            raise StopIteration
        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        self.idx += self.batch_size
        x = self.X[batch_indices]
        y = self.y[batch_indices]
        last = self.last[batch_indices]
        return x, y, last

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.sdtnet.config import CFG
from experiments.sdtnet.dataset import SdtnetDataset, N_CHANNELS
from experiments.sdtnet.model import ChannelIndependentTCNForecaster


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
        description="Train SDT-Net model (SVMD-DE-TCN)."
    )
    ap.add_argument("--preprocess_dir", default="/dataset/sdtnet_preprocess")
    ap.add_argument("--out_dir", default="/proj/k8sautoscaledl-PG0/models/sdtnet")
    ap.add_argument("--log_dir", default="/proj/k8sautoscaledl-PG0/logs/sdtnet",
                    help="Directory for training logs (default: /proj/k8sautoscaledl-PG0/logs/sdtnet)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override CFG.EPOCHS for smoke tests without editing config.py")
    ap.add_argument("--lr_schedule", type=lambda x: x.lower() == "true", default=None,
                    help="Override CFG.USE_LR_SCHEDULER")
    ap.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = setup_logging(args.log_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    epochs = args.epochs if args.epochs is not None else CFG.EPOCHS

    logging.info("=" * 60)
    logging.info("SDT-Net — Model Training on %s", device)
    logging.info("Log file: %s", log_path)
    logging.info("=" * 60)

    train_ds = SdtnetDataset(
        args.preprocess_dir, "train",
        input_len=CFG.INPUT_LEN, pred_horizon=CFG.PRED_HORIZON,
        stride=CFG.STRIDE,
        train_frac=CFG.TRAIN_FRAC, val_frac=CFG.VAL_FRAC,
    )
    val_ds = SdtnetDataset(
        args.preprocess_dir, "val",
        input_len=CFG.INPUT_LEN, pred_horizon=CFG.PRED_HORIZON,
        stride=CFG.STRIDE,
        train_frac=CFG.TRAIN_FRAC, val_frac=CFG.VAL_FRAC,
    )

    if len(train_ds) == 0:
        logging.error("Empty training dataset. Aborting.")
        return

    n_train = len(train_ds)
    n_val = len(val_ds)

    train_loader = FastTensorDataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, device=device,
    )
    val_loader = FastTensorDataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False, device=device,
    )

    del train_ds, val_ds

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    model = ChannelIndependentTCNForecaster(
        total_channels=N_CHANNELS,
        input_len=CFG.INPUT_LEN,
        pred_horizon=CFG.PRED_HORIZON,
        num_filters=CFG.TCN_NUM_FILTERS,
        kernel_size=CFG.TCN_KERNEL_SIZE,
        dilations=CFG.TCN_DILATIONS,
        dropout=CFG.TCN_DROPOUT,
        residual_prediction=CFG.RESIDUAL_PREDICTION,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    use_scheduler = args.lr_schedule if args.lr_schedule is not None else CFG.USE_LR_SCHEDULER
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_scheduler else None
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model parameters: %d", n_params)
    logging.info("Train windows: %d | Val windows: %d", n_train, n_val)
    logging.info("Epochs: %d", epochs)

    ckpt_path = os.path.join(args.out_dir, "sdtnet.pt")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_accum, n_train_count = 0.0, 0
        for x, y, last_val in train_loader:
            optimizer.zero_grad()
            with autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
                mse_loss = criterion(pred, y)
                delta_loss = criterion(
                    pred - last_val.unsqueeze(1),
                    y - last_val.unsqueeze(1),
                )
                loss = mse_loss + CFG.DELTA_LOSS_WEIGHT * delta_loss

                if CFG.DIRECTION_LOSS_WEIGHT > 0:
                    y_dir = torch.sign(y[:, -1] - last_val)
                    pred_dir = torch.sign(pred[:, -1] - last_val)
                    direction_loss = torch.relu(-y_dir * pred_dir).mean()
                    loss += CFG.DIRECTION_LOSS_WEIGHT * direction_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("NaN/Inf loss detected; skipping batch.")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            train_accum += loss.item() * x.size(0)
            n_train_count += x.size(0)
        avg_train = train_accum / max(n_train_count, 1)

        if scheduler is not None:
            scheduler.step()

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
                    "cfg": {
                        "input_len": CFG.INPUT_LEN,
                        "pred_horizon": CFG.PRED_HORIZON,
                        "stride": CFG.STRIDE,
                        "total_channels": CFG.TOTAL_CHANNELS,
                        "tcn_num_filters": CFG.TCN_NUM_FILTERS,
                        "tcn_kernel_size": CFG.TCN_KERNEL_SIZE,
                        "tcn_dilations": CFG.TCN_DILATIONS,
                        "tcn_dropout": CFG.TCN_DROPOUT,
                        "residual_prediction": CFG.RESIDUAL_PREDICTION,
                        "delta_loss_weight": CFG.DELTA_LOSS_WEIGHT,
                        "direction_loss_weight": CFG.DIRECTION_LOSS_WEIGHT,
                        "learning_rate": CFG.LEARNING_RATE,
                        "batch_size": CFG.BATCH_SIZE,
                        "weight_decay": CFG.WEIGHT_DECAY,
                        "grad_clip_norm": CFG.GRAD_CLIP_NORM,
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
        "=== SDT-Net training complete. Best val loss: %.8f | Saved: %s ===",
        best_val_loss, ckpt_path,
    )


if __name__ == "__main__":
    main()
