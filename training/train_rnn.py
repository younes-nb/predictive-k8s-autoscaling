import os
import glob
import sys
import time
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, PREPROCESSING, TRAINING, DEFAULT_CHECKPOINT_PATH


class ShardedWindowsDataset(Dataset):
    def __init__(self, windows_dir, split, input_len, horizon):
        self.input_len = input_len
        self.horizon = horizon
        self.split = split
        self.shards = []
        self.lengths = []
        self.cum_lengths = []

        pattern = os.path.join(windows_dir, f"part-*_X_{split}.npy")
        x_files = sorted(glob.glob(pattern))
        if not x_files:
            raise RuntimeError(f"No X_{split}.npy files found under {windows_dir}")

        total = 0
        for x_path in x_files:
            base = x_path.replace(f"_X_{split}.npy", "")
            y_path = base + f"_y_{split}.npy"
            if not os.path.exists(y_path):
                raise RuntimeError(
                    f"Missing matching y file for {x_path}: expected {y_path}"
                )

            X = np.load(x_path, mmap_mode="r")
            Y = np.load(y_path, mmap_mode="r")

            if X.shape[0] != Y.shape[0]:
                raise RuntimeError(f"Shape mismatch between {x_path} and {y_path}")

            if X.ndim == 2:
                if X.shape[1] != input_len:
                    print(
                        f"[WARN] {x_path} has shape {X.shape}, expected (*, {input_len})"
                    )
            elif X.ndim == 3:
                if X.shape[1] != input_len:
                    print(
                        f"[WARN] {x_path} has shape {X.shape}, expected (*, {input_len}, F)"
                    )
            else:
                raise RuntimeError(
                    f"[ERROR] {x_path} has unsupported ndim={X.ndim} (shape={X.shape})"
                )

            if Y.ndim != 2 or Y.shape[1] != horizon:
                print(f"[WARN] {y_path} has shape {Y.shape}, expected (*, {horizon})")

            self.shards.append((X, Y))
            self.lengths.append(X.shape[0])
            total += X.shape[0]
            self.cum_lengths.append(total)

        self.total_len = total
        print(
            f"[{split}] Loaded {len(self.shards)} shard(s), total windows: {self.total_len}"
        )

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError(idx)

        for shard_idx, cum in enumerate(self.cum_lengths):
            if idx < cum:
                prev_cum = 0 if shard_idx == 0 else self.cum_lengths[shard_idx - 1]
                local_idx = idx - prev_cum
                X, Y = self.shards[shard_idx]

                x_arr = np.array(X[local_idx], copy=True)
                y_arr = np.array(Y[local_idx], copy=True)

                if x_arr.ndim == 1:
                    x_arr = x_arr[:, None]

                x_tensor = torch.from_numpy(x_arr).float()
                y_tensor = torch.from_numpy(y_arr).float()
                return x_tensor, y_tensor

        raise IndexError(idx)


class RNNForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 5,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.horizon = horizon
        self.rnn_type = rnn_type.lower()
        if self.rnn_type not in ("lstm", "gru"):
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}")

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print("Using device:", device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    run_ts = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(PATHS.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(PATHS.LOGS_DIR, f"train_{run_ts}.log")
    print(f"Epoch summaries will be logged to: {log_path}")

    print("Loading datasets from:", args.windows_dir)
    print(
        f"  input_len={args.input_len}, horizon={args.pred_horizon}, rnn_type={args.rnn_type}"
    )

    train_dataset = ShardedWindowsDataset(
        windows_dir=args.windows_dir,
        split="train",
        input_len=args.input_len,
        horizon=args.pred_horizon,
    )

    try:
        val_dataset = ShardedWindowsDataset(
            windows_dir=args.windows_dir,
            split="val",
            input_len=args.input_len,
            horizon=args.pred_horizon,
        )
    except RuntimeError as e:
        print("[WARN] Could not load val split:", e)
        val_dataset = None

    first_X, _ = train_dataset.shards[0]
    if first_X.ndim == 3:
        input_size = int(first_X.shape[2])
    else:
        input_size = 1

    print(f"Inferred input_size={input_size} feature(s) from windows")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
    else:
        print("[INFO] No validation data available, skipping validation.")

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=args.pred_horizon,
        rnn_type=args.rnn_type,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_score = float("inf")

    ckpt_dir = os.path.dirname(args.checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(f"Run timestamp: {run_ts}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Windows dir: {args.windows_dir}\n")
        f.write(f"Checkpoint path: {args.checkpoint_path}\n")
        f.write(f"Hyperparams: {vars(args)}\n")
        f.write(f"Inferred input_size: {input_size}\n")
        f.write("-" * 60 + "\n")

    print("Model:")
    print(model)
    print("Checkpoint will be saved to:", args.checkpoint_path)
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        num_batches = len(train_loader)
        print(f"\nEpoch {epoch}/{args.epochs} - {num_batches} batches")

        epoch_train_start = time.time()

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader, start=1):
            batch_start = time.time()

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            batch_time = time.time() - batch_start

            bs = X_batch.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

            if batch_idx % args.log_interval == 0 or batch_idx == num_batches:
                avg_so_far = train_loss_sum / max(1, train_count)
                print(
                    f"  [Epoch {epoch:03d}] "
                    f"Batch {batch_idx:06d}/{num_batches:06d} "
                    f"Loss (batch)={loss.item():.6f} "
                    f"Avg (so far)={avg_so_far:.6f} "
                    f"BatchTime={batch_time:.3f}s"
                )

        train_time_epoch = time.time() - epoch_train_start
        train_loss = train_loss_sum / max(1, train_count)

        val_loss = None
        val_time_epoch = None

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0

            val_start = time.time()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    bs = X_batch.size(0)
                    val_loss_sum += loss.item() * bs
                    val_count += bs

            val_time_epoch = time.time() - val_start
            val_loss = val_loss_sum / max(1, val_count)

        if val_loss is not None:
            summary = (
                f"Epoch {epoch:03d} SUMMARY: "
                f"train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}, "
                f"train_time={train_time_epoch:.2f}s, "
                f"val_time={val_time_epoch:.2f}s"
            )
        else:
            summary = (
                f"Epoch {epoch:03d} SUMMARY: "
                f"train_loss={train_loss:.6f}, "
                f"train_time={train_time_epoch:.2f}s"
            )

        print(summary)
        with open(log_path, "a") as f:
            f.write(summary + "\n")

        score = val_loss if val_loss is not None else train_loss
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_time_epoch": train_time_epoch,
                    "val_time_epoch": val_time_epoch,
                    "args": {**vars(args), "input_size": input_size},
                },
                args.checkpoint_path,
            )
            print(
                f"  → Saved new best model (score={score:.6f}) to {args.checkpoint_path}"
            )

    final_msg = f"Training complete. Best score: {best_score:.6f}"
    print("\n" + final_msg)
    with open(log_path, "a") as f:
        f.write(final_msg + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train RNN (LSTM/GRU) on windowed datasets."
    )
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--pred_horizon", type=int, default=PREPROCESSING.PRED_HORIZON)
    p.add_argument("--hidden_size", type=int, default=TRAINING.HIDDEN_SIZE)
    p.add_argument("--num_layers", type=int, default=TRAINING.NUM_LAYERS)
    p.add_argument("--dropout", type=float, default=TRAINING.DROPOUT)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=TRAINING.EPOCHS)
    p.add_argument("--lr", type=float, default=TRAINING.LR)
    p.add_argument("--grad_clip", type=float, default=TRAINING.GRAD_CLIP)
    p.add_argument("--num_workers", type=int, default=TRAINING.NUM_WORKERS)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=TRAINING.SEED)
    p.add_argument("--log_interval", type=int, default=TRAINING.LOG_INTERVAL)
    p.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
