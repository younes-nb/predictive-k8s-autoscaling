import os
import glob
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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
                raise RuntimeError(f"Missing matching y file for {x_path}: expected {y_path}")

            X = np.load(x_path, mmap_mode="r")
            Y = np.load(y_path, mmap_mode="r")

            if X.shape[0] != Y.shape[0]:
                raise RuntimeError(f"Shape mismatch between {x_path} and {y_path}")

            if X.ndim != 2 or X.shape[1] != input_len:
                print(f"[WARN] {x_path} has shape {X.shape}, expected (*, {input_len})")
            if Y.ndim != 2 or Y.shape[1] != horizon:
                print(f"[WARN] {y_path} has shape {Y.shape}, expected (*, {horizon})")

            self.shards.append((X, Y))
            self.lengths.append(X.shape[0])
            total += X.shape[0]
            self.cum_lengths.append(total)

        self.total_len = total
        print(f"[{split}] Loaded {len(self.shards)} shard(s), total windows: {self.total_len}")

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
                x_arr = X[local_idx]
                y_arr = Y[local_idx]

                x_tensor = torch.from_numpy(x_arr).float().unsqueeze(-1)
                y_tensor = torch.from_numpy(y_arr).float()
                return x_tensor, y_tensor

        raise IndexError(idx)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.1, horizon=5):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        returns: (batch, horizon)
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.fc(last)
        return pred


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("Loading datasets from:", args.windows_dir)
    print(f"  input_len={args.input_len}, horizon={args.pred_horizon}")

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

    model = LSTMForecaster(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=args.pred_horizon,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_score = float("inf")

    ckpt_dir = os.path.dirname(args.checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

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

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader, start=1):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            batch_size = X_batch.size(0)
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

            if batch_idx % args.log_interval == 0 or batch_idx == num_batches:
                avg_so_far = train_loss_sum / max(1, train_count)
                print(
                    f"  [Epoch {epoch:03d}] "
                    f"Batch {batch_idx:06d}/{num_batches:06d} "
                    f"Loss (batch)={loss.item():.6f} "
                    f"Avg (so far)={avg_so_far:.6f}"
                )

        train_loss = train_loss_sum / max(1, train_count)

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    batch_size = X_batch.size(0)
                    val_loss_sum += loss.item() * batch_size
                    val_count += batch_size

            val_loss = val_loss_sum / max(1, val_count)
        else:
            val_loss = None

        if val_loss is not None:
            print(f"Epoch {epoch:03d} SUMMARY: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d} SUMMARY: train_loss={train_loss:.6f}")

        score = val_loss if val_loss is not None else train_loss
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                args.checkpoint_path,
            )
            print(f"  → Saved new best model (score={score:.6f}) to {args.checkpoint_path}")

    print("\nTraining complete. Best score:", best_score)


def parse_args():
    p = argparse.ArgumentParser(description="Train LSTM on CPU windows.")
    p.add_argument(
        "--windows_dir",
        required=True,
        help="Directory with part-*_X_{train,val,test}.npy and part-*_y_{train,val,test}.npy",
    )
    p.add_argument("--input_len", type=int, default=60)
    p.add_argument("--pred_horizon", type=int, default=5)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--cpu",
        action="store_true",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log_interval",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="/dataset1/alibaba_v2022/models/lstm_cpu_baseline.pt",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
