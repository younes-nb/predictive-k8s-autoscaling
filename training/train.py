import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, PREPROCESSING, TRAINING, DEFAULT_CHECKPOINT_PATH
from common.dataset import ShardedWindowsDataset
from common.models import RNNForecaster


def weighted_mse(preds, target, w=None):
    per_sample = ((preds - target) ** 2).mean(dim=1)
    if w is None:
        return per_sample.mean()
    w = w.clamp(min=0.1, max=50.0)
    return (w * per_sample).sum() / w.sum().clamp_min(1e-6)


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    torch.manual_seed(args.seed)

    train_ds = ShardedWindowsDataset(
        args.windows_dir, "train", args.input_len, args.pred_horizon, args.use_weights
    )
    val_ds = ShardedWindowsDataset(
        args.windows_dir, "val", args.input_len, args.pred_horizon, args.use_weights
    )

    if len(train_ds) > 0:
        first_x, _, *_ = train_ds[0]
        input_size = first_x.shape[-1]
    else:
        raise RuntimeError("Train dataset is empty.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=args.pred_horizon,
        rnn_type=args.rnn_type,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            if args.use_weights:
                x, y, w = batch
                w = w.to(device)
            else:
                x, y = batch
                w = None

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = weighted_mse(preds, y, w)
            loss.backward()

            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            train_loss += loss.item() * x.size(0)

        avg_train_loss = train_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if args.use_weights:
                    x, y, _ = batch
                else:
                    x, y = batch

                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = nn.MSELoss()(preds, y)
                val_loss += loss.item() * x.size(0)

        avg_val_loss = val_loss / len(val_ds) if len(val_ds) > 0 else 0.0

        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}"
        )

        if avg_val_loss < best_score:
            best_score = avg_val_loss
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "input_size": input_size,
                },
                args.checkpoint_path,
            )
            print(f"  -> Saved checkpoint to {args.checkpoint_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--use_weights", action="store_true")
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
    p.add_argument("--seed", type=int, default=TRAINING.SEED)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--rnn_type", default="lstm")

    train(p.parse_args())
