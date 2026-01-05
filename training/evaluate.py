import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, TRAINING
from common.dataset import ShardedWindowsDataset
from common.models import RNNForecaster
from common.uncertainty import mc_dropout_predict, compute_adaptive_thresholds


def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    ckpt_args = checkpoint["args"]

    input_size = ckpt_args.get("input_size", 1)
    hidden_size = ckpt_args["hidden_size"]
    num_layers = ckpt_args["num_layers"]
    dropout = ckpt_args["dropout"]
    horizon = ckpt_args["pred_horizon"]
    rnn_type = ckpt_args["rnn_type"]
    input_len = ckpt_args["input_len"]

    test_ds = ShardedWindowsDataset(args.windows_dir, "test", input_len, horizon)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
        rnn_type=rnn_type,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    mse_sum = 0
    tp = fp = tn = fn = 0
    total_samples = 0

    print("Starting evaluation with Adaptive Thresholds (MC Dropout)...")

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

        mu, sigma = mc_dropout_predict(model, x, repeats=args.inference_repeats)

        mse_sum += ((mu - y) ** 2).sum().item()
        theta_base = torch.full(mu.shape, args.base_threshold, device=device)

        adaptive_thr = compute_adaptive_thresholds(
            theta_base,
            sigma,
            k=args.k,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
        )

        y_true_cls = y.max(dim=1).values >= adaptive_thr.max(dim=1).values
        y_pred_cls = mu.max(dim=1).values >= adaptive_thr.max(dim=1).values

        tp += (y_pred_cls & y_true_cls).sum().item()
        fp += (y_pred_cls & ~y_true_cls).sum().item()
        tn += (~y_pred_cls & ~y_true_cls).sum().item()
        fn += (~y_pred_cls & y_true_cls).sum().item()

        total_samples += x.size(0)

        if i % 10 == 0:
            print(f"Batch {i}/{len(test_loader)} processed...")

    mse = mse_sum / (total_samples * horizon)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n=== Test Results ===")
    print(f"MSE: {mse:.6f}")
    print(f"Adaptive Classification (k={args.k}):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--inference_repeats", type=int, default=10)
    p.add_argument("--base_threshold", type=float, default=0.8)
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--theta_min", type=float, default=0.60)
    p.add_argument("--theta_max", type=float, default=0.90)

    evaluate(p.parse_args())
