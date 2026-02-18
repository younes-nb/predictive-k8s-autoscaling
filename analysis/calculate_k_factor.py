import os
import sys
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import norm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, TRAINING
from common.dataset import ShardedWindowsDataset
from common.models import RNNForecaster
from training.evaluate import setup_logging


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(
        description="Residual Calibration for Deterministic RNN"
    )
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--batch_size", type=int, default=TRAINING.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=TRAINING.NUM_WORKERS)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--out", type=str, default="k_factor_analysis.png")

    args = p.parse_args()
    setup_logging("k_factor_calc")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    test_ds = ShardedWindowsDataset(
        args.windows_dir,
        "test",
        ckpt_args.get("input_len", 60),
        ckpt_args.get("pred_horizon", 5),
    )
    first_x, _ = test_ds[0]
    input_size = first_x.shape[-1] if first_x.ndim > 1 else 1

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=ckpt_args.get("hidden_size", TRAINING.HIDDEN_SIZE),
        num_layers=ckpt_args.get("num_layers", TRAINING.NUM_LAYERS),
        dropout=ckpt_args.get("dropout", TRAINING.DROPOUT),
        horizon=ckpt_args.get("pred_horizon", 5),
        rnn_type=ckpt_args.get("rnn_type", "lstm"),
        bidirectional=ckpt_args.get("bidirectional", TRAINING.BIDIRECTIONAL),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    all_y_true = []
    all_y_pred = []

    logging.info(f"Running inference for K-calibration on {len(test_ds)} samples...")
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        mu = model(x)

        all_y_true.append(y.cpu().numpy().flatten())
        all_y_pred.append(mu.cpu().numpy().flatten())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)

    rmse = np.sqrt(np.mean(residuals**2))

    z_scores = abs_residuals / (rmse + 1e-6)

    k_emp = np.percentile(z_scores, args.coverage * 100)

    k_theory = norm.ppf(1 - (1 - args.coverage) / 2)

    logging.info("\n" + "=" * 45)
    logging.info("🛡️  K-FACTOR CALIBRATION RESULTS")
    logging.info("=" * 45)
    logging.info(f"Model RMSE (Global Sigma): {rmse:.6f}")
    logging.info(f"Empirical K-Factor:        {k_emp:.4f}")
    logging.info(f"Theoretical K (Normal):    {k_theory:.4f}")
    logging.info("-" * 45)
    logging.info(f"Final Coverage Margin:     ±{k_emp:.2f} * RMSE")
    logging.info("=" * 45)

    plt.figure(figsize=(10, 6))
    plt.hist(
        z_scores,
        bins=100,
        density=True,
        alpha=0.7,
        color="teal",
        label="Empirical standardized residuals",
    )

    x_plot = np.linspace(0, max(6, k_emp + 1), 200)
    plt.plot(
        x_plot, 2 * norm.pdf(x_plot, 0, 1), "r--", lw=2, label="Theoretical Half-Normal"
    )

    plt.axvline(k_emp, color="gold", lw=3, label=f"Empirical K ({k_emp:.2f})")
    plt.title(f"Residual Distribution Calibration (Target: {args.coverage*100}%)")
    plt.xlabel("Z-score (|Error| / RMSE)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(args.out)
    logging.info(f"Calibration plot saved to {args.out}")


if __name__ == "__main__":
    main()
