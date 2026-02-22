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
    p = argparse.ArgumentParser(description="MC Dropout K-Factor Calibration")
    p.add_argument("--windows_dir", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--batch_size", type=int, default=1)  # Small batch for MC repeats
    p.add_argument(
        "--coverage", type=float, default=0.95, help="Target confidence (e.g. 0.95)"
    )
    p.add_argument(
        "--mc_repeats", type=int, default=25, help="Must match deployment config"
    )
    p.add_argument("--out", type=str, default="k_mc_calibration.png")

    args = p.parse_args()
    setup_logging("k_mc_calibration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    test_ds = ShardedWindowsDataset(
        args.windows_dir,
        "test",
        ckpt_args.get("input_len", 60),
        ckpt_args.get("pred_horizon", 5),
    )

    model = RNNForecaster(
        input_size=ckpt_args.get("input_size", 1),
        hidden_size=ckpt_args.get("hidden_size", 128),
        num_layers=ckpt_args.get("num_layers", 3),
        dropout=ckpt_args.get("dropout", 0.3),
        horizon=ckpt_args.get("pred_horizon", 5),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.train()

    test_loader = DataLoader(test_ds, batch_size=1)

    z_scores = []

    logging.info(
        f"Calibrating K using {args.mc_repeats} MC passes on {len(test_ds)} samples..."
    )

    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y_true = y[0, -1].item()

        x_repeated = x.repeat(args.mc_repeats, 1, 1)

        preds = model(x_repeated)

        final_step_preds = preds[:, -1].cpu().numpy()

        mu_mc = np.mean(final_step_preds)
        sigma_mc = np.std(final_step_preds)

        error = np.abs(y_true - mu_mc)

        if sigma_mc > 1e-6:
            z = error / sigma_mc
            z_scores.append(z)

        if i % 100 == 0:
            logging.info(f"Processed {i}/{len(test_ds)} samples...")

    k_emp = np.percentile(z_scores, args.coverage * 100)
    k_theory = norm.ppf(1 - (1 - args.coverage) / 2)

    logging.info("\n" + "=" * 45)
    logging.info("🛡️  MC DROPOUT K-CALIBRATION RESULTS")
    logging.info("=" * 45)
    logging.info(f"Target Coverage:       {args.coverage*100}%")
    logging.info(f"Empirical K-Factor:    {k_emp:.4f}")
    logging.info(f"Theoretical K (Norm):  {k_theory:.4f}")
    logging.info(f"Usage: Thr = Base - ({k_emp:.2f} * sigma_mc)")
    logging.info("=" * 45)

    plt.figure(figsize=(10, 6))
    plt.hist(
        z_scores,
        bins=100,
        density=True,
        alpha=0.7,
        color="royalblue",
        label="Empirical Z-scores ($|Error| / \sigma_{MC}$)",
    )
    plt.axvline(
        k_emp, color="orange", linestyle="--", lw=3, label=f"Calibrated K={k_emp:.2f}"
    )
    plt.title(f"Standardized Residuals via MC Dropout ({args.coverage*100}% Coverage)")
    plt.xlabel("Z-Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(args.out)
    logging.info(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
