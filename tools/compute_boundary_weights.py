import os
import glob
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, PREPROCESSING, TRAINING

from training.train import RNNForecaster


def natural_key(p: str):
    import re

    return [
        int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", Path(p).name)
    ]


def find_shards(windows_dir: str, split: str):
    pattern = os.path.join(windows_dir, f"part-*_X_{split}.npy")
    x_files = sorted(glob.glob(pattern), key=natural_key)

    if not x_files:
        raise RuntimeError(f"No shards for split={split} found in {windows_dir}")

    shards = []
    for x_path in x_files:
        base = x_path.replace(f"_X_{split}.npy", "")
        y_path = base + f"_y_{split}.npy"
        sid_path = base + f"_sid_{split}.npy"
        if not os.path.exists(y_path) or not os.path.exists(sid_path):
            print(f"  [SKIP] Shard {base} missing Y or SID files.")
            continue
        shards.append((x_path, y_path, sid_path, base))
    return shards


def hist_update(hist: np.ndarray, values: np.ndarray, bins: int):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    values = np.clip(values, 0.0, 1.0)
    indices = (values * (bins - 1)).astype(np.int32)
    np.add.at(hist, indices, 1)


def hist_quantile(hist: np.ndarray, tau: float) -> float:
    total = hist.sum()
    if total <= 0:
        return 0.5
    cdf = np.cumsum(hist)
    target = tau * total
    bin_idx = np.searchsorted(cdf, target, side="left")
    return float(bin_idx / (len(hist) - 1))


@torch.no_grad()
def mc_dropout_sigma(
    model, X, repeats: int, horizon_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.train()
    preds = []
    for _ in range(repeats):
        out = model(X)
        preds.append(out[:, horizon_index].unsqueeze(0))

    preds_stack = torch.cat(preds, dim=0)
    return preds_stack.mean(dim=0), preds_stack.std(dim=0)


def main():
    ap = argparse.ArgumentParser(
        description="Compute Adaptive Boundary Weights for Training."
    )
    ap.add_argument("--windows_dir", default=PATHS.WINDOWS_DIR)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val"])

    ap.add_argument("--rnn_type", default="lstm")
    ap.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    ap.add_argument("--hidden_size", type=int, default=TRAINING.HIDDEN_SIZE)
    ap.add_argument("--num_layers", type=int, default=TRAINING.NUM_LAYERS)
    ap.add_argument("--dropout", type=float, default=TRAINING.DROPOUT)
    ap.add_argument("--horizon", type=int, default=PREPROCESSING.PRED_HORIZON)

    ap.add_argument(
        "--tau_base", type=float, default=0.80, help="Quantile for service baseline"
    )
    ap.add_argument("--mc_repeats", type=int, default=25)
    ap.add_argument("--k", type=float, default=2.0, help="Uncertainty multiplier")
    ap.add_argument("--gamma", type=float, default=6.0, help="Weight peak amplitude")
    ap.add_argument(
        "--delta", type=float, default=0.05, help="Width of boundary kernel"
    )
    ap.add_argument("--theta_min", type=float, default=0.60)
    ap.add_argument("--theta_max", type=float, default=0.90)

    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    shards = find_shards(args.windows_dir, args.split)

    hists: Dict[int, np.ndarray] = {}
    bins = 1000

    print(f"Pass 1: Building histograms for {len(shards)} shards...")
    for x_path, _, sid_path, _ in shards:
        X = np.load(x_path, mmap_mode="r")
        sid = np.load(sid_path, mmap_mode="r")

        u_now = X[:, -1, -1] if X.ndim == 3 else X[:, -1]

        for s in np.unique(sid):
            if s not in hists:
                hists[s] = np.zeros(bins, dtype=np.int64)
            hist_update(hists[s], u_now[sid == s], bins)

    theta_base = {int(s): hist_quantile(h, args.tau_base) for s, h in hists.items()}

    sample_x = np.load(shards[0][0], mmap_mode="r")
    input_size = sample_x.shape[2] if sample_x.ndim == 3 else 1

    model = RNNForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=args.horizon,
        rnn_type=args.rnn_type,
    ).to(device)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from {args.checkpoint_path}")

    print(f"Pass 2: Computing Gaussian boundary weights...")
    for x_path, y_path, sid_path, base in shards:
        X = np.load(x_path, mmap_mode="r")
        Y = np.load(y_path, mmap_mode="r")
        sid = np.load(sid_path, mmap_mode="r")

        out_w_path = base + f"_w_{args.split}.npy"
        W = np.lib.format.open_memmap(
            out_w_path, mode="w+", dtype=np.float32, shape=(X.shape[0],)
        )

        for i in range(0, X.shape[0], args.batch_size):
            end = i + args.batch_size

            x_batch = torch.from_numpy(X[i:end].copy()).float().to(device)
            y_batch = torch.from_numpy(Y[i:end].copy()).float().to(device)

            sid_batch = sid[i:end]

            y_target = y_batch[:, -1]

            mu, sigma = mc_dropout_sigma(model, x_batch, args.mc_repeats, -1)

            tb = torch.tensor(
                [theta_base.get(int(s), 0.7) for s in sid_batch], device=device
            )

            theta = torch.clamp(
                tb - args.k * sigma, min=args.theta_min, max=args.theta_max
            )

            dist_sq = (y_target - theta) ** 2
            w = 1.0 + args.gamma * torch.exp(-dist_sq / (2.0 * (args.delta**2)))

            W[i : i + len(w)] = w.detach().cpu().numpy()

        W.flush()
        print(f"  Saved: {Path(out_w_path).name}")

    print("Weight computation complete.")


if __name__ == "__main__":
    main()
