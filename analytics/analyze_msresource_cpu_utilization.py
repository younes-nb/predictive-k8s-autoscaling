import gc
import glob
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths, PREPROCESSING

DEFAULT_BINS = 128
CPU_RANGE = (0.0, 1.0)
CORR_RANGE = (-1.0, 1.0)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze CPU utilization from window shards with OOM-safe processing."
    )
    parser.add_argument(
        "--windows_dir",
        type=str,
        default=Paths.WINDOWS_DIR,
        help="Directory containing window shard files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Data split to analyze.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins for both histograms.",
    )
    parser.add_argument(
        "--cpu_lower_bound",
        type=float,
        default=None,
        help="Optional CPU utilization lower bound; samples below are excluded.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=Paths.LOGS_DIR,
        help="Directory to save plots.",
    )
    return parser.parse_args()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return float("nan")
    xs = x - x.mean()
    ys = y - y.mean()
    denom = np.sqrt((xs**2).sum()) * np.sqrt((ys**2).sum())
    if denom == 0.0:
        return float("nan")
    return float((xs * ys).sum() / denom)


def _bin_index(value: float, lo: float, hi: float, n_bins: int) -> int:
    v = max(lo, min(hi, float(value)))
    idx = int((v - lo) / (hi - lo) * n_bins)
    if idx == n_bins:
        idx -= 1
    return idx


def plot_cpu_histogram(bin_edges, counts, out_path):
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    plt.figure(figsize=(10, 6))
    plt.bar(
        centers,
        counts,
        width=width,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    plt.title("Per-Sample CPU Utilization Distribution")
    plt.xlabel("Average CPU Utilization (across input window steps)")
    plt.ylabel("Sample Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    log(f"CPU utilization histogram saved to {out_path}")


def plot_corr_histogram(bin_edges, counts, title, out_path):
    if np.sum(counts) == 0:
        log(f"No correlation counts available for {title}.")
        return

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    plt.figure(figsize=(10, 6))
    plt.bar(
        centers,
        counts,
        width=width,
        color="orchid",
        edgecolor="black",
        alpha=0.7,
    )
    plt.title(title)
    plt.xlabel("Correlation")
    plt.ylabel("Sample Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    log(f"Correlation histogram saved to {out_path}")


def main():
    args = parse_args()

    input_len = PREPROCESSING.INPUT_LEN
    pred_horizon = PREPROCESSING.PRED_HORIZON
    overlap = min(input_len, pred_horizon)

    n_bins = args.bins
    cpu_hist = np.zeros(n_bins, dtype=np.int64)
    corr_hist = np.zeros(n_bins, dtype=np.int64)

    cpu_count = 0
    corr_count = 0
    total_cpu_sum = 0.0
    total_cpu_n = 0
    corr_sum = 0.0

    pattern = os.path.join(args.windows_dir, f"part-*_X_{args.split}.npy")
    x_files = sorted(glob.glob(pattern))
    if not x_files:
        raise SystemExit(f"No window shards found matching {pattern}")

    log(f"Found {len(x_files)} shard(s) for split='{args.split}'")
    os.makedirs(args.out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cpu_hist_path = os.path.join(args.out_dir, f"cpu_utilization_hist_{ts}.png")
    corr_hist_path = os.path.join(args.out_dir, f"cpu_correlation_hist_{ts}.png")

    for x_path in x_files:
        base = x_path.rsplit("_X_", 1)[0]
        y_path = base + f"_y_{args.split}.npy"

        X = np.load(x_path, mmap_mode="r")
        Y = np.load(y_path, mmap_mode="r")
        N = X.shape[0]

        log(f"Processing {os.path.basename(x_path)}: {N} samples ...")

        cpu_col = X[:, :, 0]

        for i in range(N):
            cpu_vals = cpu_col[i]
            valid = cpu_vals[np.isfinite(cpu_vals)]
            if len(valid) == 0:
                continue
            if (
                args.cpu_lower_bound is not None
                and np.min(valid) < args.cpu_lower_bound
            ):
                continue

            total_cpu_sum += float(valid.sum())
            total_cpu_n += len(valid)

            sample_mean = float(np.mean(valid))
            cpu_hist[_bin_index(sample_mean, *CPU_RANGE, n_bins)] += 1
            cpu_count += 1

            x_tail = cpu_vals[-overlap:]
            y_tail = Y[i, -overlap:]
            r = _pearson(x_tail, y_tail)
            if np.isfinite(r):
                corr_sum += r
                corr_hist[_bin_index(r, *CORR_RANGE, n_bins)] += 1
                corr_count += 1

        del X, Y
        gc.collect()

    global_avg_cpu = total_cpu_sum / max(total_cpu_n, 1)
    avg_corr = corr_sum / max(corr_count, 1) if corr_count > 0 else float("nan")

    print("\n" + "=" * 60)
    print("  WINDOW SHARD CPU UTILIZATION SUMMARY")
    print("=" * 60)
    print(f"Split:                          {args.split}")
    print(f"Shards processed:               {len(x_files)}")
    print(f"Total samples with valid CPU:   {cpu_count}")
    print(f"Global avg CPU (all timesteps): {global_avg_cpu:.6f}")
    print(f"Samples with valid correlation: {corr_count}")
    if corr_count > 0:
        print(f"Avg correlation (last input vs horizon): {avg_corr:.6f}")
    else:
        print("Avg correlation:                  n/a")
    print("=" * 60 + "\n")

    cpu_bin_edges = np.linspace(*CPU_RANGE, n_bins + 1)
    plot_cpu_histogram(cpu_bin_edges, cpu_hist, cpu_hist_path)

    if corr_count > 0:
        corr_bin_edges = np.linspace(*CORR_RANGE, n_bins + 1)
        plot_corr_histogram(
            corr_bin_edges,
            corr_hist,
            "Per-Sample Correlation Distribution: Input Window vs Horizon",
            corr_hist_path
        )
    else:
        log("No valid correlations to plot.")

    log("Done.")


if __name__ == "__main__":
    main()
