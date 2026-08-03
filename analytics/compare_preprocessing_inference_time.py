#!/usr/bin/env python3
"""Benchmark per-window inference time of SWT vs CSKV preprocessing."""

import argparse
import glob
import os
import sys
import time
from dataclasses import replace

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from preprocessing.swt.config import CFG as SWT_CFG
from preprocessing.swt.decomposition import decompose_window
from preprocessing.cskv.config import CFG as CSKV_CFG, set_seed
from preprocessing.cskv.decomposition import decompose_service_signal
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.config_training_defaults import TRAINING


def generate_windows(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, length)
    base = (
        np.sin(2 * np.pi * t / 15)
        + 0.5 * np.sin(2 * np.pi * t / 5)
        + 0.3 * np.sin(2 * np.pi * t / 8)
    )
    windows = np.empty((n, length), dtype=np.float64)
    for i in range(n):
        windows[i] = base + 0.1 * rng.standard_normal(length)
    return windows


def load_windows(windows_dir: str, n: int, rng: np.random.Generator) -> np.ndarray:
    npy_files = sorted(glob.glob(os.path.join(windows_dir, "service_*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No service_*.npy files found in {windows_dir}")

    chosen_files = rng.choice(len(npy_files), size=min(n, len(npy_files)), replace=False)
    windows = []
    for idx in chosen_files:
        data = np.load(npy_files[idx])  # shape (N_timesteps, window_len)
        row = rng.integers(data.shape[0])
        windows.append(data[row])
    return np.stack(windows, axis=0)


def benchmark_swt(windows: np.ndarray, cfg) -> list[float]:
    times = []
    for i in range(len(windows)):
        t0 = time.perf_counter()
        decompose_window(windows[i], cfg)
        times.append(time.perf_counter() - t0)
    return times


def benchmark_cskv(windows: np.ndarray, cfg) -> list[float]:
    set_seed(TRAINING.SEED)
    times = []
    for i in range(len(windows)):
        t0 = time.perf_counter()
        decompose_service_signal(windows[i], cfg)
        times.append(time.perf_counter() - t0)
    return times


def compute_stats(times_ms: np.ndarray) -> dict:
    return {
        "avg": np.mean(times_ms),
        "p50": np.percentile(times_ms, 50),
        "p95": np.percentile(times_ms, 95),
        "min": np.min(times_ms),
        "max": np.max(times_ms),
    }


def print_table(n_windows: int, window_len: int, swt_stats: dict, cskv_stats: dict):
    print(f"\nPreprocessing Inference Time Benchmark (N={n_windows}, window_len={window_len})")
    print("=" * 74)
    header = f"{'Approach':<10} {'AVG(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
    print(header)
    print("-" * 74)
    for label, stats in [("SWT", swt_stats), ("CSKV", cskv_stats)]:
        print(
            f"{label:<10} {stats['avg']:>10.2f} {stats['p50']:>10.2f} "
            f"{stats['p95']:>10.2f} {stats['min']:>10.2f} {stats['max']:>10.2f}"
        )
    print("=" * 74)
    if swt_stats["avg"] > 0:
        ratio = cskv_stats["avg"] / swt_stats["avg"]
        slower, faster = ("CSKV", "SWT") if ratio > 1 else ("SWT", "CSKV")
        print(f"\n{slower} is {ratio:.2f}x slower than {faster} on average")
    else:
        print("\nSWT avg is zero; cannot compute ratio")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_windows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--windows_dir",
        default="/dataset/decomp_analysis/cpu/windows_64",
        help="Directory of service_*.npy files to load real windows from.",
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Ignore --windows_dir and generate synthetic windows instead.",
    )

    swt_g = parser.add_argument_group("SWT config overrides")
    swt_g.add_argument("--swt_input_len", type=int, default=None)
    swt_g.add_argument("--swt_pred_horizon", type=int, default=None)
    swt_g.add_argument("--swt_swt_level", type=int, default=None)

    cskv_g = parser.add_argument_group("CSKV config overrides")
    cskv_g.add_argument("--cskv_input_len", type=int, default=None)
    cskv_g.add_argument("--cskv_pred_horizon", type=int, default=None)
    cskv_g.add_argument("--cskv_n_clusters", type=int, default=None)
    cskv_g.add_argument("--cskv_ceemdan_trials", type=int, default=None)
    cskv_g.add_argument("--cskv_ceemdan_epsilon", type=float, default=None)
    cskv_g.add_argument("--cskv_vmd_k", type=int, default=None)

    args = parser.parse_args()

    swt_overrides = {}
    cskv_overrides = {}

    if args.swt_input_len is not None:
        swt_overrides["INPUT_LEN"] = args.swt_input_len
    if args.swt_pred_horizon is not None:
        swt_overrides["PRED_HORIZON"] = args.swt_pred_horizon
    if args.swt_swt_level is not None:
        swt_overrides["SWT_LEVEL"] = args.swt_swt_level

    if args.cskv_input_len is not None:
        cskv_overrides["INPUT_LEN"] = args.cskv_input_len
    if args.cskv_pred_horizon is not None:
        cskv_overrides["PRED_HORIZON"] = args.cskv_pred_horizon
    if args.cskv_n_clusters is not None:
        cskv_overrides["N_CLUSTERS"] = args.cskv_n_clusters
    if args.cskv_ceemdan_trials is not None:
        cskv_overrides["CEEMDAN_TRIALS"] = args.cskv_ceemdan_trials
    if args.cskv_ceemdan_epsilon is not None:
        cskv_overrides["CEEMDAN_EPSILON"] = args.cskv_ceemdan_epsilon
    if args.cskv_vmd_k is not None:
        cskv_overrides["VMD_K"] = args.cskv_vmd_k

    rng = np.random.default_rng(args.seed)

    if args.use_synthetic:
        synth_len = swt_overrides.get("INPUT_LEN", PREPROCESSING.INPUT_LEN)
        windows = generate_windows(args.n_windows, synth_len, rng)
        window_len = synth_len
    else:
        windows = load_windows(args.windows_dir, args.n_windows, rng)
        window_len = windows.shape[1]

    swt_overrides.setdefault("INPUT_LEN", window_len)
    cskv_overrides.setdefault("INPUT_LEN", window_len)

    swt_cfg = replace(SWT_CFG, **swt_overrides)
    cskv_cfg = replace(CSKV_CFG, **cskv_overrides)

    print(f"Running SWT benchmark ({args.n_windows} windows, len={window_len})...")
    swt_times = benchmark_swt(windows, swt_cfg)
    swt_stats = compute_stats(np.array(swt_times) * 1000)
    print(f"  done ({swt_stats['avg']:.2f} ms avg)")

    print(f"Running CSKV benchmark ({args.n_windows} windows, len={window_len})...")
    cskv_times = benchmark_cskv(windows, cskv_cfg)
    cskv_stats = compute_stats(np.array(cskv_times) * 1000)
    print(f"  done ({cskv_stats['avg']:.2f} ms avg)")

    print_table(args.n_windows, window_len, swt_stats, cskv_stats)


if __name__ == "__main__":
    main()
