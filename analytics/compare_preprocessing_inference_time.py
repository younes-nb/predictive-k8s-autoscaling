#!/usr/bin/env python3
"""Benchmark per-window inference time of SV vs CSKV preprocessing."""

import argparse
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from preprocessing.sv.config import CFG as SV_CFG
from preprocessing.sv.decomposition import decompose_window
from preprocessing.cskv.config import CFG as CSKV_CFG, set_seed
from preprocessing.cskv.decomposition import decompose_service_signal


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


def benchmark_sv(windows: np.ndarray) -> list[float]:
    times = []
    for i in range(len(windows)):
        t0 = time.perf_counter()
        decompose_window(windows[i], SV_CFG)
        times.append(time.perf_counter() - t0)
    return times


def benchmark_cskv(windows: np.ndarray) -> list[float]:
    set_seed(CSKV_CFG.SEED)
    times = []
    for i in range(len(windows)):
        t0 = time.perf_counter()
        decompose_service_signal(windows[i], CSKV_CFG)
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


def print_table(n_windows: int, window_len: int, sv_stats: dict, cskv_stats: dict):
    print(f"\nPreprocessing Inference Time Benchmark (N={n_windows}, window_len={window_len})")
    print("=" * 74)
    header = f"{'Approach':<10} {'AVG(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
    print(header)
    print("-" * 74)
    for label, stats in [("SV", sv_stats), ("CSKV", cskv_stats)]:
        print(
            f"{label:<10} {stats['avg']:>10.2f} {stats['p50']:>10.2f} "
            f"{stats['p95']:>10.2f} {stats['min']:>10.2f} {stats['max']:>10.2f}"
        )
    print("=" * 74)
    if sv_stats["avg"] > 0:
        ratio = cskv_stats["avg"] / sv_stats["avg"]
        slower, faster = ("CSKV", "SV") if ratio > 1 else ("SV", "CSKV")
        print(f"\n{slower} is {ratio:.2f}x slower than {faster} on average")
    else:
        print("\nSV avg is zero; cannot compute ratio")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_windows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    windows = generate_windows(args.n_windows, SV_CFG.INPUT_LEN, rng)

    print(f"Running SV benchmark ({args.n_windows} windows)...")
    sv_times = benchmark_sv(windows)
    sv_stats = compute_stats(np.array(sv_times) * 1000)
    print(f"  done ({sv_stats['avg']:.2f} ms avg)")

    print(f"Running CSKV benchmark ({args.n_windows} windows)...")
    cskv_times = benchmark_cskv(windows)
    cskv_stats = compute_stats(np.array(cskv_times) * 1000)
    print(f"  done ({cskv_stats['avg']:.2f} ms avg)")

    print_table(args.n_windows, SV_CFG.INPUT_LEN, sv_stats, cskv_stats)


if __name__ == "__main__":
    main()
