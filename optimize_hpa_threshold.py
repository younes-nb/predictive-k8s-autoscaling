import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional
from config.defaults import PATHS

os.makedirs(PATHS.LOGS_DIR, exist_ok=True)

CHUNK_SIZE = 1_000_000
ALPHA = 1.0
BETA = 5.0
THRESHOLDS = np.arange(0.4, 0.91, 0.05)
CRITICAL_PERCENTILE = 95
MIN_SAMPLES = 200


class OnlineServiceStats:
    __slots__ = ("count", "mean_cpu", "cpu_samples")

    def __init__(self):
        self.count = 0
        self.mean_cpu = 0.0
        self.cpu_samples = []

    def update(self, cpu: np.ndarray, reservoir_size: int = 5000):
        cpu = np.asarray(cpu, dtype=np.float64)
        cpu = cpu[~np.isnan(cpu)]
        for u in cpu:
            self.count += 1
            self.mean_cpu += (u - self.mean_cpu) / self.count

            if len(self.cpu_samples) < reservoir_size:
                self.cpu_samples.append(float(u))
            else:
                j = np.random.randint(0, self.count)
                if j < reservoir_size:
                    self.cpu_samples[j] = float(u)

    def critical_cpu(self, percentile: float) -> float:
        if not self.cpu_samples:
            return float("nan")
        return float(np.percentile(self.cpu_samples, percentile))

    def sla_risk(self, u_crit: float) -> float:
        if not self.cpu_samples or np.isnan(u_crit):
            return float("nan")
        samples = np.asarray(self.cpu_samples, dtype=np.float64)
        return float(np.mean(samples > u_crit))


def _natural_key(path: str) -> tuple:
    name = Path(path).name
    stem = Path(path).stem
    m = re.search(r"_(\d+)$", stem)
    if m:
        return (stem[: m.start()], int(m.group(1)), name)
    return (stem, float("inf"), name)


def select_files(
    raw_dir: str,
    pattern: str = "*.csv",
    first_n: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    indices: Optional[List[int]] = None,
) -> List[str]:
    files = glob.glob(os.path.join(raw_dir, pattern))
    files = sorted(files, key=_natural_key)

    if not files:
        raise RuntimeError(f"No files matched pattern='{pattern}' in {raw_dir}")

    if indices is not None and len(indices) > 0:
        picked = []
        for i in indices:
            if i < 0 or i >= len(files):
                raise ValueError(f"Index {i} out of range (0..{len(files)-1})")
            picked.append(files[i])
        return picked

    if start is not None or end is not None:
        return files[slice(start, end)]

    if first_n is not None:
        return files[:first_n]

    return files


def stream_msresource_stats(files: List[str]) -> dict:
    service_stats = defaultdict(OnlineServiceStats)

    for file_idx, file in enumerate(files):
        print(f"Processing file {file_idx + 1}/{len(files)}: {file}")

        reader = pd.read_csv(
            file,
            usecols=["msname", "cpu_utilization"],
            chunksize=CHUNK_SIZE,
        )

        for chunk_idx, chunk in enumerate(reader):
            grouped = chunk.groupby("msname")["cpu_utilization"]
            for msname, cpu_values in grouped:
                service_stats[msname].update(cpu_values.values)

            if chunk_idx % 5 == 0:
                print(
                    f"  processed chunk {chunk_idx}, "
                    f"services tracked: {len(service_stats)}"
                )

    return service_stats


def optimize_thresholds(service_stats: dict) -> pd.DataFrame:
    records = []

    for msname, stats in service_stats.items():
        if stats.count < MIN_SAMPLES:
            continue

        u_crit = stats.critical_cpu(CRITICAL_PERCENTILE)
        sla_risk = stats.sla_risk(u_crit)

        best_theta = None
        best_cost = float("inf")

        if np.isnan(u_crit) or np.isnan(sla_risk):
            continue

        for theta in THRESHOLDS:
            overprov = max(0.0, float(theta) - float(stats.mean_cpu))
            cost = ALPHA * overprov + BETA * sla_risk
            if cost < best_cost:
                best_cost = cost
                best_theta = float(theta)

        records.append(
            {
                "msname": msname,
                "optimal_threshold": best_theta,
                "cost": float(best_cost),
                "avg_cpu": float(stats.mean_cpu),
                "u_critical_p95": float(u_crit),
                "sla_risk_reservoir": float(sla_risk),
                "num_samples": int(stats.count),
            }
        )

    return pd.DataFrame(records)


def parse_args():
    p = argparse.ArgumentParser(description="Lazy threshold optimization (streaming)")
    p.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern inside RAW_MSRESOURCE",
    )
    p.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Use only first N CSV files (after sorting)",
    )
    p.add_argument(
        "--start", type=int, default=None, help="Start index for slicing files"
    )
    p.add_argument(
        "--end", type=int, default=None, help="End index (exclusive) for slicing files"
    )
    p.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated explicit file indices, e.g. '0,1,5,9'",
    )
    p.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE, help="CSV chunk size (rows)"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reservoir sampling reproducibility",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="hpa_threshold_optimization_lazy",
        help="Output file prefix",
    )
    return p.parse_args()


def main():
    args = parse_args()

    global CHUNK_SIZE
    CHUNK_SIZE = args.chunk_size
    np.random.seed(args.seed)

    indices_list = None
    if args.indices:
        indices_list = [int(x.strip()) for x in args.indices.split(",") if x.strip()]

    files = select_files(
        raw_dir=PATHS.RAW_MSRESOURCE,
        pattern=args.pattern,
        first_n=args.first_n,
        start=args.start,
        end=args.end,
        indices=indices_list,
    )

    print("Starting lazy threshold optimization...")
    print(f"Selected {len(files)} file(s). First file: {files[0]}")
    print(f"Chunk size: {CHUNK_SIZE}, seed: {args.seed}")

    service_stats = stream_msresource_stats(files)

    print("Optimizing thresholds...")
    results_df = optimize_thresholds(service_stats)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_services": int(len(results_df)),
        "threshold_mean": (
            float(results_df["optimal_threshold"].mean())
            if len(results_df)
            else float("nan")
        ),
        "threshold_median": (
            float(results_df["optimal_threshold"].median())
            if len(results_df)
            else float("nan")
        ),
        "threshold_std": (
            float(results_df["optimal_threshold"].std())
            if len(results_df)
            else float("nan")
        ),
        "alpha": ALPHA,
        "beta": BETA,
        "critical_percentile": CRITICAL_PERCENTILE,
        "chunk_size": CHUNK_SIZE,
        "files_used": len(files),
        "file_pattern": args.pattern,
        "file_slice": {
            "first_n": args.first_n,
            "start": args.start,
            "end": args.end,
            "indices": indices_list,
        },
        "reservoir_seed": args.seed,
    }

    csv_path = os.path.join(PATHS.LOGS_DIR, f"{args.out_prefix}.csv")
    json_path = os.path.join(PATHS.LOGS_DIR, f"{args.out_prefix}_summary.json")

    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Results saved to {csv_path}")
    print(f"Summary saved to {json_path}")


if __name__ == "__main__":
    main()
