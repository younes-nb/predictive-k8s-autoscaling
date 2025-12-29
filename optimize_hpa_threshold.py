import os
import glob
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from config.defaults import PATHS

os.makedirs(PATHS.LOGS_DIR, exist_ok=True)

CHUNK_SIZE = 1_000_000
ALPHA = 1.0
BETA = 5.0
THRESHOLDS = np.arange(0.4, 0.91, 0.05)
CRITICAL_PERCENTILE = 95

MIN_SAMPLES = 200


class OnlineServiceStats:
    __slots__ = (
        "count",
        "mean_cpu",
        "cpu_samples",
    )

    def __init__(self):
        self.count = 0
        self.mean_cpu = 0.0
        self.cpu_samples = []

    def update(self, cpu: np.ndarray, reservoir_size: int = 5000):
        for u in cpu:
            self.count += 1
            self.mean_cpu += (u - self.mean_cpu) / self.count

            if len(self.cpu_samples) < reservoir_size:
                self.cpu_samples.append(u)
            else:
                j = np.random.randint(0, self.count)
                if j < reservoir_size:
                    self.cpu_samples[j] = u

    def critical_cpu(self, percentile: float) -> float:
        return float(np.percentile(self.cpu_samples, percentile))

    def sla_risk(self, u_crit: float) -> float:
        samples = np.array(self.cpu_samples)
        return float(np.mean(samples > u_crit))


def stream_msresource_stats() -> dict:
    service_stats = defaultdict(OnlineServiceStats)

    files = sorted(glob.glob(os.path.join(PATHS.RAW_MSRESOURCE, "*.csv")))
    if not files:
        raise RuntimeError("No MSResource CSV files found.")

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

        for theta in THRESHOLDS:
            overprov = max(0.0, theta - stats.mean_cpu)
            cost = ALPHA * overprov + BETA * sla_risk

            if cost < best_cost:
                best_cost = cost
                best_theta = theta

        records.append(
            {
                "msname": msname,
                "optimal_threshold": best_theta,
                "cost": best_cost,
                "avg_cpu": stats.mean_cpu,
                "u_critical_p95": u_crit,
                "num_samples": stats.count,
            }
        )

    return pd.DataFrame(records)


def main():
    print("Starting lazy threshold optimization...")

    service_stats = stream_msresource_stats()

    print("Optimizing thresholds...")
    results_df = optimize_thresholds(service_stats)

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_services": int(len(results_df)),
        "threshold_mean": float(results_df["optimal_threshold"].mean()),
        "threshold_median": float(results_df["optimal_threshold"].median()),
        "threshold_std": float(results_df["optimal_threshold"].std()),
        "alpha": ALPHA,
        "beta": BETA,
        "critical_percentile": CRITICAL_PERCENTILE,
        "chunk_size": CHUNK_SIZE,
    }

    csv_path = os.path.join(LOGS_DIR, "hpa_threshold_optimization_lazy.csv")
    json_path = os.path.join(LOGS_DIR, "hpa_threshold_optimization_lazy_summary.json")

    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Results saved to {csv_path}")
    print(f"Summary saved to {json_path}")


if __name__ == "__main__":
    main()
