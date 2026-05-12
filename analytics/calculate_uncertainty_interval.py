import os
import sys
import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import Paths


def calculate_regime_interval(df, sampling_rate_sec=60, window_size=5):
    df = df.with_columns(
        (
            pl.col("providerrpc_mcr") + pl.col("http_mcr") + pl.col("providermq_mcr")
        ).alias("total_mcr")
    )

    df = df.with_columns(
        pl.col("total_mcr").rolling_std(window_size=window_size).alias("volatility")
    ).drop_nulls()

    vol_signal = df["volatility"].to_numpy()
    if len(vol_signal) < 100:
        return None

    max_lag = min(len(vol_signal) // 4, 60)
    acf_values = acf(vol_signal, nlags=max_lag, fft=True)

    decorrelation_lag = np.where(acf_values < 0.5)[0]

    if len(decorrelation_lag) == 0:
        return None

    first_lag = decorrelation_lag[0]
    interval_seconds = first_lag * sampling_rate_sec

    return interval_seconds, acf_values


def main():
    parser = argparse.ArgumentParser(description="Volatility Autocorrelation Analysis")
    parser.add_argument("--trace", type=str, default=Paths.PARQUET_MSRTMCRE)
    parser.add_argument("--out", type=str, default="volatility_acf.png")
    args = parser.parse_args()

    q = pl.scan_parquet(args.trace)

    ms_names = (
        q.group_by("msname")
        .agg(pl.count())
        .sort("count", descending=True)
        .limit(5)
        .collect()["msname"]
        .to_list()
    )

    results = []

    plt.figure(figsize=(10, 6))

    for name in ms_names:
        df_ms = q.filter(pl.col("msname") == name).sort("timestamp").collect()
        res = calculate_regime_interval(df_ms)

        if res:
            interval, acf_vals = res
            results.append(interval)
            plt.plot(acf_vals, alpha=0.5, label=f"{name} (Lag: {interval}s)")

    if not results:
        print("❌ Could not determine decorrelation time.")
        return

    final_interval = int(np.median(results))

    print("\n" + "=" * 45)
    print("🔄 REGIME STATIONARITY ANALYSIS")
    print("=" * 45)
    print(f"Median Decorrelation Time: {final_interval} seconds")
    print(f"Recommended Update Interval: {final_interval}s")
    print("-" * 45)
    print("Justification: This is the duration over which ")
    print("workload volatility remains statistically similar.")
    print("=" * 45)

    plt.axhline(0.5, color="red", linestyle="--", label="50% Correlation Threshold")
    plt.title("Autocorrelation of Workload Volatility")
    plt.xlabel("Lag (Minutes/Samples)")
    plt.ylabel("ACF Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(args.out)


if __name__ == "__main__":
    main()
