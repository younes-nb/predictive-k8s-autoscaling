import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import glob
import os
import argparse
import sys
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

DEFAULT_PLOTS_DIR = "/proj/k8sautoscaledl-PG0/analytics_out"

DEPLOYMENT_LIMITS = {
    "adservice": 0.3,
    "cartservice": 0.3,
    "checkoutservice": 0.2,
    "currencyservice": 0.2,
    "emailservice": 0.2,
    "frontend": 0.2,
    "paymentservice": 0.2,
    "productcatalogservice": 0.2,
    "recommendationservice": 0.2,
    "shippingservice": 0.2,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CPA Experiment Results")
    parser.add_argument(
        "--start", type=str, required=True, help="Start Timestamp (YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument(
        "--end", type=str, required=True, help="End Timestamp (YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory containing CSV files"
    )
    parser.add_argument(
        "--plots_dir", type=str, default=DEFAULT_PLOTS_DIR,
        help="Directory to save plot PNGs",
    )
    parser.add_argument(
        "--horizon", type=int, default=5,
        help="Forecast horizon in minutes (pred[t] targets actual[t+horizon])",
    )
    return parser.parse_args()


def get_limit(deployment_name):
    return DEPLOYMENT_LIMITS.get(deployment_name, 1.0)


def load_and_filter_data(data_dir, start_str, end_str):
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    deployment_data = {}
    global_df = pd.DataFrame()

    start_ts = pd.to_datetime(start_str)
    end_ts = pd.to_datetime(end_str)

    print(f"Filtering data from {start_ts} to {end_ts}...\n")

    for filename in tqdm(all_files, desc="Loading CSVs", unit="file"):
        deployment_name = os.path.basename(filename).replace(".csv", "")
        try:
            df = pd.read_csv(filename)
            if "timestamp" not in df.columns:
                print(
                    f"Skipping {deployment_name}: 'timestamp' column missing."
                )
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
            filtered_df = df.loc[mask].copy()

            if not filtered_df.empty:
                limit_per_pod = get_limit(deployment_name)

                safe_replicas = filtered_df["replicas"].replace(0, 1)

                total_capacity = safe_replicas * limit_per_pod

                filtered_df["cpu_actual_norm"] = (
                    filtered_df["cpu"] / total_capacity
                ).clip(upper=1.0)
                filtered_df["cpu_pred_norm"] = (
                    filtered_df["pred_cpu"] / total_capacity
                ).clip(upper=1.0)
                filtered_df["mem_actual_norm"] = (
                    filtered_df["memory"] / total_capacity
                ).clip(upper=1.0)
                filtered_df["mem_pred_norm"] = (
                    filtered_df["pred_mem"] / total_capacity
                ).clip(upper=1.0)
                filtered_df["deployment"] = deployment_name
                deployment_data[deployment_name] = filtered_df
                global_df = pd.concat([global_df, filtered_df], ignore_index=True)
            else:
                print(
                    f"{deployment_name}: No data found in the specified time range."
                )

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return deployment_data, global_df


def _mse_mae(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, mae


def _forecast_metrics(actual, pred, delta, horizon):
    """Evaluate horizon-ahead forecasts where pred[t] targets actual[t+horizon].

    Returns (final, raw, naive, n) where final/raw/naive are (mse, mae) tuples
    or None, and n is the number of aligned rows used:
      - final: the logged prediction (raw model output + residual correction)
      - raw:   the model output before residual correction (pred - delta)
      - naive: persistence baseline (actual[t] used to forecast actual[t+horizon])
    """
    frame = pd.DataFrame(
        {
            "y": actual.shift(-horizon),
            "a": actual,
            "final": pred,
            "raw": (pred - delta).clip(lower=0.0) if delta is not None else None,
        }
    ).dropna(subset=["y", "a", "final"])

    n = len(frame)
    if n < horizon:
        return None, None, None, n

    final = _mse_mae(frame["y"], frame["final"])
    naive = _mse_mae(frame["y"], frame["a"])
    raw = _mse_mae(frame["y"], frame["raw"]) if delta is not None else None
    return final, raw, naive, n


def calculate_metrics(global_df, horizon=5):
    if global_df.empty:
        print("No data available to calculate metrics.")
        return

    cpu_valid = global_df[global_df["pred_cpu"] > 0]
    mem_valid = global_df[global_df["pred_mem"] > 0]

    has_delta_cpu = "delta_cpu" in global_df.columns
    has_delta_mem = "delta_mem" in global_df.columns

    if cpu_valid.empty:
        cpu_res = (None, None, None, 0)
    else:
        cpu_res = _forecast_metrics(
            cpu_valid["cpu_actual_norm"],
            cpu_valid["cpu_pred_norm"],
            cpu_valid["delta_cpu"] if has_delta_cpu else None,
            horizon,
        )

    if mem_valid.empty:
        mem_res = (None, None, None, 0)
    else:
        mem_res = _forecast_metrics(
            mem_valid["mem_actual_norm"],
            mem_valid["mem_pred_norm"],
            mem_valid["delta_mem"] if has_delta_mem else None,
            horizon,
        )

    inf_times = global_df["inference_time_s"]
    avg_inf = inf_times.mean()
    p95_inf = inf_times.quantile(0.95)

    if "replicas" in global_df.columns:
        avg_replicas = global_df["replicas"].mean()
    else:
        avg_replicas = 0.0

    print("=" * 40)
    print("GLOBAL EXPERIMENT METRICS")
    print("=" * 40)
    print(f"Total Data Points:    {len(global_df)}")
    print(f"Forecast Horizon:     {horizon} min (pred[t] vs actual[t+{horizon}])")
    print("-" * 20)

    def _row(target, res):
        final, raw, naive, n = res
        if final is None:
            print(f"{target:5s}  no valid aligned data ({n} rows)")
            return
        f_mse, f_mae = final
        n_mse, n_mae = naive
        d_final = (f_mae - n_mae) / n_mae * 100 if n_mae > 0 else float("nan")
        line = (
            f"{target:5s}  final MSE {f_mse:.5f} MAE {f_mae:.5f} ({f_mae*100:.2f}%)"
            f"  | naive MSE {n_mse:.5f} MAE {n_mae:.5f} ({n_mae*100:.2f}%)"
            f"  | Delta {d_final:+.1f}%"
        )
        if raw is not None:
            r_mse, r_mae = raw
            d_raw = (r_mae - n_mae) / n_mae * 100 if n_mae > 0 else float("nan")
            line += (
                f"  (raw MSE {r_mse:.5f} MAE {r_mae:.5f} ({r_mae*100:.2f}%)"
                f"  Delta {d_raw:+.1f}%)"
            )
        print(line)

    _row("CPU", cpu_res)
    _row("Mem", mem_res)
    print("-" * 20)
    print(f"Avg Inference Time:   {avg_inf:.4f} s")
    print(f"P95 Inference Time:   {p95_inf:.4f} s")
    print("=" * 40)
    print(f"Avg Replicas: {avg_replicas:.2f}")
    print("=" * 40)


def _build_plot_grid(n_deployments):
    cols = 2
    rows = (n_deployments + 1) // cols
    if rows == 0:
        rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(36, 6 * rows), sharex=False)
    axes = axes.flatten()
    for j in range(n_deployments, len(axes)):
        axes[j].axis("off")
    return fig, axes


def _style_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _add_legend(ax):
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc="upper left")


def _save_figure(fig, suffix, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    tehran_tz = timezone(timedelta(hours=3, minutes=30))
    timestamp_str = datetime.now(tehran_tz).strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(plots_dir, f"load_test_results_{timestamp_str}_{suffix}.png")

    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")


def plot_cpu(deployment_data, plots_dir, horizon=5):
    fig, axes = _build_plot_grid(len(deployment_data))

    for i, (name, df) in tqdm(
        enumerate(deployment_data.items()), desc="Plotting CPU", unit="svc",
        total=len(deployment_data),
    ):
        ax = axes[i]

        ax.plot(
            df["timestamp"],
            df["cpu_actual_norm"],
            label="Actual Load (%)",
            color="blue",
            alpha=0.6,
        )
        ax.plot(
            df["timestamp"],
            df["cpu_pred_norm"].shift(horizon),
            label="Predicted Load (%)",
            color="orange",
            linestyle="--",
        )

        ax.set_title(
            f"Deployment: {name} (Limit: {get_limit(name)}m per pod)", fontweight="bold"
        )
        ax.set_ylabel("CPU Utilization (0.0 - 1.0)")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, linewidth=1)

        _style_time_axis(ax)
        _add_legend(ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.6)
    _save_figure(fig, "cpu", plots_dir)


def plot_replicas(deployment_data, plots_dir):
    fig, axes = _build_plot_grid(len(deployment_data))

    for i, (name, df) in tqdm(
        enumerate(deployment_data.items()), desc="Plotting replicas", unit="svc",
        total=len(deployment_data),
    ):
        ax = axes[i]

        ax.step(
            df["timestamp"],
            df["replicas"],
            label="Replicas",
            color="green",
            where="post",
            alpha=0.7,
        )
        ax.set_title(f"Deployment: {name}", fontweight="bold")
        ax.set_ylabel("Replicas")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        rep_min = df["replicas"].min()
        rep_max = df["replicas"].max()
        if rep_min == rep_max:
            ax.set_ylim(rep_min - 1, rep_max + 1)

        _style_time_axis(ax)
        _add_legend(ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.6)
    _save_figure(fig, "replicas", plots_dir)


def plot_memory(deployment_data, plots_dir, horizon=5):
    fig, axes = _build_plot_grid(len(deployment_data))

    for i, (name, df) in tqdm(
        enumerate(deployment_data.items()), desc="Plotting memory", unit="svc",
        total=len(deployment_data),
    ):
        ax = axes[i]

        ax.plot(
            df["timestamp"],
            df["mem_actual_norm"],
            label="Actual Memory (%)",
            color="blue",
            alpha=0.6,
        )
        ax.plot(
            df["timestamp"],
            df["mem_pred_norm"].shift(horizon),
            label="Predicted Memory (%)",
            color="orange",
            linestyle="--",
        )

        ax.set_title(
            f"Deployment: {name} (Limit: {get_limit(name)}m per pod)", fontweight="bold"
        )
        ax.set_ylabel("Memory Utilization (0.0 - 1.0)")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, linewidth=1)

        _style_time_axis(ax)
        _add_legend(ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.6)
    _save_figure(fig, "mem", plots_dir)


if __name__ == "__main__":
    args = parse_args()
    dep_data, glob_df = load_and_filter_data(args.data_dir, args.start, args.end)

    if glob_df.empty:
        print("No data available to calculate metrics.")
        sys.exit(1)

    calculate_metrics(glob_df, args.horizon)
    plot_cpu(dep_data, args.plots_dir, args.horizon)
    plot_replicas(dep_data, args.plots_dir)
    plot_memory(dep_data, args.plots_dir, args.horizon)
    plt.show()
