import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import glob
import os
import argparse
from datetime import datetime, timezone, timedelta

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
    "redis-cart": 0.125,
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
    return parser.parse_args()


def get_limit(deployment_name):
    return DEPLOYMENT_LIMITS.get(deployment_name, 1.0)


def load_and_filter_data(data_dir, start_str, end_str):
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    deployment_data = {}
    global_df = pd.DataFrame()

    start_ts = pd.to_datetime(start_str)
    end_ts = pd.to_datetime(end_str)

    print(f"🔎 Filtering data from {start_ts} to {end_ts}...\n")

    for filename in all_files:
        deployment_name = os.path.basename(filename).replace(".csv", "")
        try:
            df = pd.read_csv(filename)
            if "timestamp" not in df.columns:
                print(
                    f"⚠️  Skipping {deployment_name}: 'timestamp' column missing."
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
                    f"⚠️  {deployment_name}: No data found in the specified time range."
                )

        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    return deployment_data, global_df


def calculate_metrics(global_df):
    if global_df.empty:
        print("❌ No data available to calculate metrics.")
        return

    valid_preds = global_df[global_df["pred_cpu"] > 0]

    if valid_preds.empty:
        print("⚠️ No valid predictions (>0) found.")
        mse, mae = 0, 0
    else:
        y_true = valid_preds["cpu_actual_norm"]
        y_pred = valid_preds["cpu_pred_norm"]

        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))

    inf_times = global_df["inference_time_s"]
    avg_inf = inf_times.mean()
    p95_inf = inf_times.quantile(0.95)

    if "sigma" in global_df.columns:
        avg_sigma = global_df["sigma"].mean()
        p95_sigma = global_df["sigma"].quantile(0.95)
    else:
        avg_sigma = 0.0
        p95_sigma = 0.0
    if "replicas" in global_df.columns:
        avg_replicas = global_df["replicas"].mean()
    else:
        avg_replicas = 0.0

    print("=" * 40)
    print("📊  GLOBAL EXPERIMENT METRICS")
    print("=" * 40)
    print(f"Total Data Points:    {len(global_df)}")
    print("-" * 20)
    print(f"Prediction MSE:       {mse:.5f}")
    print(f"Prediction MAE:       {mae:.5f} ({(mae*100):.2f}%)")
    print("-" * 20)
    print(f"Avg Inference Time:   {avg_inf:.4f} s")
    print(f"P95 Inference Time:   {p95_inf:.4f} s")
    print("-" * 20)
    print(f"Avg Uncertainty: {avg_sigma:.5f}")
    print(f"P95 Uncertainty: {p95_sigma:.5f}")
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


def _save_figure(fig, suffix):
    save_dir = "/proj/k8sautoscaledl-PG0/plots"
    os.makedirs(save_dir, exist_ok=True)

    tehran_tz = timezone(timedelta(hours=3, minutes=30))
    timestamp_str = datetime.now(tehran_tz).strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(save_dir, f"load_test_results_{timestamp_str}_{suffix}.png")

    plt.savefig(output_file, dpi=300)
    print(f"✅ Plot saved to {output_file}")


def plot_cpu(deployment_data):
    fig, axes = _build_plot_grid(len(deployment_data))

    for i, (name, df) in enumerate(deployment_data.items()):
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
            df["cpu_pred_norm"].shift(5),
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
    _save_figure(fig, "cpu")


def plot_replicas(deployment_data):
    fig, axes = _build_plot_grid(len(deployment_data))

    for i, (name, df) in enumerate(deployment_data.items()):
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
    _save_figure(fig, "replicas")


def plot_memory(deployment_data):
    fig, axes = _build_plot_grid(len(deployment_data))

    for i, (name, df) in enumerate(deployment_data.items()):
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
            df["mem_pred_norm"].shift(5),
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
    _save_figure(fig, "mem")


if __name__ == "__main__":
    args = parse_args()
    dep_data, glob_df = load_and_filter_data(args.data_dir, args.start, args.end)
    calculate_metrics(glob_df)
    plot_cpu(dep_data)
    plot_replicas(dep_data)
    plot_memory(dep_data)
    plt.show()
