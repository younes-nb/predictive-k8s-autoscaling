import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import glob
import os
import argparse

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
            if "timestamp_tehran" not in df.columns:
                print(
                    f"⚠️  Skipping {deployment_name}: 'timestamp_tehran' column missing."
                )
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp_tehran"])
            mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
            filtered_df = df.loc[mask].copy()

            if not filtered_df.empty:
                limit_per_pod = get_limit(deployment_name)

                safe_replicas = filtered_df["current_replicas"].replace(0, 1)

                total_capacity = safe_replicas * limit_per_pod

                filtered_df["cpu_actual_norm"] = (
                    filtered_df["current_cpu_60th"] / total_capacity
                ).clip(upper=1.0)
                filtered_df["cpu_pred_norm"] = (
                    filtered_df["predicted_cpu_max"] / total_capacity
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

    valid_preds = global_df[global_df["predicted_cpu_max"] > 0]

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
    avg_sigma = global_df["sigma"].mean() if "sigma" in global_df.columns else 0.0

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
    print(f"Avg Sigma (Uncertainty): {avg_sigma:.5f}")
    print("=" * 40)


def plot_deployments(deployment_data):
    if not deployment_data:
        return

    n_deployments = len(deployment_data)
    cols = 2
    rows = (n_deployments + 1) // cols
    if rows == 0:
        rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), sharex=False)
    axes = axes.flatten()

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
            df["cpu_pred_norm"],
            label="Predicted Load (%)",
            color="orange",
            linestyle="--",
        )
        ax.plot(
            df["timestamp"],
            df["threshold"],
            label="Threshold",
            color="red",
            linestyle=":",
            alpha=0.5,
        )

        ax.set_title(
            f"Deployment: {name} (Limit: {get_limit(name)}m per pod)", fontweight="bold"
        )
        ax.set_ylabel("CPU Utilization (0.0 - 1.0)")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.grid(True, alpha=0.3)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, linewidth=1)

        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        ax2 = ax.twinx()
        ax2.step(
            df["timestamp"],
            df["current_replicas"],
            label="Replicas",
            color="green",
            where="post",
            alpha=0.7,
        )
        ax2.set_ylabel("Replicas")

        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        rep_min = df["current_replicas"].min()
        rep_max = df["current_replicas"].max()
        if rep_min == rep_max:
            ax2.set_ylim(rep_min - 1, rep_max + 1)

        if i == 0:
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.6)

    output_file = "experiment_results_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n✅ Plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    dep_data, glob_df = load_and_filter_data(args.data_dir, args.start, args.end)
    calculate_metrics(glob_df)
    plot_deployments(dep_data)
