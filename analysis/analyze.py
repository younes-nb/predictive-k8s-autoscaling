import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import glob
import os
import argparse


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
        y_true = valid_preds["current_cpu_60th"]
        y_pred = valid_preds["predicted_cpu_max"]

        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))

    inf_times = global_df["inference_time_s"]
    avg_inf = inf_times.mean()
    p95_inf = inf_times.quantile(0.95)

    if "sigma" in global_df.columns:
        avg_sigma = global_df["sigma"].mean()
    else:
        avg_sigma = 0.0

    print("=" * 40)
    print("📊  GLOBAL EXPERIMENT METRICS")
    print("=" * 40)
    print(f"Total Data Points:    {len(global_df)}")
    print("-" * 20)
    print(f"Prediction MSE:       {mse:.5f}")
    print(f"Prediction MAE:       {mae:.5f}")
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
            df["current_cpu_60th"],
            label="Actual CPU",
            color="blue",
            alpha=0.6,
        )
        ax.plot(
            df["timestamp"],
            df["predicted_cpu_max"],
            label="Predicted Max",
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

        ax.set_title(f"Deployment: {name}", fontweight="bold")
        ax.set_ylabel("CPU Load")
        ax.grid(True, alpha=0.3)

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

        if i == 0:
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)

    output_file = "experiment_results_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n✅ Plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    dep_data, glob_df = load_and_filter_data(args.data_dir, args.start, args.end)
    calculate_metrics(glob_df)
    plot_deployments(dep_data)
