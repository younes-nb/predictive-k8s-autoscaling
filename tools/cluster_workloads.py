import os
import json
import argparse
import sys
import time
import polars as pl
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import joblib
import pytz
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, f1_score
from kneed import KneeLocator

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, ARCHETYPES, PREPROCESSING


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds // 60:.0f}m {seconds % 60:.2f}s"


def get_tehran_timestamp():
    tehran_tz = pytz.timezone("Asia/Tehran")
    return datetime.now(tehran_tz).strftime("%Y%m%d_%H%M%S")


def save_cluster_metrics(best_k, sil_score, counts, durations):
    os.makedirs(PATHS.LOGS_DIR, exist_ok=True)
    timestamp = get_tehran_timestamp()
    log_file = os.path.join(PATHS.LOGS_DIR, f"cluster_metrics_{timestamp}.log")

    with open(log_file, "w") as f:
        f.write(f"Clustering Run Metrics - {timestamp} (Tehran Time)\n")
        f.write("-" * 50 + "\n")
        f.write(f"Optimal Clusters (K): {best_k}\n")
        f.write(f"Silhouette Score: {sil_score:.4f}\n\n")
        f.write("Cluster Distribution:\n")
        for cluster_id, count in enumerate(counts):
            f.write(f"  - Archetype {cluster_id}: {count} members\n")
        f.write("Process Durations:\n")
        for key, val in durations.items():
            f.write(f"  - {key}: {format_duration(val)}\n")

    print(f"Results saved to: {log_file}")


def extract_robust_features(
    parquet_dir: str,
    max_services: int = None,
    batch_size: int = 50,
    temp_dir: str = "/dataset/duckdb_temp",
):
    start_extraction = time.perf_counter()
    print(f"Scanning parquet files in {parquet_dir}...")

    if not os.path.exists(temp_dir):
        print(f"Creating directory: {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)

    db_path = os.path.join(temp_dir, "cluster_processing.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='20GB'")

    parquet_glob = os.path.join(parquet_dir, "*.parquet")

    print("Phase 1: Identifying unique microservices...")
    unique_ms_query = f"SELECT DISTINCT msname FROM read_parquet('{parquet_glob}')"
    unique_msnames = con.execute(unique_ms_query).df()["msname"].tolist()
    unique_msnames = [m for m in unique_msnames if pd.notna(m)]

    if max_services is not None:
        print(f"Limiting clustering to {max_services} microservices...")
        unique_msnames = unique_msnames[:max_services]

    total_services = len(unique_msnames)
    print(f"Found {total_services} unique microservices to process.")

    con.execute("""
        CREATE TABLE ms_features (
            msname VARCHAR,
            cpu_mean DOUBLE,
            cpu_std DOUBLE,
            cpu_p95 DOUBLE,
            cpu_skew DOUBLE,
            cpu_kurt DOUBLE,
            peak_to_avg DOUBLE,
            coeff_variation DOUBLE,
            sample_count BIGINT
        )
        """)

    total_batches = (total_services + batch_size - 1) // batch_size
    print(f"Phase 2: Processing in {total_batches} batches...")

    time_col = PREPROCESSING.TIME_COL

    for i in range(0, total_services, batch_size):
        batch_start_time = time.perf_counter()
        batch = unique_msnames[i : i + batch_size]
        msnames_sql_list = ", ".join([f"'{m}'" for m in batch])
        batch_num = (i // batch_size) + 1

        print(
            f"Extracting features for Batch {batch_num}/{total_batches} ({len(batch)} services)...",
            flush=True,
        )

        query = f"""
        INSERT INTO ms_features
        WITH minute_agg AS (
            SELECT msname, {time_col}, avg(cpu_utilization) as cpu_utilization
            FROM read_parquet('{parquet_glob}')
            WHERE msname IN ({msnames_sql_list})
            GROUP BY msname, {time_col}
        )
        SELECT 
            msname,
            avg(cpu_utilization) as cpu_mean,
            stddev_samp(cpu_utilization) as cpu_std,
            approx_quantile(cpu_utilization, 0.95) as cpu_p95,
            skewness(cpu_utilization) as cpu_skew,
            kurtosis(cpu_utilization) as cpu_kurt,
            (max(cpu_utilization) / NULLIF(avg(cpu_utilization), 0)) as peak_to_avg,
            (stddev_samp(cpu_utilization) / NULLIF(avg(cpu_utilization), 0)) as coeff_variation,
            count(*) as sample_count
        FROM minute_agg
        GROUP BY msname
        HAVING count(*) > 10
        """
        con.execute(query)
        batch_end_time = time.perf_counter()
        batch_duration = batch_end_time - batch_start_time
        print(f"Done in {format_duration(batch_duration)}")

    print("Feature extraction complete. Loading results into memory...")
    df_pandas = con.execute("SELECT * FROM ms_features").df()

    con.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    total_extraction_duration = time.perf_counter() - start_extraction
    return pl.from_pandas(df_pandas.fillna(0.0)), total_extraction_duration


def plot_cluster_samples(df, parquet_dir, n_clusters):
    plot_path = os.path.join(PATHS.ARCHETYPE_DIR, "plots")
    os.makedirs(plot_path, exist_ok=True)

    for cid in range(n_clusters):
        cluster_members = df.filter(pl.col("archetype_id") == cid)["msname"].to_list()
        if not cluster_members:
            continue

        samples = np.random.choice(
            cluster_members, min(2, len(cluster_members)), replace=False
        )

        fig, axes = plt.subplots(1, len(samples), figsize=(12, 4))
        if len(samples) == 1:
            axes = [axes]

        for i, ms in enumerate(samples):
            raw_data = (
                pl.scan_parquet(os.path.join(parquet_dir, "*.parquet"))
                .filter(pl.col("msname") == ms)
                .collect()
            )
            axes[i].plot(raw_data["cpu_utilization"].to_numpy(), alpha=0.7)
            axes[i].set_title(f"MS: {ms}")
            axes[i].set_ylabel("CPU Util")
            axes[i].set_ylim(0, 1)

        plt.suptitle(f"Archetype {cid} Sample Workloads")
        plt.tight_layout()
        plt.savefig(
            os.path.join(PATHS.ARCHETYPE_DIR, "plots", f"cluster_{cid}_samples.png")
        )
        plt.close()
        print(f"Cluster sample plots saved in: {plot_path}")


def analyze_label_stability(
    features_df, scaler, kmeans_model, parquet_glob, time_steps_to_test
):
    start_stability = time.perf_counter()
    con = duckdb.connect(":memory:")

    ground_truth_labels = features_df["archetype_id"].to_numpy()
    ms_names = features_df["msname"].to_list()
    stability_results = []

    print("Starting Sensitivity Analysis...")

    for n in time_steps_to_test:
        step_start = time.perf_counter()
        print(f"Testing window size: {n} minutes...")

        query = f"""
        WITH windowed_agg AS (
            SELECT msname, {PREPROCESSING.TIME_COL}, avg(cpu_utilization) as cpu_utilization
            FROM read_parquet('{parquet_glob}')
            GROUP BY msname, {PREPROCESSING.TIME_COL}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY msname ORDER BY {PREPROCESSING.TIME_COL}) <= {n}
        )
        SELECT 
            msname,
            avg(cpu_utilization) as cpu_mean,
            stddev_samp(cpu_utilization) as cpu_std,
            approx_quantile(cpu_utilization, 0.95) as cpu_p95,
            skewness(cpu_utilization) as cpu_skew,
            kurtosis(cpu_utilization) as cpu_kurt,
            (max(cpu_utilization) / NULLIF(avg(cpu_utilization), 0)) as peak_to_avg,
            (stddev_samp(cpu_utilization) / NULLIF(avg(cpu_utilization), 0)) as coeff_variation
        FROM windowed_agg
        GROUP BY msname
        """

        partial_df = con.execute(query).df().fillna(0.0)

        partial_df = pd.merge(
            pd.DataFrame({"msname": ms_names}), partial_df, on="msname", how="left"
        ).fillna(0.0)
        partial_data = partial_df.drop(columns=["msname"]).to_numpy()

        scaled_partial = scaler.transform(partial_data)
        predicted_labels = kmeans_model.predict(scaled_partial)

        score = f1_score(ground_truth_labels, predicted_labels, average="weighted")
        stability_results.append(score)
        step_end = time.perf_counter()
        print(f"Window {n}m: F1={score:.4f} ({format_duration(step_end - step_start)})")

    con.close()
    total_stability_duration = time.perf_counter() - start_stability
    return stability_results, total_stability_duration


def plot_stability_curve(time_steps, scores):
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, scores, marker="o", linestyle="-", color="teal", linewidth=2)

    kn = KneeLocator(time_steps, scores, curve="concave", direction="increasing")
    optimal_n = kn.knee or time_steps[-1]

    plt.axvline(
        x=optimal_n, color="red", linestyle="--", label=f"Optimal Window: {optimal_n}m"
    )
    plt.title("Archetype Stability vs. Observation Window")
    plt.xlabel("Minutes of Observation")
    plt.ylabel("Weighted F1-Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(PATHS.ARCHETYPE_DIR, "window_stability_curve.png"))
    print(f"Analysis complete. Suggested deployment window: {optimal_n} minutes.")


def main():
    ap = argparse.ArgumentParser(description="Cluster microservice workloads")
    ap.add_argument("--max_services", type=int, default=PREPROCESSING.MAX_SERVICES)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=50,
    )
    ap.add_argument(
        "--temp_dir",
        type=str,
        default="/dataset/duckdb_temp",
    )
    args, _ = ap.parse_known_args()

    os.makedirs(PATHS.ARCHETYPE_DIR, exist_ok=True)
    durations = {}

    features_df, durations["Total Feature Extraction"] = extract_robust_features(
        PATHS.PARQUET_MSRESOURCE,
        max_services=args.max_services,
        batch_size=args.batch_size,
        temp_dir=args.temp_dir,
    )

    print("Determining optimal K and clustering...")
    start_clustering = time.perf_counter()
    feature_cols = [
        "cpu_mean",
        "cpu_std",
        "cpu_p95",
        "cpu_skew",
        "cpu_kurt",
        "peak_to_avg",
        "coeff_variation",
    ]
    data_to_scale = features_df.select(feature_cols).to_numpy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_scale)

    wcss = []
    k_range = range(ARCHETYPES.MIN_K, ARCHETYPES.MAX_K + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled_data)
        wcss.append(km.inertia_)

    kn = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")
    best_k = kn.knee or ARCHETYPES.MIN_K
    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_km.fit_predict(scaled_data)
    durations["Clustering & Elbow Method"] = time.perf_counter() - start_clustering

    sil_score = silhouette_score(scaled_data, labels)
    print(f"Optimal K: {best_k} | Silhouette Score: {sil_score:.4f}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nTotal Number of Clusters: {best_k}")
    for cluster_id, count in zip(unique_labels, counts):
        print(f"  Cluster {cluster_id}: {count} members")
    print()

    model_save_path = os.path.join(PATHS.ARCHETYPE_DIR, "kmeans_model.joblib")
    scaler_save_path = os.path.join(PATHS.ARCHETYPE_DIR, "scaler.joblib")

    joblib.dump(final_km, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"K-Means model exported to: {model_save_path}")
    print(f"Scaler exported to: {scaler_save_path}")

    features_df = features_df.with_columns(pl.Series("archetype_id", labels))

    mapping = {
        row["msname"]: int(row["archetype_id"]) for row in features_df.to_dicts()
    }
    with open(PATHS.ARCHETYPE_MAPPING, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Archetype mapping saved to: {PATHS.ARCHETYPE_MAPPING}")

    plot_cluster_samples(features_df, PATHS.PARQUET_MSRESOURCE, best_k)

    windows = [15, 30, 60, 120, 240, 480, 720, 1440, 2880]
    scores, durations["Stability Analysis (Min Time Steps)"] = analyze_label_stability(
        features_df,
        scaler,
        final_km,
        os.path.join(PATHS.PARQUET_MSRESOURCE, "*.parquet"),
        windows,
    )
    plot_stability_curve(windows, scores)

    save_cluster_metrics(best_k, sil_score, counts, durations)

    print("\n" + "=" * 40)
    print("       FINAL RUNTIME SUMMARY")
    print("=" * 40)
    for stage, dur in durations.items():
        print(f"{stage:.<30} {format_duration(dur)}")
    print("=" * 40)


if __name__ == "__main__":
    main()
