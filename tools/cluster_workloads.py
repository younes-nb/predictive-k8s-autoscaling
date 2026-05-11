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
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    silhouette_score,
    f1_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from kneed import KneeLocator

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, PREPROCESSING


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds // 60:.0f}m {seconds % 60:.2f}s"


def get_tehran_timestamp():
    tehran_tz = pytz.timezone("Asia/Tehran")
    return datetime.now(tehran_tz).strftime("%Y%m%d_%H%M%S")


def save_cluster_metrics(
    num_clusters, sil_score, db_score, ch_score, counts, durations, cluster_stats=None
):
    os.makedirs(PATHS.LOGS_DIR, exist_ok=True)
    timestamp = get_tehran_timestamp()
    log_file = os.path.join(PATHS.LOGS_DIR, f"cluster_metrics_{timestamp}.log")

    with open(log_file, "w") as f:
        f.write(f"Clustering Run Metrics - {timestamp} (Tehran Time)\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Archetypes Found: {num_clusters}\n")
        f.write(f"Silhouette Score (Higher is better): {sil_score:.4f}\n")
        f.write(f"Davies-Bouldin Index (Lower is better): {db_score:.4f}\n")
        f.write(f"Calinski-Harabasz Index (Higher is better): {ch_score:.4f}\n\n")

        f.write("Cluster Distribution:\n")
        for cluster_id, count in zip(counts[0], counts[1]):
            name = "Noise (Label -1)" if cluster_id == -1 else f"Archetype {cluster_id}"
            f.write(f"  - {name}: {count} members\n")

        if cluster_stats is not None:
            f.write("\nAverage Feature Values per Cluster:\n")
            stats_dict = cluster_stats.to_dicts()
            for row in sorted(stats_dict, key=lambda x: x["archetype_id"]):
                f.write(f"  Archetype {row['archetype_id']}:\n")
                for feat, val in row.items():
                    if feat != "archetype_id":
                        f.write(f"    - {feat:.<20} {val:.4f}\n")

        f.write("\nProcess Durations:\n")
        for key, val in durations.items():
            f.write(f"  - {key}: {format_duration(val)}\n")

    print(f"Detailed metrics saved to: {log_file}")


def extract_robust_features(
    parquet_dir: str,
    max_services: int = None,
    batch_size: int = 64,
    temp_dir: str = "/dataset/duckdb_temp",
):
    start_extraction = time.perf_counter()
    print(f"Scanning parquet files in {parquet_dir}...")

    if not os.path.exists(temp_dir):
        print(f"📁 Creating directory: {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)

    db_path = os.path.join(temp_dir, "cluster_processing.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=12")
    con.execute("PRAGMA memory_limit='32GB'")

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
            msname VARCHAR, cpu_mean DOUBLE, cpu_std DOUBLE, cpu_p95 DOUBLE,
            cpu_skew DOUBLE, cpu_kurt DOUBLE, peak_to_avg DOUBLE,
            coeff_variation DOUBLE, cpu_autocorr DOUBLE, cpu_mad DOUBLE,
            cpu_slope DOUBLE, cpu_iqr DOUBLE, burstiness_ratio DOUBLE,
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
        ),
        lagged_agg AS (
            SELECT *, lag(cpu_utilization) OVER (PARTITION BY msname ORDER BY {time_col}) as prev_cpu,
            row_number() OVER (PARTITION BY msname ORDER BY {time_col}) as time_idx
            FROM minute_agg
        )
        SELECT 
            msname, avg(cpu_utilization), stddev_samp(cpu_utilization),
            approx_quantile(cpu_utilization, 0.95), ln(abs(skewness(cpu_utilization)) + 1),
            ln(abs(kurtosis(cpu_utilization)) + 1), (max(cpu_utilization) / NULLIF(avg(cpu_utilization), 0)),
            (stddev_samp(cpu_utilization) / NULLIF(avg(cpu_utilization), 0)), corr(cpu_utilization, prev_cpu),
            avg(abs(cpu_utilization - prev_cpu)), regr_slope(cpu_utilization, time_idx),
            (approx_quantile(cpu_utilization, 0.75) - approx_quantile(cpu_utilization, 0.25)),
            (max(cpu_utilization) / (approx_quantile(cpu_utilization, 0.5) + 0.00001)),
            count(*)
        FROM lagged_agg GROUP BY msname HAVING count(*) > 10 AND avg(cpu_utilization) > 0.005
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
    return pl.from_pandas(df_pandas.fillna(0.0)), time.perf_counter() - start_extraction


def plot_archetype_projection(pca_data, labels):
    plt.figure(figsize=(12, 8))

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = "gray" if label == -1 else cmap(i % 10)
        alpha = 0.3 if label == -1 else 0.6
        name = "General/Noise" if label == -1 else f"Archetype {label}"

        plt.scatter(
            pca_data[mask, 0],
            pca_data[mask, 1],
            c=[color],
            label=name,
            alpha=alpha,
            s=15,
            edgecolors="none",
        )

    plt.title("Workload Archetype Projection (PCA Space)", fontsize=14)
    plt.xlabel("Principal Component 1 (Shape Variance)")
    plt.ylabel("Principal Component 2 (Intensity Variance)")
    plt.legend(loc="best", markerscale=2)
    plt.grid(True, alpha=0.2)

    save_path = os.path.join(PATHS.ARCHETYPE_DIR, "archetype_projection_2d.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Cluster projection plot saved to: {save_path}")


def plot_cluster_samples(df, parquet_dir, unique_labels):
    plot_path = os.path.join(PATHS.ARCHETYPE_DIR, "plots")
    os.makedirs(plot_path, exist_ok=True)
    for cid in unique_labels:
        members = df.filter(pl.col("archetype_id") == cid)["msname"].to_list()
        if not members:
            continue
        n_samples = min(10, len(members))
        samples = np.random.choice(members, n_samples, replace=False)
        fig, axes = plt.subplots(10, 1, figsize=(20, 12))
        axes = axes.flatten()
        for i, ms in enumerate(samples):
            raw = (
                pl.scan_parquet(os.path.join(parquet_dir, "*.parquet"))
                .filter(pl.col("msname") == ms)
                .collect()
            )
            axes[i].plot(raw["cpu_utilization"].to_numpy(), color="navy")
            axes[i].set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"cluster_{cid}_samples.png"))
        plt.close()
    print(f"Cluster sample plots saved in: {plot_path}")


def analyze_label_stability(
    features_df,
    scaler,
    pca,
    classifier_model,
    parquet_glob,
    time_steps_to_test,
    clustering_cols,
):
    start_stability = time.perf_counter()
    temp_db_path = os.path.join("/dataset/duckdb_temp", "stability_check.db")
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

    con = duckdb.connect(temp_db_path)
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='32GB'")
    con.execute("PRAGMA max_temp_directory_size='300GiB'")
    con.execute("SET preserve_insertion_order=false")

    ground_truth_labels = features_df["archetype_id"].to_numpy()
    ms_names = features_df["msname"].to_list()
    stability_results = []

    print("Starting Sensitivity Analysis (PCA + HDBSCAN + KNN)...")

    for n in time_steps_to_test:
        step_start = time.perf_counter()
        print(f"Testing window size: {n} minutes...")

        query = f"""
        WITH ms_start AS (
            SELECT
                msname,
                min({PREPROCESSING.TIME_COL}) AS start_ts
            FROM read_parquet('{parquet_glob}')
            GROUP BY msname
        ),
        windowed AS (
            SELECT r.msname, r.cpu_utilization
            FROM read_parquet('{parquet_glob}') r
            JOIN ms_start s ON r.msname = s.msname
            WHERE r.{PREPROCESSING.TIME_COL} <= s.start_ts + INTERVAL '{n} minutes'
        )
        SELECT msname, 
               avg(cpu_utilization) as cpu_mean, 
               stddev_samp(cpu_utilization) as cpu_std,
               approx_quantile(cpu_utilization, 0.95) as cpu_p95, 
               ln(abs(skewness(cpu_utilization)) + 1) as cpu_skew,
               ln(abs(kurtosis(cpu_utilization)) + 1) as cpu_kurt, 
               (max(cpu_utilization) / NULLIF(avg(cpu_utilization), 0)) as peak_to_avg,
               (stddev_samp(cpu_utilization) / NULLIF(avg(cpu_utilization), 0)) as coeff_variation,
               (max(cpu_utilization) / (approx_quantile(cpu_utilization, 0.5) + 0.01)) as burstiness_ratio
        FROM windowed
        GROUP BY msname
        """

        partial_df = con.execute(query).df().fillna(0.0)

        partial_df = pd.merge(
            pd.DataFrame({"msname": ms_names}), partial_df, on="msname", how="left"
        ).fillna(0.0)

        partial_df["peak_to_avg"] = np.log1p(partial_df["peak_to_avg"])
        partial_df["burstiness_ratio"] = np.log1p(partial_df["burstiness_ratio"])

        partial_data = partial_df[clustering_cols].to_numpy()
        partial_data = np.nan_to_num(partial_data, nan=0.0)

        partial_data = np.clip(
            partial_data,
            np.percentile(partial_data, 1, axis=0),
            np.percentile(partial_data, 99, axis=0),
        )

        scaled_partial = scaler.transform(partial_data)
        pca_partial = pca.transform(scaled_partial)
        predicted_labels = classifier_model.predict(pca_partial)

        score = f1_score(ground_truth_labels, predicted_labels, average="weighted")
        stability_results.append(score)
        step_end = time.perf_counter()
        print(f"Window {n}m: F1={score:.4f} ({format_duration(step_end - step_start)})")

    con.close()
    return stability_results, time.perf_counter() - start_stability


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_services", type=int, default=PREPROCESSING.MAX_SERVICES)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--temp_dir", type=str, default="/dataset/duckdb_temp")
    args, _ = ap.parse_known_args()

    os.makedirs(PATHS.ARCHETYPE_DIR, exist_ok=True)
    durations = {}

    features_df, durations["Total Feature Extraction"] = extract_robust_features(
        PATHS.PARQUET_MSRESOURCE,
        max_services=args.max_services,
        batch_size=args.batch_size,
        temp_dir=args.temp_dir,
    )

    print("Determining archetypes using PCA + HDBSCAN...")
    start_clustering = time.perf_counter()

    feature_cols = [
        "cpu_mean",
        "cpu_std",
        "cpu_p95",
        "cpu_skew",
        "cpu_kurt",
        "peak_to_avg",
        "coeff_variation",
        "cpu_autocorr",
        "cpu_mad",
        "cpu_slope",
        "cpu_iqr",
        "burstiness_ratio",
    ]

    clustering_cols = [
        "cpu_skew",
        "cpu_kurt",
        "peak_to_avg",
        "coeff_variation",
        "burstiness_ratio",
    ]

    pdf = features_df.to_pandas()

    pdf["peak_to_avg"] = np.log1p(pdf["peak_to_avg"])
    pdf["burstiness_ratio"] = np.log1p(pdf["burstiness_ratio"])

    data_to_scale = pdf[clustering_cols].to_numpy()
    data_to_scale = np.nan_to_num(data_to_scale, nan=0.0)

    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    scaled_data = scaler.fit_transform(data_to_scale)

    pca = PCA(n_components=0.95, random_state=42)
    pca_data = pca.fit_transform(scaled_data)
    print(
        f"PCA reduced behavioral features from {len(clustering_cols)} to {pca.n_components_} components."
    )

    clusterer = HDBSCAN(
        min_cluster_size=100, min_samples=15, cluster_selection_method="leaf", copy=True
    )
    labels = clusterer.fit_predict(pca_data)

    plot_archetype_projection(pca_data, labels)

    print("Training KNN Classifier for stability analysis...")
    knn = KNeighborsClassifier(n_neighbors=5).fit(pca_data, labels)
    durations["PCA, Clustering & KNN Training"] = time.perf_counter() - start_clustering

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_mask = labels != -1

    if valid_mask.sum() > 0:
        sil_score = silhouette_score(pca_data[valid_mask], labels[valid_mask])
        db_score = davies_bouldin_score(pca_data[valid_mask], labels[valid_mask])
        ch_score = calinski_harabasz_score(pca_data[valid_mask], labels[valid_mask])
    else:
        sil_score, db_score, ch_score = 0, 0, 0

    print(f"Clusters Found: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
    print(
        f"Total Number of Clusters Found: {len(unique_labels) - (1 if -1 in unique_labels else 0)}"
    )
    print(f"Silhouette Score (Excluding Noise): {sil_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    print(f"Calinski-Harabasz Index: {ch_score:.4f}\n")

    for cluster_id, count in zip(unique_labels, counts):
        name = "Noise/General" if cluster_id == -1 else f"Archetype {cluster_id}"
        print(f"  {name}: {count} members")
    print()

    model_save_path = os.path.join(PATHS.ARCHETYPE_DIR, "hdbscan_model.joblib")
    scaler_save_path = os.path.join(PATHS.ARCHETYPE_DIR, "scaler.joblib")
    pca_save_path = os.path.join(PATHS.ARCHETYPE_DIR, "pca.joblib")
    knn_save_path = os.path.join(PATHS.ARCHETYPE_DIR, "knn_classifier.joblib")

    joblib.dump(clusterer, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump(pca, pca_save_path)
    joblib.dump(knn, knn_save_path)

    features_df = features_df.with_columns(pl.Series("archetype_id", labels))
    pdf["archetype_id"] = labels

    print("Calculating cluster feature averages...")
    cluster_stats = (
        features_df.group_by("archetype_id")
        .agg([pl.col(c).mean() for c in feature_cols])
        .sort("archetype_id")
    )

    print("\n" + "-" * 30)
    print("CLUSTER FEATURE AVERAGES")
    print("-" * 30)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(cluster_stats.to_pandas())
    print("-" * 30 + "\n")

    mapping = {
        row["msname"]: int(row["archetype_id"]) for row in features_df.to_dicts()
    }
    with open(PATHS.ARCHETYPE_MAPPING, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Archetype mapping saved to: {PATHS.ARCHETYPE_MAPPING}")

    plot_cluster_samples(features_df, PATHS.PARQUET_MSRESOURCE, unique_labels)

    windows = [15, 30, 60, 120, 240, 480, 720, 1440, 2880, 5760, 10080]
    scores, durations["Stability Analysis (Min Time Steps)"] = analyze_label_stability(
        pdf,
        scaler,
        pca,
        knn,
        os.path.join(PATHS.PARQUET_MSRESOURCE, "*.parquet"),
        windows,
        clustering_cols,
    )
    plot_stability_curve(windows, scores)

    save_cluster_metrics(
        len(unique_labels) - (1 if -1 in unique_labels else 0),
        sil_score,
        db_score,
        ch_score,
        (unique_labels, counts),
        durations,
        cluster_stats=cluster_stats,
    )

    print("\n" + "=" * 40)
    print("      FINAL RUNTIME SUMMARY")
    print("=" * 40)
    for stage, dur in durations.items():
        print(f"{stage:.<30} {format_duration(dur)}")
    print("=" * 40)


if __name__ == "__main__":
    main()
