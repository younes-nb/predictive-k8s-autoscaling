import os
import json
import argparse
import sys
import polars as pl
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS, ARCHETYPES, PREPROCESSING


def extract_robust_features(
    parquet_dir: str,
    max_services: int = None,
    batch_size: int = 50,
    temp_dir: str = "/dataset/duckdb_temp",
):
    print(f"Scanning parquet files in {parquet_dir}...")

    if not os.path.exists(temp_dir):
        print(f"📁 Creating directory: {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)

    db_path = os.path.join(temp_dir, "cluster_processing.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='20GB'")

    parquet_glob = os.path.join(parquet_dir, "*.parquet")

    print("🔍 Phase 1: Identifying unique microservices...")
    unique_ms_query = f"SELECT DISTINCT msname FROM read_parquet('{parquet_glob}')"
    unique_msnames = con.execute(unique_ms_query).df()["msname"].tolist()
    unique_msnames = [m for m in unique_msnames if pd.notna(m)]

    if max_services is not None:
        print(f"Limiting clustering to {max_services} microservices...")
        unique_msnames = unique_msnames[:max_services]

    total_services = len(unique_msnames)
    print(f"🎯 Found {total_services} unique microservices to process.")

    con.execute("""
        CREATE TABLE ms_features (
            msname VARCHAR,
            cpu_mean DOUBLE,
            cpu_std DOUBLE,
            cpu_p95 DOUBLE,
            cpu_skew DOUBLE,
            cpu_kurt DOUBLE,
            sample_count BIGINT
        )
        """)

    total_batches = (total_services + batch_size - 1) // batch_size
    print(f"🔍 Phase 2: Processing in {total_batches} batches to bypass OOM limits...")

    time_col = PREPROCESSING.TIME_COL

    for i in range(0, total_services, batch_size):
        batch = unique_msnames[i : i + batch_size]
        msnames_sql_list = ", ".join([f"'{m}'" for m in batch])
        batch_num = (i // batch_size) + 1

        print(
            f"📦 Extracting features for Batch {batch_num}/{total_batches} ({len(batch)} services)...",
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

    print("✅ Feature extraction complete. Loading results into memory...")
    df_pandas = con.execute("SELECT * FROM ms_features").df()

    con.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    df_pandas = df_pandas.fillna(0.0)

    return pl.from_pandas(df_pandas)


def plot_cluster_samples(df, labels, parquet_dir, n_clusters):
    os.makedirs(os.path.join(PATHS.ARCHETYPE_DIR, "plots"), exist_ok=True)

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

        plt.suptitle(f"Archetype {cid} Sample Workloads")
        plt.tight_layout()
        plt.savefig(
            os.path.join(PATHS.ARCHETYPE_DIR, "plots", f"cluster_{cid}_samples.png")
        )
        plt.close()


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

    features_df = extract_robust_features(
        PATHS.PARQUET_MSRESOURCE,
        max_services=args.max_services,
        batch_size=args.batch_size,
        temp_dir=args.temp_dir,
    )

    feature_cols = ["cpu_mean", "cpu_std", "cpu_p95", "cpu_skew", "cpu_kurt"]
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

    sil_score = silhouette_score(scaled_data, labels)
    print(f"Optimal K: {best_k} | Silhouette Score: {sil_score:.4f}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nTotal Number of Clusters: {best_k}")
    for cluster_id, count in zip(unique_labels, counts):
        print(f"  Cluster {cluster_id}: {count} members")
    print()

    features_df = features_df.with_columns(pl.Series("archetype_id", labels))

    mapping = {
        row["msname"]: int(row["archetype_id"]) for row in features_df.to_dicts()
    }
    with open(PATHS.ARCHETYPE_MAPPING, "w") as f:
        json.dump(mapping, f, indent=4)

    plot_cluster_samples(features_df, labels, PATHS.PARQUET_MSRESOURCE, best_k)


if __name__ == "__main__":
    main()
