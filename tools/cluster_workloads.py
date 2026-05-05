import os
import json
import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

from config.defaults import PATHS, ARCHETYPES, PREPROCESSING


def extract_robust_features(parquet_dir: str, max_services: int = None):
    print(f"Scanning parquet files in {parquet_dir}...")

    lf = (
        pl.scan_parquet(os.path.join(parquet_dir, "*.parquet"))
        .group_by(["msname", PREPROCESSING.TIME_COL])
        .agg(
            [
                pl.col("cpu_utilization").mean().alias("cpu_utilization"),
            ]
        )
    )

    features = (
        lf.group_by("msname")
        .agg(
            [
                pl.col("cpu_utilization").mean().alias("cpu_mean"),
                pl.col("cpu_utilization").std().alias("cpu_std"),
                pl.col("cpu_utilization").quantile(0.95).alias("cpu_p95"),
                pl.col("cpu_utilization").skew().alias("cpu_skew"),
                pl.col("cpu_utilization").kurtosis().alias("cpu_kurt"),
                pl.count().alias("sample_count"),
            ]
        )
        .filter(pl.col("sample_count") > 10)
        .collect()
    )

    if max_services is not None:
        print(f"Limiting clustering to {max_services} microservices...")
        features = features.head(max_services)

    return features


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
    args, _ = ap.parse_known_args()

    os.makedirs(PATHS.ARCHETYPE_DIR, exist_ok=True)

    features_df = extract_robust_features(
        PATHS.PARQUET_MSRESOURCE, max_services=args.max_services
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
