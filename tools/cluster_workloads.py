import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import glob

from config.defaults import PATHS, ARCHETYPES, PREPROCESSING


def extract_workload_features(parquet_path: str):
    print(f"Reading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    id_cols = list(PREPROCESSING.ID_COLS)

    features = (
        df.groupby(id_cols)["cpu_utilization"]
        .agg(
            [
                ("mean", "mean"),
                ("std", "std"),
                ("p95", lambda x: x.quantile(0.95)),
                ("skew", "skew"),
                ("kurtosis", lambda x: x.kurtosis()),
            ]
        )
        .reset_index()
    )

    features = features.fillna(0)

    return features


def find_optimal_k(feature_matrix: np.ndarray):
    wcss = []
    k_range = range(ARCHETYPES.MIN_K, ARCHETYPES.MAX_K + 1)

    print(f"Evaluating k from {ARCHETYPES.MIN_K} to {ARCHETYPES.MAX_K}...")

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=ARCHETYPES.RANDOM_STATE, n_init=10)
        kmeans.fit(feature_matrix)
        wcss.append(kmeans.inertia_)

    kn = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")

    optimal_k = kn.knee

    if optimal_k is None:
        print("Warning: No clear knee found. Defaulting to MIN_K.")
        optimal_k = ARCHETYPES.MIN_K

    print(f"Optimal clusters detected: {optimal_k}")

    metrics_df = pd.DataFrame({"k": list(k_range), "wcss": wcss})
    metrics_df.to_csv(PATHS.CLUSTER_STATS, index=False)

    return optimal_k


def main():
    os.makedirs(PATHS.ARCHETYPE_DIR, exist_ok=True)

    parquet_files = glob.glob(os.path.join(PATHS.PARQUET_MSRESOURCE, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {PATHS.PARQUET_MSRESOURCE}")

    features_df = extract_workload_features(parquet_files[0])

    scaler = StandardScaler()
    feature_cols = ["mean", "std", "p95", "skew", "kurtosis"]
    scaled_data = scaler.fit_transform(features_df[feature_cols])

    best_k = find_optimal_k(scaled_data)

    final_kmeans = KMeans(
        n_clusters=best_k, random_state=ARCHETYPES.RANDOM_STATE, n_init=10
    )
    features_df["archetype_id"] = final_kmeans.fit_transform(scaled_data).argmin(axis=1)

    mapping = {}
    for _, row in features_df.iterrows():
        s_id = f"{row['msname']}:{row['msinstanceid']}"
        mapping[s_id] = int(row["archetype_id"])

    with open(PATHS.ARCHETYPE_MAPPING, "w") as f:
        json.dump(mapping, f, indent=4)

    print(
        f"Successfully saved archetype mapping for {len(mapping)} services to {PATHS.ARCHETYPE_MAPPING}"
    )


if __name__ == "__main__":
    main()
