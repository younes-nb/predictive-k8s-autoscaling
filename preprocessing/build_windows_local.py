import os
import glob
import sys
import argparse
import polars as pl
import numpy as np
import gc

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import (
    PREPROCESSING,
    FEATURE_SETS,
    DATASET_TABLES,
    FEATURES,
    get_feature_set,
    tables_for_feature_set,
    table_to_feature_exprs,
)
from common.utils import windowize_multivariate, moving_average


def list_parquet_parts(parquet_dir: str):
    return sorted(glob.glob(os.path.join(parquet_dir, "part-*.parquet")))


def build_table_agg(
    df_or_lazy, time_col: str, id_cols: list, freq: str, feature_exprs: list
):
    if isinstance(df_or_lazy, pl.DataFrame):
        df_or_lazy = df_or_lazy.lazy()
    df_or_lazy = df_or_lazy.with_columns(pl.col(time_col).cast(pl.Datetime))
    agg_exprs = [
        pl.col(raw_col).last().alias(feat_name) for feat_name, raw_col in feature_exprs
    ]
    return (
        df_or_lazy.with_columns(pl.col(time_col).dt.truncate(freq).alias("_t"))
        .group_by(["_t"] + id_cols)
        .agg(agg_exprs)
        .sort(["_t"] + id_cols)
    )


def save_chunk(out_dir, shard_idx, chunk_idx, shard_data):
    base = os.path.join(out_dir, f"part-{shard_idx:04d}_chunk-{chunk_idx:04d}")
    saved_any = False
    for split, (Xs, Ys, Ss) in shard_data.items():
        if Xs:
            np.save(f"{base}_X_{split}.npy", np.concatenate(Xs))
            np.save(f"{base}_y_{split}.npy", np.concatenate(Ys))
            np.save(f"{base}_sid_{split}.npy", np.concatenate(Ss))
            saved_any = True
    return saved_any


def process_csv_as_test(csv_path, out_dir, args, feature_names, target_idx):
    """Processes the load-test CSV using the same logic as traces."""
    print(f"\n--- 🧪 Processing Test Data from CSV: {csv_path} ---")

    df = pl.read_csv(csv_path).rename(
        {
            "RPS": "mcr_diff",
            "CPU": "cpu_utilization",
            "Memory": "memory_utilization",
            "Deployment": "msname",
        }
    )

    if "msinstanceid" not in df.columns:
        df = df.with_columns(pl.lit("local-instance").alias("msinstanceid"))

    epsilon = 1e-6
    df = df.with_columns(
        (
            (pl.col("mcr_diff") + epsilon)
            .log()
            .diff()
            .over("msname")
            .fill_null(0.0)
            .tanh()
        ).alias("mcr_diff")
    )

    shard_data = {"test": ([], [], [])}
    group_cols = ["msname", "msinstanceid"]

    for _, g in df.group_by(group_cols, maintain_order=True):
        if g.height < args.input_len + args.pred_horizon:
            continue

        feat_raw = np.stack(
            [g[f].to_numpy().astype("float32") for f in feature_names], axis=1
        )
        feat_processed = np.zeros_like(feat_raw)

        for j in range(len(feature_names)):
            vals = moving_average(feat_raw[:, j], args.smoothing_window)
            feat_processed[:, j] = vals if vals is not None else 0.0

        X, Y, S = windowize_multivariate(
            feat_processed,
            feat_processed[:, target_idx],
            args.input_len,
            args.pred_horizon,
            args.stride,
        )
        shard_data["test"][0].append(X)
        shard_data["test"][1].append(Y)
        shard_data["test"][2].append(S)

    save_chunk(out_dir, 9999, 0, shard_data) 
    print("✅ Load Test CSV windows saved successfully.")


def main():
    p = argparse.ArgumentParser(
        description="Build windows for Train/Val (Parquet) and Test (CSV)."
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--csv_path", required=True)
    p.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET)
    p.add_argument("--time_col", default=PREPROCESSING.TIME_COL)
    p.add_argument("--id_cols", nargs="+", default=list(PREPROCESSING.ID_COLS))
    p.add_argument("--freq", default=PREPROCESSING.FREQ)
    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--pred_horizon", type=int, default=PREPROCESSING.PRED_HORIZON)
    p.add_argument("--stride", type=int, default=PREPROCESSING.STRIDE)
    p.add_argument(
        "--smoothing_window", type=int, default=PREPROCESSING.SMOOTHING_WINDOW
    )
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--service_col", type=str, default=PREPROCESSING.SERVICE_COL)
    args = p.parse_args()

    spec = get_feature_set(args.feature_set)
    feature_names = list(spec["features"])
    target_idx = feature_names.index(str(spec["target"]))

    needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))
    table_exprs = table_to_feature_exprs(args.feature_set)
    base_table = FEATURES[str(spec["target"])]["table"]

    os.makedirs(args.out_dir, exist_ok=True)

    print("--- 📚 Processing Train/Val Data from Parquet Traces ---")
    table_parts = {
        t: list_parquet_parts(DATASET_TABLES[t]["parquet_dir"]) for t in needed_tables
    }

    for shard_idx, base_pq in enumerate(table_parts[base_table]):
        gc.collect()
        base_id_cols = DATASET_TABLES[base_table]["key_cols"]

        lf_base = pl.scan_parquet(base_pq)
        joined_lazy = build_table_agg(
            lf_base, args.time_col, base_id_cols, args.freq, table_exprs[base_table]
        )

        joined = joined_lazy.collect()

        if "mcr_diff" in feature_names:
            epsilon = 1e-6
            joined = joined.with_columns(
                (
                    (pl.col("mcr_diff") + epsilon)
                    .log()
                    .diff()
                    .over(base_id_cols)
                    .fill_null(0.0)
                    .tanh()
                ).alias("mcr_diff")
            )

        shard_data = {"train": ([], [], []), "val": ([], [], [])}
        for _, g in joined.group_by(base_id_cols, maintain_order=True):
            if g.height < args.input_len + args.pred_horizon:
                continue

            feat_raw = np.stack(
                [g[f].to_numpy().astype("float32") for f in feature_names], axis=1
            )
            feat_processed = np.zeros_like(feat_raw)
            for j in range(len(feature_names)):
                vals = moving_average(feat_raw[:, j], args.smoothing_window)
                feat_processed[:, j] = vals if vals is not None else 0.0

            X_all, Y_all, S_all = windowize_multivariate(
                feat_processed,
                feat_processed[:, target_idx],
                args.input_len,
                args.pred_horizon,
                args.stride,
            )

            n_w = len(X_all)
            cut = max(int(n_w * 0.9), 1)
            shard_data["train"][0].append(X_all[:cut])
            shard_data["train"][1].append(Y_all[:cut])
            shard_data["train"][2].append(S_all[:cut])
            shard_data["val"][0].append(X_all[cut:])
            shard_data["val"][1].append(Y_all[cut:])
            shard_data["val"][2].append(S_all[cut:])

        save_chunk(args.out_dir, shard_idx, 0, shard_data)

    process_csv_as_test(args.csv_path, args.out_dir, args, feature_names, target_idx)


if __name__ == "__main__":
    main()
