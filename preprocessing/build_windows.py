import os
import glob
import sys
import argparse
import polars as pl
import numpy as np
import gc
import shutil
import tempfile

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
from core.utils import windowize_multivariate, moving_average


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

    out = (
        df_or_lazy.with_columns(pl.col(time_col).dt.truncate(freq).alias("_t"))
        .group_by(["_t"] + id_cols)
        .agg(agg_exprs)
        .sort(["_t"] + id_cols)
    )
    return out


def save_chunk(out_dir, shard_idx, chunk_idx, shard_data):
    base_name = f"part-{shard_idx:04d}_chunk-{chunk_idx:04d}"
    saved_any = False

    try:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp_dir:
            tmp_base = os.path.join(tmp_dir, base_name)

            for split, (Xs, Ys, Ss) in shard_data.items():
                if Xs:
                    np.save(f"{tmp_base}_X_{split}.npy", np.concatenate(Xs))
                    np.save(f"{tmp_base}_y_{split}.npy", np.concatenate(Ys))
                    np.save(f"{tmp_base}_sid_{split}.npy", np.concatenate(Ss))
                    saved_any = True

            if saved_any:
                for src_file in glob.glob(f"{tmp_base}*"):
                    file_name = os.path.basename(src_file)
                    dest_file = os.path.join(out_dir, file_name)

                    shutil.move(src_file, dest_file)

    except OSError as e:
        if "Read-only file system" in str(e):
            print(f"\nCRITICAL: /dataset drive locked up during move: {e}")
        else:
            print(f"\nStaging Error: {e}")
        raise

    return saved_any


def main():
    p = argparse.ArgumentParser(
        description="Build windows using global service batching to ensure correct splitting and bypass OOM."
    )

    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--feature_set",
        type=str,
        default=PREPROCESSING.FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
    )
    p.add_argument("--time_col", default=PREPROCESSING.TIME_COL)
    p.add_argument("--id_cols", nargs="+", default=list(PREPROCESSING.ID_COLS))
    p.add_argument("--freq", default=PREPROCESSING.FREQ)
    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--pred_horizon", type=int, default=PREPROCESSING.PRED_HORIZON)
    p.add_argument("--stride", type=int, default=PREPROCESSING.STRIDE)
    p.add_argument("--train_frac", type=float, default=PREPROCESSING.TRAIN_FRAC)
    p.add_argument("--val_frac", type=float, default=PREPROCESSING.VAL_FRAC)
    p.add_argument(
        "--smoothing_window", type=int, default=PREPROCESSING.SMOOTHING_WINDOW
    )
    p.add_argument("--service_col", type=str, default=PREPROCESSING.SERVICE_COL)
    p.add_argument("--max_services", type=int, default=PREPROCESSING.MAX_SERVICES)
    p.add_argument("--subset_seed", type=int, default=PREPROCESSING.SUBSET_SEED)
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    args = p.parse_args()

    if (
        args.train_frac <= 0
        or args.val_frac < 0
        or (args.train_frac + args.val_frac >= 1.0)
    ):
        raise SystemExit("Invalid train/val fractions.")

    spec = get_feature_set(args.feature_set)
    feature_names = list(spec["features"])
    target_feature = str(spec["target"])
    target_idx = feature_names.index(target_feature)

    needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))
    table_exprs = table_to_feature_exprs(args.feature_set)
    base_table = FEATURES[target_feature]["table"]

    table_parts: dict[str, list[str]] = {}
    for t in needed_tables:
        pq_dir = DATASET_TABLES[t]["parquet_dir"]
        parts = list_parquet_parts(pq_dir)
        if not parts:
            raise SystemExit(f"No parquet parts found for table='{t}'")
        table_parts[t] = parts

    print(
        f"Discovering unique services across all {len(table_parts[base_table])} base shards..."
    )
    all_services_df = (
        pl.scan_parquet(table_parts[base_table])
        .select(args.service_col)
        .unique()
        .collect(engine="streaming")
    )
    all_services_list = sorted(all_services_df[args.service_col].to_list())

    if args.max_services and len(all_services_list) > args.max_services:
        rng = np.random.default_rng(args.subset_seed)
        idxs = rng.choice(len(all_services_list), size=args.max_services, replace=False)
        all_services_list = sorted(np.array(all_services_list)[idxs].tolist())
        print(f"Selected subset: {len(all_services_list)} services")
    else:
        print(f"Processing all {len(all_services_list)} services globally")

    os.makedirs(args.out_dir, exist_ok=True)
    total_batches = (len(all_services_list) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(total_batches):
        gc.collect()
        start_idx = batch_idx * args.batch_size
        end_idx = start_idx + args.batch_size
        current_batch_ids = all_services_list[start_idx:end_idx]

        print(
            f"\n=== Global Batch {batch_idx+1}/{total_batches} ({len(current_batch_ids)} services) ==="
        )

        base_id_cols = DATASET_TABLES[base_table]["key_cols"]
        base_need_cols = list(
            set(
                [
                    args.time_col,
                    *base_id_cols,
                    *[raw for _, raw in table_exprs[base_table]],
                ]
            )
        )

        lf_base = (
            pl.scan_parquet(table_parts[base_table])
            .filter(pl.col(args.service_col).is_in(current_batch_ids))
            .select(base_need_cols)
        )

        joined_lazy = build_table_agg(
            lf_base, args.time_col, base_id_cols, args.freq, table_exprs[base_table]
        )

        bounds = lf_base.select(
            [
                pl.col(args.time_col).min().alias("min_t"),
                pl.col(args.time_col).max().alias("max_t"),
            ]
        ).collect()

        if bounds.height == 0 or bounds["min_t"][0] is None:
            continue
        min_t, max_t = bounds["min_t"][0], bounds["max_t"][0]

        for t in needed_tables:
            if t == base_table:
                continue
            t_id_cols = DATASET_TABLES[t]["key_cols"]
            t_need_cols = list(
                set([args.time_col, *t_id_cols, *[raw for _, raw in table_exprs[t]]])
            )

            lf_t = pl.scan_parquet(table_parts[t]).filter(
                (pl.col(args.time_col).cast(pl.Datetime) >= min_t)
                & (pl.col(args.time_col).cast(pl.Datetime) <= max_t)
            )

            t_schema = pl.scan_parquet(table_parts[t]).collect_schema().names()
            if args.service_col in t_schema:
                lf_t = lf_t.filter(pl.col(args.service_col).is_in(current_batch_ids))

            lf_t_agg = build_table_agg(
                lf_t.select(t_need_cols),
                args.time_col,
                t_id_cols,
                args.freq,
                table_exprs[t],
            )

            join_on = ["_t"] + FEATURE_SETS[args.feature_set].get("join_keys", {}).get(
                t, []
            )
            joined_lazy = joined_lazy.join(lf_t_agg, on=join_on, how="left")

        for feat in feature_names:
            is_resource = "cpu" in feat.lower() or "mem" in feat.lower()
            if is_resource and "diff" not in feat.lower():
                joined_lazy = joined_lazy.with_columns(pl.col(feat).clip(0.0, 1.0))

        sort_cols = list(
            set(args.id_cols).intersection(joined_lazy.collect_schema().names())
        ) + ["_t"]
        joined = joined_lazy.drop_nulls(feature_names).sort(sort_cols).collect()
        del joined_lazy
        gc.collect()

        if joined.height == 0:
            continue

        shard_data = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
        group_cols = [c for c in args.id_cols if c in joined.columns]

        for _, g in joined.group_by(group_cols, maintain_order=True):
            if g.height < args.input_len + args.pred_horizon:
                continue

            feat_raw = np.stack(
                [g[feat].to_numpy().astype("float32") for feat in feature_names], axis=1
            )
            feat_processed = np.zeros_like(feat_raw)
            valid_group = True

            for j in range(len(feature_names)):
                vals = moving_average(feat_raw[:, j], args.smoothing_window)
                if vals is None:
                    if j == target_idx:
                        valid_group = False
                        break
                    feat_processed[:, j] = 0.0
                    continue
                feat_processed[:, j] = vals

            if not valid_group:
                continue

            n = len(feat_processed)
            idx_tr = int(n * args.train_frac)
            idx_val = int(n * (args.train_frac + args.val_frac))

            split_configs = [
                ("train", 0, idx_tr),
                ("val", idx_tr, idx_val),
                ("test", idx_val, n),
            ]

            for split_name, start, end in split_configs:
                sub_feat = feat_processed[start:end]
                if len(sub_feat) < args.input_len + args.pred_horizon:
                    continue

                Xs, Ys, Ss = windowize_multivariate(
                    sub_feat,
                    sub_feat[:, target_idx],
                    args.input_len,
                    args.pred_horizon,
                    args.stride,
                )

                if Xs.size > 0:
                    shard_data[split_name][0].append(Xs)
                    shard_data[split_name][1].append(Ys)
                    shard_data[split_name][2].append(Ss)

        save_chunk(args.out_dir, batch_idx, 0, shard_data)
        del joined, shard_data
        gc.collect()

    print("\nAll global batches processed.")


if __name__ == "__main__":
    main()
