import os
import glob
import sys
import argparse
import shutil
import tempfile
import time
import gc

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import polars as pl
import numpy as np
from tqdm import tqdm

from core.utils import windowize_multivariate

from shared.config_paths import PATHS, DATASET_TABLES
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.features import FEATURE_SETS, get_feature_set, tables_for_feature_set, table_to_feature_exprs, FEATURES

from preprocessing.parquet_utils import list_parquet_parts, build_table_agg


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
                files_to_move = glob.glob(f"{tmp_base}*")
                for src_file in files_to_move:
                    file_name = os.path.basename(src_file)
                    dest_file = os.path.join(out_dir, file_name)
                    shutil.move(src_file, dest_file)
                    os.sync()
                    time.sleep(2.0)

    except OSError as e:
        print(f"\nStaging Error: {e}")
        raise

    return saved_any


def main():
    p = argparse.ArgumentParser(
        description="Build windows: join tables, create sliding windows, split train/val/test."
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
    p.add_argument("--service_col", type=str, default=PREPROCESSING.SERVICE_COL)
    p.add_argument("--max_services", type=int, default=PREPROCESSING.MAX_SERVICES)
    p.add_argument("--subset_seed", type=int, default=PREPROCESSING.SUBSET_SEED)
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    args = p.parse_args()
    rng = np.random.default_rng(args.subset_seed)

    if (
        args.train_frac <= 0
        or args.val_frac < 0
        or (args.train_frac + args.val_frac >= 1.0)
    ):
        raise SystemExit("Invalid train/val fractions.")

    spec = get_feature_set(args.feature_set)
    feature_names = list(spec["features"])
    target_features = list(spec["targets"])
    target_indices = [feature_names.index(f) for f in target_features]

    needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))
    table_exprs = table_to_feature_exprs(args.feature_set)
    base_table = spec.get("base_table", FEATURES[target_features[0]]["table"])

    effective_id_cols = [args.service_col]

    def agg_exprs_for_table(table_name: str):
        exprs = []
        for feat_name, raw_col in table_exprs[table_name]:
            if table_name == "msresource":
                exprs.append(pl.col(raw_col).mean().alias(feat_name))
            elif table_name == "msrtmcre":
                exprs.append(pl.col(raw_col).sum().alias(feat_name))
            else:
                exprs.append(pl.col(raw_col).last().alias(feat_name))
        return exprs

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

    pbar = tqdm(total=total_batches, desc="Building windows", unit="batch",
                bar_format=("{desc}: {percentage:5.1f}%|{bar}| "
                             "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
                             "{rate_fmt}]"))

    for batch_idx in range(total_batches):
        gc.collect()

        done_marker = os.path.join(args.out_dir, f"part-{batch_idx:04d}.done")
        if os.path.exists(done_marker):
            pbar.set_postfix_str(f"batch {batch_idx+1} cached")
            pbar.update(1)
            continue

        start_idx = batch_idx * args.batch_size
        end_idx = start_idx + args.batch_size
        current_batch_ids = all_services_list[start_idx:end_idx]

        pbar.set_postfix_str(f"{len(current_batch_ids)} services")
        batch_start_time = time.time()

        load_ids = sorted(set(current_batch_ids))

        base_id_cols = effective_id_cols
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
            .filter(pl.col(args.service_col).is_in(load_ids))
            .select(base_need_cols)
        )

        joined_lazy = build_table_agg(
            lf_base,
            args.time_col,
            base_id_cols,
            args.freq,
            table_exprs[base_table],
            agg_exprs=agg_exprs_for_table(base_table),
        )

        bounds = lf_base.select(
            [
                pl.col(args.time_col).min().alias("min_t"),
                pl.col(args.time_col).max().alias("max_t"),
            ]
        ).collect()

        if bounds.height == 0 or bounds["min_t"][0] is None:
            open(done_marker, "a").close()
            continue

        min_t, max_t = bounds["min_t"][0], bounds["max_t"][0]

        for t in needed_tables:
            if t == base_table:
                continue
            t_id_cols = effective_id_cols
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
                agg_exprs=agg_exprs_for_table(t),
            )

            join_on = ["_t"] + FEATURE_SETS[args.feature_set].get("join_keys", {}).get(
                t, []
            )
            joined_lazy = joined_lazy.join(lf_t_agg, on=join_on, how="left")

        for feat in feature_names:
            is_resource = "cpu" in feat.lower() or "mem" in feat.lower()
            if is_resource:
                joined_lazy = joined_lazy.with_columns(pl.col(feat).clip(0.0, 1.0))

        sort_cols = list(
            set(effective_id_cols).intersection(joined_lazy.collect_schema().names())
        ) + ["_t"]

        joined = (
            joined_lazy.drop_nulls(feature_names)
            .collect(engine="streaming")
            .sort(sort_cols)
        )
        del joined_lazy
        gc.collect()

        if joined.height == 0:
            open(done_marker, "a").close()
            continue

        shard_data = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
        batch_set = set(current_batch_ids)

        group_cols = [c for c in effective_id_cols if c in joined.columns]

        for ms_key, g in joined.group_by(group_cols, maintain_order=True):
            if g.height < args.input_len + args.pred_horizon:
                continue

            ms_id = ms_key if isinstance(ms_key, str) else ms_key[0]

            if ms_id not in batch_set:
                continue

            feat_arrays = {}
            for feat in feature_names:
                feat_arrays[feat] = g[feat].to_numpy().astype("float32")

            feat_raw = np.stack(
                [feat_arrays[feat] for feat in feature_names], axis=1
            )

            n = len(feat_raw)
            idx_tr = int(n * args.train_frac)
            idx_val = int(n * (args.train_frac + args.val_frac))

            split_configs = [
                ("train", 0, idx_tr),
                ("val", idx_tr, idx_val),
                ("test", idx_val, n),
            ]

            for split_name, start, end in split_configs:
                sub_feat = feat_raw[start:end]
                if len(sub_feat) < args.input_len + args.pred_horizon:
                    continue

                y_target = sub_feat[:, target_indices]
                if len(target_indices) == 1:
                    y_target = y_target[:, 0]

                Xs, Ys, Ss = windowize_multivariate(
                    sub_feat,
                    y_target,
                    args.input_len,
                    args.pred_horizon,
                    args.stride,
                )

                if Xs.size > 0:
                    shard_data[split_name][0].append(Xs)
                    shard_data[split_name][1].append(Ys)
                    shard_data[split_name][2].append(Ss)

        save_chunk(args.out_dir, batch_idx, 0, shard_data)

        open(done_marker, "a").close()

        del joined, shard_data
        gc.collect()

        pbar.update(1)

    pbar.close()
    print("\nAll global batches processed.")


if __name__ == "__main__":
    main()
