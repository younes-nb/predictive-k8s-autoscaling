import os
import glob
import sys
import argparse
import shutil
import tempfile
import time
import gc
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_n_cpus = os.cpu_count() or 1
_NUM_WORKERS = 0.9
os.environ.setdefault(
    "POLARS_MAX_THREADS", str(max(1, int(_n_cpus * _NUM_WORKERS)))
)

import polars as pl
import numpy as np
from tqdm import tqdm

from core.utils import windowize_multivariate

from shared.config_paths import PATHS, DATASET_TABLES
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.features import FEATURE_SETS, get_feature_set, tables_for_feature_set, table_to_feature_exprs, FEATURES

from preprocessing.parquet_utils import (
    list_parquet_parts,
    build_table_agg,
    discover_unique_services,
)

_WORKER_CTX = {}


def save_chunk(out_dir, shard_idx, chunk_idx, shard_data, sync=False):
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
                if sync:
                    os.sync()

    except OSError as e:
        print(f"\nStaging Error: {e}")
        raise

    return saved_any


def _agg_exprs_for_table(table_exprs, table_name):
    import polars as pl

    exprs = []
    for feat_name, raw_col in table_exprs[table_name]:
        if table_name == "msresource":
            exprs.append(pl.col(raw_col).mean().alias(feat_name))
        elif table_name == "msrtmcre":
            exprs.append(pl.col(raw_col).sum().alias(feat_name))
        else:
            exprs.append(pl.col(raw_col).last().alias(feat_name))
    return exprs


def _process_service_group(group_idx, service_ids):
    import numpy as np

    ctx = _WORKER_CTX
    args_dict = ctx["args_dict"]
    out_dir = args_dict["out_dir"]
    done_marker = os.path.join(out_dir, f"part-{group_idx:04d}.done")
    if os.path.exists(done_marker):
        return (group_idx, 0, 0.0, True)

    t0 = time.time()
    arrays = ctx["service_arrays"]
    target_indices = ctx["target_indices"]

    shard_data = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
    n_processed = 0

    for ms_id in service_ids:
        feat_raw = arrays.get(ms_id)
        if feat_raw is None:
            continue

        n = len(feat_raw)
        idx_tr = int(n * args_dict["train_frac"])
        idx_val = int(n * (args_dict["train_frac"] + args_dict["val_frac"]))

        split_configs = [
            ("train", 0, idx_tr),
            ("val", idx_tr, idx_val),
            ("test", idx_val, n),
        ]

        for split_name, start, end in split_configs:
            sub_feat = feat_raw[start:end]
            if len(sub_feat) < args_dict["input_len"] + args_dict["pred_horizon"]:
                continue

            y_target = sub_feat[:, target_indices]
            if len(target_indices) == 1:
                y_target = y_target[:, 0]

            Xs, Ys, Ss = windowize_multivariate(
                sub_feat, y_target,
                args_dict["input_len"], args_dict["pred_horizon"], args_dict["stride"],
            )

            if Xs.size > 0:
                shard_data[split_name][0].append(Xs)
                shard_data[split_name][1].append(Ys)
                shard_data[split_name][2].append(Ss)
                n_processed += 1

    save_chunk(out_dir, group_idx, 0, shard_data, sync=ctx["sync"])
    open(done_marker, "a").close()

    del shard_data
    gc.collect()

    return (group_idx, n_processed, time.time() - t0, False)


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
    p.add_argument("--batch_size", type=int, default=0,
                    help="Services per worker group; 0 = auto-size to the worker pool (default: 0)")
    p.add_argument("--num_workers", type=float, default=0.9,
                    help="Fraction of CPU cores to use (default: 0.9)")
    p.add_argument("--recompute", action="store_true",
                    help="Delete cached done markers and shards, forcing a full rebuild")
    p.add_argument("--no_service_cache", action="store_true",
                    help="Bypass the unique-service discovery cache and scan parquet directly")
    p.add_argument("--refresh_service_cache", action="store_true",
                    help="Force rebuild of the unique-service discovery cache")
    p.add_argument("--sync", action="store_true",
                    help="Run os.sync() after saving each chunk (durability; slower)")

    args = p.parse_args()

    if (
        args.train_frac <= 0
        or args.val_frac < 0
        or (args.train_frac + args.val_frac >= 1.0)
    ):
        raise SystemExit("Invalid train/val fractions.")

    n_cpus = os.cpu_count() or 1
    num_workers = max(1, int(n_cpus * args.num_workers))
    print(
        f"n_cpus={n_cpus}, worker pool={num_workers}, "
        f"polars threads capped to {pl.thread_pool_size()}"
    )

    spec = get_feature_set(args.feature_set)
    feature_names = list(spec["features"])
    target_features = list(spec["targets"])
    target_indices = [feature_names.index(f) for f in target_features]

    needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))
    table_exprs = table_to_feature_exprs(args.feature_set)
    base_table = spec.get("base_table", FEATURES[target_features[0]]["table"])

    effective_id_cols = [args.service_col]

    table_parts: dict[str, list[str]] = {}
    for t in needed_tables:
        pq_dir = DATASET_TABLES[t]["parquet_dir"]
        parts = list_parquet_parts(pq_dir)
        if not parts:
            raise SystemExit(f"No parquet parts found for table='{t}'")
        table_parts[t] = parts

    base_parts = table_parts[base_table]

    print(
        f"Discovering unique services across {len(base_parts)} base shards "
        f"(polars parallel across all cores)..."
    )
    all_services_list = discover_unique_services(
        DATASET_TABLES[base_table]["parquet_dir"],
        args.service_col,
        use_cache=not args.no_service_cache,
        refresh=args.refresh_service_cache,
    )
    print(f"Total unique services: {len(all_services_list)}")

    if args.max_services and len(all_services_list) > args.max_services:
        rng = np.random.default_rng(args.subset_seed)
        idxs = rng.choice(len(all_services_list), size=args.max_services, replace=False)
        all_services_list = sorted(np.array(all_services_list)[idxs].tolist())
        print(f"Selected subset: {len(all_services_list)} services")
    else:
        print(f"Processing all {len(all_services_list)} services globally")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.recompute:
        cached = glob.glob(os.path.join(args.out_dir, "part-*.done")) + \
                 glob.glob(os.path.join(args.out_dir, "part-*_chunk-*.npy"))
        for f in cached:
            os.remove(f)
        if cached:
            print(f"Removed {len(cached)} cached artifacts for recompute")

    if args.batch_size and args.batch_size > 0:
        group_size = args.batch_size
    else:
        group_size = max(1, math.ceil(len(all_services_list) / num_workers))
    total_groups = (len(all_services_list) + group_size - 1) // group_size

    groups_to_run = []
    for gi in range(total_groups):
        done_marker = os.path.join(args.out_dir, f"part-{gi:04d}.done")
        if os.path.exists(done_marker):
            continue
        groups_to_run.append(gi)

    if not groups_to_run:
        print("\nAll service groups processed (all cached).")
        return

    args_dict = {
        "out_dir": args.out_dir,
        "time_col": args.time_col,
        "freq": args.freq,
        "input_len": args.input_len,
        "pred_horizon": args.pred_horizon,
        "stride": args.stride,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "service_col": args.service_col,
        "feature_set": args.feature_set,
    }

    agg_frames = {}
    agg_order = [base_table] + [t for t in needed_tables if t != base_table]
    for t in agg_order:
        parts = table_parts[t]
        schema = pl.scan_parquet(parts).collect_schema().names()
        has_service = args.service_col in schema

        need_cols = list(
            set([
                args.time_col,
                *effective_id_cols,
                *[raw for _, raw in table_exprs[t]],
            ])
        )
        lf = pl.scan_parquet(parts, low_memory=True).select(need_cols)
        if has_service:
            lf = lf.filter(pl.col(args.service_col).is_in(all_services_list))

        agg_frames[t] = build_table_agg(
            lf, args.time_col, effective_id_cols, args.freq, table_exprs[t],
            agg_exprs=_agg_exprs_for_table(table_exprs, t),
        ).collect(engine="streaming")
        print(f"Table '{t}' aggregated: {agg_frames[t].height} rows")

    if base_table not in agg_frames or agg_frames[base_table].is_empty():
        print("No base-table data after aggregation; nothing to do.")
        return

    joined = agg_frames[base_table].lazy()
    join_keys = FEATURE_SETS[args.feature_set].get("join_keys", {})
    for t in agg_order:
        if t == base_table:
            continue
        t_frame = agg_frames[t]
        if args.service_col in t_frame.columns:
            t_frame = t_frame.filter(pl.col(args.service_col).is_in(all_services_list))
        join_on = ["_t"] + join_keys.get(t, [])
        joined = joined.join(t_frame.lazy(), on=join_on, how="left")

    for feat in feature_names:
        is_resource = "cpu" in feat.lower() or "mem" in feat.lower()
        if is_resource:
            joined = joined.with_columns(pl.col(feat).clip(0.0, 1.0))

    sort_cols = list(
        set(effective_id_cols).intersection(joined.collect_schema().names())
    ) + ["_t"]

    joined_df = (
        joined.drop_nulls(feature_names)
        .collect(engine="streaming")
        .sort(sort_cols)
    )
    print(f"Joined/clean table: {joined_df.height} rows")

    if joined_df.height == 0:
        print("No valid rows after join/filtering; nothing to do.")
        return

    group_cols = [c for c in effective_id_cols if c in joined_df.columns]
    service_arrays = {}
    for ms_key, g in joined_df.group_by(group_cols, maintain_order=True):
        if g.height < args.input_len + args.pred_horizon:
            continue
        ms_id = ms_key if isinstance(ms_key, str) else ms_key[0]
        feat_arrays = {
            f: g[f].to_numpy().astype("float32") for f in feature_names
        }
        service_arrays[ms_id] = np.stack(
            [feat_arrays[f] for f in feature_names], axis=1
        )
    del joined_df
    gc.collect()
    print(f"Service feature arrays: {len(service_arrays)} services")

    if not service_arrays:
        print("No services with enough data after filtering; nothing to do.")
        return

    global _WORKER_CTX
    _WORKER_CTX = {
        "args_dict": args_dict,
        "service_arrays": service_arrays,
        "target_indices": target_indices,
        "sync": args.sync,
    }

    tasks = []
    for gi in groups_to_run:
        start_idx = gi * group_size
        end_idx = min(start_idx + group_size, len(all_services_list))
        tasks.append((gi, all_services_list[start_idx:end_idx]))

    print(f"Processing {len(tasks)} groups with {num_workers} workers (group_size={group_size})")

    pbar = tqdm(total=len(tasks), desc="Building windows", unit="group",
                bar_format=("{desc}: {percentage:5.1f}%|{bar}| "
                             "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
                             "{rate_fmt}]"))

    pool_kwargs = {}
    if os.name == "posix":
        pool_kwargs["mp_context"] = mp.get_context("fork")

    if num_workers <= 1:
        for gi, ids in tasks:
            _process_service_group(gi, ids)
            pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=num_workers, **pool_kwargs) as executor:
            futures = {
                executor.submit(_process_service_group, gi, ids): gi
                for gi, ids in tasks
            }
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    pbar.close()
    print("\nAll service groups processed.")


if __name__ == "__main__":
    main()
