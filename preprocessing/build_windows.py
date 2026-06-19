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

from core.utils import windowize_multivariate, moving_average

from shared.config_paths import PATHS, DATASET_TABLES
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.config_training_defaults import TRAINING
from shared.features import FEATURE_SETS, get_feature_set, tables_for_feature_set, table_to_feature_exprs, FEATURES, is_derived_feature

from preprocessing.parquet_utils import list_parquet_parts, build_table_agg
from preprocessing.adjacency import build_adjacency_map
from shared.smote_tomek import _apply_smote_tomek


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


def compute_delta(arr: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(arr)
    delta[1:] = arr[1:] - arr[:-1]
    return delta


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
    p.add_argument(
        "--smote_tomek",
        action="store_true",
        default=PREPROCESSING.SMOTE_TOMEK,
        help="Apply SMOTE-Tomek to training windows.",
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
    target_feature = str(spec["target"])
    target_idx = feature_names.index(target_feature)
    mcr_feats = [f for f in feature_names if "mcr" in f.lower()]

    requires_callgraph = spec.get("requires_callgraph", False)
    requires_delta = spec.get("requires_delta", False)

    needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))
    table_exprs = table_to_feature_exprs(args.feature_set)
    base_table = FEATURES[target_feature]["table"]

    use_service_level = args.feature_set in ("cpu_mem_mcr", "cpu_delta_upstream")
    effective_id_cols = [args.service_col] if use_service_level else list(args.id_cols)

    def agg_exprs_for_table(table_name: str):
        exprs = []
        for feat_name, raw_col in table_exprs[table_name]:
            if use_service_level and table_name == "msresource":
                exprs.append(pl.col(raw_col).mean().alias(feat_name))
            elif use_service_level and table_name == "msrtmcre":
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

    adjacency = {}
    if requires_callgraph:
        cache_path = os.path.join(PATHS.PARQUET_ROOT, "cache", "adjacency.pkl")
        print(f"Building adjacency map from MSCallGraph...")
        adjacency = build_adjacency_map(set(all_services_list), cache_path)
        print(f"Adjacency map built: {len(adjacency)} services with upstream callers")

    mcr_minmax = {}
    if use_service_level and mcr_feats and "msrtmcre" in needed_tables:
        mcr_raw_cols = [
            raw for feat, raw in table_exprs["msrtmcre"] if feat in mcr_feats
        ]
        lf_mcr = pl.scan_parquet(table_parts["msrtmcre"]).select(
            [args.time_col, *effective_id_cols, *mcr_raw_cols]
        )
        mcr_agg = build_table_agg(
            lf_mcr,
            args.time_col,
            effective_id_cols,
            args.freq,
            table_exprs["msrtmcre"],
            agg_exprs=agg_exprs_for_table("msrtmcre"),
        )
        mcr_stats = mcr_agg.select(
            *[pl.col(f).min().alias(f"{f}_min") for f in mcr_feats],
            *[pl.col(f).max().alias(f"{f}_max") for f in mcr_feats],
        ).collect(engine="streaming")
        if mcr_stats.height > 0:
            mcr_minmax = {
                f: (float(mcr_stats[f"{f}_min"][0]), float(mcr_stats[f"{f}_max"][0]))
                for f in mcr_feats
            }

    os.makedirs(args.out_dir, exist_ok=True)
    total_batches = (len(all_services_list) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(total_batches):
        gc.collect()

        done_marker = os.path.join(args.out_dir, f"part-{batch_idx:04d}.done")
        if os.path.exists(done_marker):
            print(
                f"\n=== Skipping Global Batch {batch_idx+1}/{total_batches} (Already completed) ==="
            )
            continue

        start_idx = batch_idx * args.batch_size
        end_idx = start_idx + args.batch_size
        current_batch_ids = all_services_list[start_idx:end_idx]

        print(
            f"\n=== Global Batch {batch_idx+1}/{total_batches} ({len(current_batch_ids)} services) ==="
        )

        batch_start_time = time.time()

        load_ids = set(current_batch_ids)
        if requires_callgraph:
            for ms in current_batch_ids:
                upstream_callers = adjacency.get(ms, set())
                load_ids.update(upstream_callers)
            load_ids = sorted(load_ids)
            print(f"  Loading {len(load_ids)} services (including upstream callers)")

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

        if mcr_minmax:
            for f in mcr_feats:
                fmin, fmax = mcr_minmax[f]
                denom = (fmax - fmin) if (fmax - fmin) != 0 else 1.0
                joined_lazy = joined_lazy.with_columns(
                    ((pl.col(f) - fmin) / denom).clip(0.0, 1.0).alias(f)
                )

        upstream_data = {}
        if requires_callgraph:
            for ms in current_batch_ids:
                upstream_callers = adjacency.get(ms, set())
                upstream_data[ms] = list(upstream_callers)

        for feat in feature_names:
            if is_derived_feature(feat):
                continue
            is_resource = "cpu" in feat.lower() or "mem" in feat.lower()
            if is_resource:
                joined_lazy = joined_lazy.with_columns(pl.col(feat).clip(0.0, 1.0))

        sort_cols = list(
            set(effective_id_cols).intersection(joined_lazy.collect_schema().names())
        ) + ["_t"]

        raw_feature_names = [f for f in feature_names if not is_derived_feature(f)]
        joined = (
            joined_lazy.drop_nulls(raw_feature_names)
            .collect(engine="streaming")
            .sort(sort_cols)
        )
        del joined_lazy
        gc.collect()

        if joined.height == 0:
            open(done_marker, "a").close()
            continue

        # Build service_data for upstream lookups
        service_data = {}
        group_cols = [c for c in effective_id_cols if c in joined.columns]
        if requires_callgraph:
            for ms_key, g in joined.group_by(group_cols, maintain_order=True):
                ms_id = ms_key if isinstance(ms_key, str) else ms_key[0]
                service_data[ms_id] = {
                    "timestamps": g["_t"].to_numpy(),
                    "values": g["cpu_utilization"].to_numpy().astype("float32"),
                }

        upstream_data = {}
        if requires_callgraph:
            for ms in current_batch_ids:
                upstream_callers = adjacency.get(ms, set())
                upstream_data[ms] = list(upstream_callers)

        shard_data = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
        batch_set = set(current_batch_ids)

        for ms_key, g in joined.group_by(group_cols, maintain_order=True):
            if g.height < args.input_len + args.pred_horizon:
                continue

            ms_id = ms_key if isinstance(ms_key, str) else ms_key[0]

            if ms_id not in batch_set:
                continue

            feat_arrays = {}
            for feat in raw_feature_names:
                feat_arrays[feat] = g[feat].to_numpy().astype("float32")

            if requires_delta:
                for feat in list(feat_arrays.keys()):
                    feat_arrays[f"{feat}_delta"] = compute_delta(feat_arrays[feat])

            if requires_callgraph:
                upstream_callers = upstream_data.get(ms_id, [])
                if upstream_callers:
                    timestamps = g["_t"].to_numpy()
                    upstream_cpu = np.zeros(len(timestamps), dtype="float32")
                    valid_count = np.zeros(len(timestamps), dtype="int32")

                    for um in upstream_callers:
                        if um in service_data:
                            um_ts = service_data[um]["timestamps"]
                            um_vals = service_data[um]["values"]
                            ts_to_val = dict(zip(um_ts, um_vals))
                            for i, t in enumerate(timestamps):
                                if t in ts_to_val:
                                    upstream_cpu[i] += ts_to_val[t]
                                    valid_count[i] += 1

                    upstream_cpu_mean = np.divide(
                        upstream_cpu, valid_count,
                        out=np.zeros_like(upstream_cpu),
                        where=valid_count > 0,
                    )
                else:
                    upstream_cpu_mean = np.zeros(len(g), dtype="float32")

                feat_arrays["upstream_cpu_utilization_mean"] = upstream_cpu_mean
                feat_arrays["upstream_cpu_utilization_delta_mean"] = compute_delta(upstream_cpu_mean)

            feat_raw = np.stack(
                [feat_arrays[feat] for feat in feature_names], axis=1
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

        if args.smote_tomek:
            try:
                shard_data["train"] = _apply_smote_tomek(
                    "train",
                    shard_data["train"],
                    TRAINING.THETA_BASE,
                    rng,
                )
            except (MemoryError, RuntimeError) as e:
                print(
                    f"[WARN] SMOTE-Tomek skipped for batch {batch_idx}: {e}. "
                )
            gc.collect()

        save_chunk(args.out_dir, batch_idx, 0, shard_data)

        open(done_marker, "a").close()

        del joined, shard_data
        gc.collect()

        batch_duration = time.time() - batch_start_time
        print(f"--> Batch {batch_idx+1} completed in {batch_duration:.2f} seconds")

    print("\nAll global batches processed.")


if __name__ == "__main__":
    main()
