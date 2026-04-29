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

    out = (
        df_or_lazy.with_columns(pl.col(time_col).dt.truncate(freq).alias("_t"))
        .group_by(["_t"] + id_cols)
        .agg(agg_exprs)
        .sort(["_t"] + id_cols)
    )
    return out


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


def main():
    p = argparse.ArgumentParser(
        description="Build windows using service batching to bypass OOM limits."
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
        "--batch_size", type=int, default=50, help="Number of services to join at once"
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

    print(f"Feature set: {args.feature_set} | Base Table: {base_table}")

    selected_services = None
    if args.max_services > 0:
        base_dir = DATASET_TABLES[base_table]["parquet_dir"]
        base_parts = list_parquet_parts(base_dir)
        all_services = set()
        for part in base_parts:
            try:
                df_part_s = (
                    pl.scan_parquet(part).select(args.service_col).unique().collect()
                )
                all_services.update(df_part_s[args.service_col].to_list())
            except Exception as e:
                print(f"Warning: Could not read {part}: {e}")

        all_services_list = sorted(list(all_services))
        if len(all_services_list) > args.max_services:
            rng = np.random.default_rng(args.subset_seed)
            idxs = rng.choice(
                len(all_services_list), size=args.max_services, replace=False
            )
            selected_services = set(np.array(all_services_list)[idxs].tolist())
            print(f"Selected subset: {len(selected_services)} services")
        else:
            print(f"Using all {len(all_services_list)} services")

    table_parts: dict[str, list[str]] = {}
    for t in needed_tables:
        pq_dir = DATASET_TABLES[t]["parquet_dir"]
        parts = list_parquet_parts(pq_dir)
        if not parts:
            raise SystemExit(f"No parquet parts found for table='{t}' in {pq_dir}")
        table_parts[t] = parts

    os.makedirs(args.out_dir, exist_ok=True)

    for shard_idx, base_pq in enumerate(table_parts[base_table]):
        gc.collect()
        print(f"\n=== Shard {shard_idx:04d} (base={os.path.basename(base_pq)}) ===")

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

        try:
            lf_base_schema = pl.scan_parquet(base_pq).collect_schema().names()
            if args.service_col not in lf_base_schema:
                print(f"Skipping {base_pq}, missing {args.service_col}")
                continue

            shard_services = (
                pl.scan_parquet(base_pq)
                .select(args.service_col)
                .unique()
                .collect()[args.service_col]
                .to_list()
            )
            if selected_services is not None:
                shard_services = list(
                    set(shard_services).intersection(selected_services)
                )
        except Exception as e:
            print(f"Error reading services from {base_pq}: {e}")
            continue

        if not shard_services:
            continue

        total_batches = (len(shard_services) + args.batch_size - 1) // args.batch_size
        print(
            f"Found {len(shard_services)} services. Processing in {total_batches} batches..."
        )

        chunk_idx = 0
        for i in range(0, len(shard_services), args.batch_size):
            batch = shard_services[i : i + args.batch_size]
            batch_num = (i // args.batch_size) + 1
            print(f"  -> Batch {batch_num}/{total_batches} ({len(batch)} services)")

            try:
                lf_base = (
                    pl.scan_parquet(base_pq)
                    .select(base_need_cols)
                    .filter(pl.col(args.service_col).is_in(batch))
                )

                joined_lazy = build_table_agg(
                    lf_base,
                    args.time_col,
                    base_id_cols,
                    args.freq,
                    table_exprs[base_table],
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
                    need_cols = list(
                        set(
                            [
                                args.time_col,
                                *t_id_cols,
                                *[raw for _, raw in table_exprs[t]],
                            ]
                        )
                    )

                    lf_t = (
                        pl.scan_parquet(table_parts[t])
                        .filter(
                            (pl.col(args.time_col).cast(pl.Datetime) >= min_t)
                            & (pl.col(args.time_col).cast(pl.Datetime) <= max_t)
                        )
                        .select(need_cols)
                    )

                    if (
                        args.service_col
                        in pl.scan_parquet(table_parts[t]).collect_schema().names()
                    ):
                        lf_t = lf_t.filter(pl.col(args.service_col).is_in(batch))

                    lf_t_agg = build_table_agg(
                        lf_t, args.time_col, t_id_cols, args.freq, table_exprs[t]
                    )

                    join_on = ["_t"] + FEATURE_SETS[args.feature_set].get(
                        "join_keys", {}
                    ).get(t, [])
                    if len(join_on) == 1:
                        schema_joined = joined_lazy.collect_schema().names()
                        schema_t = lf_t_agg.collect_schema().names()
                        join_on = list(set(schema_joined).intersection(schema_t))

                    joined_lazy = joined_lazy.join(lf_t_agg, on=join_on, how="left")

                joined = (
                    joined_lazy.drop_nulls(feature_names)
                    .sort(
                        list(
                            set(args.id_cols).intersection(
                                joined_lazy.collect_schema().names()
                            )
                        )
                        + ["_t"]
                    )
                    .collect()
                )
                del joined_lazy
                gc.collect()

                if joined.height == 0:
                    continue

                shard_data = {
                    "train": ([], [], []),
                    "val": ([], [], []),
                    "test": ([], [], []),
                }
                group_cols = [c for c in args.id_cols if c in joined.columns]

                for _, g in joined.group_by(group_cols, maintain_order=True):
                    if g.height < args.input_len + args.pred_horizon:
                        continue

                    feat_raw = np.stack(
                        [
                            g[feat].to_numpy().astype("float32")
                            for feat in feature_names
                        ],
                        axis=1,
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

                    X_all, Y_all, S_all = windowize_multivariate(
                        feat_processed,
                        feat_processed[:, target_idx],
                        args.input_len,
                        args.pred_horizon,
                        args.stride,
                    )

                    n_w = len(X_all)
                    if n_w == 0:
                        continue

                    cut_tr = max(int(n_w * args.train_frac), 1)
                    cut_val = min(
                        max(int(n_w * (args.train_frac + args.val_frac)), cut_tr + 1),
                        n_w - 1,
                    )

                    for split, start, end in [
                        ("train", 0, cut_tr),
                        ("val", cut_tr, cut_val),
                        ("test", cut_val, None),
                    ]:
                        if X_all[start:end].size:
                            shard_data[split][0].append(X_all[start:end])
                            shard_data[split][1].append(Y_all[start:end])
                            shard_data[split][2].append(S_all[start:end])

                if save_chunk(args.out_dir, shard_idx, chunk_idx, shard_data):
                    chunk_idx += 1

                del joined, shard_data
                gc.collect()

            except Exception as e:
                print(f"Error processing batch {batch_num} in shard {shard_idx}: {e}")
                continue

    print("\nAll shards processed.")


if __name__ == "__main__":
    main()
