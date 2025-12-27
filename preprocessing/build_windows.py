import os
import glob
import sys
import time
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


def windowize_multivariate(
    x_feat: np.ndarray, y_target: np.ndarray, in_len: int, horizon: int, stride: int
):
    X, Y = [], []
    T = x_feat.shape[0]
    for i in range(0, T - in_len - horizon + 1, stride):
        X.append(x_feat[i : i + in_len, :])
        Y.append(y_target[i + in_len : i + in_len + horizon])
    if not X:
        return (
            np.empty((0, in_len, x_feat.shape[1]), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def moving_average(a: np.ndarray, window: int):
    n = len(a)
    if window <= 1 or n == 0:
        return a
    if window > n:
        window = n
    kernel = np.ones(window, dtype=a.dtype) / float(window)
    return np.convolve(a, kernel, mode="same")


def minmax_norm(a: np.ndarray, eps: float = 1e-8):
    a_min = float(np.min(a))
    a_max = float(np.max(a))
    denom = a_max - a_min
    if denom < eps:
        return None
    return (a - a_min) / (denom + eps)


def list_parquet_parts(parquet_dir: str):
    return sorted(glob.glob(os.path.join(parquet_dir, "part-*.parquet")))


def build_table_agg(
    df_or_lazy, time_col: str, id_cols: list, freq: str, feature_exprs: list
):
    if isinstance(df_or_lazy, pl.DataFrame):
        if df_or_lazy[time_col].dtype != pl.Datetime:
            df_or_lazy = df_or_lazy.with_columns(pl.col(time_col).cast(pl.Datetime))
    else:
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


def main():
    p = argparse.ArgumentParser(
        description="Build windows (multi-table join, feature-set aware)."
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

    args = p.parse_args()

    if args.train_frac <= 0 or args.val_frac < 0:
        raise SystemExit("train_frac must be > 0 and val_frac must be >= 0")
    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("train_frac + val_frac must be < 1.0")

    spec = get_feature_set(args.feature_set)
    feature_names = list(spec["features"])
    target_feature = str(spec["target"])
    target_idx = feature_names.index(target_feature)

    needed_tables = sorted(list(tables_for_feature_set(args.feature_set)))
    table_exprs = table_to_feature_exprs(args.feature_set)

    base_table = FEATURES[target_feature]["table"]

    print(f"Feature set: {args.feature_set}")
    print(f"Tables: {needed_tables}")
    print(f"Base join table: {base_table}")

    selected_services = None
    if args.max_services > 0:
        base_dir = DATASET_TABLES[base_table]["parquet_dir"]
        base_parts = list_parquet_parts(base_dir)
        print(f"Scanning {len(base_parts)} parts for unique services...")
        
        all_services = set()
        for part in base_parts:
            try:
                df_part_s = pl.scan_parquet(part).select(args.service_col).unique().collect()
                all_services.update(df_part_s[args.service_col].to_list())
            except Exception as e:
                print(f"Warning: Could not read {part}: {e}")

        all_services_list = sorted(list(all_services))
        print(f"Found {len(all_services_list)} unique services.")

        rng = np.random.default_rng(args.subset_seed)
        if len(all_services_list) > args.max_services:
            idxs = rng.choice(
                len(all_services_list), size=args.max_services, replace=False
            )
            selected_services = set(np.array(all_services_list)[idxs].tolist())
            print(f"Selected subset: {len(selected_services)} services")
        else:
            selected_services = set(all_services_list) if all_services_list else None
            print(f"Using all available services")

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
        print(f"\n=== Shard {shard_idx:04d} (base={base_pq}) ===")

        base_id_cols = DATASET_TABLES[base_table]["key_cols"]
        base_need_cols = [
            args.time_col,
            *base_id_cols,
            *[raw for _, raw in table_exprs[base_table]],
        ]
        base_need_cols = list(set(base_need_cols))

        try:
            lf_base = pl.scan_parquet(base_pq).select(base_need_cols)
            if selected_services is not None and args.service_col in lf_base.collect_schema().names():
                lf_base = lf_base.filter(pl.col(args.service_col).is_in(list(selected_services)))
            
            df_base = lf_base.collect()
        except Exception as e:
            print(f"Error reading base parquet {base_pq}: {e}")
            continue

        if df_base.height == 0:
            print("  Base shard empty; skipping.")
            continue
        
        if df_base[args.time_col].dtype != pl.Datetime:
            df_base = df_base.with_columns(pl.col(args.time_col).cast(pl.Datetime))

        min_t = df_base[args.time_col].min()
        max_t = df_base[args.time_col].max()
        print(f"  Time range: {min_t} to {max_t}")

        df_base_agg = build_table_agg(
            df_base, args.time_col, base_id_cols, args.freq, table_exprs[base_table]
        )
        
        del df_base
        gc.collect()

        joined = df_base_agg

        for t in needed_tables:
            if t == base_table:
                continue
            
            t_id_cols = DATASET_TABLES[t]["key_cols"]
            need_cols = [
                args.time_col,
                *t_id_cols,
                *[raw for _, raw in table_exprs[t]],
            ]
            need_cols = list(set(need_cols))

            try:
                lf_t = pl.scan_parquet(table_parts[t])
                lf_t = lf_t.filter(
                    pl.col(args.time_col).cast(pl.Datetime) >= min_t,
                    pl.col(args.time_col).cast(pl.Datetime) <= max_t
                ).select(need_cols)

                lf_t_agg = build_table_agg(
                    lf_t, args.time_col, t_id_cols, args.freq, table_exprs[t]
                )
                
                df_t_agg = lf_t_agg.collect(engine="streaming")
                
            except Exception as e:
                print(f"  Error processing table '{t}': {e}. Skipping join.")
                continue

            feature_spec = FEATURE_SETS[args.feature_set]
            spec_join_keys = feature_spec.get("join_keys", {})
            
            if t in spec_join_keys:
                join_on = ["_t"] + spec_join_keys[t]
            else:
                common = set(joined.columns).intersection(df_t_agg.columns)
                join_on = list(common)
            
            missing_keys = [k for k in join_on if k not in joined.columns or k not in df_t_agg.columns]
            if missing_keys:
                print(f"  Cannot join '{t}': missing keys {missing_keys}. Skipping.")
                continue

            joined = joined.join(df_t_agg, on=join_on, how="left")
            del df_t_agg
            gc.collect()

        joined = joined.drop_nulls(feature_names)
        if joined.height == 0:
            print("  No rows left after drop_nulls; skipping.")
            continue

        shard_train_X, shard_train_y = [], []
        shard_val_X, shard_val_y = [], []
        shard_test_X, shard_test_y = [], []

        group_cols = [c for c in args.id_cols if c in joined.columns]
        joined = joined.sort(group_cols + ["_t"])

        for _, g in joined.group_by(group_cols, maintain_order=True):
            T = g.height
            if T < args.input_len + args.pred_horizon:
                continue

            series = [g[feat].to_numpy().astype("float32") for feat in feature_names]
            feat_raw = np.stack(series, axis=1)

            F = feat_raw.shape[1]
            feat_norm = np.zeros_like(feat_raw, dtype=np.float32)
            target_norm = None

            for j in range(F):
                s = feat_raw[:, j]
                s_smooth = moving_average(s, args.smoothing_window)
                s_norm = minmax_norm(s_smooth)

                if j == target_idx:
                    if s_norm is None: break
                    target_norm = s_norm.astype("float32")
                    feat_norm[:, j] = target_norm
                else:
                    feat_norm[:, j] = (0.0 if s_norm is None else s_norm.astype("float32"))

            if target_norm is None: continue

            X_all, Y_all = windowize_multivariate(
                feat_norm, target_norm, args.input_len, args.pred_horizon, args.stride
            )
            n_w = X_all.shape[0]
            if n_w == 0: continue

            cut_train = max(int(n_w * args.train_frac), 1)
            cut_val = min(max(int(n_w * (args.train_frac + args.val_frac)), cut_train + 1), n_w - 1)

            if X_all[:cut_train].size:
                shard_train_X.append(X_all[:cut_train]); shard_train_y.append(Y_all[:cut_train])
            if X_all[cut_train:cut_val].size:
                shard_val_X.append(X_all[cut_train:cut_val]); shard_val_y.append(Y_all[cut_train:cut_val])
            if X_all[cut_val:].size:
                shard_test_X.append(X_all[cut_val:]); shard_test_y.append(Y_all[cut_val:])

        def cat_or_empty(chunks, shape_tail):
            if not chunks: return np.empty((0, *shape_tail), dtype="float32")
            return np.concatenate(chunks, axis=0).astype("float32")

        if shard_train_X or shard_val_X or shard_test_X:
            Xtr = cat_or_empty(shard_train_X, (args.input_len, len(feature_names)))
            ytr = cat_or_empty(shard_train_y, (args.pred_horizon,))
            Xv = cat_or_empty(shard_val_X, (args.input_len, len(feature_names)))
            yv = cat_or_empty(shard_val_y, (args.pred_horizon,))
            Xte = cat_or_empty(shard_test_X, (args.input_len, len(feature_names)))
            yte = cat_or_empty(shard_test_y, (args.pred_horizon,))

            base = os.path.join(args.out_dir, f"part-{shard_idx:04d}")
            np.save(base + "_X_train.npy", Xtr); np.save(base + "_y_train.npy", ytr)
            np.save(base + "_X_val.npy", Xv); np.save(base + "_y_val.npy", yv)
            np.save(base + "_X_test.npy", Xte); np.save(base + "_y_test.npy", yte)
            print(f"  Saved: train {Xtr.shape}, val {Xv.shape}, test {Xte.shape}")
            
        del joined, shard_train_X, shard_val_X, shard_test_X
        gc.collect()

    print("\nAll shards processed.")

if __name__ == "__main__":
    main()