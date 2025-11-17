import os
import glob
import sys
import argparse

import polars as pl
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import PREPROCESSING


def windowize(arr, in_len, horizon, stride):
    X, Y = [], []
    N = len(arr)
    for i in range(0, N - in_len - horizon + 1, stride):
        X.append(arr[i : i + in_len])
        Y.append(arr[i + in_len : i + in_len + horizon])
    return np.asarray(X), np.asarray(Y)


def moving_average(a, window):
    n = len(a)
    if window <= 1 or n == 0:
        return a
    if window > n:
        window = n
    kernel = np.ones(window, dtype=a.dtype) / float(window)
    return np.convolve(a, kernel, mode="same")


def main():
    p = argparse.ArgumentParser(
        description="Build windows with smoothing and normalization (sharded per Parquet)."
    )
    p.add_argument("--parquet_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--time_col", default=PREPROCESSING.TIME_COL)
    p.add_argument("--target_col", default=PREPROCESSING.TARGET_COL)
    p.add_argument(
        "--id_cols",
        nargs="+",
        default=list(PREPROCESSING.ID_COLS),
    )
    p.add_argument("--freq", default=PREPROCESSING.FREQ)

    p.add_argument("--input_len", type=int, default=PREPROCESSING.INPUT_LEN)
    p.add_argument("--pred_horizon", type=int, default=PREPROCESSING.PRED_HORIZON)
    p.add_argument("--stride", type=int, default=PREPROCESSING.STRIDE)
    p.add_argument("--train_frac", type=float, default=PREPROCESSING.TRAIN_FRAC)
    p.add_argument("--val_frac", type=float, default=PREPROCESSING.VAL_FRAC)
    p.add_argument(
        "--smoothing_window",
        type=int,
        default=PREPROCESSING.SMOOTHING_WINDOW,
    )
    p.add_argument(
        "--service_col",
        type=str,
        default=PREPROCESSING.SERVICE_COL,
        help="Column that identifies the service for subsetting (default: msname).",
    )
    p.add_argument(
        "--max_services",
        type=int,
        default=PREPROCESSING.MAX_SERVICES,
        help="Max number of services to keep globally (0 means use all services).",
    )
    p.add_argument(
        "--subset_seed",
        type=int,
        default=PREPROCESSING.SUBSET_SEED,
        help="Random seed used when sampling a subset of services.",
    )

    args = p.parse_args()

    if args.train_frac <= 0 or args.val_frac < 0:
        raise SystemExit("train_frac must be > 0 and val_frac must be >= 0")
    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("train_frac + val_frac must be < 1.0")

    os.makedirs(args.out_dir, exist_ok=True)

    parts = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    if not parts:
        raise SystemExit("No parquet found")

    selected_services = None
    if args.max_services and args.max_services > 0:
        service_col = args.service_col
        print(
            f"Selecting a global subset of services from column '{service_col}' "
            f"(max_services={args.max_services}, seed={args.subset_seed})"
        )

        all_services = set()
        for pq in parts:
            try:
                df_services = pl.read_parquet(pq, columns=[service_col])
            except Exception as e:
                print(
                    f"[ERROR] Unable to read column '{service_col}' from {pq}: {e}",
                    file=sys.stderr,
                )
                raise SystemExit(
                    f"Service column '{service_col}' not found in parquet schema."
                )

            uniq = df_services[service_col].unique()
            all_services.update(uniq.to_list())

        all_services_list = sorted(all_services)
        total_services = len(all_services_list)

        if total_services == 0:
            raise SystemExit(
                f"No services found in column '{service_col}' across all parquet files."
            )

        if total_services <= args.max_services:
            selected_services = all_services_list
            print(
                f"Found {total_services} services (<= max_services). "
                f"Using all of them."
            )
        else:
            rng = np.random.default_rng(args.subset_seed)
            indices = rng.choice(
                total_services,
                size=args.max_services,
                replace=False,
            )
            services_array = np.array(all_services_list, dtype=object)
            selected_services = services_array[indices].tolist()
            print(
                f"Found {total_services} services total, "
                f"randomly selected {len(selected_services)}."
            )

        selected_services = set(selected_services)
        sample_preview = list(selected_services)[:10]
        print(f"Example selected services (up to 10): {sample_preview}")
    # -------------------------------------------------------------------------

    for shard_idx, pq in enumerate(parts):
        print(f"=== Processing parquet shard {shard_idx}: {pq}")
        df = pl.read_parquet(pq)

        if selected_services is not None:
            if args.service_col not in df.columns:
                raise SystemExit(
                    f"service_col '{args.service_col}' not present in shard {pq}"
                )
            before_rows = df.height
            df = df.filter(pl.col(args.service_col).is_in(list(selected_services)))
            after_rows = df.height
            print(
                f"  Filtered shard rows by service subset: {before_rows} -> {after_rows}"
            )
            if after_rows == 0:
                print("  Shard has no rows after service filtering, skipping.")
                continue

        if df[args.time_col].dtype != pl.Datetime:
            df = df.with_columns(pl.col(args.time_col).cast(pl.Datetime))

        df = (
            df.with_columns(pl.col(args.time_col).dt.truncate(args.freq).alias("_t"))
            .group_by("_t", *args.id_cols)
            .agg(pl.col(args.target_col).last().alias(args.target_col))
            .sort(["_t", *args.id_cols])
        )

        if df.height == 0:
            print("  Shard has no rows after aggregation, skipping.")
            continue

        t0 = df["_t"].min()
        df = df.with_columns(
            ((pl.col("_t") - t0).dt.total_minutes().round(0).cast(pl.Int32)).alias(
                "_minute"
            )
        )

        shard_train_X, shard_train_y = [], []
        shard_val_X, shard_val_y = [], []
        shard_test_X, shard_test_y = [], []

        for _, g in df.group_by(args.id_cols, maintain_order=True):
            g = g.sort("_minute")

            a_raw = g[args.target_col].to_numpy().astype("float32")
            N = len(a_raw)
            if N < args.input_len + args.pred_horizon:
                continue

            a_smooth = moving_average(a_raw, args.smoothing_window)

            a_min = float(a_smooth.min())
            a_max = float(a_smooth.max())
            if a_max - a_min < 1e-8:
                continue
            a_norm = (a_smooth - a_min) / (a_max - a_min + 1e-8)

            X_all, Y_all = windowize(
                a_norm,
                in_len=args.input_len,
                horizon=args.pred_horizon,
                stride=args.stride,
            )
            n_w = X_all.shape[0]
            if n_w == 0:
                continue

            cut_train = int(n_w * args.train_frac)
            cut_val = int(n_w * (args.train_frac + args.val_frac))

            cut_train = max(cut_train, 1)
            cut_val = max(cut_val, cut_train + 1)
            cut_val = min(cut_val, n_w - 1)

            Xtr_g = X_all[:cut_train]
            Ytr_g = Y_all[:cut_train]
            Xv_g = X_all[cut_train:cut_val]
            Yv_g = Y_all[cut_train:cut_val]
            Xte_g = X_all[cut_val:]
            Yte_g = Y_all[cut_val:]

            if Xtr_g.size:
                shard_train_X.append(Xtr_g)
                shard_train_y.append(Ytr_g)
            if Xv_g.size:
                shard_val_X.append(Xv_g)
                shard_val_y.append(Yv_g)
            if Xte_g.size:
                shard_test_X.append(Xte_g)
                shard_test_y.append(Yte_g)

        def cat_or_empty(chunks, shape_tail):
            if not chunks:
                return np.empty((0, *shape_tail), dtype="float32")
            return np.concatenate(chunks, axis=0).astype("float32")

        if shard_train_X or shard_val_X or shard_test_X:
            Xtr = cat_or_empty(shard_train_X, (args.input_len,))
            ytr = cat_or_empty(shard_train_y, (args.pred_horizon,))
            Xv = cat_or_empty(shard_val_X, (args.input_len,))
            yv = cat_or_empty(shard_val_y, (args.pred_horizon,))
            Xte = cat_or_empty(shard_test_X, (args.input_len,))
            yte = cat_or_empty(shard_test_y, (args.pred_horizon,))

            base = os.path.join(args.out_dir, f"part-{shard_idx:04d}")
            np.save(base + "_X_train.npy", Xtr)
            np.save(base + "_y_train.npy", ytr)
            np.save(base + "_X_val.npy", Xv)
            np.save(base + "_y_val.npy", yv)
            np.save(base + "_X_test.npy", Xte)
            np.save(base + "_y_test.npy", yte)

            print(f"  Saved shard {shard_idx}:")
            print(f"    train: {Xtr.shape}, {ytr.shape}")
            print(f"    val:   {Xv.shape}, {yv.shape}")
            print(f"    test:  {Xte.shape}, {yte.shape}")
        else:
            print(f"  No windows produced for shard {shard_idx}, skipping saves")

    print("All shards processed.")


if __name__ == "__main__":
    main()
