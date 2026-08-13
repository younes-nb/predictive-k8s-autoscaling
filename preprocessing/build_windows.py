import os
import glob
import sys
import argparse
import shutil
import tempfile
import time
import gc
import json
import hashlib
import math
import multiprocessing as mp
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_n_cpus = os.cpu_count() or 1
_NUM_WORKERS = 0.9


def _pre_scan_polars_threads():
    for i, a in enumerate(sys.argv):
        if a == "--polars_threads" and i + 1 < len(sys.argv):
            try:
                return max(1, int(sys.argv[i + 1]))
            except ValueError:
                return None
        if a.startswith("--polars_threads="):
            try:
                return max(1, int(a.split("=", 1)[1]))
            except ValueError:
                return None
    return None


_polars_threads = _pre_scan_polars_threads()
if _polars_threads is None:
    _polars_threads = int(os.environ.get("POLARS_MAX_THREADS") or 48)
os.environ["POLARS_MAX_THREADS"] = str(_polars_threads)

import polars as pl
import numpy as np
from tqdm import tqdm

_WINDOW_DECIMALS = 2
_WINDOW_DTYPE = np.float16

_CSV_COLUMN_MAP = {
    "cpu_utilization": "CPU",
    "memory_utilization": "Memory",
}

from core.utils import windowize_multivariate

from shared.config_paths import PATHS, DATASET_TABLES
from shared.config_preprocessing_defaults import PREPROCESSING
from shared.features import FEATURE_SETS, get_feature_set, tables_for_feature_set, table_to_feature_exprs, FEATURES

from preprocessing.parquet_utils import (
    list_parquet_parts,
    discover_unique_services,
    _parquet_fingerprint,
)

_WORKER_CTX = {}


def _quantize_windows(arr):
    return np.round(arr, _WINDOW_DECIMALS).astype(_WINDOW_DTYPE)


def save_chunk(out_dir, shard_idx, chunk_idx, shard_data, sync=False,
               quantize_cols=None):
    base_name = f"part-{shard_idx:04d}_chunk-{chunk_idx:04d}"
    saved_any = False

    try:
        # Stage in out_dir (same filesystem): the move below becomes an atomic
        # rename, and we avoid filling /dev/shm when 57 workers write concurrently.
        with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
            tmp_base = os.path.join(tmp_dir, base_name)

            for split, (Xs, Ys, Ss) in shard_data.items():
                if Xs:
                    x_arr = np.concatenate(Xs)
                    y_arr = np.concatenate(Ys)
                    # Round utilization to 1e-2 and store as float16 when every
                    # channel is a [0,1]-bounded resource feature; mixed feature
                    # sets (unbounded call-rate columns) stay float32 to avoid
                    # float16 overflow.
                    if quantize_cols is not None and len(quantize_cols) >= x_arr.shape[-1]:
                        x_arr = _quantize_windows(x_arr)
                    if quantize_cols is not None:
                        y_arr = _quantize_windows(y_arr)
                    np.save(f"{tmp_base}_X_{split}.npy", x_arr)
                    np.save(f"{tmp_base}_y_{split}.npy", y_arr)
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


def _part_agg_plan(table_exprs, table_name, time_col):
    """Return (per-part agg exprs, fold exprs, final exprs) for exact per-part aggregation.

    Per-part frames keep intermediate columns (_s/_c/_l/_t). The fold merge re-aggregates
    them incrementally (associative for mean/sum/max), keeping at most two part frames in
    memory; final exprs project the intermediate columns to the real feature names.
    """
    part_exprs = []
    fold_exprs = []
    final_exprs = []
    for feat_name, raw_col in table_exprs[table_name]:
        if table_name == "msresource":
            s, c = f"{feat_name}_s", f"{feat_name}_c"
            part_exprs += [
                pl.col(raw_col).sum().alias(s),
                pl.col(raw_col).count().alias(c),
            ]
            fold_exprs += [pl.col(s).sum().alias(s), pl.col(c).sum().alias(c)]
            final_exprs.append((pl.col(s) / pl.col(c)).alias(feat_name))
        elif table_name == "msrtmcre":
            s = f"{feat_name}_s"
            part_exprs.append(pl.col(raw_col).sum().alias(s))
            fold_exprs.append(pl.col(s).sum().alias(s))
            final_exprs.append(pl.col(s).alias(feat_name))
        else:
            l, t = f"{feat_name}_l", f"{feat_name}_t"
            part_exprs += [
                pl.col(raw_col).last().alias(l),
                pl.col(time_col).max().alias(t),
            ]
            fold_exprs += [
                pl.col(l).sort_by(pl.col(t), descending=True).first().alias(l),
                pl.col(t).max().alias(t),
            ]
            final_exprs.append(pl.col(l).alias(feat_name))
    return part_exprs, fold_exprs, final_exprs


def _merge_part_frames(part_frames, fold_exprs, id_cols):
    if not part_frames:
        return None
    merged = part_frames[0]
    for frame in part_frames[1:]:
        combined = pl.concat([merged, frame], how="vertical")
        merged = (
            combined.lazy()
            .group_by(["_t"] + id_cols)
            .agg(fold_exprs)
            .collect(engine="streaming")
        )
        del combined
        gc.collect()
    return merged.sort(["_t"] + id_cols)


def _finalize_merged(merged, final_exprs, id_cols):
    key_cols = [c for c in ["_t"] + id_cols if c in merged.columns]
    return merged.select([*[pl.col(c) for c in key_cols], *final_exprs])


def _process_service_group(group_idx, service_ids):
    import numpy as np

    ctx = _WORKER_CTX
    args_dict = ctx["args_dict"]
    out_dir = args_dict["out_dir"]
    done_marker = os.path.join(out_dir, f"part-{group_idx:04d}.done")
    if os.path.exists(done_marker):
        return (group_idx, 0, 0.0, True)

    t0 = time.time()
    arrays = ctx.get("service_arrays")
    big = ctx.get("big_array")
    index = ctx.get("service_index")
    target_indices = ctx["target_indices"]

    shard_data = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
    n_processed = 0

    for ms_id in service_ids:
        if big is not None:
            pos = index.get(ms_id)
            if pos is None:
                continue
            feat_raw = big[pos[0]:pos[0] + pos[1]]
        else:
            feat_raw = arrays.get(ms_id)
            if feat_raw is None:
                continue

        n = len(feat_raw)
        if args_dict.get("split_mode") == "hours":
            rph = args_dict["rows_per_hour"]
            idx_tr = min(n, int(round(args_dict["train_hours"] * rph)))
            idx_val = min(n, idx_tr + int(round(args_dict["val_hours"] * rph)))
            idx_end = n
            if args_dict.get("test_hours"):
                idx_end = min(n, idx_val + int(round(args_dict["test_hours"] * rph)))
            split_configs = [
                ("train", 0, idx_tr),
                ("val", idx_tr, idx_val),
                ("test", idx_val, idx_end),
            ]
        else:
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

    save_chunk(out_dir, group_idx, 0, shard_data, sync=ctx["sync"],
               quantize_cols=ctx.get("resource_indices"))
    open(done_marker, "a").close()

    del shard_data
    gc.collect()

    return (group_idx, n_processed, time.time() - t0, False)


def _service_arrays_paths(out_dir):
    return (
        os.path.join(out_dir, "_service_arrays.npy"),
        os.path.join(out_dir, "_service_index.json"),
    )


def _csv_fingerprint(csv_path):
    st = os.stat(csv_path)
    return (st.st_mtime_ns, st.st_size)


def _freq_seconds(freq):
    """Parse a polars duration string like '1m'/'60s'/'2h' into seconds, or
    None if it can't be parsed (gap check is then disabled)."""
    if not freq:
        return None
    unit = freq[-1].lower()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit)
    if mult is None:
        return None
    try:
        return int(freq[:-1]) * mult
    except ValueError:
        return None


def _split_params(args):
    """Return split-mode keys for args_dict. With any --*_hours flag set the
    split is done by hours (each service's rows [0, train_hours), then
    val_hours, then test_hours; the remainder falls to test); otherwise the
    classic per-service train/val fractions apply."""
    hours = [args.train_hours, args.val_hours, args.test_hours]
    if any(h is not None for h in hours):
        if args.train_hours is None or args.train_hours <= 0:
            raise SystemExit(
                "--train_hours must be set (>0) when splitting by hours "
                "(--val_hours/--test_hours also enable hour-based splitting)."
            )
        if (args.val_hours or 0) < 0 or (args.test_hours or 0) < 0:
            raise SystemExit("Hours splits must be >= 0.")
        step_sec = _freq_seconds(args.freq)
        if not step_sec:
            raise SystemExit(
                f"Cannot split by hours: could not parse --freq '{args.freq}' into seconds."
            )
        return {
            "split_mode": "hours",
            "rows_per_hour": 3600.0 / step_sec,
            "train_hours": float(args.train_hours),
            "val_hours": float(args.val_hours or 0.0),
            "test_hours": float(args.test_hours) if args.test_hours is not None else None,
        }
    return {
        "split_mode": "frac",
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
    }


def _arrays_signature(args, base_table):
    if args.csv_path:
        fp = _csv_fingerprint(args.csv_path)
    else:
        fp = _parquet_fingerprint(DATASET_TABLES[base_table]["parquet_dir"])
    blob = {
        "feature_set": args.feature_set,
        "freq": args.freq,
        "time_col": args.time_col,
        "service_col": args.service_col,
        "input_len": args.input_len,
        "pred_horizon": args.pred_horizon,
        "max_services": args.max_services,
        "subset_seed": args.subset_seed,
        "base_parts": fp,
    }
    return hashlib.md5(json.dumps(blob, sort_keys=True).encode()).hexdigest()


def _arrays_cache_valid(arrays_path, index_path, signature):
    if not (os.path.exists(arrays_path) and os.path.exists(index_path)):
        return False
    try:
        with open(index_path, "r") as f:
            data = json.load(f)
        return data.get("signature") == signature
    except (OSError, ValueError):
        return False


def _save_service_arrays(service_arrays, arrays_path, index_path, signature):
    total = sum(len(a) for a in service_arrays.values())
    channels = next(iter(service_arrays.values())).shape[1]
    big = np.empty((total, channels), dtype="float32")
    index = {}
    off = 0
    for ms_id in sorted(service_arrays):
        a = service_arrays[ms_id]
        big[off:off + len(a)] = a
        index[ms_id] = [int(off), int(len(a))]
        off += len(a)
    np.save(arrays_path, big)
    with open(index_path, "w") as f:
        json.dump({"signature": signature, "index": index}, f)
    del big
    gc.collect()
    print(f"Service arrays cached: {os.path.basename(arrays_path)} "
          f"({total} rows x {channels} ch, {len(service_arrays)} services)",
          flush=True)


def _load_csv_service_arrays(csv_path, feature_names, time_col, id_col,
                             tz_name, input_len, pred_horizon, freq):
    """Load per-service feature arrays from an HPA-logs CSV.

    Each feature is resolved to a CSV column via _CSV_COLUMN_MAP. Rows are
    ordered by timestamp (parsed as tz-aware then epoch), and a service is
    skipped if its timestamps have gaps or duplicates (windowize is positional
    and would silently misalign rows otherwise).
    """
    df = pl.read_csv(csv_path)
    df_cols = set(df.columns)
    if time_col not in df_cols or id_col not in df_cols:
        raise SystemExit(
            f"CSV '{csv_path}' missing required column '{time_col}' or "
            f"'{id_col}' (got {sorted(df_cols)})"
        )

    col_map = {}
    for f in feature_names:
        if f not in _CSV_COLUMN_MAP:
            raise SystemExit(
                f"Feature '{f}' has no CSV column mapping; add it to "
                f"_CSV_COLUMN_MAP (csv_path={csv_path})"
            )
        col = _CSV_COLUMN_MAP[f]
        if col not in df_cols:
            raise SystemExit(
                f"CSV '{csv_path}' missing column '{col}' needed for feature "
                f"'{f}' (got {sorted(df_cols)})"
            )
        col_map[f] = col

    tz = ZoneInfo(tz_name)
    ts_list = []
    n_unparsed = 0
    for s in df[time_col].to_list():
        try:
            ts_list.append(
                datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=tz)
                .timestamp()
            )
        except ValueError:
            ts_list.append(None)
            n_unparsed += 1
    if n_unparsed:
        print(f"  CSV: dropped {n_unparsed} rows with unparseable timestamps",
              flush=True)
    df = df.with_columns(pl.Series("_ts", [int(v) if v is not None else None for v in ts_list], dtype=pl.Int64))
    df = df.drop_nulls("_ts")

    step_sec = _freq_seconds(freq)
    df = df.select([id_col, "_ts"] + [col_map[f] for f in feature_names])
    df = df.sort([id_col, "_ts"])

    service_arrays = {}
    skipped = []
    for key, g in df.group_by([id_col], maintain_order=True):
        svc = key[0] if isinstance(key, (list, tuple)) else key
        ts = g["_ts"].to_numpy()
        diffs = np.diff(ts)
        if step_sec is not None:
            bad = diffs[diffs != step_sec]
        else:
            bad = diffs[diffs <= 0]
        if bad.size:
            skipped.append(svc)
            continue
        arr = np.stack(
            [g[col_map[f]].to_numpy().astype("float32") for f in feature_names],
            axis=1,
        )
        if arr.shape[0] < input_len + pred_horizon:
            skipped.append(svc)
            continue
        service_arrays[svc] = arr

    if skipped:
        print(f"  CSV: skipped {len(skipped)} services (timestamp gaps/dups "
              f"or too short): {', '.join(sorted(skipped)[:20])}",
              flush=True)
    print(f"  CSV: {len(service_arrays)} services loaded from "
          f"{os.path.basename(csv_path)}", flush=True)
    return service_arrays


def _run_csv_source(args, spec, feature_names, target_indices,
                    resource_indices, num_workers):
    """CSV-source build: build service arrays from the CSV, then reuse the
    exact same windows phase as the parquet path."""
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV path not found: {csv_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(args.out_dir, "tmp*")):
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)

    if args.recompute:
        cached = glob.glob(os.path.join(args.out_dir, "part-*.done")) + \
                 glob.glob(os.path.join(args.out_dir, "part-*_chunk-*.npy"))
        for f in cached:
            os.remove(f)
        if cached:
            print(f"Removed {len(cached)} cached artifacts for recompute",
                  flush=True)

    arrays_path, index_path = _service_arrays_paths(args.out_dir)
    signature = _arrays_signature(args, spec.get("base_table"))
    cache_valid = _arrays_cache_valid(arrays_path, index_path, signature)

    if args.phase == "windows":
        if not cache_valid:
            raise SystemExit(
                "--phase windows but CSV service-array cache is missing or "
                "stale; run without --phase first"
            )
        with open(index_path, "r") as f:
            all_services_list = sorted(json.load(f)["index"].keys())
        print(f"CSV service-array cache is valid: {len(all_services_list)} "
              f"services", flush=True)
    elif cache_valid:
        with open(index_path, "r") as f:
            all_services_list = sorted(json.load(f)["index"].keys())
        print(f"CSV service-array cache is valid: {len(all_services_list)} "
              f"services", flush=True)
    else:
        service_arrays = _load_csv_service_arrays(
            csv_path, feature_names, args.csv_time_col, args.csv_id_col,
            args.csv_tz, args.input_len, args.pred_horizon, args.freq,
        )
        if args.max_services and len(service_arrays) > args.max_services:
            rng = np.random.default_rng(args.subset_seed)
            idxs = rng.choice(
                len(service_arrays), size=args.max_services, replace=False
            )
            keep = set(np.array(sorted(service_arrays.keys()))[idxs].tolist())
            service_arrays = {k: v for k, v in service_arrays.items() if k in keep}
            print(f"Selected subset: {len(service_arrays)} services")
        else:
            print(f"Processing all {len(service_arrays)} services")
        if not service_arrays:
            print("No services with enough data in CSV; nothing to do.")
            return
        all_services_list = sorted(service_arrays.keys())
        _save_service_arrays(service_arrays, arrays_path, index_path, signature)

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
        "time_col": args.csv_time_col,
        "freq": args.freq,
        "input_len": args.input_len,
        "pred_horizon": args.pred_horizon,
        "stride": args.stride,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "service_col": args.csv_id_col,
        "feature_set": args.feature_set,
        "resource_indices": resource_indices,
    }
    args_dict.update(_split_params(args))

    _phase_windows(args, args_dict, target_indices, all_services_list,
                   groups_to_run, group_size, num_workers,
                   arrays_path, index_path)


def _reexec():
    """Re-exec a fresh interpreter so the ~50GB aggregation memory is reclaimed
    (polars/mimalloc arenas are NOT returned to the OS on del/gc)."""
    tail = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--phase":
            skip_next = True
            continue
        if a.startswith("--phase="):
            continue
        tail.append(a)
    tail += ["--phase", "windows"]
    argv = [sys.executable, os.path.abspath(__file__)] + tail
    print("Aggregation done; re-executing in a fresh process to reclaim memory...",
          flush=True)
    os.execv(sys.executable, argv)


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
    p.add_argument("--train_hours", type=float, default=None,
                    help="Split by hours instead of --train_frac/--val_frac: train on the first "
                         "N hours of each service's series. Enables hour-based splitting and is "
                         "required when --val_hours/--test_hours are set. Changing any split "
                         "param on an existing out_dir requires --recompute.")
    p.add_argument("--val_hours", type=float, default=None,
                    help="Val split size in hours when splitting by hours (default: 0, i.e. no val).")
    p.add_argument("--test_hours", type=float, default=None,
                    help="Test split size in hours when splitting by hours; rows beyond "
                         "train+val+test also fall to test when unset (default).")
    p.add_argument("--service_col", type=str, default=PREPROCESSING.SERVICE_COL)
    p.add_argument("--max_services", type=int, default=PREPROCESSING.MAX_SERVICES)
    p.add_argument("--subset_seed", type=int, default=PREPROCESSING.SUBSET_SEED)
    p.add_argument("--batch_size", type=int, default=0,
                    help="Services per worker group; 0 = auto-size to the worker pool (default: 0)")
    p.add_argument("--num_workers", type=float, default=0.9,
                    help="Fraction of CPU cores to use (default: 0.9)")
    p.add_argument("--polars_threads", type=int, default=48,
                    help="Polars thread pool size for aggregation/sort; aggregation is CPU-bound and scales "
                         "~linearly with threads (peak RAM barely changes), default: %(default)s")
    p.add_argument("--recompute", action="store_true",
                    help="Delete cached done markers and shards, forcing a full rebuild")
    p.add_argument("--no_service_cache", action="store_true",
                    help="Bypass the unique-service discovery cache and scan parquet directly")
    p.add_argument("--refresh_service_cache", action="store_true",
                    help="Force rebuild of the unique-service discovery cache")
    p.add_argument("--sync", action="store_true",
                    help="Run os.sync() after saving each chunk (durability; slower)")
    p.add_argument("--csv_path", default=None,
                    help="Path to an HPA-logs CSV (e.g. hpa_historical_logs.csv). "
                         "When set, windows are built from the CSV instead of the "
                         "parquet tables; each feature maps to a CSV column via "
                         "_CSV_COLUMN_MAP (so cpu/cpu_mem/cpu_mem_both work unchanged).")
    p.add_argument("--csv_time_col", default="Timestamp",
                    help="CSV timestamp column (default: %(default)s)")
    p.add_argument("--csv_id_col", default="Deployment",
                    help="CSV service-id column (default: %(default)s)")
    p.add_argument("--csv_tz", default="Asia/Tehran",
                    help="Timezone of CSV timestamps (default: %(default)s)")
    p.add_argument("--phase", choices=["auto", "windows"], default="auto",
                    help=argparse.SUPPRESS)

    args = p.parse_args()

    if args.train_hours is None and (
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
    resource_indices = [
        i for i, f in enumerate(feature_names)
        if "cpu" in f.lower() or "mem" in f.lower()
    ]

    if args.csv_path:
        _run_csv_source(args, spec, feature_names, target_indices,
                        resource_indices, num_workers)
        return

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
        f"(polars parallel across all cores)...", flush=True
    )
    all_services_list = discover_unique_services(
        DATASET_TABLES[base_table]["parquet_dir"],
        args.service_col,
        use_cache=not args.no_service_cache,
        refresh=args.refresh_service_cache,
    )
    print(f"Total unique services: {len(all_services_list)}", flush=True)

    if args.max_services and len(all_services_list) > args.max_services:
        rng = np.random.default_rng(args.subset_seed)
        idxs = rng.choice(len(all_services_list), size=args.max_services, replace=False)
        all_services_list = sorted(np.array(all_services_list)[idxs].tolist())
        print(f"Selected subset: {len(all_services_list)} services")
    else:
        print(f"Processing all {len(all_services_list)} services globally")

    os.makedirs(args.out_dir, exist_ok=True)

    for stale in glob.glob(os.path.join(args.out_dir, "tmp*")):
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)

    if args.recompute:
        cached = glob.glob(os.path.join(args.out_dir, "part-*.done")) + \
                 glob.glob(os.path.join(args.out_dir, "part-*_chunk-*.npy"))
        for f in cached:
            os.remove(f)
        if cached:
            print(f"Removed {len(cached)} cached artifacts for recompute", flush=True)

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
        "resource_indices": resource_indices,
    }
    args_dict.update(_split_params(args))

    arrays_path, index_path = _service_arrays_paths(args.out_dir)
    signature = _arrays_signature(args, base_table)

    if args.phase == "windows" or _arrays_cache_valid(arrays_path, index_path, signature):
        _phase_windows(args, args_dict, target_indices, all_services_list,
                       groups_to_run, group_size, num_workers,
                       arrays_path, index_path)
    else:
        _phase_aggregate(args, args_dict, target_indices, feature_names,
                         table_parts, needed_tables, table_exprs, base_table,
                         effective_id_cols, all_services_list,
                         arrays_path, index_path, signature)
        _reexec()


def _phase_aggregate(args, args_dict, target_indices, feature_names,
                     table_parts, needed_tables, table_exprs, base_table,
                     effective_id_cols, all_services_list,
                     arrays_path, index_path, signature):
    agg_frames = {}
    agg_order = [base_table] + [t for t in needed_tables if t != base_table]
    for t in agg_order:
        t0 = time.time()
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
        part_exprs, fold_exprs, final_exprs = _part_agg_plan(table_exprs, t, args.time_col)
        part_frames = []
        with tqdm(total=len(parts), desc=f"Aggregating table '{t}'", unit="part",
                  bar_format=("{desc}: {percentage:5.1f}%|{bar}| "
                              "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
                              "{rate_fmt}]")) as pbar:
            for part in parts:
                lf = (
                    pl.scan_parquet(part, low_memory=True)
                    .select(need_cols)
                    .with_columns(pl.col(args.time_col).cast(pl.Datetime))
                )
                if has_service:
                    lf = lf.filter(pl.col(args.service_col).is_in(all_services_list))
                lf = lf.with_columns(
                    pl.col(args.time_col).dt.truncate(args.freq).alias("_t")
                )
                part_frames.append(
                    lf.group_by(["_t"] + effective_id_cols)
                    .agg(part_exprs)
                    .collect(engine="streaming")
                )
                pbar.update(1)

        merged = _merge_part_frames(part_frames, fold_exprs, effective_id_cols)
        agg_frames[t] = _finalize_merged(merged, final_exprs, effective_id_cols)
        print(f"  Table '{t}' aggregated: {agg_frames[t].height} rows "
              f"in {time.time() - t0:.1f}s", flush=True)

    if base_table not in agg_frames or agg_frames[base_table].is_empty():
        print("No base-table data after aggregation; nothing to do.")
        return

    t_join = time.time()
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

    # No re-sort here: _merge_part_frames already orders by (_t, id_cols), so rows
    # within each service are in _t order; group_by below preserves it.
    joined_df = joined.drop_nulls(feature_names).collect(engine="streaming")
    print(f"Joined/clean table: {joined_df.height} rows "
          f"in {time.time() - t_join:.1f}s", flush=True)

    if joined_df.height == 0:
        print("No valid rows after join/filtering; nothing to do.")
        return

    group_cols = [c for c in effective_id_cols if c in joined_df.columns]
    service_arrays = {}
    pbar_arr = tqdm(desc="Building service arrays", unit="svc",
                    bar_format=("{desc}: {elapsed} [{rate_fmt}]"))
    n_svc = 0
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
        n_svc += 1
        pbar_arr.set_description(f"Building service arrays ({n_svc})")
        pbar_arr.update(1)
    pbar_arr.close()
    del joined_df
    gc.collect()
    print(f"Service feature arrays: {len(service_arrays)} services "
          f"in {time.time() - t_join:.1f}s", flush=True)

    if not service_arrays:
        print("No services with enough data after filtering; nothing to do.")
        return

    # Free the big aggregated frames before writing the cache file.
    del agg_frames
    del joined
    gc.collect()

    _save_service_arrays(service_arrays, arrays_path, index_path, signature)


def _phase_windows(args, args_dict, target_indices, all_services_list,
                   groups_to_run, group_size, num_workers,
                   arrays_path, index_path):
    with open(index_path, "r") as f:
        data = json.load(f)
    big = np.load(arrays_path, mmap_mode="r")
    index = data["index"]
    print(f"Loaded service arrays (mmap): {big.shape[0]} rows x {big.shape[1]} ch, "
          f"{len(index)} services", flush=True)

    global _WORKER_CTX
    _WORKER_CTX = {
        "args_dict": args_dict,
        "big_array": big,
        "service_index": index,
        "target_indices": target_indices,
        "resource_indices": args_dict["resource_indices"],
        "sync": args.sync,
    }

    tasks = []
    for gi in groups_to_run:
        start_idx = gi * group_size
        end_idx = min(start_idx + group_size, len(all_services_list))
        tasks.append((gi, all_services_list[start_idx:end_idx]))

    print(f"Processing {len(tasks)} groups with {num_workers} workers "
          f"(group_size={group_size})", flush=True)

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
        # Retry loop: groups are idempotent (done-marker guarded), so if the pool
        # breaks (worker OOM-killed/terminated), recreate it and re-run leftovers.
        # Give up loudly only if the pool breaks 3x with no progress (livelock).
        remaining = {gi: ids for gi, ids in tasks}
        last_progress = len(remaining)
        consecutive_breaks = 0
        while remaining:
            try:
                executor = ProcessPoolExecutor(max_workers=num_workers, **pool_kwargs)
                try:
                    futures = {
                        executor.submit(_process_service_group, gi, ids): gi
                        for gi, ids in remaining.items()
                    }
                    for future in as_completed(futures):
                        gi = futures[future]
                        future.result()
                        pbar.update(1)
                        remaining.pop(gi, None)
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
            except BrokenExecutor:
                if len(remaining) < last_progress:
                    consecutive_breaks = 0
                last_progress = len(remaining)
                consecutive_breaks += 1
                if consecutive_breaks > 3:
                    raise
                print(f"\n[WARN] Worker pool broke; {len(remaining)} groups left, "
                      f"retrying...", flush=True)
                time.sleep(5)

    pbar.close()
    print("\nAll service groups processed.")


if __name__ == "__main__":
    main()
