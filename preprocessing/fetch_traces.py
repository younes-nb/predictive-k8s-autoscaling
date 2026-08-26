import argparse
import glob
import os
import sys
import subprocess
import time

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import (
    DATASET_TABLES,
    PREPROCESSING,
    tables_for_feature_set,
    table_to_raw_columns,
)

BASE_URL = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces"


def parse_dh(s: str):
    if "d" not in s:
        raise ValueError("Bad time spec: %r (expected like '0d0' or '1d12')" % s)
    d_str, h_str = s.split("d", 1)
    return int(d_str), int(h_str)


def compute_indices(start_date: str, end_date: str, ratio_min: int):
    sd, sh = parse_dh(start_date)
    ed, eh = parse_dh(end_date)
    start_idx = (sd * 24 * 60 + sh * 60) // ratio_min
    end_idx = (ed * 24 * 60 + eh * 60) // ratio_min - 1
    return start_idx, max(start_idx, end_idx)


def download_file(url: str, dst_path: str, max_retries: int = 5) -> bool:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["curl", "-fL", "--retry", "3", "--retry-delay", "5", "--max-time", "300",
                 "-o", dst_path, url],
                timeout=400,
                capture_output=True
            )
            if result.returncode == 0 and os.path.exists(dst_path):
                size = os.path.getsize(dst_path)
                if size > 1_000_000:
                    return True
                else:
                    print(f"  Download too small ({size} bytes)", file=sys.stderr)
            else:
                print(f"  Download failed (attempt {attempt+1}/{max_retries}): {result.stderr.decode()[:100] if result.stderr else 'Unknown error'}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"  Download timeout (attempt {attempt+1}/{max_retries})", file=sys.stderr)
        except Exception as e:
            print(f"  Download error (attempt {attempt+1}/{max_retries}): {e}", file=sys.stderr)

        try:
            if os.path.exists(dst_path):
                os.remove(dst_path)
        except OSError:
            pass

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    return False


def extract_tar(tar_path: str, raw_dir: str) -> str:
    try:
        result = subprocess.run(
            ["tar", "-xzf", tar_path, "-C", raw_dir],
            timeout=180,
            capture_output=True
        )
        if result.returncode == 0:
            os.remove(tar_path)
            basename = os.path.basename(tar_path)
            parts = basename.split("_")
            if len(parts) >= 3:
                idx = parts[1]
                csv_name = f"CallGraph_{idx}.csv"
                csv_path = os.path.join(raw_dir, csv_name)
                if os.path.exists(csv_path):
                    return csv_path
            csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv') and f.startswith('CallGraph_')]
            if csv_files:
                return os.path.join(raw_dir, csv_files[0])
        else:
            print(f"  Extract failed: {result.stderr.decode()[:200] if result.stderr else 'Unknown error'}", file=sys.stderr)
    except Exception as e:
        print(f"  Extract error: {e}", file=sys.stderr)
    return None


def process_batch(pending_csv_paths, out_dir, part_num, key_cols, feature_cols, table, worker_id):
    if not pending_csv_paths:
        return part_num

    out_dir_abs = DATASET_TABLES[table]["parquet_dir"]
    existing_parts = glob.glob(os.path.join(out_dir_abs, f"part-*_w{worker_id}.parquet"))
    current_part_num = len(existing_parts)
    out_path = os.path.join(out_dir_abs, f"part-{current_part_num:05d}_w{worker_id}.parquet")
    tmp_path = out_path + ".tmp"

    try:
        all_dfs = []
        total_rows = 0

        for csv_path in pending_csv_paths:
            try:
                print(f"  Processing {os.path.basename(csv_path)} ...", end=" ", flush=True)

                df = pl.read_csv(
                    csv_path,
                    low_memory=True,
                    try_parse_dates=False,
                    infer_schema_length=0,
                    truncate_ragged_lines=True,
                    ignore_errors=True
                )

                if df.height == 0:
                    print("empty")
                    continue

                if table == "mscallgraph":
                    S = {"MS_15135", "MS_58542", "MS_30441", "MS_7951", "MS_48031", "MS_60792"}
                    df = df.filter(
                        pl.col("dm").is_in(S) | pl.col("um").is_in(S)
                    )

                if df.height == 0:
                    print("filtered empty")
                    continue

                df = df.with_columns(
                    pl.col("timestamp")
                    .str.strip_chars()
                    .cast(pl.Int64)
                    .alias("ts_int")
                )
                df = df.with_columns(
                    pl.from_epoch(pl.col("ts_int") // 1000, time_unit="s")
                    .alias("timestamp_dt")
                )

                select_cols = ["timestamp", "timestamp_dt", "traceid", "rpc_id", "um", "dm", "rpctype", "rt", "service", "interface", "uminstanceid", "dminstanceid"]
                available_cols = [c for c in select_cols if c in df.columns]
                df = df.select(available_cols)
                df = df.sort(["timestamp_dt", "traceid", "rpc_id"])

                if df.height == 0:
                    print("no data after processing")
                    continue

                all_dfs.append(df)
                total_rows += df.height
                print(f"{df.height} rows")

            except Exception as e:
                print(f"ERROR: {e}")

        if all_dfs:
            combined = pl.concat(all_dfs) if len(all_dfs) > 1 else all_dfs[0]
            if combined.height > 0:
                combined.write_parquet(tmp_path, compression="zstd")
                os.rename(tmp_path, out_path)
                print(f"  Wrote {total_rows} rows to {os.path.basename(out_path)}")
                return current_part_num + 1
            else:
                print("  No data to write")
                return part_num
        else:
            print("  No valid dataframes")
            return part_num

    except Exception as e:
        print(f"  Batch processing error: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        return part_num


def run_worker(args, worker_id):
    needed_tables = args.tables or sorted(list(tables_for_feature_set(args.feature_set)))

    for table in needed_tables:
        cfg = DATASET_TABLES[table]
        raw_dir = cfg["raw_dir"]
        out_dir = cfg["parquet_dir"]
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        start_idx, end_idx = compute_indices(args.start_date, args.end_date, int(cfg["ratio_min"]))

        all_indices = list(range(start_idx, end_idx + 1))
        my_indices = [idx for i, idx in enumerate(all_indices) if i % args.n_workers == worker_id]
        print(f"Worker {worker_id}: processing {len(my_indices)} chunks (indices {my_indices[0]} to {my_indices[-1]} if any)")

        key_cols = list(cfg.get("key_cols", []))
        feature_cols = table_to_raw_columns(args.feature_set).get(table, [])

        part_num = 0
        pending = []
        pending_indices = []

        def _flush_batch(label):
            nonlocal part_num, pending, pending_indices
            if not pending:
                return
            print(f"  {label} of {len(pending)} files...")
            old_part = part_num
            part_num = process_batch(pending_csv_paths=pending, out_dir=out_dir, part_num=part_num, key_cols=key_cols, feature_cols=feature_cols, table=table, worker_id=worker_id)
            if part_num > old_part:
                for di in pending_indices:
                    dp = os.path.join(out_dir, f"{table}_{di}.csv_done")
                    with open(dp, 'w') as f:
                        f.write("")
                if args.ingest:
                    for csv in pending:
                        try:
                            os.remove(csv)
                        except OSError:
                            pass
            pending = []
            pending_indices = []

        skipped = 0
        for idx in my_indices:
            csv_done = os.path.join(out_dir, f"{table}_{idx}.csv_done")
            if os.path.exists(csv_done):
                skipped += 1
                continue

            url = f"{BASE_URL}/{cfg['prefix']}_{idx}.tar.gz"
            tar_path = os.path.join(raw_dir, f"{table}_{idx}_w{worker_id}.tar.gz")
            csv_path = os.path.join(raw_dir, f"CallGraph_{idx}.csv")

            if not os.path.exists(csv_path):
                print(f"  Downloading chunk {idx}...", end=" ", flush=True)
                if download_file(url, tar_path):
                    print("extracting...", end=" ", flush=True)
                    result_csv_path = extract_tar(tar_path, raw_dir)
                    if result_csv_path and os.path.exists(result_csv_path):
                        print("done")
                    else:
                        print("extract failed")
                        continue
                else:
                    print("download failed")
                    continue

            pending.append(csv_path)
            pending_indices.append(idx)
            if len(pending) >= args.batch_size:
                _flush_batch("Processing batch")

        if skipped:
            print(f"  Skipped {skipped} already-parquetified chunks")
        _flush_batch("Processing final batch")


def count_done_markers(out_dir, table, total):
    count = 0
    for f in os.listdir(out_dir):
        if f.startswith(f"{table}_") and f.endswith(".csv_done"):
            count += 1
    return min(count, total)


def spawn_workers(args):
    needed_tables = args.tables or sorted(list(tables_for_feature_set(args.feature_set)))
    total_chunks = 0
    for table in needed_tables:
        cfg = DATASET_TABLES[table]
        start_idx, end_idx = compute_indices(args.start_date, args.end_date, int(cfg["ratio_min"]))
        total_chunks += end_idx - start_idx + 1

    print(f"Spawning {args.n_workers} workers for {total_chunks} total chunks")

    cmd_base = [
        sys.executable, os.path.abspath(__file__),
        "--feature_set", args.feature_set,
        "--start_date", args.start_date,
        "--end_date", args.end_date,
        "--batch_size", str(args.batch_size),
        "--n_workers", str(args.n_workers),
    ]
    if args.tables:
        cmd_base += ["--tables"] + args.tables
    if args.ingest:
        cmd_base.append("--ingest")

    workers = []
    for wid in range(args.n_workers):
        cmd = cmd_base + ["--_worker_id", str(wid)]
        log_path = f"/tmp/fetch_traces_w{wid}.log"
        log_fh = open(log_path, "w")
        p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        workers.append((p, log_fh, wid, log_path))
        print(f"  Worker {wid} started (pid={p.pid}, log={log_path})")

    from tqdm import tqdm
    with tqdm(total=total_chunks, desc="Fetching", unit="chunk", ncols=80) as pbar:
        while any(p.poll() is None for p, _, _, _ in workers):
            done = 0
            for table in needed_tables:
                out_dir = DATASET_TABLES[table]["parquet_dir"]
                cfg = DATASET_TABLES[table]
                start_idx, end_idx = compute_indices(args.start_date, args.end_date, int(cfg["ratio_min"]))
                done += count_done_markers(out_dir, table, end_idx - start_idx + 1)
            pbar.n = done
            pbar.refresh()

            alive = [p for p, _, _, _ in workers if p.poll() is None]
            if not alive:
                break
            time.sleep(3)

        for p, _, _, _ in workers:
            p.wait()
        pbar.n = total_chunks
        pbar.refresh()

    for _, fh, wid, log_path in workers:
        fh.close()

    failed = [(wid, log_path) for p, _, wid, log_path in workers if p.returncode != 0]
    if failed:
        print(f"\n{len(failed)} worker(s) failed:")
        for wid, log_path in failed:
            print(f"  Worker {wid}: see {log_path}")
    else:
        print(f"\nAll {args.n_workers} workers completed successfully.")


def main():
    ap = argparse.ArgumentParser(description="Download and ingest Alibaba trace chunks.")
    ap.add_argument("--start_date", default="0d0")
    ap.add_argument("--end_date", default="7d0")
    ap.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET if hasattr(PREPROCESSING, 'FEATURE_SET') else "cpu_mem_both")
    ap.add_argument("--tables", nargs="+")
    ap.add_argument("--ingest", action="store_true", help="Delete CSVs after parquetification")
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--_worker_id", type=int, default=None, dest="worker_id",
                    help=argparse.SUPPRESS)

    args = ap.parse_args()

    if args.worker_id is not None:
        worker_id = args.worker_id
        print(f"Worker {worker_id}/{args.n_workers} (feature_set={args.feature_set})")
        run_worker(args, worker_id)
        print(f"\nWorker {worker_id} done.")
    else:
        needed_tables = args.tables or sorted(list(tables_for_feature_set(args.feature_set)))
        print(f"Feature set: {args.feature_set}")
        print(f"Tables: {needed_tables}")
        print(f"Range: {args.start_date} -> {args.end_date}")
        print(f"Workers: {args.n_workers}")
        print(f"Ingest (rm CSVs): {args.ingest}")
        spawn_workers(args)
        print("\nDone.")


if __name__ == "__main__":
    main()
