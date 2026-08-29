import argparse
import glob
import os
import sys
import subprocess
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

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


def get_all_indices(args):
    needed_tables = args.tables or sorted(list(tables_for_feature_set(args.feature_set)))
    result = {}
    for table in needed_tables:
        cfg = DATASET_TABLES[table]
        start_idx, end_idx = compute_indices(args.start_date, args.end_date, int(cfg["ratio_min"]))
        result[table] = list(range(start_idx, end_idx + 1))
    return needed_tables, result


import ctypes
_ALIGN = 4096
_mmap = __import__("mmap")


def _odirect_write_stream(proc, csv_path):
    """Write proc.stdout (decompressed CSV stream) to csv_path using O_DIRECT
    aligned writes. Returns bytes written or raises on error."""
    BLK = 4 * 1024 * 1024
    fd = os.open(csv_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    total = 0
    try:
        while True:
            chunk = proc.stdout.read(BLK)
            if not chunk:
                break
            n = ((len(chunk) + _ALIGN - 1) // _ALIGN) * _ALIGN
            buf = _mmap.mmap(-1, n)
            try:
                buf[:len(chunk)] = chunk
                view = memoryview(buf)
                off = 0
                while off < n:
                    w = os.write(fd, view[off:off + 0x200000])
                    off += w
                del view
            finally:
                buf.close()
            total += len(chunk)
        os.fsync(fd)
    finally:
        if os.path.exists(csv_path) and os.path.getsize(csv_path) != total:
            os.truncate(csv_path, total)
        os.close(fd)
    return total


def _pool_extract_one(args):
    tar_path, raw_dir, idx, use_pigz, odirect = args
    csv_path = os.path.join(raw_dir, f"CallGraph_{idx}.csv")
    try:
        if odirect:
            member = f"CallGraph_{idx}.csv"
            cmd = ["tar", "-xOzf", tar_path, member] if not use_pigz else \
                  ["tar", "-xO", "--use-compress-program=pigz", "-f", tar_path, member]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                _odirect_write_stream(proc, csv_path)
            except Exception:
                try:
                    os.remove(csv_path)
                except OSError:
                    pass
                proc.kill()
                proc.wait()
                raise
            rc = proc.wait()
            if rc != 0:
                err = proc.stderr.read().decode()[:200] if proc.stderr else "unknown"
                try:
                    os.remove(csv_path)
                except OSError:
                    pass
                return idx, None, f"tar failed: {err}"
            try:
                os.remove(tar_path)
            except OSError:
                pass
            return idx, csv_path, None
        if use_pigz:
            cmd = ["tar", "-xf", tar_path, "-C", raw_dir,
                   "--use-compress-program=pigz"]
        else:
            cmd = ["tar", "-xzf", tar_path, "-C", raw_dir]
        result = subprocess.run(cmd, timeout=None, capture_output=True)
        if result.returncode == 0:
            try:
                os.remove(tar_path)
            except OSError:
                pass
            if os.path.exists(csv_path):
                return idx, csv_path, None
            return idx, None, "CSV not found after extract"
        else:
            err = result.stderr.decode()[:200] if result.stderr else "unknown"
            try:
                os.remove(csv_path)
            except OSError:
                pass
            return idx, None, f"tar failed: {err}"
    except Exception as e:
        try:
            os.remove(csv_path)
        except OSError:
            pass
        return idx, None, str(e)


def _pool_ingest_one(args):
    csv_paths, out_dir, table, worker_id, batch_idx = args
    out_dir_abs = DATASET_TABLES[table]["parquet_dir"]
    out_path = os.path.join(out_dir_abs, f"part-{batch_idx:05d}_w{worker_id}.parquet")
    tmp_path = out_path + ".tmp"

    try:
        all_dfs = []
        total_rows = 0
        done_indices = []

        for csv_path, idx in csv_paths:
            try:
                df = pl.read_csv(
                    csv_path,
                    low_memory=True,
                    try_parse_dates=False,
                    infer_schema_length=0,
                    truncate_ragged_lines=True,
                    ignore_errors=True
                )
                if df.height == 0:
                    done_indices.append(idx)
                    continue

                if table == "mscallgraph":
                    S = {"MS_15135", "MS_58542", "MS_30441", "MS_7951", "MS_48031", "MS_60792"}
                    df = df.filter(
                        pl.col("dm").is_in(S) | pl.col("um").is_in(S)
                    )

                if df.height == 0:
                    done_indices.append(idx)
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

                if df.height > 0:
                    all_dfs.append(df)
                    total_rows += df.height
                done_indices.append(idx)
            except Exception as e:
                done_indices.append(idx)

        if all_dfs:
            combined = pl.concat(all_dfs) if len(all_dfs) > 1 else all_dfs[0]
            if combined.height > 0:
                combined.write_parquet(tmp_path, compression="zstd")
                os.rename(tmp_path, out_path)

        return done_indices, total_rows, None

    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        return [idx for _, idx in csv_paths], 0, str(e)


def _tar_ok(tar_path):
    try:
        r = subprocess.run(["pigz", "-t", tar_path], capture_output=True)
        return r.returncode == 0
    except Exception:
        return False


def _find_corrupt_tars(args, table, raw_dir, indices):
    from tqdm import tqdm
    cfg = DATASET_TABLES[table]
    cache = os.path.join(cfg["parquet_dir"], f"_{table}_validated.txt")
    validated = set()
    if os.path.exists(cache):
        with open(cache) as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    validated.add(int(line))

    to_val = []
    for idx in indices:
        tar_path = os.path.join(raw_dir, f"{table}_{idx}.tar.gz")
        csv_path = os.path.join(raw_dir, f"CallGraph_{idx}.csv")
        csv_done = os.path.join(cfg["parquet_dir"], f"{table}_{idx}.csv_done")
        if os.path.exists(csv_done) or os.path.exists(csv_path):
            continue
        if os.path.exists(tar_path):
            if idx in validated:
                continue
            to_val.append((idx, tar_path))

    if not to_val:
        print(f"  [{table}] No new tars to validate")
        return []

    print(f"  [{table}] Validating {len(to_val)} tars in parallel...")
    corrupt = []
    count = 0
    with ThreadPoolExecutor(max_workers=args.recheck_workers) as pool:
        futs = {pool.submit(_tar_ok, tp): idx for idx, tp in to_val}
        with tqdm(total=len(futs), desc=f"  [{table}] Validate", unit="tar", ncols=80, dynamic_ncols=True) as pbar:
            for fut in as_completed(futs):
                idx = futs[fut]
                if fut.result():
                    validated.add(idx)
                else:
                    corrupt.append((idx, f"{BASE_URL}/{cfg['prefix']}_{idx}.tar.gz"))
                pbar.update(1)
                count += 1
                if count % args.recheck_workers == 0:
                    with open(cache, "w") as f:
                        for i in sorted(validated):
                            f.write(f"{i}\n")

    with open(cache, "w") as f:
        for idx in sorted(validated):
            f.write(f"{idx}\n")
    print(f"  [{table}] Validated {len(validated)} OK, {len(corrupt)} corrupt")
    return corrupt


def _collect_downloads(args, table, raw_dir, indices, include_corrupt):
    cfg = DATASET_TABLES[table]
    urls_to_download = []
    for idx in indices:
        tar_path = os.path.join(raw_dir, f"{table}_{idx}.tar.gz")
        csv_path = os.path.join(raw_dir, f"CallGraph_{idx}.csv")
        csv_done = os.path.join(cfg["parquet_dir"], f"{table}_{idx}.csv_done")
        if os.path.exists(csv_done) or os.path.exists(csv_path):
            continue
        if os.path.exists(tar_path):
            if include_corrupt:
                if not _tar_ok(tar_path):
                    urls_to_download.append((idx, f"{BASE_URL}/{cfg['prefix']}_{idx}.tar.gz"))
            continue
        url = f"{BASE_URL}/{cfg['prefix']}_{idx}.tar.gz"
        urls_to_download.append((idx, url))
    return urls_to_download


def phase1_download(args, needed_tables, all_indices):
    from tqdm import tqdm

    for table in needed_tables:
        cfg = DATASET_TABLES[table]
        raw_dir = cfg["raw_dir"]
        os.makedirs(raw_dir, exist_ok=True)

        indices = all_indices[table]
        urls_to_download = _collect_downloads(args, table, raw_dir, indices, include_corrupt=False)

        if not urls_to_download:
            print(f"  [{table}] All chunks already downloaded/extracted, skipping phase 1")
            continue

        print(f"  [{table}] Downloading {len(urls_to_download)} chunks with aria2c...")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp') as f:
            input_file = f.name
            for idx, url in urls_to_download:
                f.write(f"{url}\n")
                f.write(f"  out={table}_{idx}.tar.gz\n")
                f.write(f"  dir={raw_dir}\n")

        try:
            cmd = [
                "aria2c",
                "-i", input_file,
                "-j", str(args.aria_concurrent),
                "-x", str(args.aria_connections),
                "-s", str(args.aria_connections),
                "-k", "10M",
                "-c",
                "--max-tries=10",
                "--retry-wait=10",
                "--timeout=600",
                "--auto-file-renaming=false",
                "--console-log-level=warn",
                "--summary-interval=30",
            ]
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"  [{table}] aria2c exited with code {proc.returncode}")
        finally:
            try:
                os.remove(input_file)
            except OSError:
                pass


def phase2_extract(args, needed_tables, all_indices):
    from tqdm import tqdm

    for table in needed_tables:
        cfg = DATASET_TABLES[table]
        raw_dir = cfg["raw_dir"]

        if args.recheck:
            print(f"  [{table}] Validating tars (recheck corrupt)...")
            corrupt = _find_corrupt_tars(args, table, raw_dir, all_indices[table])
            if corrupt:
                print(f"  [{table}] Re-downloading {len(corrupt)} corrupt tars...")
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp') as f:
                    input_file = f.name
                    for idx, url in corrupt:
                        for stale in (raw_dir + f"/{table}_{idx}.tar.gz",
                                      raw_dir + f"/{table}_{idx}.tar.gz.aria2"):
                            try:
                                if os.path.exists(stale):
                                    os.remove(stale)
                            except OSError:
                                pass
                        f.write(f"{url}\n")
                        f.write(f"  out={table}_{idx}.tar.gz\n")
                        f.write(f"  dir={raw_dir}\n")
                try:
                    cmd = [
                        "aria2c",
                        "-i", input_file,
                        "-j", str(args.aria_concurrent),
                        "-x", str(args.aria_connections),
                        "-s", str(args.aria_connections),
                        "-k", "10M",
                        "--max-tries=10",
                        "--retry-wait=10",
                        "--timeout=600",
                        "--auto-file-renaming=false",
                        "--console-log-level=warn",
                        "--summary-interval=30",
                    ]
                    subprocess.run(cmd)
                finally:
                    try:
                        os.remove(input_file)
                    except OSError:
                        pass
                print(f"  [{table}] Done re-downloading. Re-validating...")

        tarballs = []
        bad_tars = []
        cache = os.path.join(cfg["parquet_dir"], f"_{table}_validated.txt")
        validated = set()
        if os.path.exists(cache):
            with open(cache) as f:
                for line in f:
                    if line.strip().isdigit():
                        validated.add(int(line.strip()))
        for idx in all_indices[table]:
            csv_done = os.path.join(cfg["parquet_dir"], f"{table}_{idx}.csv_done")
            csv_path = os.path.join(raw_dir, f"CallGraph_{idx}.csv")
            tar_path = os.path.join(raw_dir, f"{table}_{idx}.tar.gz")
            if os.path.exists(csv_done):
                continue
            if os.path.exists(csv_path):
                continue
            if os.path.exists(tar_path):
                tarballs.append((tar_path, raw_dir, idx, args.use_pigz, args.extract_odirect))

        if len(validated) > 0:
            with open(cache, "w") as f:
                for idx in sorted(validated):
                    f.write(f"{idx}\n")

        if not tarballs:
            print(f"  [{table}] No tarballs to extract, skipping phase 2")
            continue

        print(f"  [{table}] Extracting {len(tarballs)} tarballs ({args.extract_workers} threads)...")

        failed = []
        if args.live_ingest:
            out_dir = cfg["parquet_dir"]
            os.makedirs(out_dir, exist_ok=True)
            existing_parts = glob.glob(os.path.join(out_dir, f"part-*_w*.parquet"))
            part_counter = [len(existing_parts)]
            pending_buf = []
            ingest_futures = {}
            ingest_paths = {}
            ingest_pg = tqdm(total=0, desc=f"  [{table}] Live-ingest", unit="csv", ncols=80, position=1) if args.live_ingest else None

            def drain_ingests():
                for fut in list(ingest_futures):
                    if not fut.done():
                        continue
                    done_indices, total_rows, err = ingest_futures.pop(fut).result()
                    if err:
                        print(f"\n  Ingest error: {err}", file=sys.stderr)
                    for idx in done_indices:
                        dp = os.path.join(out_dir, f"{table}_{idx}.csv_done")
                        try:
                            with open(dp, 'w') as f:
                                f.write("")
                        except OSError:
                            pass
                    if args.ingest:
                        for _, (csv_path, _) in ingest_paths.pop(fut, []):
                            try:
                                os.remove(csv_path)
                            except OSError:
                                pass
                    ingest_pg.update(len(done_indices))

            with ThreadPoolExecutor(max_workers=args.extract_workers) as pool:
                futures = {pool.submit(_pool_extract_one, t): t for t in tarballs}
                with ProcessPoolExecutor(max_workers=args.ingest_workers) as igpool:
                    with tqdm(total=len(futures), desc=f"  [{table}] Extract", unit="tar", ncols=80) as pbar:
                        for future in as_completed(futures):
                            idx, csv_path, err = future.result()
                            if err:
                                failed.append((idx, f"{BASE_URL}/{cfg['prefix']}_{idx}.tar.gz"))
                            else:
                                pending_buf.append((csv_path, idx))
                                if len(pending_buf) >= args.batch_size:
                                    batch = pending_buf[:]
                                    pending_buf = []
                                    fut = igpool.submit(_pool_ingest_one,
                                        (batch, out_dir, table, 0, part_counter[0]))
                                    part_counter[0] += 1
                                    ingest_futures[fut] = batch
                                    ingest_paths[fut] = batch
                                    ingest_pg.total += len(batch)
                            pbar.update(1)
                            drain_ingests()

                # drain remaining buffer + any in-flight ingests
                if pending_buf:
                    batch = pending_buf[:]
                    pending_buf = []
                    fut = igpool.submit(_pool_ingest_one, (batch, out_dir, table, 0, part_counter[0]))
                    part_counter[0] += 1
                    ingest_futures[fut] = batch
                    ingest_paths[fut] = batch
                    ingest_pg.total += len(batch)
                while ingest_futures:
                    drain_ingests()
                    time.sleep(0.5)
        else:
            with ThreadPoolExecutor(max_workers=args.extract_workers) as pool:
                futures = {pool.submit(_pool_extract_one, t): t for t in tarballs}
                with tqdm(total=len(futures), desc=f"  [{table}] Extract", unit="tar", ncols=80) as pbar:
                    for future in as_completed(futures):
                        idx, csv_path, err = future.result()
                        if err:
                            failed.append((idx, f"{BASE_URL}/{cfg['prefix']}_{idx}.tar.gz"))
                        pbar.update(1)

        if failed:
            print(f"  [{table}] {len(failed)} tars failed extraction, re-downloading...")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp') as f:
                input_file = f.name
                for idx, url in failed:
                    for stale in (raw_dir + f"/{table}_{idx}.tar.gz",
                                  raw_dir + f"/{table}_{idx}.tar.gz.aria2"):
                        try:
                            if os.path.exists(stale):
                                os.remove(stale)
                        except OSError:
                            pass
                    f.write(f"{url}\n")
                    f.write(f"  out={table}_{idx}.tar.gz\n")
                    f.write(f"  dir={raw_dir}\n")
                f.flush()
            try:
                cmd = [
                    "aria2c",
                    "-i", input_file,
                    "-j", str(args.aria_concurrent),
                    "-x", str(args.aria_connections),
                    "-s", str(args.aria_connections),
                    "-k", "10M",
                    "--max-tries=10",
                    "--retry-wait=10",
                    "--timeout=600",
                    "--auto-file-renaming=false",
                    "--console-log-level=warn",
                    "--summary-interval=30",
                ]
                subprocess.run(cmd)
            finally:
                try:
                    os.remove(input_file)
                except OSError:
                    pass

            retry = []
            for idx, url in failed:
                tar_path = os.path.join(raw_dir, f"{table}_{idx}.tar.gz")
                csv_path = os.path.join(raw_dir, f"CallGraph_{idx}.csv")
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                retry.append((tar_path, raw_dir, idx, args.use_pigz, args.extract_odirect))
            if retry:
                with ThreadPoolExecutor(max_workers=args.extract_workers) as pool:
                    futures = {pool.submit(_pool_extract_one, t): t for t in retry}
                    with tqdm(total=len(futures), desc=f"  [{table}] Re-extract", unit="tar", ncols=80) as pbar:
                        for future in as_completed(futures):
                            idx, csv_path, err = future.result()
                            if err:
                                print(f"\n  STILL FAILED idx={idx}: {err}", file=sys.stderr)
                            pbar.update(1)


def phase3_ingest(args, needed_tables, all_indices):
    from tqdm import tqdm

    for table in needed_tables:
        cfg = DATASET_TABLES[table]
        raw_dir = cfg["raw_dir"]
        out_dir = cfg["parquet_dir"]
        os.makedirs(out_dir, exist_ok=True)

        key_cols = list(cfg.get("key_cols", []))
        feature_cols = table_to_raw_columns(args.feature_set).get(table, [])

        indices_to_ingest = []
        for idx in all_indices[table]:
            csv_done = os.path.join(out_dir, f"{table}_{idx}.csv_done")
            if os.path.exists(csv_done):
                continue
            csv_path = os.path.join(raw_dir, f"CallGraph_{idx}.csv")
            if os.path.exists(csv_path):
                indices_to_ingest.append((csv_path, idx))

        if not indices_to_ingest:
            print(f"  [{table}] All chunks already ingested, skipping phase 3")
            continue

        print(f"  [{table}] Ingesting {len(indices_to_ingest)} CSVs into parquet ({args.ingest_workers} workers)...")

        existing_parts = glob.glob(os.path.join(out_dir, f"part-*_w*.parquet"))
        batch_size = args.batch_size
        batches = []
        for i in range(0, len(indices_to_ingest), batch_size):
            batch = indices_to_ingest[i:i+batch_size]
            batches.append((batch, out_dir, table, 0, len(existing_parts) + len(batches)))

        with ProcessPoolExecutor(max_workers=args.ingest_workers) as pool:
            futures = {}
            for batch_args in batches:
                csv_paths, od, tbl, wid, batch_idx = batch_args
                future = pool.submit(_pool_ingest_one, (csv_paths, od, tbl, wid, batch_idx))
                futures[future] = batch_args

            with tqdm(total=len(indices_to_ingest), desc=f"  [{table}] Ingest", unit="csv", ncols=80) as pbar:
                for future in as_completed(futures):
                    done_indices, total_rows, err = future.result()
                    if err:
                        print(f"\n  Ingest error: {err}", file=sys.stderr)
                    for idx in done_indices:
                        dp = os.path.join(out_dir, f"{table}_{idx}.csv_done")
                        with open(dp, 'w') as f:
                            f.write("")
                    if args.ingest:
                        csv_paths_batch, _, _, _, _ = futures[future]
                        for csv_path, _ in csv_paths_batch:
                            try:
                                os.remove(csv_path)
                            except OSError:
                                pass
                    pbar.update(len(done_indices))


def main():
    ap = argparse.ArgumentParser(description="Download and ingest Alibaba trace chunks (3-phase pipeline).")
    ap.add_argument("--start_date", default="0d0")
    ap.add_argument("--end_date", default="7d0")
    ap.add_argument("--feature_set", default=PREPROCESSING.FEATURE_SET if hasattr(PREPROCESSING, 'FEATURE_SET') else "cpu_mem_both")
    ap.add_argument("--tables", nargs="+")
    ap.add_argument("--ingest", action="store_true", help="Delete CSVs after parquetification")
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--aria_concurrent", type=int, default=4, help="aria2c concurrent downloads (-j)")
    ap.add_argument("--aria_connections", type=int, default=4, help="aria2c connections per file (-x/-s)")
    ap.add_argument("--extract_workers", type=int, default=8, help="Thread pool size for tar extraction")
    ap.add_argument("--extract_odirect", action="store_true", default=False, help="Write extracted CSVs with O_DIRECT (avoids page-cache/journal overload that triggers SAN read-only remounts)")
    ap.add_argument("--live_ingest", action="store_true", default=False, help="Ingest extracted CSVs to parquet in batches as they complete, instead of one big ingest pass after all extraction")
    ap.add_argument("--ingest_workers", type=int, default=4, help="Process pool size for CSV->parquet")
    ap.add_argument("--use_pigz", action="store_true", default=True, help="Use pigz for parallel tar decompression")
    ap.add_argument("--recheck", action="store_true", help="Validate existing tars with pigz -t and re-download corrupt ones before extraction")
    ap.add_argument("--recheck_workers", type=int, default=16, help="Parallel workers for tar validation")
    ap.add_argument("--skip_download", action="store_true", help="Skip phase 1 (download)")
    ap.add_argument("--skip_extract", action="store_true", help="Skip phase 2 (extract)")
    ap.add_argument("--skip_ingest", action="store_true", help="Skip phase 3 (ingest)")

    args = ap.parse_args()

    needed_tables, all_indices = get_all_indices(args)
    total = sum(len(v) for v in all_indices.values())

    print(f"Feature set: {args.feature_set}")
    print(f"Tables: {needed_tables}")
    print(f"Range: {args.start_date} -> {args.end_date}")
    print(f"Total chunks: {total}")
    print(f"aria2c: -j {args.aria_concurrent} -x {args.aria_connections}")
    print(f"Extract threads: {args.extract_workers}")
    print(f"Ingest workers: {args.ingest_workers}")
    print(f"Ingest (rm CSVs): {args.ingest}")

    t0 = time.time()

    if not args.skip_download:
        print(f"\n--- Phase 1: Download ---")
        phase1_download(args, needed_tables, all_indices)

    if not args.skip_extract:
        print(f"\n--- Phase 2: Extract ---")
        phase2_extract(args, needed_tables, all_indices)

    if not args.skip_ingest:
        print(f"\n--- Phase 3: Ingest ---")
        phase3_ingest(args, needed_tables, all_indices)

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\nDone in {h}h {m}m {s}s.")


if __name__ == "__main__":
    main()
