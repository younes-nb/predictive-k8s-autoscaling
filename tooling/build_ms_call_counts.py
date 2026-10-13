"""Build per-minute per-MS inbound (received) call counts from MSCallGraph.

Input : /dataset/parquet/mscallgraph/part-*_w0.parquet (1.3T, 6239 files)
Output: /dataset/ms_call_counts/
  - per-minute aggregates sharded parquet:  minute (=timestamp_ms//60000),
    dm (receiver MS), call_count (all rpctypes summed: user+http+rpc+mq+db+mc+...)
  - totals per MS + metadata

Cleaning applied per file:
  1. drop corrupted rows where rpctype looks like an rpc_id (CSV-shift artifact, ~0.04%)
  2. dedup double-recorded spans on (traceid, rpc_id, um, dm) (~12-13% dups:
     each remote call is logged once UM-side and once DM-side)
  3. NO rpctype filtering otherwise: http/rpc/mq/db/mc/UNKNOWN/USER-originated
     are all summed together as requested ("only received calls, sum all together").
"""
import os
import glob
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import polars as pl

SRC_DIR = "/dataset/parquet/mscallgraph"
OUT_DIR = "/dataset/ms_call_counts"
TMP_DIR = os.path.join(OUT_DIR, "_tmp_parts")
FINAL_DIR = os.path.join(OUT_DIR, "per_minute")
N_WORKERS = 6
# Cap per-process thread pools: without this each of the N_WORKERS processes
# lets polars/rayon grab all 40 cores -> oversubscription + memory spikes.
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")


def list_sources():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "part-*_w0.parquet")))
    # exclude any temp/hidden files
    files = [f for f in files if os.path.isfile(f) and not f.endswith(".tmp")]
    return files


def process_one(args):
    src, idx = args
    try:
        t0 = time.time()
        df = (
            pl.scan_parquet(src)
            .select(["timestamp", "traceid", "rpc_id", "um", "dm", "rpctype"])
            .collect()
        )
        n_raw = len(df)
        # timestamp is stored as string ms
        df = df.with_columns(pl.col("timestamp").cast(pl.Int64, strict=False).alias("ts"))
        df = df.drop_nulls(subset=["ts", "dm"])
        # drop corrupted rows: rpctype holds an rpc_id (column shift artifact)
        df = df.filter(~pl.col("rpctype").str.contains(r"^[0-9.]+$"))
        df = df.with_columns((pl.col("ts") // 60000).alias("minute"))
        # Memory guard: encode high-cardinality strings as Categorical before
        # the dedup (dict encoding: int codes instead of repeated strings),
        # and dedup/group per minute chunk (each file spans ~3 min) so peak
        # memory is ~1/3 of whole-file unique().
        df = df.with_columns(
            pl.col("traceid").cast(pl.Categorical),
            pl.col("rpc_id").cast(pl.Categorical),
            pl.col("um").cast(pl.Categorical),
            pl.col("dm").cast(pl.Categorical),
        )
        parts = []
        for m in df["minute"].unique().to_list():
            c = df.filter(pl.col("minute") == m)
            # dedup UM-side + DM-side double recording
            c = c.unique(subset=["traceid", "rpc_id", "um", "dm"], keep="first")
            g = c.group_by(["minute", "dm"]).agg(pl.len().alias("call_count"))
            parts.append(g)
        out = pl.concat(parts)
        out = out.with_columns(
            (pl.col("minute") * 60000).alias("minute_start_ms"),
            pl.col("dm").cast(pl.String),
        )
        tmp_path = os.path.join(TMP_DIR, f"agg-{idx:05d}.parquet")
        out.write_parquet(tmp_path + ".tmp", compression="zstd")
        os.replace(tmp_path + ".tmp", tmp_path)  # atomic: no 0-byte files on crash
        dt = time.time() - t0
        return (idx, src, n_raw, len(out), dt, None)
    except Exception as e:
        return (idx, src, 0, 0, 0.0, f"{e}\n{traceback.format_exc()}")


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)
    files = list_sources()
    print(f"Found {len(files)} source files", flush=True)
    if not files:
        raise SystemExit(f"No source parquet found in {SRC_DIR}")

    # skip already-done parts (resume support)
    todo = []
    done_idx = set()
    for i, f in enumerate(files):
        tp = os.path.join(TMP_DIR, f"agg-{i:05d}.parquet")
        if os.path.isfile(tp) and os.path.getsize(tp) > 0:
            done_idx.add(i)
        else:
            todo.append((f, i))
    print(f"Already done: {len(done_idx)}, todo: {len(todo)}", flush=True)

    t_all = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            for j, res in enumerate(ex.map(process_one, todo, chunksize=1)):
                idx, src, n_raw, n_grp, dt, err = res
                if err:
                    print(f"[FAIL] {idx} {os.path.basename(src)}: {err}", flush=True)
                elif j % 200 == 0 or j == len(todo) - 1:
                    print(
                        f"[{j+1}/{len(todo)}] {os.path.basename(src)} "
                        f"raw={n_raw} groups={n_grp} {dt:.1f}s",
                        flush=True,
                    )
    print(f"Per-file stage done in {(time.time()-t_all)/60:.1f} min", flush=True)

    # ---- merge stage ----
    print("Merging intermediates...", flush=True)
    t0 = time.time()
    q = pl.scan_parquet(os.path.join(TMP_DIR, "agg-*.parquet")).select(
        ["minute", "dm", "call_count"]
    )
    # minutes are disjoint across source files, but re-aggregate to be safe
    merged = q.group_by(["minute", "dm"]).agg(pl.sum("call_count").alias("call_count"))
    merged = merged.sort(["minute", "dm"]).with_columns(
        (pl.col("minute") * 60000).alias("minute_start_ms")
    )
    # write sharded final output (~one file per ~1000 minutes)
    df_all = merged.collect()
    print(f"Merged rows: {len(df_all)} in {time.time()-t0:.1f}s", flush=True)
    n_min = df_all["minute"].n_unique()
    print(f"Minutes covered: {n_min} [{df_all['minute'].min()}..{df_all['minute'].max()}]", flush=True)
    print(f"Distinct receiver MS (dm): {df_all['dm'].n_unique()}", flush=True)

    # shard by minute to keep files ~manageable
    shard_size = 1000  # minutes per shard
    mins = sorted(df_all["minute"].unique().to_list())
    for s, k in enumerate(range(0, len(mins), shard_size)):
        chunk_mins = set(mins[k : k + shard_size])
        part = df_all.filter(pl.col("minute").is_in(chunk_mins)).sort(["minute", "dm"])
        part.write_parquet(
            os.path.join(FINAL_DIR, f"call_counts_min{k//shard_size:04d}.parquet"),
            compression="zstd",
        )
        print(f"  wrote shard {s} rows={len(part)}", flush=True)

    # per-MS grand totals
    totals = df_all.group_by("dm").agg(pl.sum("call_count").alias("total_calls")).sort(
        "total_calls", descending=True
    )
    totals.write_parquet(os.path.join(OUT_DIR, "totals_per_ms.parquet"), compression="zstd")
    totals.write_csv(os.path.join(OUT_DIR, "totals_per_ms.csv"))

    meta = {
        "src_dir": SRC_DIR,
        "n_source_files": len(files),
        "cleaning": "drop rpctype~rpc_id corruption; dedup (traceid,rpc_id,um,dm); sum all rpctypes incl. USER/MQ/UNKNOWN",
        "key": "minute = floor(timestamp_ms/60000), dm = receiver MS",
        "total_rows": int(len(df_all)),
        "n_minutes": int(n_min),
        "minute_range": [int(df_all["minute"].min()), int(df_all["minute"].max())],
        "n_distinct_dm": int(df_all["dm"].n_unique()),
        "grand_total_calls": int(df_all["call_count"].sum()),
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"ALL DONE in {(time.time()-t_all)/60:.1f} min -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
