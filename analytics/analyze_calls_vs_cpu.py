#!/usr/bin/env python
"""Per-MS CPU vs received-call-count relationship (Alibaba cluster-trace-v2022, 13 days).

Pipeline
--------
S1 map   (ProcessPool, env-tuned): /dataset/parquet/msresource part files
         -> per (minute, msname): CPU = mean of cpu_utilization across all instances
         sampled that minute (0-1 fraction, max-min normalized per the official
         release), n_inst = number of instances sampled. minute = timestamp/60000
         (msresource timestamps are exact-minute ms, same minute base as call counts).
S2 assemble (ProcessPool): tmp map shards -> sharded cpu_ms_per_minute/cpu-range-*
         (instance-count-weighted re-aggregation over the bucket's minute range).
S3 universe+join (ProcessPool): from /dataset/ms_call_counts/per_minute (built once
         by tooling/build_ms_call_counts.py; not rebuilt here)
         - per-MS call totals (dropping dm == UNKNOWN: unofficial release artifact,
           ~4.9% of raw calls, has no MSResource counterpart possible)
         - per-MS CPU coverage from assembled S2 shards
         - inner join (minute, msname) -> joined_per_minute/join-range-*.parquet.
S4 metrics (main proc, numpy/scipy): for EVERY MS present in both tables:
         n minute-samples, coverage, moments + p95/p99 of per-minute call counts,
         moments of CPU, Pearson r, Spearman rho, best |lag-corr| over lags
         -lag_max..+lag_max minutes (k>0 means CPU follows calls by k minutes).
S5 report + plots: services in the highest call-sample coverage tier with the
         largest std of their per-minute call count -> twin-axis timelines
         (calls left, CPU% right) + calls-vs-CPU hexbins; metrics CSV, markdown
         summary, metadata.json.

Call-count semantics (inherited cleaning from tooling/build_ms_call_counts.py):
minute = floor(call timestamp_ms/60000), receiver MS = dm, all rpctypes summed
(user+http+rpc+mq+db+mc...), dedup on (traceid, rpc_id, um, dm).

Smoke mode (--smoke): verified real-data window minutes [9000, 11000) from
msresource part files [300, 368) -> /tmp/opencode/calls_vs_cpu_smoke (~2-4 min).

Full run:  micromamba run -n venv python analytics/analyze_calls_vs_cpu.py
Resume-safe (reuses existing tmp shards / stage outputs); --force clears them.
"""
from __future__ import annotations

import os

# ---- cap thread pools BEFORE importing polars (each pool worker re-reads these) ----
os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import glob
import gc
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.defaults import PATHS  # noqa: E402  (env-overridable dataset paths)

import polars as pl  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

CC_BUCKET = 1000  # /dataset/ms_call_counts/per_minute shards are 1000-min apart

S1_DIR = "_tmp_mshards"
S2_DIR = "cpu_ms_per_minute"
UNI_DIR = "universe"
JOIN_DIR = "joined_per_minute"

FINAL_FILES = [
    "summary.md", "metadata.json", "metrics_all_ms.csv", "top5_series.parquet",
    "universe_calls_per_ms.parquet", "universe_cpu_per_ms.parquet",
    "code_table.parquet", "stage1_complete.json",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_dur(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m{s % 60:02d}s"


def atoi_path(p: str) -> int:
    base = os.path.basename(p)
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else 0


def atomic_write_parquet(df: pl.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, path)


def make_pool(workers: int, threads: int) -> ProcessPoolExecutor:
    """Spawn pool; children re-import this module and size polars from os.environ."""
    t = max(1, int(threads))
    os.environ["POLARS_MAX_THREADS"] = str(t)
    os.environ["RAYON_NUM_THREADS"] = str(t)
    os.environ["OMP_NUM_THREADS"] = str(t)
    return ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))


# --------------------------------------------------------------------------
# Stage 1: map one msresource parquet -> per (minute, msname) mean over instances
# --------------------------------------------------------------------------
def s1_map_msresource_file(task) -> dict:
    (path, idx, tmp_dir) = task
    shard = os.path.join(tmp_dir, f"mshard-{idx:05d}.parquet")
    sidecar = shard + ".json"
    if os.path.isfile(shard) and os.path.isfile(sidecar):
        with open(sidecar) as f:
            return {"idx": idx, "skipped": True, **json.load(f)}
    try:
        df = (
            pl.scan_parquet(path)
            .select(["timestamp", "msname", "cpu_utilization"])
            .filter(pl.col("msname").is_not_null(), pl.col("cpu_utilization").is_not_null())
            .with_columns((pl.col("timestamp") // 60000).cast(pl.Int32).alias("minute"))
            .collect(engine="streaming")
        )
        agg = (
            df.group_by(["minute", "msname"])
            .agg(
                pl.col("cpu_utilization").mean().cast(pl.Float32).alias("cpu_mean"),
                pl.len().cast(pl.UInt32).alias("n_inst"),
            )
            .sort(["minute", "msname"])
        )
        atomic_write_parquet(agg, shard)
        meta = {
            "rows_src": int(len(df)),
            "rows": int(len(agg)),
            "min_minute": int(agg["minute"].min()),
            "max_minute": int(agg["minute"].max()),
            "n_ms": int(agg["msname"].n_unique()),
        }
        with open(sidecar + ".tmp", "w") as f:
            json.dump(meta, f)
        os.replace(sidecar + ".tmp", sidecar)
        del df, agg
        return {"idx": idx, "skipped": False, **meta}
    except Exception:
        return {"idx": idx, "error": traceback.format_exc()}


# --------------------------------------------------------------------------
# Stage 2: assemble tmp map shards -> one sharded per-(minute,msname) CPU table
# --------------------------------------------------------------------------
def s2_assemble_cpu_bucket(task) -> dict:
    (bucket_id, lo, hi, tmp_paths, s2_dir) = task
    out = os.path.join(s2_dir, f"cpu-range-{bucket_id:04d}.parquet")
    if os.path.isfile(out):
        return {"bucket": bucket_id, "skipped": True}
    try:
        flt = (pl.col("minute") >= lo) & (pl.col("minute") < hi)
        q = None
        for p in tmp_paths:
            lf = pl.scan_parquet(p).filter(flt)
            q = lf if q is None else pl.concat([q, lf], how="vertical")
        agg = (
            q.group_by(["minute", "msname"])
            .agg(
                (pl.col("cpu_mean") * pl.col("n_inst").cast(pl.Float64)).sum().alias("sc"),
                pl.col("n_inst").sum().cast(pl.UInt32).alias("n_inst"),
            )
            .with_columns(
                (pl.col("sc") / pl.col("n_inst").cast(pl.Float64)).cast(pl.Float32).alias("cpu_mean")
            )
            .select(
                pl.col("minute").cast(pl.Int32),
                pl.col("msname"),
                pl.col("cpu_mean"),
                pl.col("n_inst"),
            )
            .sort(["minute", "msname"])
            .collect(engine="streaming")
        )
        atomic_write_parquet(agg, out)
        n_inst_mean = float(np.mean(agg["n_inst"].to_numpy())) if len(agg) else 0.0
        return {
            "bucket": bucket_id,
            "rows": int(len(agg)),
            "n_ms": int(agg["msname"].n_unique()),
            "mean_inst_per_ms_minute": n_inst_mean,
        }
    except Exception:
        return {"bucket": bucket_id, "error": traceback.format_exc()}


# --------------------------------------------------------------------------
# Stage 3a: per-MS call totals over one minute bucket (drops dm == UNKNOWN)
# --------------------------------------------------------------------------
def s3_universe_calls(task) -> dict:
    (bucket_id, cc_paths, uni_dir, lo, hi) = task
    out = os.path.join(uni_dir, f"calls-part-{bucket_id:04d}.parquet")
    unk_marker = os.path.join(uni_dir, f"unknown-part-{bucket_id:04d}.json")
    if os.path.isfile(out) and os.path.isfile(unk_marker):
        return {"bucket": bucket_id, "skipped": True}
    try:
        flt = (pl.col("minute") >= lo) & (pl.col("minute") < hi)
        base = pl.concat([pl.scan_parquet(p) for p in cc_paths], how="vertical").filter(flt)
        cc = base.group_by(["minute", "dm"]).agg(pl.col("call_count").cast(pl.UInt64).sum())
        known = (
            cc.filter(pl.col("dm") != "UNKNOWN")
            .group_by("dm")
            .agg(
                pl.col("call_count").sum().alias("calls_total"),
                pl.len().cast(pl.UInt64).alias("n_minute"),
            )
            .sort("calls_total", descending=True)
            .collect(engine="streaming")
            .rename({"dm": "msname"})
        )
        atomic_write_parquet(known, out)
        unknown = cc.filter(pl.col("dm") == "UNKNOWN").select(
            pl.col("call_count").sum().alias("total"),
            pl.len().cast(pl.UInt64).alias("rows"),
        ).collect(engine="streaming")
        u_total = int(unknown["total"][0] or 0)
        u_rows = int(unknown["rows"][0] or 0)
        with open(unk_marker + ".tmp", "w") as f:
            json.dump({"unknown_calls": u_total, "unknown_rows": u_rows}, f)
        os.replace(unk_marker + ".tmp", unk_marker)
        return {
            "bucket": bucket_id,
            "n_ms": int(len(known)),
            "unknown_calls": u_total,
            "unknown_rows": u_rows,
        }
    except Exception:
        return {"bucket": bucket_id, "error": traceback.format_exc()}


# --------------------------------------------------------------------------
# Stage 3b: per-MS CPU coverage from one assembled cpu shard
# --------------------------------------------------------------------------
def s3_universe_cpu(task) -> dict:
    (cpu_path, uni_dir) = task
    bucket_id = atoi_path(cpu_path)
    out = os.path.join(uni_dir, f"cpu-part-{bucket_id:04d}.parquet")
    if os.path.isfile(out):
        return {"bucket": bucket_id, "skipped": True}
    try:
        q = (
            pl.scan_parquet(cpu_path)
            .group_by("msname")
            .agg(pl.len().cast(pl.UInt32).alias("n_minute"))
            .sort("msname")
            .collect(engine="streaming")
        )
        atomic_write_parquet(q, out)
        return {"bucket": bucket_id, "n_ms": int(len(q))}
    except Exception:
        return {"bucket": bucket_id, "error": traceback.format_exc()}


# --------------------------------------------------------------------------
# Stage 3c: inner-join call counts with per-minute MS CPU for one minute bucket
# --------------------------------------------------------------------------
def s3_join_bucket(task) -> dict:
    (bucket_id, lo, hi, cc_paths, cpu_path, join_dir) = task
    out = os.path.join(join_dir, f"join-range-{bucket_id:04d}.parquet")
    if os.path.isfile(out):
        return {"bucket": bucket_id, "skipped": True}
    try:
        flt = (pl.col("minute") >= lo) & (pl.col("minute") < hi)
        cc_lf = (
            pl.concat([pl.scan_parquet(p) for p in cc_paths], how="vertical")
            .filter(flt)
            .filter(pl.col("dm") != "UNKNOWN")
            .group_by(["minute", "dm"])
            .agg(pl.col("call_count").cast(pl.UInt64).sum())
            .with_columns(
                pl.col("minute").cast(pl.Int32),
                pl.col("dm").alias("msname"),
            )
            .group_by(["minute", "msname"])
            .agg(pl.col("call_count").sum())
        )
        cpu_lf = pl.scan_parquet(cpu_path).filter(flt)
        joined = (
            cc_lf.join(
                cpu_lf.select(["minute", "msname", "cpu_mean"]),
                on=["minute", "msname"],
                how="inner",
            )
            .select(
                pl.col("minute").cast(pl.Int32),
                pl.col("msname"),
                pl.col("call_count").cast(pl.UInt32),
                pl.col("cpu_mean").cast(pl.Float32),
            )
            .sort(["minute", "msname"])
            .collect(engine="streaming")
        )
        atomic_write_parquet(joined, out)
        cc_stats = (
            cc_lf.select(pl.len().alias("n"), pl.col("call_count").sum().alias("s"))
            .collect(engine="streaming")
        )
        return {
            "bucket": bucket_id,
            "cc_rows": int(cc_stats["n"][0]),
            "cc_calls": int(cc_stats["s"][0] or 0),
            "joined_rows": int(len(joined)),
            "joined_calls": int(joined["call_count"].sum()),
            "n_ms": int(joined["msname"].n_unique()),
        }
    except Exception:
        return {"bucket": bucket_id, "error": traceback.format_exc()}


# --------------------------------------------------------------------------
# pool runner
# --------------------------------------------------------------------------
def pool_entry(payload: tuple):
    kind, task = payload
    if kind == "s1":
        return s1_map_msresource_file(task)
    if kind == "s2":
        return s2_assemble_cpu_bucket(task)
    if kind == "uni_calls":
        return s3_universe_calls(task)
    if kind == "uni_cpu":
        return s3_universe_cpu(task)
    if kind == "join":
        return s3_join_bucket(task)
    raise ValueError(f"unknown task kind {kind}")


def run_pool(tasks, workers: int, threads: int, kind: str) -> list:
    results = []
    if not tasks:
        return results
    t0 = time.time()
    log(f"[{kind}] {len(tasks)} tasks | {workers} workers x {threads} polars-threads")
    with make_pool(workers, threads) as ex:
        for i, res in enumerate(ex.map(pool_entry, tasks, chunksize=1)):
            results.append(res)
            if "error" in res:
                log(f"  [FAIL] {res['error']}")
                raise SystemExit(f"task failed in stage {kind}")
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks) - 1:
                log(f"  [{i + 1}/{len(tasks)}] {res}")
    log(f"[{kind}] done in {fmt_dur(time.time() - t0)}")
    return results


# --------------------------------------------------------------------------
# Stage 4: per-MS metrics from joined shards (numpy / scipy)
# --------------------------------------------------------------------------
def stage4_metrics(joined_paths, code_tbl: pl.DataFrame, span_lo: int, span_hi: int,
                   lag_max: int, lag_min_samples: int, metrics_csv: str):
    log(f"[S4] reading {len(joined_paths)} joined shards...")
    t0 = time.time()
    ms_list, minute_list, calls_list, cpu_list = [], [], [], []
    for p in joined_paths:
        df = (
            pl.read_parquet(p)
            .join(code_tbl, on="msname", how="inner")
            .select(
                pl.col("code").cast(pl.Int32),
                pl.col("minute").cast(pl.Int32),
                pl.col("call_count").cast(pl.UInt32),
                pl.col("cpu_mean").cast(pl.Float32),
            )
            .sort(["code", "minute"])
        )
        ms_list.append(df["code"].to_numpy())
        minute_list.append(df["minute"].to_numpy())
        calls_list.append(df["call_count"].to_numpy())
        cpu_list.append(df["cpu_mean"].to_numpy())
        del df
    ms = np.concatenate(ms_list)
    minute = np.concatenate(minute_list)
    calls = np.concatenate(calls_list)
    cpu = np.concatenate(cpu_list)
    del ms_list, minute_list, calls_list, cpu_list
    gc.collect()
    log(f"[S4] {len(ms):,} joined minute-samples in {fmt_dur(time.time() - t0)}")
    minute = (minute - span_lo).astype(np.int32)

    order = np.lexsort((minute, ms))
    ms, minute, calls, cpu = ms[order], minute[order], calls[order], cpu[order]
    uniq = np.unique(ms)
    starts = np.searchsorted(ms, uniq, side="left")
    ends = np.searchsorted(ms, uniq, side="right")
    del order
    gc.collect()

    n_groups = len(starts)
    names = code_tbl["msname"].to_list()
    span = int(span_hi - span_lo)
    rows: list[dict] = []
    n_lag_tested = 0
    n_lag_nonzero = 0
    log(f"[S4] metrics for {n_groups:,} MSs over span {span:,} min (lags +-{lag_max})")
    t1 = time.time()
    for gi in range(n_groups):
        b, e = starts[gi], ends[gi]
        x = calls[b:e].astype(np.float64)
        y = cpu[b:e].astype(np.float64)
        n = e - b
        rec = {
            "msname": names[int(ms[b])],
            "n_minute_samples": n,
            "coverage_frac": n / span,
            "call_count_mean": float(x.mean()),
            "call_count_median": float(np.median(x)),
            "call_count_std": float(x.std()),
            "call_count_min": float(x.min()),
            "call_count_p95": float(np.percentile(x, 95)),
            "call_count_p99": float(np.percentile(x, 99)),
            "call_count_max": float(x.max()),
            "cpu_mean": float(y.mean()),
            "cpu_median": float(np.median(y)),
            "cpu_std": float(y.std()),
            "cpu_min": float(y.min()),
            "cpu_p95": float(np.percentile(y, 95)),
            "cpu_max": float(y.max()),
            "pearson_r": None,
            "pearson_r2": None,
            "spearman_rho": None,
            "best_lag_abs_corr": None,
            "best_lag_min": None,
        }
        sx, sy = x.std(), y.std()
        if sx > 0 and sy > 0:
            r = float(np.corrcoef(x, y)[0, 1])
            rec["pearson_r"], rec["pearson_r2"] = r, r * r
            rank_x, rank_y = rankdata(x), rankdata(y)
            if rank_x.std() > 0 and rank_y.std() > 0:
                rec["spearman_rho"] = float(np.corrcoef(rank_x, rank_y)[0, 1])

        if n >= lag_min_samples:
            m0, m1 = int(minute[b]), int(minute[e - 1])
            T = m1 - m0 + 1
            if T <= 18_720 + 4 * lag_max:
                loc = minute[b:e] - m0
                x_dense = np.zeros(T, dtype=np.float64)
                y_dense = np.zeros(T, dtype=np.float64)
                px = np.zeros(T, dtype=bool)
                py = np.zeros(T, dtype=bool)
                x_dense[loc] = x
                y_dense[loc] = y
                px[loc] = True
                py[loc] = True
                best_abs, best_k, best_r = 0.0, 0, None
                for k in range(-lag_max, lag_max + 1):
                    s = abs(k)
                    if s >= T:
                        continue
                    if k >= 0:  # pairs (x_t, y_{t+k}), k > 0 = CPU follows calls
                        a = x_dense[: T - s]
                        bb = y_dense[s:]
                        mask = px[: T - s] & py[s:]
                    else:  # pairs (x_t, y_{t-s}), CPU leads calls by s minutes
                        a = x_dense[s:]
                        bb = y_dense[: T - s]
                        mask = px[s:] & py[: T - s]
                    if int(mask.sum()) < 50:
                        continue
                    av, bv = a[mask], bb[mask]
                    if av.std() > 0 and bv.std() > 0:
                        r_k = float(np.corrcoef(av, bv)[0, 1])
                        if abs(r_k) >= best_abs:
                            best_abs, best_k, best_r = abs(r_k), k, r_k
                n_lag_tested += 1
                if best_r is not None:
                    rec["best_lag_abs_corr"] = best_r
                    rec["best_lag_min"] = best_k
                    if best_k != 0:
                        n_lag_nonzero += 1
        rows.append(rec)

    out = pl.DataFrame(rows).sort("msname")
    out.write_csv(metrics_csv)
    log(f"[S4] done in {fmt_dur(time.time() - t1)} -> {metrics_csv}")
    return out, n_lag_tested, n_lag_nonzero


# --------------------------------------------------------------------------
# Stage 5: top-5 selection, series extraction, plots, summary
# --------------------------------------------------------------------------
def select_top5(mt: pl.DataFrame, top_n: int) -> pl.DataFrame:
    if mt.is_empty():
        raise SystemExit("no joined services: no MS has both CPU and call data in window")
    max_n = mt["n_minute_samples"].max()
    tier = mt.filter(pl.col("n_minute_samples") >= 0.9 * max_n)
    log(f"[S5] coverage tier n >= {0.9 * max_n:.1f} -> {len(tier)} of {len(mt)} MSs")
    return tier.sort(
        ["call_count_std", "n_minute_samples", "msname"], descending=[True, True, False]
    ).head(top_n)


def load_top5_series(joined_paths, names: list[str]) -> pl.DataFrame:
    s = None
    for p in joined_paths:
        f = pl.scan_parquet(p).filter(pl.col("msname").is_in(names)).sort("minute")
        s = f if s is None else pl.concat([s, f], how="vertical")
    return s.sort(["msname", "minute"]).collect(engine="streaming")


def _fmt_opt(v, spec: str = ".3f") -> str:
    if v is None:
        return "n/a"
    return f"{v:+d}" if spec == "+d" else format(float(v), spec)


def plot_top5(top5: pl.DataFrame, top5_series: pl.DataFrame, out_dir: str) -> None:
    names = top5["msname"].to_list()
    for name in names:
        s = top5_series.filter(pl.col("msname") == name).sort("minute")
        calls = s["call_count"].to_numpy().astype(np.float64)
        cpu = s["cpu_mean"].to_numpy().astype(np.float64)
        x = s["minute"].to_numpy() / 1440.0
        fig, ax = plt.subplots(figsize=(12, 6))
        ax2 = ax.twinx()
        ax.plot(x, np.maximum(calls, 0.5), color="steelblue", lw=0.7)
        ax.set_yscale("log")
        ax.set_ylabel("calls/min", color="steelblue")
        ax2.plot(x, cpu, color="darkorange", lw=0.7)
        ax2.set_ylabel("MS CPU (normalized 0–1)", color="darkorange")
        ax2.set_ylim(0, 1.0)
        ax.grid(alpha=0.25)
        ax.set_xlabel("trace day (minute/1440)")
        ax.set_title(name, fontsize=12)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{name}_timeseries.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


def plot_top5_curve(top5: pl.DataFrame, top5_series: pl.DataFrame, out_dir: str) -> None:
    """Calls (x, log) vs CPU (y, 0-1): the binned median CPU/count curve per MS."""
    names = top5["msname"].to_list()
    for name in names:
        s = top5_series.filter(pl.col("msname") == name)
        calls = s["call_count"].to_numpy().astype(np.float64)
        cpu = s["cpu_mean"].to_numpy().astype(np.float64)
        x = 1.0 + calls
        mask = (calls > 0) & np.isfinite(cpu)
        x, cpu = x[mask], cpu[mask]
        fig, ax = plt.subplots(figsize=(10, 6))
        if len(x) > 0:
            lo, hi = float(np.min(x)), float(np.max(x))
            edges = np.logspace(np.log10(lo), np.log10(hi * 1.0001), 41)
            which = np.digitize(x, edges) - 1
            bx, med = [], []
            for b in range(40):
                m = which == b
                if m.sum() < 5:
                    continue
                bx.append(float(np.median(x[m])))
                med.append(float(np.median(cpu[m])))
            if bx:
                bx, med = np.array(bx), np.array(med)
                ax.plot(bx, med, color="darkorange", lw=2.2)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("MS CPU (normalized 0–1)")
        ax.set_xlabel("calls/min (log scale; x plotted as 1+calls)")
        ax.grid(alpha=0.25)
        ax.set_title(name, fontsize=12)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{name}_curve_calls_cpu.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


def build_summary(args, span, timings, uni: dict, mt: pl.DataFrame, top5: pl.DataFrame,
                  n_lag_tested: int, n_lag_nonzero: int, out_dir: str) -> str:
    def f3(v):
        return "n/a" if v is None else f"{v:.3f}"

    md = ["# CPU vs call counts — per-MS analysis", ""]
    md.append(f"*Run: {datetime.now().isoformat(timespec='seconds')} | script: `analytics/analyze_calls_vs_cpu.py`*")
    md.append("")
    md.append(f"**Analysis window:** minutes {span[0]:,}..{span[1]:,} ({(span[1] - span[0]) / 1440:.2f} days)")
    md.append("")
    md.append(f"- call-count source: `{args.call_counts_dir}` — per-minute received calls per MS (dm), cleaned & deduped by `tooling/build_ms_call_counts.py`")
    md.append(f"- CPU source: `{args.ms_resource_dir}` — mean of `cpu_utilization` across instances per (minute, msname); **max-min normalized (0–1, not cores)**, reported in %")
    md.append("")
    md.append("## Universe accounting")
    md.append("")
    md.append("| quantity | value |")
    md.append("|---|---:|")
    md.append(f"| services with call-count data (UNKNOWN dropped) | {uni['n_ms_calls']:,} |")
    md.append(f"| services with CPU series | {uni['n_ms_cpu']:,} |")
    md.append(f"| services analyzed (both) | {uni['n_ms_both']:,} |")
    md.append(f"| services with calls but no CPU | {uni['n_ms_calls_only']:,} |")
    md.append(f"| services with CPU but no calls | {uni['n_ms_cpu_only']:,} |")
    md.append(f"| joined (minute, MS) samples | {uni['joined_rows']:,} |")
    md.append(f"| total call volume (non-UNKNOWN receivers) | {uni['total_calls_nonunknown']:,} |")
    md.append(f"| total call-volume analyzed (joined) | {uni['joined_calls']:,} |")
    md.append(f"| call volume in dm=UNKNOWN (dropped) | {uni['unknown_calls']:,} |")
    md.append(f"| UNKNOWN share of dropped+analyzed volume | {100.0 * uni['unknown_calls'] / max(1, uni['joined_calls'] + uni['unknown_calls']):.1f}% |")
    md.append("")
    md.append("*(most call-only services are stateful DB/MC endpoints from other dedicated clusters, plus column-shift artifacts — they have no MSResource CPU row by construction; note the released call graph also spans auxiliary clusters whose containers were not released in MSResource, so several of the highest-volume receivers carry no CPU row)*")
    if uni.get("top_unmatched"):
        md.append("")
        md.append("### Top receivers by call volume WITHOUT any CPU row (excluded from the analysis)")
        md.append("")
        md.append("| rank | msname | total calls | minutes seen |")
        md.append("|---:|---|---:|---:|")
        for k, r in enumerate(uni["top_unmatched"]):
            md.append(f"| {k + 1} | {r['msname']} | {r['calls_total']:,} | {r['n_minute']:,} |")
    md.append("")
    md.append("## Per-MS metric distributions (all analyzed services)")
    md.append("")
    md.append("| metric | median | p10 | p90 | min | max |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for col in ["pearson_r", "spearman_rho", "call_count_std", "coverage_frac", "best_lag_abs_corr"]:
        vals = mt[col].drop_nulls()
        if len(vals) == 0:
            continue
        v = vals.to_numpy().astype(np.float64)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        md.append(
            f"| {col} | {f3(float(np.median(v)))} | {f3(float(np.percentile(v, 10)))} | "
            f"{f3(float(np.percentile(v, 90)))} | {f3(float(v.min()))} | {f3(float(v.max()))} |"
        )
    r_vec = mt["pearson_r"].drop_nulls().to_numpy().astype(np.float64)
    r_vec = r_vec[np.isfinite(r_vec)]
    if len(r_vec):
        md.append("")
        md.append(
            f"- |pearson_r| > 0.3 for {100 * float((np.abs(r_vec) > 0.3).mean()):.1f}% of analyzed services; "
            f"> 0.6 for {100 * float((np.abs(r_vec) > 0.6).mean()):.1f}%"
        )
    if n_lag_tested:
        md.append(
            f"- lag search ran for {n_lag_tested:,} services (n >= {args.lag_min_samples}); "
            f"best lag != 0 for {100.0 * n_lag_nonzero / n_lag_tested:.1f}% "
            f"(positive lag = CPU follows calls by that many minutes)"
        )
    md.append("")
    md.append("## Top-5 (coverage tier: n >= 90% of max n, then highest call_count_std)")
    md.append("")
    md.append("| rank | msname | n | call std | call mean | call p95 | cpu mean | cpu std | pearson r | spearman rho | best lag | abs corr@lag |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, row in enumerate(top5.iter_rows(named=True)):
        md.append(
            f"| {i + 1} | {row['msname']} | {row['n_minute_samples']} | {row['call_count_std']:.1f} | "
            f"{row['call_count_mean']:.1f} | {row['call_count_p95']:.1f} | {100 * row['cpu_mean']:.1f}% | "
            f"{100 * row['cpu_std']:.1f}% | {f3(row['pearson_r'])} | {f3(row['spearman_rho'])} | "
            f"{_fmt_opt(row['best_lag_min'], '+d')} | {f3(row['best_lag_abs_corr'])} |"
        )
    md.append("")
    md.append("## Timing")
    md.append("")
    for k, v in timings.items():
        md.append(f"- {k}: {fmt_dur(v)}")
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    return path


# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CPU vs call-count relation per microservice (13-day Alibaba v2022 trace).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cpu_n = os.cpu_count() or 32
    p.add_argument("--ms-resource-dir", default=os.environ.get("PARQUET_MSRESOURCE", PATHS.PARQUET_MSRESOURCE),
                   help="msresource part-*.parquet dir")
    p.add_argument("--call-counts-dir", default=os.environ.get("CALL_COUNTS_DIR",
                   os.path.join(os.environ.get("DATASET_ROOT", "/dataset"), "ms_call_counts", "per_minute")),
                   help="per-minute per-MS call-count shard dir")
    p.add_argument("--out-dir", default=os.environ.get("ANALYTICS_OUT_DIR", PATHS.ANALYTICS_OUT_DIR) + "/calls_vs_cpu",
                   help="output dir")
    p.add_argument("--workers", type=int, default=max(4, min(cpu_n // 2, 24)),
                   help="process-pool workers (default ~ nproc/2)")
    p.add_argument("--threads-per-worker", type=int, default=None,
                   help="polars threads per worker (default: nproc/workers)")
    p.add_argument("--minutes", type=str, default=None,
                   help="analysis minute window lo:hi, e.g. 9000:11000 (default: full span)")
    p.add_argument("--msr-file-range", type=str, default=None,
                   help="limit msresource part files to index range i:j (smoke)")
    p.add_argument("--lag-max", type=int, default=15, help="lag search range +/- minutes")
    p.add_argument("--lag-min-samples", type=int, default=1000, help="min n for the lag search")
    p.add_argument("--top-n", type=int, default=5, help="number of services to plot")
    p.add_argument("--force", action="store_true", help="wipe stage outputs before running")
    p.add_argument("--smoke", action="store_true", help="small verified window + /tmp outputs")
    args = p.parse_args(argv)

    if args.smoke:
        if args.minutes is None:
            args.minutes = "9000:11000"
        if args.msr_file_range is None:
            args.msr_file_range = "300:368"
        args.workers = min(args.workers, 12)
        args.out_dir = "/tmp/opencode/calls_vs_cpu_smoke"
        args.lag_min_samples = min(args.lag_min_samples, 200)

    args.threads_per_worker = args.threads_per_worker or max(1, cpu_n // max(1, args.workers))

    if args.minutes:
        lo, hi = (int(x) for x in args.minutes.split(":"))
        args.minutes = (lo, hi)
    return args


def parse_bucket_plan(args, cc_paths):
    """Buckets of <= CC_BUCKET minutes, aligned at the window start."""
    if args.minutes is None:
        span_lo = 0
        probe = pl.scan_parquet(cc_paths[-1]).select(pl.col("minute").max()).collect(engine="streaming")
        span_hi = int(probe["minute"][0]) + 1
    else:
        span_lo, span_hi = args.minutes
    buckets = []
    k = span_lo
    while k < span_hi:
        buckets.append((k, min(k + CC_BUCKET, span_hi)))
        k += CC_BUCKET
    return buckets, (span_lo, span_hi)


def jsonable(o):
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return str(o)


def main():
    args = parse_args()
    t_start = time.time()

    out_dir = args.out_dir
    tmp_dir = os.path.join(out_dir, S1_DIR)
    s2_dir = os.path.join(out_dir, S2_DIR)
    uni_dir = os.path.join(out_dir, UNI_DIR)
    join_dir = os.path.join(out_dir, JOIN_DIR)

    if args.force:
        for d in (tmp_dir, s2_dir, uni_dir, join_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)
        for f in FINAL_FILES:
            fp = os.path.join(out_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
    for d in (out_dir, tmp_dir, s2_dir, uni_dir, join_dir):
        os.makedirs(d, exist_ok=True)

    msr_files = sorted(glob.glob(os.path.join(args.ms_resource_dir, "part-*.parquet")))
    if args.msr_file_range:
        i, j = (int(x) for x in args.msr_file_range.split(":"))
        msr_files = msr_files[i:j]
    if not msr_files:
        raise SystemExit(f"no msresource parquet in {args.ms_resource_dir}")
    cc_paths = sorted(glob.glob(os.path.join(args.call_counts_dir, "*.parquet")))
    if not cc_paths:
        raise SystemExit(f"no call-count shards in {args.call_counts_dir}")
    log(f"msresource files: {len(msr_files)} | call-count shards: {len(cc_paths)} | out: {out_dir} "
        f"| workers={args.workers} x {args.threads_per_worker} threads")

    buckets, (span_lo, span_hi) = parse_bucket_plan(args, cc_paths)
    log(f"minute span [{span_lo:,}, {span_hi:,}) in {len(buckets)} buckets of <= {CC_BUCKET} min")
    timings: dict[str, float] = {}

    # ---- S1: map msresource files ----
    t1 = time.time()
    tasks = [("s1", (p, i, tmp_dir)) for i, p in enumerate(msr_files)]
    run_pool(tasks, args.workers, args.threads_per_worker, "S1-map")
    shards_glob = sorted(glob.glob(os.path.join(tmp_dir, "mshard-*.parquet")))
    if len(shards_glob) != len(msr_files):
        raise SystemExit(f"S1 incomplete: {len(shards_glob)}/{len(msr_files)} map shards exist")
    timings["S1_map"] = time.time() - t1
    with open(os.path.join(out_dir, "stage1_complete.json"), "w") as f:
        json.dump({"n_files": len(msr_files)}, f)

    # ---- S2: assemble per-bucket CPU tables ----
    t2 = time.time()
    shard_ranges = {}
    for sc in sorted(glob.glob(os.path.join(tmp_dir, "mshard-*.json"))):
        with open(sc) as f:
            m = json.load(f)
        shard_ranges[os.path.join(tmp_dir, f"mshard-{atoi_path(sc):05d}.parquet")] = (m["min_minute"], m["max_minute"])
    tasks = []
    for bucket_id, (lo, hi) in enumerate(buckets):
        mine = [p for p, (a, b) in shard_ranges.items() if a < hi and b >= lo]
        if not mine:
            raise SystemExit(f"no tmp shards overlap bucket {bucket_id} [{lo},{hi})")
        tasks.append(("s2", (bucket_id, lo, hi, mine, s2_dir)))
    run_pool(tasks, args.workers, args.threads_per_worker, "S2-assemble")
    timings["S2_assemble"] = time.time() - t2

    cpu_paths = sorted(glob.glob(os.path.join(s2_dir, "cpu-range-*.parquet")))
    if len(cpu_paths) != len(buckets):
        raise SystemExit(f"S2 incomplete: {len(cpu_paths)}/{len(buckets)} cpu shards exist")

    # ---- S3: universe + join ----
    t3 = time.time()
    tasks = []
    for bucket_id, (lo, hi) in enumerate(buckets):
        mine_cc = [
            p for p in cc_paths
            if atoi_path(p) * CC_BUCKET < hi and (atoi_path(p) + 1) * CC_BUCKET > lo
        ]
        if not mine_cc:
            raise SystemExit(f"no call-count shard overlaps bucket {bucket_id} [{lo},{hi})")
        tasks.append(("uni_calls", (bucket_id, mine_cc, uni_dir, lo, hi)))
        tasks.append(("join", (bucket_id, lo, hi, mine_cc, cpu_paths[bucket_id], join_dir)))
    for p in cpu_paths:
        tasks.append(("uni_cpu", (p, uni_dir)))
    s3_results = run_pool(tasks, args.workers, args.threads_per_worker, "S3-universe+join")
    timings["S3_universe_join"] = time.time() - t3

    join_paths = sorted(glob.glob(os.path.join(join_dir, "join-range-*.parquet")))
    if len(join_paths) != len(buckets):
        raise SystemExit(f"S3 incomplete: {len(join_paths)}/{len(buckets)} join shards exist")

    # ---- merge universe tables + code table ----
    t_u = time.time()
    calls_parts = sorted(glob.glob(os.path.join(uni_dir, "calls-part-*.parquet")))
    cpu_parts = sorted(glob.glob(os.path.join(uni_dir, "cpu-part-*.parquet")))
    calls_ms = (
        pl.concat([pl.read_parquet(p) for p in calls_parts], how="vertical")
        .group_by("msname")
        .agg(pl.col("calls_total").cast(pl.UInt64).sum(), pl.col("n_minute").cast(pl.UInt64).sum())
        .sort("msname")
    )
    cpu_ms = (
        pl.concat([pl.read_parquet(p) for p in cpu_parts], how="vertical")
        .group_by("msname")
        .agg(pl.col("n_minute").cast(pl.UInt64).sum())
        .sort("msname")
    )
    calls_ms.write_parquet(os.path.join(out_dir, "universe_calls_per_ms.parquet"), compression="zstd")
    cpu_ms.write_parquet(os.path.join(out_dir, "universe_cpu_per_ms.parquet"), compression="zstd")
    code_tbl = (
        pl.concat([calls_ms.select("msname"), cpu_ms.select("msname")], how="vertical")
        .unique()
        .sort("msname")
        .with_row_index("code")
        .with_columns(pl.col("code").cast(pl.Int32))
    )
    code_tbl.write_parquet(os.path.join(out_dir, "code_table.parquet"), compression="zstd")
    timings["S3_tables_merge"] = time.time() - t_u

    join_res = {r["bucket"]: r for r in s3_results if "joined_calls" in r}
    joined_calls = 0
    joined_rows = 0
    for bucket_id in range(len(buckets)):
        r = join_res.get(bucket_id)
        if r is not None:
            joined_calls += r["joined_calls"]
            joined_rows += r["joined_rows"]
            continue
        jf = join_paths[bucket_id]
        q = (
            pl.scan_parquet(jf)
            .select(
                pl.len().alias("n"),
                pl.col("call_count").cast(pl.UInt64).sum().alias("s"),
            )
            .collect(engine="streaming")
        )
        joined_rows += int(q["n"][0])
        joined_calls += int(q["s"][0] or 0)
    unknown_calls = 0
    for un in sorted(glob.glob(os.path.join(uni_dir, "unknown-part-*.json"))):
        with open(un) as f:
            unknown_calls += int(json.load(f)["unknown_calls"])

    n_ms_calls = int(calls_ms.shape[0])
    n_ms_cpu = int(cpu_ms.shape[0])
    n_ms_both = int(
        calls_ms.select("msname").join(cpu_ms.select("msname"), on="msname").shape[0]
    )
    cpu_names = cpu_ms["msname"].to_list()
    top_unmatched = (
        calls_ms.filter(~pl.col("msname").is_in(cpu_names))
        .sort("calls_total", descending=True)
        .head(10)
    )
    top_unmatched_rows = [dict(r) for r in top_unmatched.iter_rows(named=True)]

    # ---- S4: per-MS metrics ----
    t4 = time.time()
    mt, n_lag_tested, n_lag_nonzero = stage4_metrics(
        join_paths, code_tbl, span_lo, span_hi, args.lag_max, args.lag_min_samples,
        os.path.join(out_dir, "metrics_all_ms.csv"),
    )
    timings["S4_metrics"] = time.time() - t4

    # ---- S5: top-5, plots, summary, metadata ----
    t5 = time.time()
    top5 = select_top5(mt, args.top_n)
    top5_series = load_top5_series(join_paths, top5["msname"].to_list())
    top5_series.write_parquet(os.path.join(out_dir, "top5_series.parquet"), compression="zstd")
    plot_top5(top5, top5_series, out_dir)
    plot_top5_curve(top5, top5_series, out_dir)
    timings["S5_plots"] = time.time() - t5
    timings["total_wall"] = time.time() - t_start

    total_calls_nonunknown = int(calls_ms["calls_total"].sum())
    uni = {
        "n_ms_calls": n_ms_calls,
        "n_ms_cpu": n_ms_cpu,
        "n_ms_both": n_ms_both,
        "n_ms_calls_only": n_ms_calls - n_ms_both,
        "n_ms_cpu_only": n_ms_cpu - n_ms_both,
        "joined_rows": joined_rows,
        "joined_calls": joined_calls,
        "unknown_calls": unknown_calls,
        "total_calls_nonunknown": total_calls_nonunknown,
        "unmatched_calls": total_calls_nonunknown - joined_calls,
        "top_unmatched": top_unmatched_rows,
    }
    summary_path = build_summary(
        args, (span_lo, span_hi), timings, uni, mt, top5, n_lag_tested, n_lag_nonzero, out_dir
    )
    meta = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "args": jsonable({k: v for k, v in vars(args).items()}),
        "span_minutes": [span_lo, span_hi],
        "timings_sec": jsonable(timings),
        "universe": uni,
        "top5": [jsonable(r) for r in top5.iter_rows(named=True)],
        "libs": {"polars": pl.__version__, "numpy": np.__version__,
                 "python": sys.version.split()[0]},
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log(f"ALL DONE in {fmt_dur(time.time() - t_start)}")
    log(f"summary: {summary_path}")
    log(f"metrics: {os.path.join(out_dir, 'metrics_all_ms.csv')} ({len(mt)} services)")
    log(f"plots:   {out_dir}/*_timeseries.png and {out_dir}/*_curve_calls_cpu.png for top-5 MSs")
    with open(summary_path) as f:
        sys.stdout.write("\n" + f.read() + "\n")


if __name__ == "__main__":
    main()
