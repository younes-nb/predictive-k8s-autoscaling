"""Select high-variance-CPM microservices for Madu training.

Paper motivation (§2.2, Fig.3-4): workload uncertainty (variance relative
to the mean) is the key difficulty. We rank by coefficient of variation
(CV = std/mean) of the per-minute CPM series, restricted to MSs that also
have CPU/memory metrics in msresource (needed for §4.2 profiling).
"""
import argparse
import glob
import json
import os
import sys

import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def msresource_ms_set(msresource_dir, stride=24):
    """Union of msnames over every `stride`-th part (parts are time-ordered,
    the MS set is stable across time, so striding is sufficient)."""
    files = sorted(glob.glob(os.path.join(msresource_dir, "part-*.parquet")))
    names = set()
    for f in files[::stride]:
        part = pl.read_parquet(f, columns=["msname"])["msname"].unique().to_list()
        names.update(part)
    return names


def cpm_stats(cpm_dir, candidates):
    """Per-MS mean/std of CPM over the full trace (u64-safe)."""
    q = (
        pl.scan_parquet(os.path.join(cpm_dir, "*.parquet"))
        .filter(pl.col("dm").is_in(list(candidates)))
        .with_columns(pl.col("call_count").cast(pl.UInt64))
        .group_by("dm")
        .agg([
            pl.mean("call_count").alias("mean"),
            pl.std("call_count").alias("std"),
            pl.len().alias("n_minutes"),
        ])
        .collect()
    )
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpm_dir", default="/dataset/ms_call_counts/per_minute")
    p.add_argument("--msresource_dir", default="/dataset/parquet/msresource")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--min_minutes", type=int, default=10000,
                   help="Require at least this many minutes present.")
    p.add_argument("--out", default="/proj/k8sautoscaledl-PG0/madu/ms_list.json")
    args = p.parse_args()

    print("Collecting msresource MS set...", flush=True)
    have_res = msresource_ms_set(args.msresource_dir)
    print(f"msresource MSs: {len(have_res)}", flush=True)
    print("Computing CPM stats...", flush=True)
    stats = cpm_stats(args.cpm_dir, have_res)
    stats = stats.filter(pl.col("n_minutes") >= args.min_minutes)
    stats = stats.with_columns(
        (pl.col("std") / pl.max_horizontal(pl.col("mean"), pl.lit(1.0))).alias("cv")
    ).sort("cv", descending=True)
    top = stats.head(args.top_k)
    print(top.select(["dm", "mean", "std", "cv", "n_minutes"]).head(15))
    msnames = [d for d in top["dm"].to_list() if d != "UNKNOWN"][:args.top_k]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"msnames": msnames,
                   "criterion": "cv=std/mean of per-minute CPM"}, f, indent=2)
    print(f"Wrote {len(msnames)} MSs -> {args.out}")


if __name__ == "__main__":
    main()
