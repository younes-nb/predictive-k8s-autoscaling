"""Per-minute CPU/memory/container aggregation from msresource.

For the §4.2 performance-profiling regression we need, per (minute, MS):
mean CPU utilization, mean memory utilization and the container count
(distinct msinstanceid). msresource parts are time-ordered (~30 min each),
so this loops parts, filters to the wanted MSs and concatenates grouped
partials (peak memory stays small).
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms_list", required=True, help="JSON from madu/select_ms.py")
    p.add_argument("--msresource_dir", default="/dataset/parquet/msresource")
    p.add_argument("--out", default="/proj/k8sautoscaledl-PG0/madu/util_per_minute.parquet")
    args = p.parse_args()

    with open(args.ms_list) as f:
        msnames = json.load(f)["msnames"]
    print(f"Aggregating {len(msnames)} MSs", flush=True)
    files = sorted(glob.glob(os.path.join(args.msresource_dir, "part-*.parquet")))
    parts = []
    for i, fp in enumerate(files):
        df = (
            pl.scan_parquet(fp)
            .filter(pl.col("msname").is_in(msnames))
            .select(["timestamp", "msname", "msinstanceid",
                     "cpu_utilization", "memory_utilization"])
            .with_columns((pl.col("timestamp") // 60000).alias("minute"))
            .group_by(["minute", "msname"])
            .agg([
                pl.mean("cpu_utilization").alias("cpu"),
                pl.mean("memory_utilization").alias("mem"),
                pl.n_unique("msinstanceid").alias("n_inst"),
                pl.len().alias("n_rows"),
            ])
            .collect()
        )
        if len(df):
            parts.append(df)
        if i % 100 == 0:
            print(f"[{i}/{len(files)}] kept rows so far: "
                  f"{sum(len(d) for d in parts)}", flush=True)
    out = pl.concat(parts).sort(["msname", "minute"])
    # parts are time-disjoint; no re-aggregation needed, but be safe:
    out = out.group_by(["minute", "msname"]).agg([
        pl.mean("cpu").alias("cpu"),
        pl.mean("mem").alias("mem"),
        pl.max("n_inst").alias("n_inst"),
        pl.sum("n_rows").alias("n_rows"),
    ]).sort(["msname", "minute"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.write_parquet(args.out, compression="zstd")
    print(f"Wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
