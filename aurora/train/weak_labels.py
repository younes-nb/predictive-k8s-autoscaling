"""Telemetry-derived weak supervision for bottleneck labels.

No hand-labelled dataset is required. For each service, each raw feature is
converted to a **within-service percentile** over that service's own history, so
"high" means "unusually high for this service" rather than "above a universal
constant". That keeps the labelling rule independent of the workload and of any
fixed threshold.

Priority order resolves ties. The result is a weak label per (service, tick).

Rule sketch:

  transition        young pods / pending replicas -> starting up
  memory            working set at the service's high end AND (page faults or OOM)
  cpu               cpu near the limit AND throttling at the service's high end
  network           rx/tx at the service's high end
  storage           fs/io at the service's high end
  shared_dependency downstream latency high while local cpu/mem are calm
  queue             own p90 high while cpu is not the constraint
  healthy           nothing above
  unknown           insufficient data
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.groups import derive, RAW_FEATURES          # noqa: E402

CLASSES = ["cpu", "memory", "network", "storage", "queue",
           "shared_dependency", "app_problem", "infra_problem",
           "transition", "outage", "blind", "healthy", "unknown"]


def _pct_rank(s):
    """Per-service percentile rank in [0,1]; NaN stays NaN."""
    return s.rank(pct=True, method="average")


def add_percentiles(df):
    df = df.copy()
    # derived features first
    derived_cols = {}
    for col in ("success_rate", "cpu_limit_pct", "cpu_throttled_frac",
                "mem_limit_pct", "cpu_per_request", "lat_ratio_p90_p50",
                "inflight", "pending"):
        derived_cols[col] = np.nan
    d = df.apply(lambda r: derive({k: r.get(k) for k in RAW_FEATURES
                                   if k in df.columns}), axis=1)
    for col in derived_cols:
        df[col] = [x.get(col) for x in d]

    for col in df.columns:
        if col in ("service", "ts", "window"):
            continue
        df[col + "__pct"] = df.groupby("service")[col].transform(_pct_rank)
    return df


def label_row(r, p_hi=0.90, p_vhi=0.97):
    """Weak label for one row from its percentile features."""
    def hi(col, thr=p_hi):
        v = r.get(col + "__pct")
        return bool(v is not None and v == v and v >= thr)

    def vhi(col):
        return hi(col, p_vhi)

    # --- lifecycle ------------------------------------------------------
    pend = r.get("pending")
    if pend is not None and pend == pend and pend > 0:
        return "transition"
    age = r.get("pod_age_s")
    if age is not None and age == age and age < 60:
        return "transition"

    # outage: requests are arriving but not completing. Distinct from "slow":
    # a slow service still completes.
    comp = r.get("completion_ratio")
    req_rate = r.get("req_rate")
    if (comp is not None and comp == comp and req_rate is not None
            and req_rate > 0 and comp < 0.5):
        return "outage"

    # blind: arrivals with no latency sample. Not a fast service -- an
    # unmeasured one.
    lat = r.get("lat_p90")
    if req_rate is not None and req_rate > 0 and (lat is None or lat != lat):
        return "blind"

    cpu_hot = hi("cpu_limit_pct") and hi("cpu_throttled_frac")
    mem_hot = vhi("mem_limit_pct") and (hi("mem_page_faults") or hi("oom_events"))
    net_hot = hi("net_rx") or hi("net_tx")
    io_hot = hi("fs_write") or hi("fs_read") or hi("io_time")
    lat_hot = vhi("lat_p90") or vhi("lat_p99")

    if cpu_hot:
        return "cpu"
    if mem_hot:
        return "memory"
    if lat_hot and not cpu_hot and hi("lat_p90"):
        # own latency high but not CPU-bound
        if hi("cpu_request_utilization"):
            return "queue"
        return "shared_dependency"
    if net_hot:
        return "network"
    if io_hot:
        return "storage"
    return "healthy"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus.parquet")
    ap.add_argument("--out", default="../data/corpus_labelled.parquet")
    a = ap.parse_args()

    df = pd.read_parquet(a.corpus)
    print(f"[labels] {len(df)} rows, {df['service'].nunique()} services")
    df = add_percentiles(df)
    df["bottleneck"] = df.apply(label_row, axis=1)
    print(df["bottleneck"].value_counts().to_string())
    df.to_parquet(a.out, index=False)
    print(f"[labels] wrote {a.out}")


if __name__ == "__main__":
    main()