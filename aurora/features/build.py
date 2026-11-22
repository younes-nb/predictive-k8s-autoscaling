"""Build model-ready windows from a harvested/labelled dataframe.

Shared by training and calibration so the encoder sees exactly the same tensor
layout it will see in the control loop.
"""

import numpy as np

from .groups import FEATURE_ORDER, derive

RAW = [f for f in FEATURE_ORDER if f not in (
    "success_rate", "cpu_limit_pct", "cpu_throttled_pct", "mem_limit_pct",
    "cpu_per_request", "lat_ratio_p90_p50", "inflight", "pending",
    "cpu_per_replica", "req_per_replica")]


def row_vector(rec):
    """Raw + derived feature vector and its availability mask for one row."""
    raw = {k: rec.get(k) for k in RAW}
    d = derive(raw)
    vec, mask = [], []
    for k in FEATURE_ORDER:
        v = raw.get(k) if k in raw else d.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            vec.append(0.0)
            mask.append(0.0)
        else:
            vec.append(float(v))
            mask.append(1.0)
    return np.asarray(vec, dtype=np.float32), np.asarray(mask, dtype=np.float32)


def group_windows(df, t_len=24, stride=None):
    """Yield (service, matrix[T,F], mask[T,F]) for one service's rows."""
    stride = stride or max(1, t_len // 2)
    for svc, g in df.groupby("service"):
        g = g.sort_values("ts")
        rows = [row_vector(r) for r in g.to_dict("records")]
        if len(rows) < t_len:
            continue
        x = np.stack([r[0] for r in rows])
        m = np.stack([r[1] for r in rows])
        for i in range(0, len(rows) - t_len + 1, stride):
            yield svc, x[i:i + t_len], m[i:i + t_len]


def stack_windows(df, t_len=24, stride=None):
    xs, ms, svcs = [], [], []
    for svc, x, m in group_windows(df, t_len, stride):
        xs.append(x)
        ms.append(m)
        svcs.append(svc)
    if not xs:
        return (np.zeros((0, t_len, len(FEATURE_ORDER)), dtype=np.float32),
                np.zeros((0, t_len, len(FEATURE_ORDER)), dtype=np.float32), [])
    return np.stack(xs), np.stack(ms), svcs