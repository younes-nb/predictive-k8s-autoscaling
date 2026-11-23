"""Build model-ready windows from a harvested/labelled dataframe.

Shared by training and calibration so the encoder sees exactly the same tensor
layout it will see in the control loop.
"""

import numpy as np

from .groups import FEATURE_ORDER


def row_vector(rec, now=None):
    """Raw + derived feature vector and mask for one row.

    Delegates to the single shared transform so training and live inference
    cannot drift apart.
    """
    from .transform import transform
    return transform(rec, now)


def group_windows(df, t_len=24, stride=None, max_gap_s=None, label_col=None):
    """Yield (service, matrix[T,F], mask[T,F]) for one service's rows.

    Sequences are built per (run, service) and broken wherever the timestamp gap
    exceeds ``max_gap_s``. Previously windows were built across the whole corpus
    per service, so the end of one experiment could be joined to the start of
    another as if they were consecutive 5 s observations.
    """
    stride = stride or max(1, t_len // 2)
    max_gap_s = max_gap_s if max_gap_s is not None else 15
    group_cols = [c for c in ("window", "service") if c in df.columns]
    for _, g in df.groupby(group_cols or ["service"]):
        g = g.sort_values("ts")
        rows = [row_vector(r) for r in g.to_dict("records")]
        recs = g.to_dict("records")
        ts = [r["ts"] for r in recs]
        labels = list(g[label_col]) if (label_col and label_col in g.columns) else None
        if len(rows) < t_len:
            continue
        # split at discontinuities
        breaks = [0] + [i for i in range(1, len(rows))
                        if ts[i] - ts[i - 1] > max_gap_s] + [len(rows)]
        for a, b in zip(breaks, breaks[1:]):
            if b - a < t_len:
                continue
            xs = np.stack([r[0] for r in rows[a:b]])
            ms = np.stack([r[1] for r in rows[a:b]])
            for i in range(0, (b - a) - t_len + 1, stride):
                svc = recs[a + i]["service"]
                if labels is not None:
                    yield svc, xs[i:i + t_len], ms[i:i + t_len], labels[a + i + t_len - 1]
                else:
                    yield svc, xs[i:i + t_len], ms[i:i + t_len]


def stack_windows(df, t_len=24, stride=None):
    xs, ms, svcs = [], [], []
    for item in group_windows(df, t_len, stride):
        svc, x, m = item[0], item[1], item[2]
        xs.append(x)
        ms.append(m)
        svcs.append(svc)
    if not xs:
        return (np.zeros((0, t_len, len(FEATURE_ORDER)), dtype=np.float32),
                np.zeros((0, t_len, len(FEATURE_ORDER)), dtype=np.float32), [])
    return np.stack(xs), np.stack(ms), svcs