"""Temporal window manager and feature processor.

Responsibilities:

  1. keep a rolling per-service history at the native 5 s resolution;
  2. align metrics that arrive from different sources onto one grid, using
     each sample's *own* timestamp;
  3. mark rows where a sample is missing or stale instead of inventing a zero;
  4. normalise features with deployment-independent, resource-relative
     statistics (per-service running mean/std shrunk toward a global prior).
"""

import math
from collections import defaultdict, deque

from telemetry.schema import Batch

EPS = 1e-9


class RunningNorm:
    """Per-service running mean/std with a global prior.

    ``update`` is Welford; ``prior_strength`` pseudo-counts pull a service with
    few observations toward the global statistics, which is what lets a brand
    new deployment use the pretrained model sensibly before it has its own
    baseline.
    """

    def __init__(self, prior_strength=20.0):
        self.prior_strength = prior_strength
        self.global_mean = {}
        self.global_m2 = {}
        self.global_n = 0
        self.local = defaultdict(lambda: {"mean": {}, "m2": {}, "n": 0})

    def update(self, service, values):
        g = self.global_mean
        gm2 = self.global_m2
        self.global_n += 1
        n = self.global_n
        for k, v in values.items():
            if v is None:
                continue
            d = v - g.get(k, 0.0)
            g[k] = g.get(k, 0.0) + d / n
            gm2[k] = gm2.get(k, 0.0) + d * (v - g[k])

        loc = self.local[service]
        loc["n"] += 1
        ln = loc["n"]
        for k, v in values.items():
            if v is None:
                continue
            d = v - loc["mean"].get(k, 0.0)
            loc["mean"][k] = loc["mean"].get(k, 0.0) + d / ln
            loc["m2"][k] = loc["m2"].get(k, 0.0) + d * (v - loc["mean"][k])

    def stats(self, service, key):
        loc = self.local[service]
        n = loc["n"]
        w = self.prior_strength
        if n <= 0:
            gk = self.global_n or 1
            return self.global_mean.get(key, 0.0), max(
                (self.global_m2.get(key, 0.0) / gk) ** 0.5, EPS)
        lm = loc["mean"].get(key, 0.0)
        lv = loc["m2"].get(key, 0.0) / max(n - 1, 1)
        gm = self.global_mean.get(key, 0.0)
        gv = (self.global_m2.get(key, 0.0) / max(self.global_n - 1, 1)
              if self.global_n > 1 else lv)
        mean = (n * lm + w * gm) / (n + w)
        var = (n * (lv + (lm - mean) ** 2) + w * (gv + (gm - mean) ** 2)) / (n + w)
        return mean, max(var ** 0.5, EPS)

    def normalise(self, service, key, value):
        mean, std = self.stats(service, key)
        return (value - mean) / std


class WindowManager:
    """Rolling history per service, aligned on the telemetry grid."""

    def __init__(self, cfg, norm=None):
        self.cfg = cfg
        self.step = cfg.telemetry_step_s
        # keep a couple of spare samples: the grid is snapped to the
        # telemetry boundary, so the oldest grid point can sit one step inside
        # the fetched range and would otherwise be trimmed away
        self.length = cfg.history_len + 3
        self.history = defaultdict(lambda: defaultdict(deque))
        self.norm = norm or RunningNorm()
        self.last_ingest = {}

    # ------------------------------------------------------------------
    def ingest(self, service, timestamp, features):
        """Append one observation. ``features`` maps name -> float or None."""
        h = self.history[service]
        for key, value in features.items():
            if value is None:
                continue
            dq = h[key]
            if dq and timestamp <= dq[-1][0]:
                # out-of-order sample: replace if newer value at same ts
                if timestamp == dq[-1][0]:
                    dq[-1] = (timestamp, value)
                continue
            dq.append((timestamp, value))
            while len(dq) > self.length:
                dq.popleft()
        self.norm.update(service, features)
        self.last_ingest[service] = timestamp

    # ------------------------------------------------------------------
    def is_stale(self, service, now):
        last = self.last_ingest.get(service)
        if last is None:
            return True
        age_intervals = (now - last) / self.step
        return age_intervals > self.cfg.stale_after_intervals

    # ------------------------------------------------------------------
    def batch(self, service, now, n_steps=None, features=None):
        """Build the model input window ending at ``now``.

        Missing rows keep mask=0. A row is marked stale (mask=0 for every
        feature) when nothing was ingested within the staleness bound.
        """
        n_steps = n_steps or self.length
        features = features or []
        # Snap the grid to the telemetry boundary. Anchoring it to the wall
        # clock puts the newest grid point up to one step ahead of the newest
        # sample, which then falls outside the tolerance and marks every row
        # missing (measured: 100 % of services "stale" every cycle).
        anchor = math.floor(now / self.step) * self.step
        grid = [anchor - (n_steps - 1 - i) * self.step for i in range(n_steps)]
        h = self.history.get(service, {})
        tol = self.step * 0.75

        matrix, mask, complete = [], [], True
        for ts in grid:
            row, mrow = [], []
            any_ok = False
            for key in features:
                dq = h.get(key)
                found = None
                if dq:
                    # deque is time-ordered; scan backwards (bounded, small)
                    for pts_ts, val in reversed(dq):
                        if abs(pts_ts - ts) <= tol:
                            found = val
                            break
                        if pts_ts < ts - tol:
                            break
                if found is None:
                    row.append(0.0)
                    mrow.append(False)
                else:
                    row.append(self.norm.normalise(service, key, found))
                    mrow.append(True)
                    any_ok = True
            if not any_ok:
                complete = False
            matrix.append(row)
            mask.append(mrow)

        return Batch(timestamps=grid, matrix=matrix, mask=mask,
                     service=service, complete=complete)

    def newest_raw(self, service, key):
        dq = self.history.get(service, {}).get(key)
        return dq[-1][1] if dq else None