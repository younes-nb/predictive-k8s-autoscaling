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
    """Per-feature running mean/std with a global prior.

    Three defects in the previous version are fixed here:

    1. ``global_m2`` was loaded from a stored *variance* but treated as Welford's
       M2 (a sum of squared deviations), which divided the standard deviation by
       ~sqrt(n). The prior is now stored as a proper pseudo-count: ``mean`` and
       ``var`` with ``prior_n`` pseudo-observations, i.e. ``M2 = var * prior_n``.
    2. Counts are per feature, not one shared counter incremented per ingest
       call. Previously the effective sample count depended on how many metric
       series happened to be ingested.
    3. Duplicate timestamps are not counted twice (see WindowManager.ingest,
       which only calls ``update`` for genuinely new samples).
    """

    def __init__(self, prior_strength=20.0, prior_n=1000.0):
        self.prior_strength = prior_strength
        self.prior_n = prior_n
        # When frozen, ``stats`` returns only the training prior. The encoder
        # then receives exactly the representation it was trained on, while the
        # local statistics keep accumulating for baselines, drift and capacity.
        self.frozen = False
        self.global_mean = {}
        self.global_var = {}
        self.global_n = {}
        self.local = {}

    # ------------------------------------------------------------------
    def set_prior(self, feature_mean, feature_var, n=None):
        """Seed the global prior from a frozen training transform."""
        n = self.prior_n if n is None else float(n)
        for key, mean in feature_mean.items():
            self.global_mean[key] = float(mean)
            self.global_var[key] = max(float(feature_var.get(key, 1.0)), 1e-6)
            self.global_n[key] = n

    # ------------------------------------------------------------------
    def update(self, service, values):
        """Fold one observation. ``values`` maps feature -> float.

        Callers must only pass genuinely new samples; this method does not
        attempt to detect duplicates.

        When the normaliser is frozen, the *frozen* statistics are never
        touched -- they are the training representation and the encoder must
        see exactly what it was trained on. Only the per-service local baseline
        keeps accumulating, which is what drift detection and any future local
        calibration read. The previous implementation folded every new sample
        into ``global_mean``/``global_var`` even while frozen, so the
        normalisation used by the model drifted during a run.
        """
        frozen = self.frozen
        for key, v in values.items():
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue

            if not frozen:
                n = self.global_n.get(key, 0.0) + 1.0
                mean = self.global_mean.get(key, 0.0)
                delta = v - mean
                self.global_mean[key] = mean + delta / n
                self.global_var[key] = (
                    (self.global_var.get(key, 1.0) * max(self.global_n.get(key, 0.0), 1.0)
                     + delta * (v - self.global_mean[key]))
                    / max(n, 1.0)
                )
                self.global_n[key] = n

            loc = self.local.setdefault(service, {"mean": {}, "var": {}, "n": {}})
            ln = loc["n"].get(key, 0.0) + 1.0
            lm = loc["mean"].get(key, 0.0)
            ld = v - lm
            loc["mean"][key] = lm + ld / ln
            loc["var"][key] = (
                (loc["var"].get(key, 1.0) * max(loc["n"].get(key, 0.0), 1.0)
                 + ld * (v - loc["mean"][key]))
                / max(ln, 1.0)
            )
            loc["n"][key] = ln

    # ------------------------------------------------------------------
    def freeze(self):
        """Freeze the encoder-input statistics at the training representation.

        After this call ``global_mean``/``global_var`` are read-only: ``update``
        keeps feeding the per-service local baseline (for drift) but never the
        frozen pair, so the model's input distribution cannot drift at runtime.
        """
        self.frozen = True

    def stats(self, service, key):
        # The frozen pair is what the encoder was trained with. It is never
        # written after freeze(), so inference uses the training representation
        # verbatim.
        if self.frozen:
            return (self.global_mean.get(key, 0.0),
                    max(math.sqrt(self.global_var.get(key, 1.0)), 1e-6))
        loc = self.local.get(service)
        n = loc["n"].get(key, 0.0) if loc else 0.0
        g_n = self.global_n.get(key, 0.0)
        gm = self.global_mean.get(key, 0.0)
        gv = self.global_var.get(key, 1.0)
        if n <= 0:
            return gm, max(math.sqrt(gv), 1e-6)
        lm = loc["mean"].get(key, 0.0)
        lv = loc["var"].get(key, 1.0)
        w = self.prior_strength
        mean = (n * lm + w * gm) / (n + w)
        var = (n * (lv + (lm - mean) ** 2) + w * (gv + (gm - mean) ** 2)) / (n + w)
        return mean, max(math.sqrt(max(var, 1e-12)), 1e-6)

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
        self.length = cfg.history_len
        self.keep = cfg.history_len + 3
        self.history = defaultdict(lambda: defaultdict(deque))
        self.norm = norm or RunningNorm()
        self.last_ingest = {}
        # (timestamp, arrivals, latency_present) per service, for the
        # blindness estimate: a tick with arrivals but no latency sample is
        # "blind", not "fast".
        self.availability = defaultdict(deque)

    # ------------------------------------------------------------------
    def ingest(self, service, timestamp, features):
        """Append one observation. ``features`` maps name -> float or None.

        Returns the subset of features that were genuinely new. The normaliser
        is updated with exactly that subset: the controller re-reads overlapping
        history every 15 s, and counting those repeats would make the statistics
        depend on how many times a window was fetched rather than on how many
        observations exist.
        """
        h = self.history[service]
        fresh = {}
        for key, value in features.items():
            if value is None:
                continue
            dq = h[key]
            if dq and timestamp < dq[-1][0]:
                continue                      # older than what we have
            if dq and timestamp == dq[-1][0]:
                if dq[-1][1] == value:
                    continue                  # exact duplicate
                dq[-1] = (timestamp, value)   # corrected value
                fresh[key] = value
                continue
            dq.append((timestamp, value))
            while len(dq) > self.keep:
                dq.popleft()
            fresh[key] = value
        if fresh:
            self.norm.update(service, fresh)
        self.last_ingest[service] = timestamp
        return fresh

    # ------------------------------------------------------------------
    def note_availability(self, service, timestamp, arrivals, latency_present):
        dq = self.availability[service]
        if dq and timestamp <= dq[-1][0]:
            return
        dq.append((timestamp, float(arrivals or 0.0), bool(latency_present)))
        while len(dq) > self.keep:
            dq.popleft()

    def blindness_ratio(self, service, min_arrivals=1.0):
        """Fraction of recent ticks that had arrivals but no latency sample.

        High values mean the latency signal is untrustworthy: requests are
        arriving and (we hope) completing, but nothing is measuring them. That
        is exactly the SLO's blind spot, so the estimator must treat latency as
        missing rather than as fast.
        """
        dq = self.availability.get(service)
        if not dq:
            return 0.0
        considered = [a for _, a, _ in dq if a >= min_arrivals]
        if not considered:
            return 0.0
        blind = sum(1 for _, a, p in dq if a >= min_arrivals and not p)
        return blind / len(considered)

    def latency_available(self, service):
        dq = self.availability.get(service)
        if not dq:
            return False
        return bool(dq[-1][2])

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