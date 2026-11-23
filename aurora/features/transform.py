"""The single feature transform used by BOTH training and live inference.

This module exists because the two paths had drifted apart:

  * training computed derived features in ``features/build.py`` while its RAW
    list excluded only 10 of the 14 derived names, so four derived features were
    treated as (absent) raw inputs;
  * the live loop built its batch from raw metrics *before* deriving anything,
    so the encoder saw those features as unavailable at inference time.

Both now call :func:`transform`. Any future feature change must be made here and
nowhere else; ``tests/test_parity.py`` asserts the two paths agree.
"""

import math

from .groups import FEATURE_ORDER, derive

# Names that are computed from other features rather than fetched.
DERIVED_NAMES = {
    "success_rate", "completion_ratio", "mem_limit_pct", "pod_age_s",
    "cpu_throttled_frac", "cpu_per_request", "lat_ratio_p90_p50",
    "inflight", "inflight_per_replica", "pending", "cpu_per_replica",
    "req_per_replica", "mem_headroom", "cpu_request_utilization",
}

RAW_NAMES = [f for f in FEATURE_ORDER if f not in DERIVED_NAMES]

# ``pod_start_time`` is an absolute epoch and meaningless to a model. It is
# converted to ``pod_age_s`` (seconds), which IS a feature.
_PASS_THROUGH_DROP = {"pod_start_time"}


def transform(raw, now=None, latency_available=None, blindness_ratio=None):
    """Return ``(values, mask)`` aligned to ``FEATURE_ORDER``.

    ``values`` is a list of floats, ``mask`` a list of 0/1. A feature that is
    genuinely zero gets mask 1; a feature that is unavailable or non-finite gets
    mask 0 and value 0.0. The two are never conflated.
    """
    src = {k: v for k, v in raw.items() if k not in _PASS_THROUGH_DROP}
    if now is not None:
        st = raw.get("pod_start_time")
        if st is not None:
            try:
                src["pod_age_s"] = max(0.0, float(now) - float(st))
            except (TypeError, ValueError):
                pass

    derived = derive(src)

    values, mask = [], []
    for key in FEATURE_ORDER:
        # ``pod_age_s`` is computed above from pod_start_time and lives in src,
        # not in the derived dict; every other derived name comes from derive().
        val = derived.get(key)
        if val is None:
            val = src.get(key)

        if key == "latency_available" and latency_available is not None:
            values.append(1.0 if latency_available else 0.0)
            mask.append(1)
            continue
        if key == "blindness_ratio" and blindness_ratio is not None:
            values.append(float(blindness_ratio))
            mask.append(1)
            continue

        if val is None:
            values.append(0.0)
            mask.append(0)
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            values.append(0.0)
            mask.append(0)
            continue
        if not math.isfinite(f):
            values.append(0.0)
            mask.append(0)
            continue
        values.append(f)
        mask.append(1)
    return values, mask


def vector(raw, now=None, **kw):
    """Convenience: just the values list."""
    return transform(raw, now, **kw)[0]


def availability(raw, now=None, **kw):
    """Convenience: just the mask list."""
    return transform(raw, now, **kw)[1]