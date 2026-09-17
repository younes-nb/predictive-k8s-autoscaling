"""Adaptive CPA threshold driven by recent model errors.

Threshold = BASE_THRESHOLD (80%) shifted by the recency-weighted bias of
recent prediction errors, clipped to [BASE - RANGE, BASE + RANGE]
(e.g. RANGE=10 -> live threshold in [70, 90]).

- Signed error: e = actual - predicted (point forecast). Positive bias means
  the model UNDER-predicts -> lower the threshold (scale earlier/safer).
  Negative bias (over-prediction) -> raise the threshold (avoid waste).
- Window: last ADAPTIVE_ERROR_WINDOW errors (defaults to WINDOW_SIZE).
- Newest errors weigh most: w_i = exp(-age/tau), tau = N/3, so the oldest
  sample carries ~5% of the newest sample's weight.
- Source of truth is Prometheus (cpa_actual_cpu / cpa_pred_cpu / mem series
  scraped via the metrics-exporter + PodMonitor), so error history survives
  CPA pod restarts. Pod-local CSV is only a cold-start fallback when
  Prometheus has < 2 samples.

Because the CPA interval is aligned to the prediction horizon
(EVAL_INTERVAL_SECONDS == HORIZON * 60), a prediction written at cycle t
targets cycle t+1, so errors align as actual[i+1] - pred[i].
"""

import csv
import os
import time

import numpy as np

import config
import utils


def exp_weights(n):
    """Recency weights, oldest -> newest. Newest weighs most."""
    n = int(n)
    if n <= 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)
    tau = max(1.0, n / 3.0)
    ages = np.arange(n - 1, -1, -1, dtype=float)  # oldest has largest age
    return np.exp(-ages / tau)


def weighted_bias(errors):
    """Recency-weighted mean of signed errors (oldest -> newest)."""
    arr = np.asarray(list(errors), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    w = exp_weights(arr.size)
    return float(np.sum(w * arr) / np.sum(w))


def compute_threshold(errors_cpu, errors_mem=None):
    """Map recent errors to a live threshold in [MIN, MAX].

    Returns (threshold, combined_bias, bias_cpu, bias_mem).
    Combined bias uses max() so the most under-predicted signal dominates
    (conservative: scale earlier when either CPU or memory is under-shot).
    """
    bias_cpu = weighted_bias(errors_cpu)
    bias_mem = weighted_bias(errors_mem) if errors_mem is not None else 0.0
    if errors_mem is not None and len(list(errors_mem)) > 0:
        combined = max(bias_cpu, bias_mem)
    else:
        combined = bias_cpu
    threshold = config.BASE_THRESHOLD - combined
    threshold = float(
        np.clip(
            threshold,
            config.ADAPTIVE_THRESHOLD_MIN,
            config.ADAPTIVE_THRESHOLD_MAX,
        )
    )
    return threshold, combined, bias_cpu, bias_mem


def _series_values(result):
    """Pick the richest series from a range-query result -> float list."""
    best = []
    for series in result or []:
        vals = []
        for _, v in series.get("values", []):
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                vals.append(f)
        if len(vals) > len(best):
            best = vals
    return best


def _query_metric(metric, deployment, window_s, step_s):
    now = time.time()
    q = f'avg({metric}{{deployment="{deployment}"}})'
    result = utils.query_prometheus_range(q, now - window_s, now, step_s)
    return _series_values(result)


def get_recent_errors_from_prometheus():
    """Fetch horizon-aligned errors from Prometheus.

    Returns (errors_cpu, errors_mem, n_points). Errors are oldest -> newest,
    at most ADAPTIVE_ERROR_WINDOW entries.
    """
    n = int(config.ADAPTIVE_ERROR_WINDOW)
    step = int(config.EVAL_INTERVAL_SECONDS)
    window_s = (n + 1) * step
    dep = config.DEPLOYMENT

    actual_cpu = _query_metric("cpa_actual_cpu", dep, window_s, step)
    pred_cpu = _query_metric("cpa_pred_cpu", dep, window_s, step)
    errors_cpu = []
    if len(actual_cpu) >= 2 and len(pred_cpu) >= 2:
        m = min(len(actual_cpu), len(pred_cpu))
        # pred[t] targets actual[t+1] (interval == horizon)
        errors_cpu = [
            actual_cpu[i + 1] - pred_cpu[i] for i in range(m - 1)
        ][-n:]

    errors_mem = []
    if config.NUM_TARGETS > 1:
        actual_mem = _query_metric("cpa_actual_memory", dep, window_s, step)
        pred_mem = _query_metric("cpa_pred_mem", dep, window_s, step)
        if len(actual_mem) >= 2 and len(pred_mem) >= 2:
            m = min(len(actual_mem), len(pred_mem))
            errors_mem = [
                actual_mem[i + 1] - pred_mem[i] for i in range(m - 1)
            ][-n:]

    return errors_cpu, errors_mem, min(len(actual_cpu), len(pred_cpu))


def _fallback_errors_from_csv():
    """Cold-start fallback: align last CSV rows as actual[i+1] - pred[i]."""
    path = config.EXPERIMENT_METRICS_FILE
    n = int(config.ADAPTIVE_ERROR_WINDOW)
    if not os.path.exists(path):
        return [], []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f.read().splitlines())
            rows = list(reader)[-(n + 1):]
        cpu, mem = [], []
        for r in rows:
            try:
                cpu.append((float(r["cpu"]), float(r["pred_cpu"])))
                if "memory" in r and "pred_mem" in r:
                    mem.append((float(r["memory"]), float(r["pred_mem"])))
            except (KeyError, TypeError, ValueError):
                continue
        errors_cpu = [a_next - p for (_, p), (a_next, _) in zip(cpu[:-1], cpu[1:])][-n:]
        errors_mem = (
            [a_next - p for (_, p), (a_next, _) in zip(mem[:-1], mem[1:])][-n:]
            if config.NUM_TARGETS > 1 and mem
            else []
        )
        return errors_cpu, errors_mem
    except Exception:
        return [], []


def get_adaptive_threshold():
    """Compute the live threshold.

    Primary source: Prometheus error history (persistent). Falls back to the
    pod-local CSV on cold start, then to BASE_THRESHOLD.
    Returns (threshold, info dict).
    """
    n = int(config.ADAPTIVE_ERROR_WINDOW)
    errors_cpu, errors_mem, n_points = get_recent_errors_from_prometheus()
    source = "prometheus"
    if len(errors_cpu) < 1:
        errors_cpu, errors_mem = _fallback_errors_from_csv()
        source = "csv-fallback" if errors_cpu else "base-default"
    threshold, bias, bias_cpu, bias_mem = compute_threshold(
        errors_cpu, errors_mem if config.NUM_TARGETS > 1 else None
    )
    info = {
        "source": source,
        "n_errors": len(errors_cpu),
        "n_window": n,
        "bias": bias,
        "bias_cpu": bias_cpu,
        "bias_mem": bias_mem,
    }
    return threshold, info
