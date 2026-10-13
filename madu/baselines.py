"""Baseline predictors from the paper's Table 1 (subset).

Implemented: ARIMA and a persistence (naive) baseline. The Seq2Seq and
DUBNN baselines are covered by training MaduSeq2Seq with
uncertainty=False (deterministic Seq2Seq, MSE) — see madu/train.py
--no_uncertainty. BNN is omitted (prior work, not the paper's
contribution; DUBNN's dropout approximation is discussed in §5.1.2).
"""
import numpy as np


def naive_forecast(last_values, m):
    """Persistence: repeat the last observed CPM for all m steps."""
    last_values = np.asarray(last_values, dtype=float)
    return np.tile(last_values[:, None], (1, m))


def arima_rolling_forecast(series, starts, n, m, order=(5, 1, 0),
                           refit_every=288, history=2016, verbose=False):
    """Rolling-origin ARIMA forecasts aligned with test windows.

    series: full raw CPM array; starts: window start indices into `series`
    (window [s, s+n) is history, [s+n, s+n+m) is the target).
    A statsmodels ARIMA(p,1,0) (no drift) is fit on trailing `history`
    points every `refit_every` windows (paper refits ARIMA at runtime;
    refitting every window is prohibitive, default = once per day for
    m=5); between refits the frozen AR params are recursed on fresh
    history, which is exact for multi-step AR forecasting.
    Only (p,1,0) orders are supported. Multi-step levels are clamped at
    0 (CPM is non-negative; same convention as the model-output clamp).
    """
    from statsmodels.tsa.arima.model import ARIMA

    p, d, q = order
    if not (d == 1 and q == 0):
        raise ValueError("only (p,1,0) orders are supported")
    series = np.asarray(series, dtype=float)
    preds = np.zeros((len(starts), m), dtype=float)
    phi = None
    for i, s in enumerate(starts):
        if i % refit_every == 0 or phi is None:
            hist = series[max(0, s + n - history):s + n]
            try:
                phi = np.asarray(
                    ARIMA(hist, order=order, trend="n").fit().arparams,
                    dtype=float,
                )
            except Exception as e:
                if verbose:
                    print(f"[arima] fit failed at window {i}: {e}")
                phi = None
        if phi is None:
            preds[i] = series[s + n - 1]  # fall back to persistence
            continue
        w = np.diff(series[s + n - p - 1:s + n])  # last p differences
        w = w[-p:]
        w_hat, acc = [], float(series[s + n - 1])
        for k in range(m):
            seq = list(w) + w_hat
            wk = float(np.dot(phi, seq[-p:][::-1]))
            w_hat.append(wk)
            acc += wk
            preds[i, k] = max(acc, 0.0)
    return preds
