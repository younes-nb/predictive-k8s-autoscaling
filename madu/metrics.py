"""Evaluation metrics from the paper (§5.1.1).

All metrics operate on raw (denormalized) CPM.
"""
import numpy as np

GROUPS = [("0-50%", 0, 50), ("50-95%", 50, 95), ("95-100%", 95, 100)]
ACC_EPS = 1.0  # CPM floor: accuracy = 1 - |y-yhat| / max(y, 1).
# Paper Eq.(15) divides by the raw CPM; the floor keeps idle (0-CPM)
# minutes defined instead of dividing by zero (documented deviation).


def prediction_accuracy(y_true, y_pred):
    """Eq.(15): mean(1 - |y - yhat| / y), averaged over all test samples."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(y_true, ACC_EPS)
    return float(np.mean(1.0 - np.abs(y_true - y_pred) / denom))


def median_accuracy(y_true, y_pred):
    """Robust supplement to Eq.(15): median instead of mean (our traces
    contain near-idle minutes where one huge relative error would
    dominate the mean; the paper's busy MSs did not)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(y_true, ACC_EPS)
    return float(np.median(1.0 - np.abs(y_true - y_pred) / denom))


def overestimation_accuracy(y_true, y_pred):
    """Eq.(16): mean(y / yhat) over samples with yhat > y."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_pred > y_true
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(y_true[mask] / np.maximum(y_pred[mask], ACC_EPS)))


def accurate_fraction(y_true, y_pred, tol=0.10):
    """Share of predictions within (1 ± tol) x ground truth (Fig.9a)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), ACC_EPS)
    return float(np.mean(np.abs(y_true - y_pred) / denom <= tol))


def evaluate_by_percentile(y_true, y_pred):
    """Accuracy per CPM-percentile group + average (Table 2 layout)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    out = {}
    for name, lo, hi in GROUPS:
        q_lo, q_hi = np.percentile(y_true, [lo, hi])
        if lo == 0:
            mask = y_true <= q_hi
        else:
            mask = (y_true > q_lo) & (y_true <= q_hi)
            if hi == 100:
                mask = mask | (y_true > q_hi)
        if int(mask.sum()) == 0:
            out[name] = {"accuracy": float("nan"),
                         "median_accuracy": float("nan"), "n": 0}
        else:
            out[name] = {
                "accuracy": prediction_accuracy(y_true[mask], y_pred[mask]),
                "median_accuracy": median_accuracy(y_true[mask], y_pred[mask]),
                "n": int(mask.sum()),
            }
    vals = [v["accuracy"] for v in out.values() if not np.isnan(v["accuracy"])]
    meds = [v["median_accuracy"] for v in out.values()
            if not np.isnan(v["median_accuracy"])]
    out["avg"] = {"accuracy": float(np.mean(vals)) if vals else float("nan"),
                  "median_accuracy": float(np.mean(meds)) if meds else float("nan"),
                  "n": int(len(y_true))}
    # NOTE (Table 2): the paper's "Avg" row is NOT this mean-of-groups. It
    # matches the sample-pooled accuracy ("overall" below), averaged across
    # microservices — see evaluate.summarize_results, which aggregates the
    # per-MS "overall" values for the cross-MS Avg row.
    # pooled (micro-average) over all samples — the headline number when
    # groups are imbalanced (tiny groups would otherwise dominate "avg")
    out["overall"] = {"accuracy": prediction_accuracy(y_true, y_pred),
                      "median_accuracy": median_accuracy(y_true, y_pred),
                      "n": int(len(y_true))}
    out["accurate_frac_10pct"] = accurate_fraction(y_true, y_pred)
    out["overestimation_accuracy"] = overestimation_accuracy(y_true, y_pred)
    out["mse"] = float(np.mean((y_true - y_pred) ** 2))
    out["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    return out
