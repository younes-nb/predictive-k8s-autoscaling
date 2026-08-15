import torch
import numpy as np


def find_max_inference_batch_size(
    model, input_size, args, device, starting_batch=16384
):
    batch_size = starting_batch
    model.eval()

    while batch_size > 0:
        try:
            dummy_x = torch.randn(batch_size, args.input_len, input_size, device=device)
            with torch.no_grad():
                _ = model(dummy_x)

            del dummy_x
            torch.cuda.empty_cache()
            return batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                batch_size //= 2
            else:
                raise e

    raise RuntimeError("Could not find a batch size that fits in memory.")


def _compute_one_step(y_pred_step, y_true_step, y_last, y_second_last=None):
    err = y_pred_step - y_true_step
    abs_err = np.abs(err)

    under_mask = y_pred_step < y_true_step
    over_mask = y_pred_step > y_true_step
    n = len(y_true_step)
    n_under = int(np.sum(under_mask))
    n_over = int(np.sum(over_mask))

    mse = float(np.mean(err ** 2))
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true_step - np.mean(y_true_step)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    nonzero = np.abs(y_true_step) > 1e-12
    if int(np.sum(nonzero)) > 0:
        mape = float(np.mean(np.abs(err[nonzero]) / np.abs(y_true_step[nonzero]))) * 100.0
    else:
        mape = 0.0

    actual_dir = np.sign(y_true_step - y_last)
    if y_second_last is not None:
        pred_dir = np.sign(y_last - y_second_last)
    else:
        pred_dir = np.sign(y_pred_step - y_last)
    mda = float(np.mean(actual_dir == pred_dir))

    under_rate = (n_under / n * 100.0) if n > 0 else 0.0
    over_rate = (n_over / n * 100.0) if n > 0 else 0.0

    if n_under > 0:
        mean_under = float(np.mean(y_true_step[under_mask] - y_pred_step[under_mask]))
        max_under = float(np.max(y_true_step[under_mask] - y_pred_step[under_mask]))
    else:
        mean_under = 0.0
        max_under = 0.0

    if n_over > 0:
        mean_over = float(np.mean(y_pred_step[over_mask] - y_true_step[over_mask]))
        max_over = float(np.max(y_pred_step[over_mask] - y_true_step[over_mask]))
    else:
        mean_over = 0.0
        max_over = 0.0

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "MAPE (%)": mape,
        "MDA (%)": mda * 100.0,
        "Under-Pred Rate (%)": under_rate,
        "Over-Pred Rate (%)": over_rate,
        "Mean Under Error": mean_under,
        "Mean Over Error": mean_over,
        "Max Under Error": max_under,
        "Max Over Error": max_over,
    }


METRIC_NAMES = [
    "MSE", "MAE", "RMSE", "R²", "MAPE (%)", "MDA (%)",
    "Under-Pred Rate (%)", "Over-Pred Rate (%)",
    "Mean Under Error", "Mean Over Error",
    "Max Under Error", "Max Over Error",
]

PCT_METRICS = {"MAPE (%)", "MDA (%)", "Under-Pred Rate (%)", "Over-Pred Rate (%)"}


def _delta_pct(model_val, naive_val, is_pct_metric=False):
    if is_pct_metric:
        diff = model_val - naive_val
        return f"{diff:+.1f}"
    denom = abs(naive_val)
    if denom < 1e-12:
        return "N/A"
    pct = (model_val - naive_val) / denom * 100.0
    return f"{pct:+.1f}"


def compute_metrics(
    y_pred, y_true, y_last, horizon, total_samples, log_info,
    target_name=None, y_second_last=None,
):
    if total_samples == 0:
        log_info("No samples found in test set.")
        return {}

    last_step = _compute_one_step(y_pred[:, -1], y_true[:, -1], y_last)

    avg_steps = {}
    for name in METRIC_NAMES:
        vals = []
        for h in range(horizon):
            step_metrics = _compute_one_step(y_pred[:, h], y_true[:, h], y_last)
            vals.append(step_metrics[name])
        avg_steps[name] = float(np.mean(vals))

    naive = _compute_one_step(y_last, y_true[:, -1], y_last, y_second_last=y_second_last)

    header = f"=== Evaluation{f': {target_name}' if target_name else ''} ==="
    log_info(f"\n{header}")
    log_info("")
    log_info(
        f"{'Metric':<26s} {'Last Step':>14s} {'Avg Steps':>14s} "
        f"{'Naive':>14s} {'Δ (%)':>10s}"
    )
    log_info("-" * 82)

    results = {}
    for name in METRIC_NAMES:
        ls = last_step[name]
        av = avg_steps[name]
        nv = naive[name]
        d = _delta_pct(ls, nv, is_pct_metric=(name in PCT_METRICS))

        if name in PCT_METRICS:
            log_info(
                f"{name:<26s} {ls:>13.8f}% {av:>13.8f}% {nv:>13.8f}% {d:>10s}"
            )
        else:
            log_info(
                f"{name:<26s} {ls:>14.8e} {av:>14.8e} {nv:>14.8e} {d:>10s}"
            )

        results[name] = {"last_step": ls, "avg_steps": av, "naive": nv, "delta_pct": d}

    log_info("-" * 82)

    return results


def _pearson(a, b):
    a = np.asarray(a, dtype=float).ravel() - np.asarray(a, dtype=float).ravel().mean()
    b = np.asarray(b, dtype=float).ravel() - np.asarray(b, dtype=float).ravel().mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(np.sum(a * b) / denom) if denom > 1e-12 else float("nan")


def compute_persistence_diagnostics(y_pred_last, y_true_last, y_last, log_info, target_name=None):
    """Compare the horizon-ahead forecast against the persistence baseline
    (predicting the current load for the future)."""
    y_pred_last = np.asarray(y_pred_last, dtype=float).ravel()
    y_true_last = np.asarray(y_true_last, dtype=float).ravel()
    y_last = np.asarray(y_last, dtype=float).ravel()

    n = len(y_pred_last)
    if n < 2:
        log_info("Too few samples for persistence diagnostics.")
        return {}

    corr_pred_true = _pearson(y_pred_last, y_true_last)
    corr_pred_last = _pearson(y_pred_last, y_last)
    corr_last_true = _pearson(y_last, y_true_last)

    mse_pred = float(np.mean((y_pred_last - y_true_last) ** 2))
    mse_naive = float(np.mean((y_last - y_true_last) ** 2))
    r2_vs_persistence = 1.0 - mse_pred / mse_naive if mse_naive > 1e-12 else float("nan")

    mae_pred = float(np.mean(np.abs(y_pred_last - y_true_last)))
    mae_naive = float(np.mean(np.abs(y_last - y_true_last)))
    mae_vs_persistence = mae_pred / mae_naive if mae_naive > 1e-12 else float("nan")

    abs_pred = np.abs(y_pred_last - y_true_last)
    abs_naive = np.abs(y_last - y_true_last)
    beat_persistence = float(np.mean(abs_pred < abs_naive) * 100.0)

    header = f"=== Persistence Diagnostics{f' ({target_name})' if target_name else ''} ==="
    log_info(f"\n{header}")
    log_info("-" * 62)
    log_info(f"{'ρ(pred, truth)':<28s} {corr_pred_true:>10.4f}   corr of prediction with the true target")
    log_info(f"{'ρ(pred, current)':<28s} {corr_pred_last:>10.4f}   corr of prediction with the current input load")
    log_info(f"{'ρ(current, truth)':<28s} {corr_last_true:>10.4f}   corr of persistence baseline with the true target")
    log_info(f"{'R² vs persistence':<28s} {r2_vs_persistence:>10.4f}   1 - MSE(pred)/MSE(current); <=0 means no skill over persistence")
    log_info(f"{'Beat-persistence (%)':<28s} {beat_persistence:>10.2f}   % of samples where pred is closer to truth than current load is")
    log_info(f"{'MAE vs persistence':<28s} {mae_vs_persistence:>10.4f}   MAE(pred)/MAE(current); <1 = better than persistence")
    log_info("-" * 62)

    return {
        "corr_pred_true": corr_pred_true,
        "corr_pred_last": corr_pred_last,
        "corr_last_true": corr_last_true,
        "r2_vs_persistence": r2_vs_persistence,
        "beat_persistence": beat_persistence,
        "mae_vs_persistence": mae_vs_persistence,
    }
