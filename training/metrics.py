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


def _compute_one_step(y_pred_step, y_true_step, y_last):
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
    y_pred, y_true, y_last, horizon, total_samples, log_info, target_name=None
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

    naive = _compute_one_step(y_last, y_true[:, -1], y_last)

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
