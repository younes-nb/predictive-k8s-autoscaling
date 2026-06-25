import torch
import numpy as np
from scipy.stats import pearsonr


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


def compute_metrics(
    y_pred, y_true, y_last, horizon, total_samples, log_info, target_name=None
):
    if total_samples == 0:
        log_info("No samples found in test set.")
        return

    target_idx = horizon - 1

    y_true_target = y_true[:, target_idx]
    y_pred_target = y_pred[:, target_idx]

    mse = np.mean((y_pred_target - y_true_target) ** 2)
    mae = np.mean(np.abs(y_pred_target - y_true_target))

    mse_naive = np.mean((y_true_target - y_last) ** 2)

    skill_score = 1.0 - (mse / mse_naive)

    actual_dir = np.sign(y_true_target - y_last)
    pred_dir = np.sign(y_pred_target - y_last)
    mda = np.mean(actual_dir == pred_dir)

    corr_0, _ = pearsonr(y_true_target, y_pred_target)
    corr_1, _ = pearsonr(y_last, y_pred_target)

    is_shadowing = (skill_score < 0.05) or (corr_1 > corr_0) or (mda < 0.55)

    header = f"=== Performance Metrics{f' [{target_name}]' if target_name else ''} ==="
    log_info(f"\n{header}")
    log_info("-" * 30)
    log_info(f"MSE:                   {mse:.4f}")
    log_info(f"MAE:                   {mae:.4f}")
    log_info("-" * 30)
    log_info(">>> SHADOWING DIAGNOSTICS <<<")
    log_info(f"Skill Score (vs Naive): {skill_score:.4f}  (Ideal: > 0.1)")
    log_info(f"Directional Acc (MDA):  {mda:.2%} (Ideal: > 60%)")
    log_info(f"Correlation (Lag 0):    {corr_0:.4f}")
    log_info(f"Correlation (Lag -1):   {corr_1:.4f}")

    if is_shadowing:
        log_info(
            "!! WARNING: Model shows signs of SHADOWING (overfitting to last step) !!"
        )
    else:
        log_info("PASSED: Model appears to have learned temporal dynamics.")

    log_info("-" * 30)
