import torch
import torch.nn as nn


def per_target_huber_loss(
    preds,
    target,
    cpu_beta: float = 0.01,
    mem_beta: float = 0.002,
    lambda_cpu: float = 0.5,
    lambda_mem: float = 0.5,
    rel_w: float = 0.0,
    rel_eps: float = 1e-6,
):
    """Per-target weighted Huber loss for beating the naive CPU/memory forecaster.

    preds/target: (B, H, T). T==2 assumes [cpu, mem] ordering.
    Both CPU and memory use Huber (smooth L1) with independent betas: errors
    below beta are MSE-like (scaled quadratic), errors above beta are MAE-like
    (linear). beta is the MSE<->MAE balance knob per branch. An optional
    relative (MAPE) term for memory is available but off by default.
    Returns lambda_cpu*CPU_loss + lambda_mem*mem_loss (default 0.5/0.5).
    """
    t = preds.shape[-1]
    if t == 1:
        return nn.functional.mse_loss(preds, target)
    cpu_loss = nn.functional.smooth_l1_loss(
        preds[..., 0], target[..., 0], beta=cpu_beta
    )
    p_mem = preds[..., 1]
    t_mem = target[..., 1]
    mem_loss = nn.functional.smooth_l1_loss(p_mem, t_mem, beta=mem_beta)
    if rel_w:
        rel = torch.abs(p_mem - t_mem) / (torch.abs(t_mem) + rel_eps)
        mem_loss = mem_loss + rel_w * rel.mean()
    return lambda_cpu * cpu_loss + lambda_mem * mem_loss


def per_target_loss(preds, target, mem_mode="mse"):
    """Per-target loss so the memory head gets full gradient signal.

    preds/target: (B, H, T). T==2 assumes [cpu, mem] ordering.
    CPU is always MSE; memory uses `mem_mode` ("mse" or "l1").
    Returns CPU_loss + mem_loss (equal weight).
    """
    t = preds.shape[-1]
    if t == 1:
        return nn.functional.mse_loss(preds, target)
    cpu_loss = nn.functional.mse_loss(preds[..., 0], target[..., 0])
    if mem_mode == "l1":
        mem_loss = nn.functional.l1_loss(preds[..., 1], target[..., 1])
    else:
        mem_loss = nn.functional.mse_loss(preds[..., 1], target[..., 1])
    return cpu_loss + mem_loss


def asymmetric_huber_loss(
    preds,
    target,
    cpu_beta: float = 0.01,
    mem_beta: float = 0.002,
    lambda_cpu: float = 0.5,
    lambda_mem: float = 0.5,
    under_weight_cpu: float = 1.0,
    under_weight_mem: float = 1.0,
    rel_w: float = 0.0,
    rel_eps: float = 1e-6,
):
    """Asymmetric Huber loss that penalizes underprediction more heavily.

    For HPA: underprediction (pred < target) causes premature scale-down,
    which is dangerous. Overprediction is safe (proactive scale-up).

    Args:
        under_weight_cpu: Multiplier for CPU underprediction errors (default 3.0)
        under_weight_mem: Multiplier for memory underprediction errors (default 1.0)
    """
    t = preds.shape[-1]
    if t == 1:
        return nn.functional.smooth_l1_loss(preds, target, beta=cpu_beta)

    # CPU branch
    cpu_pred = preds[..., 0]
    cpu_target = target[..., 0]
    cpu_error = cpu_target - cpu_pred  # positive = underprediction
    cpu_huber = nn.functional.smooth_l1_loss(cpu_pred, cpu_target, beta=cpu_beta, reduction='none')
    # Weight underprediction more heavily
    cpu_weight = torch.where(cpu_error > 0, under_weight_cpu, 1.0)
    cpu_loss = (cpu_huber * cpu_weight).mean()

    # Memory branch
    p_mem = preds[..., 1]
    t_mem = target[..., 1]
    mem_error = t_mem - p_mem
    mem_huber = nn.functional.smooth_l1_loss(p_mem, t_mem, beta=mem_beta, reduction='none')
    mem_weight = torch.where(mem_error > 0, under_weight_mem, 1.0)
    mem_loss = (mem_huber * mem_weight).mean()

    if rel_w:
        rel = torch.abs(p_mem - t_mem) / (torch.abs(t_mem) + rel_eps)
        mem_loss = mem_loss + rel_w * rel.mean()

    return lambda_cpu * cpu_loss + lambda_mem * mem_loss
