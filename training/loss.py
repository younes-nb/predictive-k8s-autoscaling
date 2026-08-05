from typing import Sequence

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


def weighted_mse(preds, target, w=None, under_penalty=5.0):
    diff = preds - target
    sq_err = diff**2
    under_mask = (preds < target).float()
    asym_weight = 1.0 + (under_mask * (under_penalty - 1.0))
    value_loss = (sq_err * asym_weight).mean(dim=1)

    if w is None:
        return value_loss.mean()

    w = w.clamp(min=0.1, max=15.0)
    return (w * value_loss).sum() / w.sum().clamp_min(1e-6)


class PinballLoss(nn.Module):

    def __init__(self, quantiles: Sequence[float]):
        super().__init__()
        q = torch.tensor([float(q) for q in quantiles], dtype=torch.float32)
        self.register_buffer("quantiles", q)

    def forward(
        self, preds: torch.Tensor, target: torch.Tensor, w=None
    ) -> torch.Tensor:
        if preds.dim() != 3:
            raise ValueError(
                "PinballLoss expects preds with shape (batch, horizon, q)."
            )
        q = self.quantiles.view(1, 1, -1)
        diff = target.unsqueeze(-1) - preds
        loss = torch.maximum(q * diff, (1.0 - q) * (-diff))
        per_sample = loss.mean(dim=(1, 2))

        if w is None:
            return per_sample.mean()

        w = w.clamp(min=0.1, max=15.0)
        return (w * per_sample).sum() / w.sum().clamp_min(1e-6)
