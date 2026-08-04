from typing import Sequence

import torch
import torch.nn as nn


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
