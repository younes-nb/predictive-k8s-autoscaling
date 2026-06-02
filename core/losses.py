from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def weighted_mse(
    preds: torch.Tensor,
    target: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    under_penalty: float = 5.0,
    clamp_weights: bool = True,
) -> torch.Tensor:
    diff = preds - target
    sq_err = diff**2
    under_mask = (preds < target).float()
    asym_weight = 1.0 + (under_mask * (under_penalty - 1.0))
    per_sample = (sq_err * asym_weight).mean(dim=1)

    if w is None:
        return per_sample.mean()

    if clamp_weights:
        w = w.clamp(min=0.1, max=15.0)
    return (w * per_sample).sum() / w.sum().clamp_min(1e-6)


class PinballLoss(nn.Module):
    def __init__(self, quantiles: Sequence[float]):
        super().__init__()
        q = torch.tensor([float(q) for q in quantiles], dtype=torch.float32)
        self.register_buffer("quantiles", q)

    def forward(self, preds: torch.Tensor, target: torch.Tensor, w=None) -> torch.Tensor:
        if preds.dim() != 3:
            raise ValueError("PinballLoss expects preds with shape (batch, horizon, q).")
        q = self.quantiles.view(1, 1, -1)
        diff = target.unsqueeze(-1) - preds
        loss = torch.maximum(q * diff, (1.0 - q) * (-diff))
        per_sample = loss.mean(dim=(1, 2))

        if w is None:
            return per_sample.mean()

        w = w.clamp(min=0.1, max=15.0)
        return (w * per_sample).sum() / w.sum().clamp_min(1e-6)
