import torch
import numpy as np


def compute_adaptive_thresholds(
    theta_base, sigma, k=2.0, theta_min=0.60, theta_max=0.90
):
    adaptive = theta_base - k * sigma
    return torch.clamp(adaptive, min=theta_min, max=theta_max)


@torch.no_grad()
def mc_dropout_predict(model, x, repeats=30, horizon_index=-1):
    model.train()

    preds = []
    for _ in range(repeats):
        out = model(x)
        preds.append(out.unsqueeze(0))

    stack = torch.cat(preds, dim=0)

    mu = stack.mean(dim=0)
    sigma = stack.std(dim=0, unbiased=True)

    return mu, sigma
