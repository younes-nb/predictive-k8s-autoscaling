"""Losses for the Madu predictor."""
import torch


def gaussian_nll(mu, sigma, y, eps=1e-6):
    """Negative log-likelihood of Eq.(1), averaged over batch and steps.

    L = mean( (y-mu)^2 / (2 sigma^2) + log sigma + 0.5 log(2 pi) ).
    Higher predicted variance is encouraged exactly when the mean
    deviates from the observation (data-dependent uncertainty).
    """
    sigma = sigma.clamp_min(eps)
    return (
        (y - mu) ** 2 / (2.0 * sigma ** 2)
        + torch.log(sigma)
        + 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
    ).mean()
