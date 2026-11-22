"""Shared self-supervised temporal encoder (TS2Vec-style, compact CPU build).

Learns general representations of resource behaviour from *unlabelled*
telemetry across many services, so a new deployment does not need a new encoder.

Forward returns per-timestep embeddings (B, T, D); the sequence vector is the
masked mean over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _augment(x, mask, jitter=0.05, time_mask=0.1, seed=None):
    """TS2Vec augmentation: small jitter + random temporal masking."""
    g = None
    if seed is not None:
        g = torch.Generator(device=x.device).manual_seed(int(seed))
    if mask is not None and mask.dim() == 3:
        mask = mask.any(dim=-1).to(x.dtype)
    out = x
    if jitter:
        noise = torch.randn(x.shape, generator=g, device=x.device) * jitter
        # do not perturb features that are masked out
        out = out + noise * mask.unsqueeze(-1).to(out.dtype) if mask is not None else out + noise
    if time_mask:
        b, t, _ = out.shape
        drop = (torch.rand((b, t, 1), generator=g, device=out.device) < time_mask)
        keep = (~drop).to(out.dtype)
        out = out * keep
        if mask is not None:
            mask = mask * keep.squeeze(-1)
    return out, mask


class TS2VecEncoder(nn.Module):
    def __init__(self, n_features, d_model=64, kernel=3,
                 dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(n_features, d_model)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel,
                      padding=dilation * (kernel - 1) // 2, dilation=dilation)
            for dilation in dilations
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B, T, F) -> (B, T, D).

        ``mask`` may be per-feature (B, T, F) or per-timestep (B, T); a
        timestep counts as valid if any of its features is available.
        """
        if mask is not None and mask.dim() == 3:
            mask = mask.any(dim=-1).to(x.dtype)
        h = self.proj(x).transpose(1, 2)                 # B,D,T
        for conv, norm in zip(self.convs, self.norms):
            y = conv(h)
            if y.shape[-1] != h.shape[-1]:
                y = y[..., :h.shape[-1]]
            h = norm((F.gelu(y) + h).transpose(1, 2)).transpose(1, 2)
        h = h.transpose(1, 2)                            # B,T,D
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)
        return h

    def sequence(self, x, mask=None):
        h = self.forward(x, mask)
        if mask is None:
            return h.mean(dim=1)
        if mask.dim() == 3:
            mask = mask.any(dim=-1)
        w = mask.to(h.dtype).unsqueeze(-1)
        return (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def ts2vec_loss(x, mask, encoder, temperature=0.07, jitter=0.05, time_mask=0.1):
    """Instance-level contrastive loss: same timestamp across two augmented
    views is the positive; other timestamps in the batch are negatives."""
    a, ma = _augment(x, mask, jitter=jitter, time_mask=time_mask, seed=1)
    b, mb = _augment(x, mask, jitter=jitter, time_mask=time_mask, seed=2)
    za = F.normalize(encoder(a, ma), dim=-1)             # B,T,D
    zb = F.normalize(encoder(b, mb), dim=-1)
    bsz, t, _ = za.shape
    logits = torch.einsum("btd,bsd->bts", za, zb) / temperature
    # mask out anchors/positives whose observation is unavailable, but always
    # keep the diagonal: a timestep is always its own positive
    if ma.dim() == 3:
        ma = ma.any(dim=-1).to(za.dtype)
        mb = mb.any(dim=-1).to(za.dtype)
    valid = (ma.unsqueeze(-1) * mb.unsqueeze(1)) > 0     # B,T,T
    eye = torch.eye(t, device=logits.device, dtype=logits.dtype).unsqueeze(0)
    valid = valid | (eye > 0)
    logits = logits.masked_fill(~valid, -1e4)
    target = torch.arange(t, device=logits.device).unsqueeze(0).expand(bsz, t)
    return F.cross_entropy(logits.reshape(bsz * t, t), target.reshape(-1),
                           ignore_index=-1)


def masked_reconstruction_loss(x, mask, encoder, decoder=None):
    """Auxiliary objective: reconstruct the **observed** values of the same
    window from the embedding.

    This is autoencoding of the current window, not forecasting: the target is
    the input itself at the same timestamps. It keeps the representation
    grounded in real magnitudes instead of pure instance discrimination, which
    matters for a 42-service corpus. Unused by default (``decoder=None``).
    """
    h = encoder.sequence(x, mask)
    if decoder is None:
        return torch.tensor(0.0, device=x.device)
    recon = decoder(h).unsqueeze(1).expand(-1, x.shape[1], -1)
    if mask.dim() == 3:
        w = mask.any(dim=-1).to(x.dtype).unsqueeze(-1)     # B,T,1
    else:
        w = mask.to(x.dtype).unsqueeze(-1)
    return (((recon - x) ** 2) * w).sum() / w.sum().clamp_min(1.0)