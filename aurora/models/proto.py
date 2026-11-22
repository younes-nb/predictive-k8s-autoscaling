"""Few-shot heads.

Two separate outputs, as required:

  * ``BottleneckHead`` -- prototypical network over bottleneck classes. Adding a
    new service/deployment means adding or updating prototype vectors; the
    temporal encoder is never retrained for it.
  * ``AddressabilityHead`` -- an independent binary head answering "would more
    replicas help?", so a detected bottleneck is never automatically treated as
    horizontally addressable.

Both emit a confidence and an out-of-distribution score computed against the
distance distribution recorded at calibration time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


BOTTLENECK_CLASSES = [
    "cpu",
    "memory",
    "network",
    "storage",
    "queue",
    "shared_dependency",
    "app_problem",
    "infra_problem",
    "transition",
    "healthy",
    "unknown",
]

# Which bottlenecks extra replicas can plausibly relieve. Anything not listed
# is not assumed addressable; the learned head still has the final say.
ADDRESSABLE_PRIOR = {
    "cpu", "memory", "queue", "storage",
}


class PrototypicalHead(nn.Module):
    def __init__(self, d_model, classes=None, temperature=0.1):
        super().__init__()
        self.classes = list(classes or BOTTLENECK_CLASSES)
        self.prototypes = nn.Parameter(torch.zeros(len(self.classes), d_model))
        nn.init.normal_(self.prototypes, std=0.02)
        self.temperature = temperature
        self.oood_quantile = 0.99
        self._dist_threshold = None
        # log class prior, subtracted from the logits. Without it the majority
        # classes (here healthy/transition, ~85% of the corpus) win every
        # comparison and the head collapses to a single label -- measured: a
        # euclidean/cosine prototype head labelled a 25 s-latency service
        # "transition". Standard balanced-prior correction for imbalanced
        # few-shot classification.
        self.register_buffer("log_prior",
                             torch.zeros(len(self.classes)), persistent=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_prototype(self, class_name, vector, ema=0.2):
        idx = self.classes.index(class_name)
        new = torch.as_tensor(vector, dtype=self.prototypes.dtype)
        self.prototypes[idx] = (1 - ema) * self.prototypes[idx] + ema * new

    @torch.no_grad()
    def add_prototype(self, class_name, vector):
        if class_name not in self.classes:
            self.classes.append(class_name)
            p = torch.zeros((len(self.classes), self.prototypes.shape[1]),
                            dtype=self.prototypes.dtype)
            p[:self.prototypes.shape[0]] = self.prototypes
            self.prototypes = nn.Parameter(p)
        self.set_prototype(class_name, vector, ema=1.0)

    # ------------------------------------------------------------------
    def distances(self, z):
        """z: (B, D) -> (B, K) cosine distance to each prototype.

        Cosine rather than euclidean: embeddings for different services live at
        different magnitudes, and a euclidean prototype is dominated by the
        majority class. Cosine compares direction, which is what a bottleneck
        *pattern* is.
        """
        zn = F.normalize(z, dim=-1)
        pn = F.normalize(self.prototypes, dim=-1)
        return 1.0 - zn @ pn.t()

    def set_log_prior(self, counts):
        """Record log class frequencies so the logits can be prior-corrected."""
        tot = float(sum(counts.values())) or 1.0
        lp = torch.tensor([-math.log(max(counts.get(c, 1), 1) / tot)
                           for c in self.classes], dtype=self.prototypes.dtype)
        with torch.no_grad():
            self.log_prior.copy_(lp)

    def logits(self, z):
        return -self.distances(z) / self.temperature - self.log_prior

    def calibrate(self, z, labels):
        """Record the distance threshold above which an embedding is OOD."""
        d = self.distances(z)
        y = torch.as_tensor([self.classes.index(c) if c in self.classes else -1
                             for c in labels], device=z.device)
        ok = y >= 0
        if ok.any():
            own = d[ok, y[ok]]
            k = max(1, int(math.ceil(self.oood_quantile * own.numel())) - 1)
            self._dist_threshold = float(torch.kthvalue(own.flatten(), k).values)
        return self._dist_threshold

    # ------------------------------------------------------------------
    def forward(self, z):
        d = self.distances(z)
        probs = F.softmax(self.logits(z), dim=-1)
        conf, idx = probs.max(dim=-1)
        thr = self._dist_threshold if self._dist_threshold is not None else float(
            d.min(dim=-1).values.mean())
        nearest = d.min(dim=-1).values
        ood = (nearest > thr).float()
        return {
            "class_idx": idx,
            "classes": [self.classes[i] for i in idx.tolist()],
            "probabilities": probs,
            "confidence": conf,
            "ood": ood,
            "nearest_distance": nearest,
            "threshold": thr,
        }


class AddressabilityHead(nn.Module):
    """Independent P(adding replicas improves the service)."""

    def __init__(self, d_model, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        logit = self.net(z).squeeze(-1)
        return {
            "addressable_prob": torch.sigmoid(logit),
            "logit": logit,
        }


class AuroraModel(nn.Module):
    """Bundled encoder + both heads, with the output contract the controller uses."""

    def __init__(self, encoder, bottleneck_head=None, addressability_head=None):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck_head or PrototypicalHead(encoder.d_model)
        self.addressability = addressability_head or AddressabilityHead(
            encoder.d_model)

    @torch.no_grad()
    def decide(self, x, mask=None):
        """Full per-service inference.

        Returns one dict per row of the batch with the five required outputs:
        representation, bottleneck, addressability, confidence, OOD flag.
        """
        z = self.encoder.sequence(x, mask)
        bn = self.bottleneck(z)
        ad = self.addressability(z)
        out = []
        for i in range(z.shape[0]):
            out.append({
                "representation": z[i],
                "bottleneck": bn["classes"][i],
                "bottleneck_probs": bn["probabilities"][i],
                "addressable_prob": float(ad["addressable_prob"][i]),
                "confidence": float(bn["confidence"][i]),
                "ood": bool(bn["ood"][i]),
                "nearest_distance": float(bn["nearest_distance"][i]),
            })
        return out