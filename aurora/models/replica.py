"""Replica-Change Model.

Separate from the diagnosis model. It sees the diagnosis output plus the current
system state and selects an action and a target replica count.

    inputs  = [diagnosis embedding | root_pb | affected | degradation(9)
               | provisioning(4) | log(offered+1) | log(capacity+1) | sigma
               | replicas | committed | min | max | completion | p99_ratio]
    outputs = action (hold | out | in) with a confidence
              target logits over [min_pods, max_pods] with a confidence

There is no forecasting input and no forecasting output: ``offered`` is the
observed current rate, ``capacity`` is the observed goodput per replica and its
uncertainty, and the target is a count, not a predicted resource value.

Status: **bootstrap**. The model is currently trained on telemetry-derived
targets because no validated scaling outcomes exist yet. Its recommendations
must not be read as learned scaling-benefit decisions until the ledger contains
settled before/after outcomes (see ``feedback/ledger.py``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ["hold", "out", "in"]
N_ACTIONS = len(ACTIONS)
# Widest target the model can express. Actual targets are clamped to the
# per-service [min_pods, max_pods] supplied by the caller, which comes from the
# live (or saved) HPA manifests -- never from a global constant.
MAX_TARGET = 64

# Number of scalar features appended after the diagnosis block.
N_SCALARS = 9


class ReplicaChangeModel(nn.Module):
    def __init__(self, d_model, n_degradation=9, n_provision=4, hidden=64,
                 max_target=MAX_TARGET):
        super().__init__()
        self.max_target = max_target
        in_dim = (d_model + 2 + n_degradation + n_provision + N_SCALARS)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head_action = nn.Linear(hidden, N_ACTIONS)
        self.head_target = nn.Linear(hidden, max_target + 1)

    # ------------------------------------------------------------------
    @staticmethod
    def build_input(diag, offered, capacity, sigma, replicas, committed,
                    min_pods, max_pods, completion, p99_ratio):
        import math
        parts = [
            diag["node_embedding"],
            diag["root_pb"].unsqueeze(-1),
            diag["affected"].unsqueeze(-1),
            diag["degradation"],
            diag["provisioning"],
        ]

        def s(x, d=0.0):
            if x is None:
                x = d
            try:
                x = float(x)
            except (TypeError, ValueError):
                x = d
            if x != x:
                x = d
            return torch.tensor([x])

        scalar = torch.tensor([
            math.log1p(max(float(offered or 0.0), 0.0)),
            math.log1p(max(float(capacity or 0.0), 0.0)),
            max(float(sigma or 0.0), 0.0),
            s(replicas),
            s(committed),
            s(min_pods),
            s(max_pods),
            min(max(float(completion if completion is not None else 1.0), 0.0), 1.0),
            float(p99_ratio if p99_ratio is not None else 1.0),
        ])
        return torch.cat(parts + [scalar.unsqueeze(0)], dim=-1)

    # ------------------------------------------------------------------
    def forward(self, x, min_pods=None, max_pods=None):
        h = self.net(x)
        action_logits = self.head_action(h)
        target_logits = self.head_target(h)
        lo = int(min_pods) if min_pods is not None else 0
        hi = int(min(max_pods, self.max_target)) if max_pods is not None \
            else self.max_target
        hi = max(hi, lo)
        mask = torch.zeros_like(target_logits)
        mask[:, lo:hi + 1] = 1.0
        target_logits = target_logits.masked_fill(mask == 0, -1e9)
        return {
            "action_logits": action_logits,
            "action": F.softmax(action_logits, dim=-1),
            "target_logits": target_logits,
            "target_probs": F.softmax(target_logits, dim=-1),
        }

    # ------------------------------------------------------------------
    def decide(self, x, min_pods, max_pods):
        """Return (action, target, info).

        ``info`` carries the action and target probabilities so the caller can
        apply its own confidence floor -- the diagnosis confidence must not be
        used as a proxy for the replica decision's confidence.
        """
        with torch.no_grad():
            out = self.forward(x, min_pods, max_pods)
        a = out["action"].argmax(dim=-1).item()
        t = out["target_probs"].argmax(dim=-1).item()
        info = {
            "action_conf": float(out["action"].max().item()),
            "target_conf": float(out["target_probs"].max().item()),
            "action_logits": out["action_logits"],
            "target_logits": out["target_logits"],
        }
        return ACTIONS[a], int(t), info