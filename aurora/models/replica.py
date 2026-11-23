"""Replica-Change Model.

Separate from the diagnosis model. It sees the diagnosis output plus the current
system state and selects an action and a target replica count.

    inputs  = [diagnosis embedding | root_pb | affected | degradation(9)
               | provisioning(4) | log(offered+1) | log(capacity+1) | sigma
               | replicas | committed | min | max | completion | p99_ratio]
    outputs = action (hold | out | in)
              target logits over [0, max_replicas]

There is no forecasting input and no forecasting output: ``offered`` is the
observed current rate, ``capacity`` is the posterior over observed goodput per
replica, and the target is a count, not a predicted resource value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ["hold", "out", "in"]
N_ACTIONS = len(ACTIONS)
MAX_TARGET = 32          # logits are produced for 0..MAX_TARGET and masked


class ReplicaChangeModel(nn.Module):
    def __init__(self, d_model, n_degradation=9, n_provision=4, hidden=64,
                 max_target=MAX_TARGET):
        super().__init__()
        self.max_target = max_target
        in_dim = (d_model + 2 + n_degradation + n_provision + 9)
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
            return torch.tensor([x if x is not None else d])
        scalar = torch.tensor([
            math.log1p(max(offered, 0.0)),
            math.log1p(max(capacity, 0.0)),
            max(sigma, 0.0),
            float(replicas), float(committed), float(min_pods), float(max_pods),
            min(max(completion, 0.0), 1.0),
            float(p99_ratio if p99_ratio is not None else 1.0),
        ])
        return torch.cat(parts + [scalar.unsqueeze(0)], dim=-1)

    # ------------------------------------------------------------------
    def forward(self, x, min_pods=None, max_pods=None):
        h = self.net(x)
        action_logits = self.head_action(h)
        target_logits = self.head_target(h)
        if min_pods is not None and max_pods is not None:
            mask = torch.zeros_like(target_logits)
            lo = int(min_pods)
            hi = min(int(max_pods), self.max_target)
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
        import torch
        with torch.no_grad():
            out = self.forward(x, min_pods, max_pods)
        a = out["action"].argmax(dim=-1).item()
        t = out["target_probs"].argmax(dim=-1).item()
        return ACTIONS[a], int(t), out