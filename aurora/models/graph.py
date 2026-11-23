"""Shared Graph Diagnosis Model.

A directed, heterogeneous message-passing network over the service graph.

  node history  (N, T, F)   -> temporal encoder -> h_i
  edge history  (E, T, FE)  -> edge encoder     -> e_ij
  2 message-passing layers over the DIRECTED graph
  heads:
    root_pb         P(this service is the root performance bottleneck)
    affected        P(this service is degraded *because of* a dependency)
    degradation     multi-label local cause (9 independent sigmoids)
    provisioning    4-way: under / adequate / over / unknown
    ood             distance from the training distribution

The model is shared across services and deployments: one set of weights, a
variable-size graph, and no service-name features.

It never outputs a forecast. Everything it produces describes the graph as it
is observed in this cycle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEGRADATION_CLASSES = [
    "cpu", "memory", "network", "storage", "queue",
    "shared_dependency", "application", "infrastructure", "unknown",
]
PROVISIONING_CLASSES = ["under_provisioned", "adequate", "over_provisioned", "unknown"]


class GraphDiagnosisModel(nn.Module):
    def __init__(self, n_node_features, n_edge_features, d_model=64,
                 n_degradation=len(DEGRADATION_CLASSES)):
        super().__init__()
        self.d_model = d_model
        self.n_degradation = n_degradation

        self.node_proj = nn.Linear(n_node_features, d_model)
        self.node_rnn = nn.GRU(d_model, d_model, num_layers=1,
                               batch_first=True)
        self.edge_proj = nn.Linear(n_edge_features, d_model)
        self.edge_rnn = nn.GRU(d_model, d_model, num_layers=1,
                               batch_first=True)

        self.msg_in = nn.Linear(2 * d_model, d_model)
        self.msg_out = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.head_root = nn.Linear(d_model, 1)
        self.head_affected = nn.Linear(d_model, 1)
        self.head_degradation = nn.Linear(d_model, n_degradation)
        self.head_provision = nn.Linear(d_model, len(PROVISIONING_CLASSES))

    # ------------------------------------------------------------------
    def encode_nodes(self, x, mask=None):
        """(N, T, F) -> (N, D): one embedding per service."""
        h, _ = self.node_rnn(self.node_proj(x))          # N,T,D
        if mask is None:
            return self.norm(h[:, -1])
        m = mask.any(dim=-1).to(h.dtype).unsqueeze(-1)   # N,T,1
        s = (h * m).sum(1) / m.sum(1).clamp_min(1.0)     # N,D
        return self.norm(s)

    def encode_edges(self, x, mask=None):
        h, _ = self.edge_rnn(self.edge_proj(x))
        h = h.mean(dim=1)
        if mask is not None:
            m = mask.any(dim=-1).to(h.dtype).unsqueeze(-1)
            h = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return h

    # ------------------------------------------------------------------
    def forward(self, node_x, node_mask, adj, edge_h, node_valid):
        """node_x (N,T,F); adj (N,N) dense directed; edge_h (N,N,D)."""
        h = self.encode_nodes(node_x, node_mask)
        # Two directed message-passing layers. ``adj[j, i] = 1`` means j -> i.
        for _ in range(2):
            # incoming: aggregate over predecessors j -> i
            agg_in = torch.einsum("ji,jd->id", adj, h)
            m_in = self.msg_in(torch.cat([h, self.norm(agg_in)], dim=-1))
            # outgoing: aggregate over successors i -> j
            agg_out = torch.einsum("ij,jd->id", adj, h)
            m_out = self.msg_out(torch.cat([h, self.norm(agg_out)], dim=-1))
            h = self.norm(h + F.gelu(m_in) + F.gelu(m_out)) * node_valid.unsqueeze(-1)

        return {
            "node_embedding": h,
            "root_pb": torch.sigmoid(self.head_root(h).squeeze(-1)),
            "affected": torch.sigmoid(self.head_affected(h).squeeze(-1)),
            "degradation": torch.sigmoid(self.head_degradation(h)),
            "provisioning_logits": self.head_provision(h),
            "provisioning": F.softmax(self.head_provision(h), dim=-1),
        }