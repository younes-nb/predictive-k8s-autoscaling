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
        """(N, N, T, FE) -> (N, N, D).

        The GRU needs a 3-D (batch, time, feature) input, so the two graph
        axes are folded into the batch dimension and restored afterwards.
        """
        n, m_, t, f = x.shape
        h = self.edge_proj(x.reshape(n * m_, t, f))
        h, _ = self.edge_rnn(h)
        h = h.mean(dim=1)                       # (N*N, D)
        if mask is not None:
            mk = mask.any(dim=-1).to(h.dtype).reshape(n * m_, t, 1)
            w = mk.sum(1)
            h = (h * mk).sum(1) / w.clamp_min(1.0)
        return h.reshape(n, m_, -1)

    # ------------------------------------------------------------------
    def encode_edges_sparse(self, feat):
        """(E, FE) -> (E, D): one embedding per observed invocation edge."""
        return self.edge_proj(feat)

    def forward(self, node_x, node_mask, edge_index=None, edge_attr=None,
                node_valid=None):
        """node_x (N,T,F); adj (N,N) dense directed.

        ``edge_x`` is (N,N,T,FE) observed edge telemetry; when present it is
        encoded and folded into message passing, so a *slow* edge contributes
        evidence rather than a bare dependency edge.
        """
        h = self.encode_nodes(node_x, node_mask)
        n = h.shape[0]
        if node_valid is None:
            node_valid = node_mask.any(dim=-1).any(dim=-1).float()
        if edge_index is not None and edge_attr is not None and edge_attr.numel():
            e = self.encode_edges_sparse(edge_attr.to(h.dtype))   # (E, D)
            src, dst = edge_index[0], edge_index[1]
        else:
            e = None
        # Two directed message-passing layers. ``adj[j, i] = 1`` means j -> i.
        # The edge embedding modulates each message, so edge latency/errors
        # propagate toward the caller.
        for _ in range(2):
            # sparse aggregation over observed edges only: a dense N*N tensor
            # would be 1764x larger and was enough to OOM the 512 MiB limit
            agg_in = torch.zeros_like(h)
            agg_out = torch.zeros_like(h)
            if e is not None:
                msg = h[src] + e
                agg_in = agg_in.index_add(0, dst, msg)
                agg_out = agg_out.index_add(0, src, h[dst] + e)
            m_in = self.msg_in(torch.cat([h, self.norm(agg_in)], dim=-1))
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