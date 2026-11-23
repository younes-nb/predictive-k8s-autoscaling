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
        """(N, T, F) -> (N, D): one embedding per service.

        ``mask`` is (N, T, F). A timestep with no available feature must not
        influence the recurrent state: it is zeroed *before* the GRU, so the
        hidden state only ever advances on real observations. The previous
        implementation masked after the GRU, which let missing steps alter the
        state that later valid steps were computed from.
        """
        if mask is None:
            h, _ = self.node_rnn(self.node_proj(x))
            return self.norm(h[:, -1])
        m = mask.to(x.dtype)
        xm = x * m                                  # unavailable -> 0
        h, _ = self.node_rnn(self.node_proj(xm))
        # A step is valid when ANY feature is available. If nothing at all is
        # available, no step is valid and the embedding is the zero vector.
        tm = m.any(dim=-1).to(h.dtype) if m.numel() else h.new_zeros(h.shape[:2])
        if tm.sum() <= 0:
            return self.norm(torch.zeros_like(h[:, 0]))
        tm = tm.unsqueeze(-1)                       # N,T,1
        s = (h * tm).sum(1) / tm.sum(1).clamp_min(1.0)
        return self.norm(s)

    # ------------------------------------------------------------------
    def encode_edges_dense(self, x, mask=None):
        """(E, T, FE) -> (E, D) -- the temporal edge encoder.

        ``x`` is one history per observed edge. The GRU is run over the time
        axis and the *valid* timesteps are pooled; a missing timestep is zeroed
        before the GRU and excluded from the pooling average (mask applied to
        the time axis, not after a mean that already included the gaps).
        """
        if x.dim() == 2:
            # A single observation per edge: still one GRU step so the edge
            # embedding lives in the same space as the temporal path.
            x = x.unsqueeze(1)
            mask = None if mask is None else mask.reshape(x.shape[0], 1, -1)
        if mask is None:
            h, _ = self.edge_rnn(self.edge_proj(x))
            return self.norm(h[:, -1])
        m = mask.to(x.dtype)
        h, _ = self.edge_rnn(self.edge_proj(x * m))
        # Pool over the time axis only. ``mask`` is per-feature (E, T, FE), so
        # the time validity is ``any`` over the feature axis -- reducing over
        # the wrong axis produced a (E, FE)-shaped denominator against an
        # (E, D) numerator.
        tm = m.any(dim=-1).to(h.dtype)                  # (E, T)
        w = tm.unsqueeze(-1)                            # (E, T, 1)
        s = (h * w).sum(1) / w.sum(1).clamp_min(1.0)    # (E, D)
        return self.norm(s)

    def encode_edges_sparse(self, feat, mask=None):
        """(E, FE) or (E, T, FE) -> (E, D): one embedding per observed edge.

        Uses the SAME temporal encoder as the dense path so training and
        inference share one representation.
        """
        return self.encode_edges_dense(feat, mask)

    # ------------------------------------------------------------------
    def forward(self, node_x, node_mask, edge_index=None, edge_attr=None,
                node_valid=None, edge_mask=None):
        """Diagnose one graph snapshot.

        ``node_x``    (N, T, F)   node feature history
        ``node_mask`` (N, T, F)   per-feature availability
        ``edge_index`` (2, E)     long tensor of (src, dst) node indices
        ``edge_attr``  (E, T, FE) or (E, FE)  edge feature history
        ``edge_mask``  (E, T, FE) per-feature availability for edges
        ``node_valid`` (N,)       optional explicit validity gate

        Nodes with no valid observation are *held out*: their embedding and
        every output are forced to a neutral "unknown" state rather than being
        read off a zero embedding (a linear head has a bias, so a zero input
        still produces a confident-looking probability).
        """
        h = self.encode_nodes(node_x, node_mask)
        n = h.shape[0]
        node_ok = node_mask.any(dim=-1).any(dim=-1)
        if node_valid is not None:
            node_ok = node_ok & (node_valid > 0.5)
        if edge_index is not None and edge_attr is not None and edge_attr.numel():
            e = self.encode_edges_sparse(edge_attr.to(h.dtype), edge_mask)
            src, dst = edge_index[0], edge_index[1]
            # Keep only edges whose endpoints are both valid.
            keep = node_ok[src] & node_ok[dst]
            src, dst, e = src[keep], dst[keep], e[keep]
        else:
            e = None

        # Two directed message-passing layers. ``edge_index[0] -> edge_index[1]``
        # means caller -> callee. Messages are degree-normalized (mean, not sum)
        # so a service with many neighbours does not accumulate a systematically
        # larger embedding than one with two.
        for _ in range(2):
            agg_in = torch.zeros_like(h)
            agg_out = torch.zeros_like(h)
            if e is not None and e.numel():
                # A single edge observation per edge is broadcast over the
                # time axis; a history is already (E, T, D) and is pooled.
                if e.dim() == 2:
                    msg = h[src] + e
                    nbrs = h[dst] + e
                else:
                    msg = h[src].unsqueeze(1) + e
                    nbrs = h[dst].unsqueeze(1) + e
                agg_in = agg_in.index_add(0, dst, msg)
                agg_out = agg_out.index_add(0, src, nbrs)
                ones = torch.ones(src.shape[0], device=h.device, dtype=h.dtype)
                deg_in = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
                deg_out = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
                deg_in = deg_in.index_add(0, dst, ones)
                deg_out = deg_out.index_add(0, src, ones)
                agg_in = agg_in / deg_in.clamp_min(1.0).unsqueeze(-1)
                agg_out = agg_out / deg_out.clamp_min(1.0).unsqueeze(-1)
            m_in = self.msg_in(torch.cat([h, self.norm(agg_in)], dim=-1))
            m_out = self.msg_out(torch.cat([h, self.norm(agg_out)], dim=-1))
            h = self.norm(h + F.gelu(m_in) + F.gelu(m_out))

        # Invalid nodes are masked out of the message passing so they neither
        # send nor receive evidence, and their outputs are neutralised below.
        h = h * node_ok.unsqueeze(-1).to(h.dtype)

        deg = torch.sigmoid(self.head_degradation(h))
        prov_logits = self.head_provision(h)
        prov = F.softmax(prov_logits, dim=-1)
        unknown_deg = len(DEGRADATION_CLASSES) - 1
        return {
            "node_embedding": h,
            "node_valid": node_ok.to(h.dtype),
            "root_pb": torch.sigmoid(self.head_root(h).squeeze(-1)),
            "affected": torch.sigmoid(self.head_affected(h).squeeze(-1)),
            "degradation": deg,
            "provisioning_logits": prov_logits,
            "provisioning": prov,
            # Explicit states for an unusable observation. Callers must check
            # ``node_valid`` and must not read the probabilities above as a
            # diagnosis.
            "provisioning_idx": prov.argmax(dim=-1),
            "unknown_degradation_idx": unknown_deg,
        }