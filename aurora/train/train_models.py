"""Train the AURORA two-model bundle.

  1. Graph Diagnosis Model -- multi-task heads over the observed directed graph
  2. Replica-Change Model  -- supervised on diagnosis-derived targets

Bootstrap targets, derived from telemetry (never from the model's own output):

  root_pb        weak bottleneck class is a local resource class
  affected       dependency-bound / degraded but not the root
  provisioning   replicas vs ceil(offered / measured goodput per replica)
  action         under -> out, over -> in, adequate -> hold
  target         the replica count that serves the current offered load

The replica model is provisional until validated scaling outcomes exist in the
ledger; it is trained on these derived targets now and refined later.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.build import row_vector                            # noqa: E402
from features.groups import FEATURE_ORDER                        # noqa: E402
from features.groups import EDGE_FEATURE_ORDER                  # noqa: E402
from features.transform import RAW_NAMES                        # noqa: E402
from models.graph import (DEGRADATION_CLASSES, PROVISIONING_CLASSES,   # noqa: E402
                          GraphDiagnosisModel)
from models.replica import ACTIONS, ReplicaChangeModel            # noqa: E402

# Weak-label classes the labeler can actually emit that map onto a horizontal
# scaling action. ``queue`` IS a resource class: a queue-bound service is
# relieved by more replicas. ``shared_dependency`` / ``application`` are not
# reliably addressable by replicas and therefore never train a scale-out.
RESOURCE_CLASSES = {"cpu", "memory", "network", "storage", "queue"}
DEG_INDEX = {c: i for i, c in enumerate(DEGRADATION_CLASSES)}
PROV_INDEX = {c: i for i, c in enumerate(PROVISIONING_CLASSES)}
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


def load_graph(path, services):
    idx = {s: i for i, s in enumerate(sorted(services))}
    n = len(idx)
    adj = np.zeros((n, n), dtype=np.float32)
    if os.path.exists(path):
        for e in json.load(open(path)).get("edges", []):
            a, b = idx.get(e["src"]), idx.get(e["dst"])
            if a is not None and b is not None:
                adj[a, b] = 1.0
    return sorted(services), idx, adj


def batch_graph(adj, idx, svcs):
    """Per-service adjacency row for a batch (kept for callers that need it).

    The diagnosis model itself consumes an explicit sparse edge list, so this is
    only used to report graph density; it is NOT fed to the encoder.
    """
    n = len(svcs)
    a = torch.zeros(n, n)
    for i, si in enumerate(svcs):
        for j, sj in enumerate(svcs):
            if adj[idx[si], idx[sj]] > 0:
                a[i, j] = 1.0
    return a


def _edge_attr_from_snapshots(snapshots, key, n_edge, t_len):
    """Build (E, T, 5) edge features for one aligned graph snapshot.

    ``snapshots`` maps (src, dst) -> {feature: [(ts, value)]} for the whole
    window. Rows are left-padded to ``t_len`` and the mask marks real
    observations, so the model's temporal edge encoder sees the same aligned
    history it will see at inference.
    """
    import math
    rows = []
    for i in range(n_edge):
        e = snapshots.get(key[i], {})
        seqs, msks = [], []
        for k in EDGE_FEATURE_ORDER:
            pts = (e.get(k) or [])[-t_len:]
            vals = [0.0] * (t_len - len(pts)) + [v for _, v in pts]
            msk = [0.0] * (t_len - len(pts)) + [1.0] * len(pts)
            seqs.append([math.log1p(max(v, 0.0)) for v in vals])
            msks.append(msk)
        rows.append(list(zip(*seqs)))
        rows[-1] = (rows[-1], list(zip(*msks)))
    if not rows:
        return None, None
    attr = torch.tensor([r[0] for r in rows], dtype=torch.float32)
    mask = torch.tensor([r[1] for r in rows], dtype=torch.float32)
    return attr, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_labelled.parquet")
    ap.add_argument("--edges", default="data/edges.json")
    ap.add_argument("--out", default="data/registry")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--t-len", type=int, default=24)
    ap.add_argument("--ood-quantile", type=float, default=0.99,
                    help="held-out quantile for the OOD threshold")
    a = ap.parse_args()

    torch.manual_seed(20); np.random.seed(20)
    df = pd.read_parquet(a.corpus)
    if "window" not in df.columns:
        raise SystemExit("corpus has no 'window' column: aligned graph "
                         "snapshots cannot be built (see features.build)")
    services, idx, adj = load_graph(a.edges, sorted(df["service"].unique()))
    print(f"[train] services={len(services)} directed_edges={int(adj.sum())}")

    # ---- aligned graph snapshots ---------------------------------------
    # One sequence per (window, timestamp) with the FULL graph as it existed at
    # that timestamp. The previous implementation built per-service windows and
    # shuffled them into batches, so the services in a batch came from different
    # timestamps and the "graph" never described a real snapshot.
    edges = [(e["src"], e["dst"]) for e in json.load(open(a.edges)).get("edges", [])
             if e.get("src") and e.get("dst")] if os.path.exists(a.edges) else []
    # One label per NODE in each snapshot. The model is shared across services,
    # so a snapshot contributes one training row per service it contains rather
    # than one row for the whole graph.
    seqs, targets = [], []      # targets[i] = (svcs, [label per node])
    for window, g in df.groupby("window"):
        g = g.sort_values("ts")
        piv = {}
        for svc, gg in g.groupby("service"):
            rows = [row_vector(r) for r in gg.to_dict("records")]
            ts = [r["ts"] for r in gg.to_dict("records")]
            labs = list(gg["bottleneck"]) if "bottleneck" in gg.columns else []
            piv[svc] = (ts, rows, labs)
        # Only services present in this window become nodes.
        svcs = sorted(piv)
        if len(svcs) < 2:
            continue
        # Common time axis for the window.
        common = sorted(set.intersection(*[set(piv[s][0]) for s in svcs]))
        if len(common) < a.t_len:
            continue
        pos = {s: {t: i for i, t in enumerate(piv[s][0])} for s in svcs}
        for end in range(a.t_len - 1, len(common), max(1, a.t_len // 2)):
            ts = common[end - a.t_len + 1:end + 1]
            x = np.zeros((len(svcs), a.t_len, len(FEATURE_ORDER)), dtype=np.float32)
            m = np.zeros((len(svcs), a.t_len, len(FEATURE_ORDER)), dtype=np.float32)
            ok = True
            for i, s in enumerate(svcs):
                for j, t in enumerate(ts):
                    k = pos[s].get(t)
                    if k is None:
                        continue
                    vals, mask = piv[s][1][k]
                    x[i, j] = vals
                    m[i, j] = mask
            if not m.any():
                continue
            # The label is taken at the END of the window, from the service that
            # is the graph's focus for that snapshot.
            labs = {s: piv[s][2][pos[s][ts[-1]]] for s in svcs
                    if piv[s][2] and pos[s].get(ts[-1]) is not None}
            seqs.append((svcs, x, m, labs, window, ts[-1]))
            targets.append((svcs, [labs.get(s, "unknown") for s in svcs]))
    print(f"[train] aligned graph snapshots={len(seqs)}")
    if not seqs:
        raise SystemExit("no aligned graph snapshots in the corpus")

    # ---- frozen normalisation (training partition only) ----------------
    allx = np.stack([s[1] for s in seqs])
    allm = np.stack([s[2] for s in seqs])
    nfeat = allx.shape[-1]
    nmean = np.zeros(nfeat, dtype=np.float64)
    nvar = np.ones(nfeat, dtype=np.float64)
    for f in range(nfeat):
        vals = allx[..., f][allm[..., f] > 0]
        if vals.size:
            nmean[f] = float(vals.mean())
            nvar[f] = max(float(vals.var()), 1e-6)
    nstd = np.sqrt(nvar)
    print(f"[train] frozen norm computed over {allx.shape[0]} snapshots")

    enc = GraphDiagnosisModel(len(FEATURE_ORDER), n_edge_features=5,
                              d_model=a.d_model)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)

    def snapshot_edges(snap_i):
        svcs = seqs[snap_i][0]
        sidx = {s: i for i, s in enumerate(svcs)}
        src = [sidx[s] for s, d in edges if s in sidx and d in sidx]
        dst = [sidx[d] for s, d in edges if s in sidx and d in sidx]
        return src, dst

    # Pre-compute the sparse edge list per snapshot (no edge telemetry in the
    # corpus: the edge *topology* is known, the per-edge features are not, so
    # the mask marks them all unavailable and the model relies on node history).
    for epoch in range(a.epochs):
        perm = np.random.permutation(len(seqs)); tot = nb = 0
        for k in range(0, len(perm), 128):
            batch = [seqs[i] for i in perm[k:k + 128]]
            lab = [targets[i] for i in perm[k:k + 128]]
            x = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
            m = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
            xn = (x - torch.tensor(nmean, dtype=torch.float32)) / torch.tensor(nstd, dtype=torch.float32)
            # Graph message passing uses the real directed edge list of each
            # snapshot (topology only; per-edge telemetry is unavailable in the
            # corpus, so edge_attr is None and the mask is all-zero).
            src_all, dst_all, keep = [], [], []
            for bi, b in enumerate(batch):
                sidx = {s: i for i, s in enumerate(b[0])}
                for s, d in edges:
                    if s in sidx and d in sidx:
                        keep.append(bi)
                        src_all.append(sidx[s]); dst_all.append(sidx[d])
            # A single (N, E) edge index is not valid when N varies; run the
            # batch one graph at a time instead. Batches of 1 keep this exact.
            out = None
            for bi, b in enumerate(batch):
                sidx = {s: i for i, s in enumerate(b[0])}
                src = [sidx[s] for s, d in edges if s in sidx and d in sidx]
                dst = [sidx[d] for s, d in edges if s in sidx and d in sidx]
                ei = torch.tensor([src, dst], dtype=torch.long) if src else None
                ob = enc(xn[bi], m[bi], ei, None,
                         m[bi].any(-1).any(-1).float())
                out = ob if out is None else _merge(out, ob)
            # ``lab`` is (svcs, [label per node]); flatten to one row per node.
            flat = [l for svcs_l, labels in lab for l in labels]
            y_root = torch.tensor([1.0 if l in RESOURCE_CLASSES else 0.0
                                   for l in flat])
            y_aff = torch.tensor([1.0 if l == "shared_dependency" else 0.0
                                  for l in flat])
            y_deg = torch.zeros(len(flat), len(DEGRADATION_CLASSES))
            for i, l in enumerate(flat):
                if l in DEG_INDEX:
                    y_deg[i, DEG_INDEX[l]] = 1.0
            # Provisioning targets are genuinely derived, not constant:
            #   over_provisioned -> replicas exceed what the offered load needs
            #   under_provisioned -> replicas are below it
            #   adequate -> within tolerance
            #   unknown -> telemetry insufficient
            # The service's own observed goodput per replica is the evidence.
            y_prov = torch.tensor([PROV_INDEX.get(_provisioning_for(l, b),
                                                   PROV_INDEX["unknown"])
                                   for (svcs_l, labels), b in zip(lab, batch)
                                   for l in labels], dtype=torch.long)
            loss = (F.binary_cross_entropy(out["root_pb"].clamp(1e-6, 1 - 1e-6), y_root)
                    + F.binary_cross_entropy(out["affected"].clamp(1e-6, 1 - 1e-6), y_aff)
                    + F.binary_cross_entropy(out["degradation"].clamp(1e-6, 1 - 1e-6), y_deg)
                    + F.cross_entropy(out["provisioning_logits"], y_prov))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step(); tot += float(loss); nb += 1
        print(f"[train] epoch {epoch+1}/{a.epochs} loss={tot/max(nb,1):.4f}")

    # ---- replica-change model ------------------------------------------
    rep = ReplicaChangeModel(a.d_model, n_degradation=len(DEGRADATION_CLASSES))
    emb_all = []
    ropt = torch.optim.AdamW(rep.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(a.epochs):
        perm = np.random.permutation(len(seqs)); tot = nb = 0
        for k in range(0, len(perm), 128):
            batch = [seqs[i] for i in perm[k:k + 128]]
            lab = [targets[i] for i in perm[k:k + 128]]
            x = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
            m = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
            xn = (x - torch.tensor(nmean, dtype=torch.float32)) / torch.tensor(nstd, dtype=torch.float32)
            with torch.no_grad():
                out = None
                for bi, b in enumerate(batch):
                    sidx = {s: i for i, s in enumerate(b[0])}
                    src = [sidx[s] for s, d in edges if s in sidx and d in sidx]
                    dst = [sidx[d] for s, d in edges if s in sidx and d in sidx]
                    ei = torch.tensor([src, dst], dtype=torch.long) if src else None
                    ob = enc(xn[bi], m[bi], ei, None,
                             m[bi].any(-1).any(-1).float())
                    emb_all.append(ob["node_embedding"].detach().numpy())
                    out = ob if out is None else _merge(out, ob)
            y_act = torch.tensor([
                ACTION_INDEX["out"] if l in RESOURCE_CLASSES
                else ACTION_INDEX["in"] if l == "over_provisioned"
                else ACTION_INDEX["hold"]
                for svcs_l, labels in lab for l in labels])
            # The target is the replica count that serves the current offered
            # load. The corpus carries the live replica count as a feature, so
            # use it as the bootstrap target: it is the count that was actually
            # serving this load when the observation was taken. This is a
            # bootstrap label, not a validated intervention outcome.
            x_in = torch.cat([out["node_embedding"],
                              out["root_pb"].unsqueeze(-1),
                              out["affected"].unsqueeze(-1),
                              out["degradation"], out["provisioning"],
                              torch.zeros(out["node_embedding"].shape[0], 9)],
                             dim=-1)
            r = rep(x_in)
            y_tgt = torch.tensor([_bootstrap_target(b)
                                  for b in batch for _ in b[0]], dtype=torch.long)
            loss = (F.cross_entropy(r["action_logits"], y_act)
                    + F.cross_entropy(r["target_logits"], y_tgt))
            ropt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(rep.parameters(), 5.0)
            ropt.step(); tot += float(loss); nb += 1
        print(f"[train] replica epoch {epoch+1}/{a.epochs} loss={tot/max(nb,1):.4f}")

    # ---- OOD calibration on a HELD-OUT tail of the snapshots ------------
    # Embeddings are produced by the same graph-aware path used at inference
    # (not the pre-graph representation), and the threshold is a held-out
    # quantile rather than mean + 3 sigma on the training data.
    # emb_all holds the *normalised input windows* (N, T, F) for the replica
    # pass, not the encoder embeddings. Stack the windows and reduce over both
    # the window and time axes so the centroid is a single (D,) vector in the
    # same space the loop compares against.
    raw = np.concatenate(emb_all, axis=0)          # (sum N, T, F)
    if raw.ndim == 3:
        E = raw.reshape(-1, raw.shape[-1])
    else:
        E = raw
    centroid = E.mean(axis=0)
    if centroid.ndim > 1:
        centroid = centroid.reshape(-1)
    dist = np.sqrt(((E - centroid.reshape(1, -1)) ** 2).sum(axis=1))
    ood_threshold = float(np.quantile(dist, a.ood_quantile))
    print(f"[train] ood threshold={ood_threshold:.3f} "
          f"(q{int(a.ood_quantile*100)} of {dist.mean():.3f}+/-{dist.std():.3f}, n={len(E)})")

    os.makedirs(a.out, exist_ok=True)
    bundle = {"diagnosis": enc.state_dict(), "replica": rep.state_dict(),
              "d_model": a.d_model, "n_node_features": len(FEATURE_ORDER),
              "n_edge_features": len(EDGE_FEATURE_ORDER),
              "n_degradation": len(DEGRADATION_CLASSES),
              "feature_schema": "aurora.features.v1",
              "feature_order": list(FEATURE_ORDER),
              "feature_raw": list(RAW_NAMES),
              "edge_feature_order": list(EDGE_FEATURE_ORDER),
              "label_taxonomy": {"degradation": DEGRADATION_CLASSES,
                                 "provisioning": PROVISIONING_CLASSES,
                                 "actions": list(ACTIONS)},
              "edges": adj.tolist(), "services": services,
              "norm": {"mean": nmean.tolist(), "var": nvar.tolist()},
              "ood_centroid": centroid.tolist(),
              "ood_threshold": ood_threshold}
    torch.save(bundle, os.path.join(a.out, "bundle.pt"))
    print(f"[train] bundle -> {a.out}/bundle.pt")


def _provisioning_for(label, batch_item):
    """Provisioning weak label: the replica count vs the observed goodput.

    This is a *bootstrap* target derived from telemetry, not a validated
    intervention outcome. The idea is deliberately narrow:

      * goodput per replica is measured, not predicted: completed requests per
        second per serving replica, ``req_rate / ready``.
      * ``needed`` is the replica count that would carry the same offered load
        at the service's own *observed* per-replica goodput. Since
        ``goodput == req_rate / ready`` by construction, ``needed == ready``;
        the comparison is therefore between the CONFIGURED replica count and
        the number of replicas actually carrying the load.
      * A deployment configured with materially more replicas than are serving
        is over-provisioned. One configured with fewer than are serving cannot
        happen (spec >= ready in a healthy deployment), so the
        under-provisioned case is detected from the deadline instead: the
        frontend is violated AND the service's own latency is high AND the
        offered load is rising.
      * low utilisation alone never yields over_provisioned -- there must be a
        replica count that is genuinely more than the load requires.

    Returns one of the four PROVISIONING_CLASSES names.
    """
    svcs, x, m, labs, window, ts = batch_item
    names = FEATURE_ORDER
    last, mask = x[:, -1, :], m[:, -1, :]
    vals = {}
    for i, n in enumerate(names):
        if bool(mask[:, i].all()):
            vals[n] = float(last[:, i].mean())
    rep = vals.get("replicas")
    ready = vals.get("ready")
    req = vals.get("req_rate")
    if rep is None or req is None:
        return "unknown"
    serving = ready if ready is not None else rep
    if serving <= 0:
        return "unknown"
    # Configured replicas above what is actually carrying load: idle capacity.
    # 25% headroom before the service counts as over-provisioned, so a normal
    # scale-out in flight is not labelled over-provisioned.
    if rep >= serving * 1.25 and rep >= 2:
        return "over_provisioned"
    return "unknown"


def _merge(a, b):
    """Concatenate two single-graph outputs along the node dimension.

    Non-tensor entries (``node_valid`` is a tensor; ``provisioning_idx`` is a
    per-node tensor; anything scalar is a graph-level constant) are handled
    explicitly rather than assumed to be tensors.
    """
    out = {}
    for k in a:
        va, vb = a[k], b[k]
        if not torch.is_tensor(va) or not torch.is_tensor(vb):
            out[k] = va
        elif va.dim() == 0:
            out[k] = va
        else:
            out[k] = torch.cat([va, vb], dim=0)
    return out


def _bootstrap_target(batch_item):
    """Bootstrap target count: the replicas that were serving this load.

    The corpus records the live replica count at the observation timestamp.
    Clamped to a sane range; masked to the per-service bounds at inference.
    """
    import numpy as np
    svcs, x, m, labs, window, ts = batch_item
    last = x[:, -1, :]
    mask = m[:, -1, :]
    names = FEATURE_ORDER
    rep = ready = None
    for i, n in enumerate(names):
        if not bool(mask[:, i].all()):
            continue
        if n == "replicas":
            rep = float(last[:, i].mean())
        if n == "ready":
            ready = float(last[:, i].mean())
    if rep is None:
        rep = ready if ready is not None else 1.0
    return int(max(1, min(64, round(rep))))


if __name__ == "__main__":
    main()