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
from models.graph import (DEGRADATION_CLASSES, PROVISIONING_CLASSES,   # noqa: E402
                          GraphDiagnosisModel)
from models.replica import ACTIONS, ReplicaChangeModel            # noqa: E402

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
    n = len(svcs)
    a = torch.zeros(n, n)
    for i, si in enumerate(svcs):
        for j, sj in enumerate(svcs):
            if adj[idx[si], idx[sj]] > 0:
                a[i, j] = 1.0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_labelled.parquet")
    ap.add_argument("--edges", default="data/edges.json")
    ap.add_argument("--out", default="data/registry")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--t-len", type=int, default=24)
    a = ap.parse_args()

    torch.manual_seed(20); np.random.seed(20)
    df = pd.read_parquet(a.corpus)
    services, idx, adj = load_graph(a.edges, sorted(df["service"].unique()))
    print(f"[train] services={len(services)} directed_edges={int(adj.sum())}")

    seqs, targets = [], []
    for svc in services:
        g = df[df["service"] == svc].sort_values("ts")
        rows = [row_vector(r) for r in g.to_dict("records")]
        if len(rows) < a.t_len:
            continue
        x = np.stack([r[0] for r in rows]); m = np.stack([r[1] for r in rows])
        labs = list(g["bottleneck"]) if "bottleneck" in g.columns else []
        for i in range(0, len(rows) - a.t_len + 1, max(1, a.t_len // 2)):
            seqs.append((svc, x[i:i + a.t_len], m[i:i + a.t_len]))
            targets.append(labs[i + a.t_len - 1] if labs else "unknown")
    print(f"[train] sequences={len(seqs)}")

    # Frozen normalisation: computed on the training corpus and shipped in the
    # bundle, so inference uses exactly the representation the encoder saw.
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
    print(f"[train] frozen norm computed over {allx.shape[0]} windows")

    enc = GraphDiagnosisModel(len(FEATURE_ORDER), n_edge_features=5,
                              d_model=a.d_model)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(a.epochs):
        perm = np.random.permutation(len(seqs)); tot = nb = 0
        for k in range(0, len(perm), 128):
            batch = [seqs[i] for i in perm[k:k + 128]]
            lab = [targets[i] for i in perm[k:k + 128]]
            svcs = [b[0] for b in batch]
            x = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
            m = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
            adj_b = batch_graph(adj, idx, svcs)
            valid = m.any(-1).any(-1).float()
            xn = (x - torch.tensor(nmean, dtype=torch.float32)) / torch.tensor(nstd, dtype=torch.float32)
            out = enc(xn, m, None, None, valid)

            y_root = torch.tensor([1.0 if l in RESOURCE_CLASSES else 0.0 for l in lab])
            y_aff = torch.tensor([1.0 if l == "shared_dependency" else 0.0 for l in lab])
            y_deg = torch.zeros(len(lab), len(DEGRADATION_CLASSES))
            for i, l in enumerate(lab):
                if l in DEG_INDEX:
                    y_deg[i, DEG_INDEX[l]] = 1.0
            y_prov = torch.full((len(lab),), PROV_INDEX["unknown"], dtype=torch.long)

            loss = (F.binary_cross_entropy(out["root_pb"].clamp(1e-6, 1 - 1e-6), y_root)
                    + F.binary_cross_entropy(out["affected"].clamp(1e-6, 1 - 1e-6), y_aff)
                    + F.binary_cross_entropy(out["degradation"].clamp(1e-6, 1 - 1e-6), y_deg)
                    + F.cross_entropy(out["provisioning_logits"], y_prov))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step(); tot += float(loss); nb += 1
        print(f"[train] epoch {epoch+1}/{a.epochs} loss={tot/max(nb,1):.4f}")

    # ---- replica-change model -------------------------------------------
    rep = ReplicaChangeModel(a.d_model, n_degradation=len(DEGRADATION_CLASSES))
    emb_all = []
    ropt = torch.optim.AdamW(rep.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(a.epochs):
        perm = np.random.permutation(len(seqs)); tot = nb = 0
        for k in range(0, len(perm), 128):
            batch = [seqs[i] for i in perm[k:k + 128]]
            lab = [targets[i] for i in perm[k:k + 128]]
            svcs = [b[0] for b in batch]
            x = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
            m = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
            adj_b = batch_graph(adj, idx, svcs)
            with torch.no_grad():
                xn = (x - torch.tensor(nmean, dtype=torch.float32)) / torch.tensor(nstd, dtype=torch.float32)
                out = enc(xn, m, None, None, m.any(-1).any(-1).float())
            y_act = torch.tensor([
                ACTION_INDEX["out"] if l in RESOURCE_CLASSES
                else ACTION_INDEX["in"] if l == "over_provisioned"
                else ACTION_INDEX["hold"] for l in lab])
            emb_all.append(out["node_embedding"].numpy())
            x_in = torch.cat([out["node_embedding"],
                              out["root_pb"].unsqueeze(-1),
                              out["affected"].unsqueeze(-1),
                              out["degradation"], out["provisioning"],
                              torch.zeros(len(lab), 9)], dim=-1)
            r = rep(x_in)
            y_tgt = torch.zeros(len(lab), dtype=torch.long)   # masked by bounds
            loss = (F.cross_entropy(r["action_logits"], y_act)
                    + F.cross_entropy(r["target_logits"], y_tgt))
            ropt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(rep.parameters(), 5.0)
            ropt.step(); tot += float(loss); nb += 1
        print(f"[train] replica epoch {epoch+1}/{a.epochs} loss={tot/max(nb,1):.4f}")

    # ---- OOD calibration: distance of training embeddings from their centroid
    E = np.concatenate(emb_all, axis=0)
    centroid = E.mean(axis=0)
    dist = np.sqrt(((E - centroid) ** 2).sum(axis=1))
    ood_threshold = float(dist.mean() + 3.0 * dist.std())
    print(f"[train] ood threshold={ood_threshold:.3f} (mean={dist.mean():.3f} "
          f"std={dist.std():.3f}, n={len(E)})")

    os.makedirs(a.out, exist_ok=True)
    torch.save({"diagnosis": enc.state_dict(), "replica": rep.state_dict(),
                "d_model": a.d_model, "n_node_features": len(FEATURE_ORDER),
                "n_degradation": len(DEGRADATION_CLASSES),
                "edges": adj.tolist(), "services": services,
                "norm": {"mean": nmean.tolist(), "var": nvar.tolist()},
                "ood_centroid": centroid.tolist(),
                "ood_threshold": ood_threshold,
                "ood_centroid": centroid.tolist(),
                "ood_threshold": ood_threshold},
               os.path.join(a.out, "bundle.pt"))
    print(f"[train] bundle -> {a.out}/bundle.pt")


if __name__ == "__main__":
    main()