"""Self-supervised pretraining of the shared temporal encoder.

TS2Vec-style instance discrimination over unlabelled 5 s windows from every
service in the corpus. No labels are used here: the bottleneck knowledge lives
in the prototypical head, which is calibrated separately.

Outputs (written next to ``--out``):

  * ``encoder.pt``          -- state_dict
  * ``norm.json``           -- global feature mean/var used for the prior
  * ``feature_schema.json`` -- feature order and groups
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.build import stack_windows                    # noqa: E402
from features.groups import FEATURE_ORDER                   # noqa: E402
from models.encoder import TS2VecEncoder, ts2vec_loss       # noqa: E402


def fit_norm(x, mask):
    """Global mean/var per feature over observed entries."""
    n_feat = x.shape[-1]
    mean = np.zeros(n_feat, dtype=np.float64)
    var = np.ones(n_feat, dtype=np.float64)
    for f in range(n_feat):
        vals = x[..., f][mask[..., f] > 0]
        if vals.size:
            mean[f] = float(vals.mean())
            var[f] = max(float(vals.var()), 1e-6)
    return {"mean": mean.tolist(), "var": var.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus_labelled.parquet")
    ap.add_argument("--out", default="../data/encoder.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--t-len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    import pandas as pd
    df = pd.read_parquet(a.corpus)
    x, m, svcs = stack_windows(df, a.t_len)
    print(f"[selfsup] windows={x.shape[0]} T={x.shape[1]} F={x.shape[2]} "
          f"services={len(set(svcs))}")
    if x.shape[0] < 8:
        raise SystemExit("not enough windows to train")

    norm = fit_norm(x, m)
    mean = torch.tensor(norm["mean"], dtype=torch.float32)
    std = torch.tensor(np.sqrt(norm["var"]), dtype=torch.float32)
    xn = (x - mean.numpy()) / std.numpy()

    xt = torch.tensor(xn, dtype=torch.float32)
    mt = torch.tensor(m, dtype=torch.float32)

    enc = TS2VecEncoder(x.shape[2], d_model=a.d_model).train()
    opt = torch.optim.AdamW(enc.parameters(), lr=a.lr, weight_decay=1e-4)

    n = xt.shape[0]
    for epoch in range(a.epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, a.batch_size):
            idx = perm[i:i + a.batch_size]
            xb, mb = xt[idx], mt[idx]
            if xb.shape[0] < 4:
                continue
            loss = ts2vec_loss(xb, mb, enc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        if nb:
            print(f"[selfsup] epoch {epoch + 1}/{a.epochs} loss={tot / nb:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"state_dict": enc.state_dict(),
                "d_model": a.d_model,
                "n_features": x.shape[2],
                "norm": norm,
                "feature_order": FEATURE_ORDER}, a.out)
    with open(a.out.replace(".pt", "_norm.json"), "w") as fh:
        json.dump(norm, fh, indent=2)
    print(f"[selfsup] wrote {a.out}")


if __name__ == "__main__":
    main()