#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

L = 128
H = 5
STRIDE = 5
DELTA = 0.2


def log(msg):
    print(msg, flush=True)


class GraphForecaster(nn.Module):
    def __init__(self, in_ch=50, hid=32, horizon=H, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        self.enc = nn.GRU(in_ch, hid, batch_first=True)
        self.attn = nn.Linear(2 * hid, 1)
        self.mix = nn.Linear(2 * hid, hid)
        self.head = nn.Linear(hid, horizon)

    def forward(self, x, adj, w):
        B, V, _, _ = x.shape
        h = self.enc(x.reshape(B * V, L, -1))[1].squeeze(0)
        h = h.view(B, V, -1)
        if self.use_graph:
            Wh = h
            hi = Wh.unsqueeze(2).expand(B, V, V, -1)
            hj = Wh.unsqueeze(1).expand(B, V, V, -1)
            e = self.attn(torch.cat([hi, hj], dim=-1)).squeeze(-1)
            e = e + torch.log1p(w)
            e = e.masked_fill(adj.unsqueeze(0) == 0, float("-inf"))
            a = torch.softmax(e, dim=1)
            a = torch.nan_to_num(a, nan=0.0)
            m = torch.einsum("bdS,bSh->bdh", a, Wh)
            h = torch.relu(self.mix(torch.cat([h, m], dim=-1)))
        return self.head(h)


def pr_scores(y_true, scores):
    order = np.argsort(-scores)
    tp = np.cumsum(np.asarray(y_true, dtype=float)[order] == 1)
    rec = tp / max(1, np.asarray(y_true).sum())
    prec = tp / (np.arange(len(y_true)) + 1)
    pm = prec[rec >= 0.5]
    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:])) if len(rec) > 1 else 0.0
    return ap, (round(float(pm.max()), 4) if len(pm) else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    svcs = sorted(df.msname.unique())
    V = len(svcs)
    si = {s: i for i, s in enumerate(svcs)}
    feat = [c for c in df.columns if c not in ("timestamp", "msname")]
    ci = feat.index("cpu_utilization")
    n = len(df[df.msname == svcs[0]])
    A = np.stack([df[df.msname == s].reset_index(drop=True)[feat]
                  .ffill().bfill().to_numpy(np.float32) for s in svcs])
    Craw = None
    log(f"nodes={V} rows/service={n} channels={len(feat)}")
    ed = pd.read_csv(args.edges, parse_dates=["timestamp"])
    grid = df[df.msname == svcs[0]].reset_index(drop=True)["timestamp"]
    E = np.zeros((len(grid), V, V), dtype=np.float32)
    for (t, s, d), gr in ed.groupby(["timestamp", "src", "dst"]):
        if s in si and d in si:
            j = grid.searchsorted(pd.Timestamp(t))
            if 0 <= j < len(grid) and grid.iloc[j] == pd.Timestamp(t):
                E[j, si[s], si[d]] = float(gr["rps"].max())
    E = np.nan_to_num(E)
    ntr = int(len(grid) * 0.70)
    adj = (E[:ntr].max(axis=0) > 1e-9).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    log(f"static edges (train zone): {int(adj.sum())} directed")
    Craw = A[:, :, ci]

    starts = list(range(0, n - L - H + 1, STRIDE))
    npos = len(starts)
    cut_tr, cut_te = int(npos * 0.70), int(npos * 0.80)
    tr_s, te_s = starts[:cut_tr], starts[cut_te:]

    def batch_of(ss):
        X = np.stack([A[:, s:s + L, :] for s in ss])
        W = np.stack([E[s + L - 1] for s in ss])
        Y = np.stack([Craw[:, s + L:s + L + H] for s in ss])
        return (torch.from_numpy(X), torch.from_numpy(W),
                torch.from_numpy(Y).float())

    dev = torch.device("cpu")
    model = GraphForecaster(len(feat), args.hidden, H,
                            use_graph=not args.no_graph).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    adj_t = torch.from_numpy(adj)
    nG = "graph" if not args.no_graph else "no-graph"
    for ep in range(args.epochs):
        model.train()
        tot, cnt, perm = 0.0, 0, np.random.permutation(tr_s)
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            X, W, Y = batch_of(b)
            X, W, Y = X.to(dev), W.to(dev), Y.to(dev)
            opt.zero_grad()
            loss = ((model(X, adj_t, W) - Y) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        log(f"[{nG}] epoch {ep + 1}/{args.epochs} train MSE: {tot / cnt:.6f}")
    model.eval()
    JP, YT, LT = [], [], []
    with torch.no_grad():
        for i in range(0, len(te_s), args.batch_size):
            X, W, Y = batch_of(te_s[i:i + args.batch_size])
            P = model(X.to(dev), adj_t, W.to(dev)).cpu().numpy()
            JP.append(P[:, :, -1] - X[:, :, -1, ci].numpy())
            YT.append(Y[:, :, -1].numpy())
            LT.append(X[:, :, -1, ci].numpy())
    J = np.concatenate(JP).ravel()
    Y = np.concatenate(YT).ravel()
    Lc = np.concatenate(LT).ravel()
    log(f"test rows: {len(Y)}")
    jump = Y - Lc
    spike = (jump >= args.delta).astype(int)
    drop = (jump <= -args.delta).astype(int)
    res = {"mode": nG, "n": len(Y), "delta": args.delta,
           "static_edges": int(adj.sum())}
    for name, lab, score in (("spike", spike, J), ("drop", drop, -J)):
        m = lab == 1
        ap, p50 = pr_scores(lab, score)
        mae_m = float(np.abs((Lc + J)[m] - Y[m]).mean())
        mae_p = float(np.abs(Lc[m] - Y[m]).mean())
        res[name] = {"n_events": int(lab.sum()),
                     "base_rate": round(float(lab.mean()), 5),
                     "pr_auc": round(ap, 4), "prec_at_rec50": p50,
                     "mae_ratio": round(mae_m / (mae_p + 1e-12), 4)}
    mae_m = float(np.abs((Lc + J) - Y).mean())
    mae_p = float(np.abs(Lc - Y).mean())
    res["guard"] = {"mae_ratio": round(mae_m / (mae_p + 1e-12), 4),
                    "pass": bool(mae_m <= mae_p)}
    for t in ("spike", "drop"):
        r = res[t]
        log(f"{t:>6}: n={r['n_events']} PR-AUC={r['pr_auc']} "
            f"P@R50={r['prec_at_rec50']} transMAE={r['mae_ratio']}")
    log(f"GUARD {res['guard']}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

