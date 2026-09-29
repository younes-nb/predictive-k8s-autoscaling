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
DELTA = 0.2


def log(msg):
    print(msg, flush=True)


class MLPAE(nn.Module):
    def __init__(self, dim, latent=64, hidden=512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        b = x.shape[0]
        return self.dec(self.enc(x.reshape(b, -1))).reshape(x.shape)


def windows_of(arr, starts):
    return np.stack([arr[s:s + L] for s in starts])


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
    ap.add_argument("--out", default=None)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    feat = [c for c in df.columns if c not in ("timestamp", "msname")]
    ci = feat.index("cpu_utilization")
    Xtr, Xte = [], []
    LBLspike, LBLdrop = [], []
    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 3000:
            continue
        A = g[feat].ffill().bfill().to_numpy(np.float32)
        cpu = A[:, ci]
        ntr, nte = int(n * 0.70), int(n * 0.80)
        for s in range(0, ntr - L - H + 1, args.stride):
            if abs(cpu[s + L + H - 1] - cpu[s + L - 1]) < args.delta:
                Xtr.append(A[s:s + L])
        for s in range(nte, n - L - H + 1, args.stride):
            Xte.append(A[s:s + L])
            j = cpu[s + L + H - 1] - cpu[s + L - 1]
            LBLspike.append(int(j >= args.delta))
            LBLdrop.append(int(j <= -args.delta))
    Xtr = np.stack(Xtr)
    Xte = np.stack(Xte)
    log(f"train normal windows: {len(Xtr)}  test windows: {len(Xte)}")

    y_spike = np.array(LBLspike, dtype=int)
    y_drop = np.array(LBLdrop, dtype=int)

    dev = torch.device("cpu")
    ae = MLPAE(Xtr.shape[1] * Xtr.shape[2], latent=args.latent).to(dev)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr)),
                    batch_size=args.batch_size, shuffle=True)
    ae.train()
    for ep in range(args.epochs):
        tot = 0.0
        for (b,) in dl:
            b = b.to(dev)
            opt.zero_grad()
            loss = ((ae(b) - b) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        log(f"epoch {ep + 1}/{args.epochs} recon MSE: {tot / len(Xtr):.6f}")
    ae.eval()
    with torch.no_grad():
        err, err_last = [], []
        for i in range(0, len(Xte), args.batch_size):
            b = torch.from_numpy(Xte[i:i + args.batch_size]).to(dev)
            sq = (ae(b) - b) ** 2
            err.append(sq.mean(dim=(1, 2)).cpu().numpy())
            err_last.append(sq[:, -H:, :].mean(dim=(1, 2)).cpu().numpy())
    err = np.concatenate(err)
    err_last = np.concatenate(err_last)
    res = {"n_test": len(Xte), "delta": args.delta, "latent": args.latent}
    for name, lab in (("spike", y_spike), ("drop", y_drop)):
        ap_, p50 = pr_scores(lab, err)
        ap_l, p50_l = pr_scores(lab, err_last)
        res[name] = {"n_events": int(lab.sum()),
                     "base_rate": round(float(lab.mean()), 5),
                     "pr_auc": round(ap_, 4), "prec_at_rec50": p50,
                     "pr_auc_trailH": round(ap_l, 4),
                     "prec_at_rec50_trailH": p50_l}
        log(f"AE {name}: n={int(lab.sum())} base={lab.mean():.5f} "
            f"PR-AUC={ap_:.4f} P@R50={p50} "
            f"[trail-H err: PR-AUC={ap_l:.4f} P@R50={p50_l}]")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

