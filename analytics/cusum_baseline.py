#!/usr/bin/env python3
"""Two-sided CUSUM detector baseline (classical quickest detection).

Page's CUSUM with reference k=0.5 (train-z-scored units): g+ accumulates
upward drift, g- downward drift. Score at t = statistic value (detection
frame, same as analytics/bocpd_baseline.py: onset in [t-H, t]).

CUSUM/SR procedures are minimax-optimal for detection delay under a false
alarm constraint (Lorden/Pollak/Tartakovsky) — the classical answer to
"react as fast as possible", complementing BOCPD's Bayesian view.

Run from repo root:
    python analytics/cusum_baseline.py --csv <tier0.csv> \\
        --out analytics/data/transitions/cusum.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

H = 5
DELTA = 0.2
K = 0.5


def log(msg):
    print(msg, flush=True)


def cusum_scores(x, k=K):
    x = np.asarray(x, float)
    gp = np.zeros_like(x)
    gm = np.zeros_like(x)
    for t in range(1, len(x)):
        gp[t] = max(0.0, gp[t - 1] + x[t] - k)
        gm[t] = max(0.0, gm[t - 1] - x[t] - k)
    return gp, gm


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
    ap.add_argument("--k", type=float, default=K)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    S, D = [], []
    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 3000:
            continue
        cpu = g["cpu_utilization"].ffill().bfill().to_numpy(float)
        cut = int(n * 0.80)
        tr, te = cpu[:int(n * 0.70)], cpu[cut:]
        mu = float(np.nanmedian(tr))
        sd = float(np.nanstd(tr)) + 1e-12
        gp, gm = cusum_scores((te - mu) / sd, k=args.k)
        jump = np.concatenate([np.full(H, np.nan), te[H:] - te[:-H]])
        spike = (jump >= args.delta).astype(int)
        drop = (jump <= -args.delta).astype(int)
        S.append((spike, gp))
        D.append((drop, gm))
    for name, pairs in (("spike", S), ("drop", D)):
        y = np.concatenate([p[0] for p in pairs])
        s = np.concatenate([p[1] for p in pairs])
        m = np.isfinite(y) & np.isfinite(s)
        y, s = y[m].astype(int), s[m]
        ap_, p50 = pr_scores(y, s)
        log(f"CUSUM {name}: n={int(y.sum())} base={y.mean():.5f} "
            f"PR-AUC={ap_:.4f} P@R50={p50}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"delta": args.delta, "horizon": H, "k": K}, f)


if __name__ == "__main__":
    main()
