#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import pandas as pd

H = 5
DELTA = 0.2


def log(msg):
    print(msg, flush=True)


def bocpd_scores(x, var, hazard=1.0 / 250.0, mu0=None):
    x = np.asarray(x, float)
    n = len(x)
    if mu0 is None or not np.isfinite(mu0):
        mu0 = float(np.nanmedian(x))
    kappa0 = 1.0
    max_r = min(n, 600)
    R = np.full(max_r + 1, -np.inf)
    R[0] = 0.0
    mu = np.full(max_r + 1, mu0)
    kappa = np.full(max_r + 1, kappa0)
    out = np.zeros(n)
    log_h = np.log(hazard)
    log_1mh = np.log(1.0 - hazard)
    for t in range(n):
        xt = x[t]
        pred = np.full(max_r + 1, -np.inf)
        valid = np.isfinite(R) & np.isfinite(mu)
        pred[valid] = (-0.5 * np.log(2 * np.pi * var * (1 + 1 / kappa[valid]))
                       - 0.5 * (xt - mu[valid]) ** 2 / (var * (1 + 1 / kappa[valid])))
        grow = R + pred + log_1mh
        cp = float(np.logaddexp.reduce(R + pred)) + log_h if np.isfinite(
            np.logaddexp.reduce(R + pred)) else -np.inf
        R2 = np.full(max_r + 1, -np.inf)
        R2[0] = cp
        R2[1:] = grow[:-1]
        lse = np.logaddexp.reduce(R2[np.isfinite(R2)])
        R2 -= lse
        mu2 = np.full(max_r + 1, mu0)
        kap2 = np.full(max_r + 1, kappa0)
        grown = np.isfinite(R2[1:])
        mu2[1:][grown] = ((kappa[:-1][grown] * mu[:-1][grown] + xt)
                          / (kappa[:-1][grown] + 1))
        kap2[1:][grown] = kappa[:-1][grown] + 1
        R, mu, kappa = R2, mu2, kap2
        out[t] = float(np.exp(R[:H + 1][np.isfinite(R[:H + 1])]).sum())
    return out


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
        te = cpu[cut:]
        tr = cpu[:int(n * 0.70)]
        var = float((1.4826 * np.median(np.abs(np.diff(tr)))) ** 2) + 1e-10
        sc = bocpd_scores(te, var)
        jump = np.concatenate([np.full(H, np.nan), te[H:] - te[:-H]])
        spike = (jump >= args.delta).astype(int)
        drop = (jump <= -args.delta).astype(int)
        S.append((spike, sc))
        D.append((drop, sc))
    for name, pairs in (("spike", S), ("drop", D)):
        y = np.concatenate([p[0] for p in pairs])
        s = np.concatenate([p[1] for p in pairs])
        m = np.isfinite(y) & np.isfinite(s)
        y, s = y[m].astype(int), s[m]
        ap_, p50 = pr_scores(y, s)
        log(f"BOCPD {name}: n={int(y.sum())} base={y.mean():.5f} "
            f"PR-AUC={ap_:.4f} P@R50={p50}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"note": "detection frame (onset in [t-H,t]); "
                               "forecast models use (t,t+H]",
                       "delta": args.delta, "horizon": H}, f)


if __name__ == "__main__":
    main()

