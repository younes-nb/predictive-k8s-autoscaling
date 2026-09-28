#!/usr/bin/env python3
"""Echo State Network forecaster (reservoir dynamics, ridge readout).

ESNs (Jaeger 2001) are the cheap dynamical-systems answer to bursty
regimes: a fixed random reservoir expands [cpu, rps] history into a rich
nonlinear state; only a Ridge readout is fitted (train rows only, no leak).
Literature reports strong performance on chaotic/bursting systems with
little data (Vlachas, Hassanzadeh, Chattopadhyay) — worth testing where
backprop nets overfit the train regime.

Scored with the same cold-transition metric (level-implied jumps).

Run from repo root:
    python analytics/esn_forecast.py --csv <tier0.csv> \\
        --out analytics/data/transitions/esn.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

H = 5
DELTA = 0.2


def log(msg):
    print(msg, flush=True)


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
    ap.add_argument("--units", type=int, default=300)
    ap.add_argument("--rho", type=float, default=0.9,
                    help="reservoir spectral radius")
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    # train-zone input scaling per service (leak-free)
    P = {"jump": [], "truth": [], "last": []}
    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 3000:
            continue
        raw = np.stack([g["cpu_utilization"].ffill().bfill().to_numpy(float),
                        g["rps_total"].ffill().bfill().to_numpy(float)], axis=1)
        ntr = int(n * 0.70)
        mu, sd = raw[:ntr].mean(axis=0), raw[:ntr].std(axis=0) + 1e-12
        U = (raw - mu) / sd
        # fixed reservoir (no fitting -> no leak possible here)
        Win = rng.uniform(-1, 1, (args.units, 2)) * 0.5
        W = rng.uniform(-1, 1, (args.units, args.units))
        W[rng.random((args.units, args.units)) > args.density] = 0.0
        W *= args.rho / max(1e-9, abs(np.linalg.eigvals(W)).max())
        x = np.zeros(args.units)
        S = np.empty((n, args.units))
        for t in range(n):
            x = ((1 - args.leak) * x + args.leak
                 * np.tanh(Win @ U[t] + W @ x))
            S[t] = x
        cut_tr, cut_te = int(n * 0.70), int(n * 0.80)
        tr = np.arange(0, cut_tr - H)
        te = np.arange(cut_te, n - H)
        if len(te) < 200 or len(tr) < 800:
            continue
        from sklearn.linear_model import Ridge
        r = Ridge(alpha=args.ridge).fit(S[tr], raw[tr + H, 0])
        hat = r.predict(S[te])
        P["jump"].append(hat - raw[te, 0])
        P["truth"].append(raw[te + H, 0])
        P["last"].append(raw[te, 0])
    J = np.concatenate(P["jump"])
    Y = np.concatenate(P["truth"])
    L = np.concatenate(P["last"])
    log(f"test rows: {len(Y)}")
    jump_true = Y - L
    spike = (jump_true >= args.delta).astype(int)
    drop = (jump_true <= -args.delta).astype(int)
    res = {"n": len(Y), "delta": args.delta, "units": args.units,
           "rho": args.rho}
    for name, lab, score in (("spike", spike, J), ("drop", drop, -J)):
        m = lab == 1
        ap, p50 = pr_scores(lab, score)
        mae_m = float(np.abs((L + J)[m] - Y[m]).mean())
        mae_p = float(np.abs(L[m] - Y[m]).mean())
        res[name] = {"n_events": int(lab.sum()),
                     "base_rate": round(float(lab.mean()), 5),
                     "pr_auc": round(ap, 4), "prec_at_rec50": p50,
                     "mae_ratio": round(mae_m / (mae_p + 1e-12), 4)}
    mae_m = float(np.abs((L + J) - Y).mean())
    mae_p = float(np.abs(L - Y).mean())
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
