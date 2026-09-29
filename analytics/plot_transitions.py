#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd

LAG_COLS = ["cpu_utilization", "rps_total", "active_in", "active_for",
            "throttle_ratio", "p99_latency", "req_byte_rate", "vol_rps",
            "vol_cpu", "concurrency", "frontend_rps", "mesh_rps",
            "caller_sum", "rps_slope5"]
DELTA = 0.2


def log(msg):
    print(msg, flush=True)


def bocpd_mass(x, var, h, hazard=1.0 / 250.0, mu0=None, cap=600):
    x = np.asarray(x, float)
    n = len(x)
    if mu0 is None or not np.isfinite(mu0):
        mu0 = float(np.nanmedian(x))
    max_r = min(n, cap)
    R = np.full(max_r + 1, -np.inf)
    R[0] = 0.0
    mu = np.full(max_r + 1, mu0)
    kappa = np.full(max_r + 1, 1.0)
    out = np.zeros(n)
    lh, l1 = np.log(hazard), np.log(1.0 - hazard)
    for t in range(n):
        xt = x[t]
        ok = np.isfinite(R) & np.isfinite(mu)
        pred = np.full(max_r + 1, -np.inf)
        pred[ok] = (-0.5 * np.log(2 * np.pi * var * (1 + 1 / kappa[ok]))
                    - 0.5 * (xt - mu[ok]) ** 2 / (var * (1 + 1 / kappa[ok])))
        grow = R + pred + l1
        tot = np.logaddexp.reduce(R[ok] + pred[ok]) if ok.any() else -np.inf
        R2 = np.full(max_r + 1, -np.inf)
        R2[0] = tot + lh
        R2[1:] = grow[:-1]
        R2 -= np.logaddexp.reduce(R2[np.isfinite(R2)])
        mu2 = np.full(max_r + 1, mu0)
        kap2 = np.full(max_r + 1, 1.0)
        g = np.isfinite(R2[1:])
        mu2[1:][g] = ((kappa[:-1][g] * mu[:-1][g] + xt)
                      / (kappa[:-1][g] + 1))
        kap2[1:][g] = kappa[:-1][g] + 1
        R, mu, kappa = R2, mu2, kap2
        out[t] = float(np.exp(R[:h + 1][np.isfinite(R[:h + 1])]).sum())
    return out


def pr_curve(y_true, scores, n_pts=200):
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-scores)
    tp = np.cumsum(y_true[order] == 1)
    rec = tp / max(1, y_true.sum())
    prec = tp / (np.arange(len(y_true)) + 1)
    idx = np.unique(np.linspace(0, len(rec) - 1, n_pts).astype(int))
    return rec[idx], prec[idx]


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="analytics/data/plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    base_cols = [c for c in df.columns if c not in ("timestamp", "msname")]
    cfgs = [("spike", 6), ("drop", 12)]
    dump = {}
    for kind, H in cfgs:
        Xtr_l, ytr_l, Xte_l, yte_l, meta = [], [], [], [], []
        for svc, g in df.groupby("msname"):
            g = g.reset_index(drop=True)
            n = len(g)
            if n < 500:
                continue
            F0 = g[base_cols].ffill(limit=12).bfill().to_numpy(float)
            blocks = [F0]
            for l in (3, 6, 12):
                R = np.full_like(F0, np.nan)
                R[l:] = F0[:-l]
                keep = [base_cols.index(c) for c in LAG_COLS if c in base_cols]
                blocks.append(R[:, keep])
            F = np.concatenate(blocks, axis=1)
            ok = np.isfinite(F).all(axis=1)
            cpu = g["cpu_utilization"].to_numpy(float)
            lab = np.zeros(n, dtype=int)
            if kind == "spike":
                lab[:-H] = (cpu[H:] - cpu[:-H] >= DELTA).astype(int)
            else:
                lab[:-H] = (cpu[H:] - cpu[:-H] <= -DELTA).astype(int)
            trm = np.arange(n) < int(n * 0.70)
            tem = (np.arange(n) >= int(n * 0.80)) & (np.arange(n) + H < n)
            Xtr_l.append(F[trm & ok])
            ytr_l.append(lab[trm & ok])
            Xte_l.append(F[tem & ok])
            yte_l.append(lab[tem & ok])
            meta.append((svc, g["timestamp"].iloc[tem & ok].to_numpy()))
        Xtr, ytr = np.concatenate(Xtr_l), np.concatenate(ytr_l)
        Xte, yte = np.concatenate(Xte_l), np.concatenate(yte_l)
        log(f"{kind} H={H}: train {len(ytr)} (pos {ytr.sum()}), "
            f"test {len(yte)} (pos {yte.sum()})")
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=100, class_weight="balanced", random_state=42)
        clf.fit(Xtr, ytr)
        pr = clf.predict_proba(Xte)[:, 1]
        dump[(kind, H)] = (yte, pr, meta)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (kind, H) in zip(axes, cfgs):
        yte, pr, _ = dump[(kind, H)]
        r, p = pr_curve(yte, pr)
        ax.plot(r, p, "b-", lw=2, label="HGB+lag")
        ax.axhline(yte.mean(), color="k", ls="--", lw=1, label="base rate")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(f"{kind} H={H} (n={yte.sum()})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/pr_curves.png", dpi=120)
    log("saved pr_curves.png")

    best = None
    for svc, g in df.groupby("msname"):
        if any(p in svc for p in ("mongo", "mysql")) or svc == "ts-voucher-service":
            continue
        g = g.reset_index(drop=True)
        n = len(g)
        cpu = g["cpu_utilization"].to_numpy(float)
        te = np.arange(int(n * 0.80), n - 12)
        j = np.abs(cpu[te + 6] - cpu[te])
        i = int(np.argmax(j))
        if best is None or j[i] > best[0]:
            best = (j[i], svc, g, te[i])
    amp, svc, g, t0 = best
    lo, hi = max(0, t0 - 200), min(len(g), t0 + 200)
    gw = g.iloc[lo:hi]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(gw["timestamp"], gw["cpu_utilization"], "b-", lw=1.2,
            label="actual cpu")
    ax.axvline(g["timestamp"].iloc[t0], color="k", ls=":", lw=1,
               label=f"biggest jump (+{amp:.2f})")
    ax.set_title(f"{svc} — zoom on biggest test transition")
    ax.set_ylabel("cpu_utilization")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/zoom_transition.png", dpi=130)
    log(f"saved zoom_transition.png ({svc} +{amp:.2f})")

    n = len(g)
    cpu = g["cpu_utilization"].to_numpy(float)
    H = 6
    lab = np.zeros(n, dtype=int)
    lab[:-H] = (cpu[H:] - cpu[:-H] >= DELTA).astype(int)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(gw["timestamp"], gw["cpu_utilization"], "b-", lw=1.2, label="cpu")
    ev = gw.index[lab[lo:hi] == 1]
    ax.scatter(gw["timestamp"].iloc[ev - lo], gw["cpu_utilization"].iloc[ev - lo],
               c="red", s=25, zorder=5, label="spike labels (H=6)")
    ax.set_title(f"{svc} — cpu with spike-label markers")
    ax.set_ylabel("cpu_utilization")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/timeline_labels.png", dpi=130)
    log("saved timeline_labels.png")


if __name__ == "__main__":
    main()

