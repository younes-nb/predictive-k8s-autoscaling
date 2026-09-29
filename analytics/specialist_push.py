#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd

DELTA = 0.2
LAG_COLS = ["cpu_utilization", "rps_total", "active_in", "active_for",
            "throttle_ratio", "p99_latency", "req_byte_rate", "vol_rps",
            "vol_cpu", "concurrency", "frontend_rps", "mesh_rps",
            "caller_sum", "rps_slope5"]


def log(msg):
    print(msg, flush=True)


def pr_full(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-scores)
    tp = np.cumsum(y_true[order] == 1)
    rec = tp / max(1, y_true.sum())
    prec = tp / (np.arange(len(y_true)) + 1)

    def pat(th):
        m = prec[rec >= th]
        return round(float(m.max()), 4) if len(m) else 0.0

    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:])) if len(rec) > 1 else 0.0
    return ap, pat(0.5), pat(0.8)


def add_cp_features(g):
    g = g.copy()
    rep = g["replicas"].to_numpy(float)
    chg = np.nonzero(np.diff(rep) != 0)[0]
    tss = np.zeros(len(g))
    sdir = np.zeros(len(g))
    last_c, last_d = -10 ** 9, 0.0
    ci = 0
    chg = sorted(chg.tolist())
    for i in range(len(g)):
        while ci < len(chg) and chg[ci] < i:
            last_c = chg[ci]
            last_d = float(np.sign(rep[chg[ci] + 1] - rep[chg[ci]]))
            ci += 1
        tss[i] = min(i - last_c, 360)
        sdir[i] = last_d if i - last_c <= 360 else 0.0
    g["time_since_scale"] = tss
    g["scale_dir"] = sdir
    g["demand"] = g["cpu_utilization"] * g["replicas"]
    g["demand_slope"] = g["demand"].diff(12).fillna(0.0)
    return g


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="analytics/data/special")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    base_cols = [c for c in df.columns if c not in ("timestamp", "msname")]
    cp_cols = ["replicas", "time_since_scale", "scale_dir", "demand",
               "demand_slope"]

    counts = []
    per = {}
    for svc, g in df.groupby("msname"):
        if any(p in svc for p in ("mongo", "mysql")) or svc == "ts-voucher-service":
            continue
        g = add_cp_features(g.reset_index(drop=True))
        n = len(g)
        if n < 500:
            continue
        per[svc] = g
        cpu = g["cpu_utilization"].to_numpy(float)
        te = np.arange(int(n * 0.80), n - 6)
        counts.append((svc, int(((cpu[te + 6] - cpu[te]) >= DELTA).sum())))
    counts.sort(key=lambda t: -t[1])
    top = [s for s, c in counts[:args.top]]
    log(f"top services by test spikes: {[(s, c) for s, c in counts[:args.top]]}")

    H = 6
    rows = []
    for svc in top:
        g = per[svc]
        n = len(g)
        F0 = g[base_cols].ffill(limit=12).bfill().to_numpy(float)
        blocks = [F0]
        for l in (3, 6, 12):
            R = np.full_like(F0, np.nan)
            R[l:] = F0[:-l]
            keep = [base_cols.index(c) for c in LAG_COLS if c in base_cols]
            blocks.append(R[:, keep])
        Fl = np.concatenate(blocks, axis=1)
        Fc = g[cp_cols].ffill(limit=12).bfill().to_numpy(float)
        F = np.concatenate([Fl, Fc], axis=1)
        ok = np.isfinite(F).all(axis=1)
        cpu = g["cpu_utilization"].to_numpy(float)
        lab = np.zeros(n, dtype=int)
        lab[:-H] = (cpu[H:] - cpu[:-H] >= DELTA).astype(int)
        trm = np.arange(n) < int(n * 0.70)
        tem = (np.arange(n) >= int(n * 0.80)) & (np.arange(n) + H < n)
        for name, cols in (("snapshot", F0.shape[1]),
                           ("+lag", Fl.shape[1]),
                           ("+lag+cp", F.shape[1])):
            X = F[:, :cols]
            okc = np.isfinite(X).all(axis=1)
            Xtr, ytr = X[trm & okc], lab[trm & okc]
            Xte, yte = X[tem & okc], lab[tem & okc]
            if yte.sum() < 10 or ytr.sum() < 10:
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
                min_samples_leaf=100, class_weight="balanced", random_state=42)
            clf.fit(Xtr, ytr)
            ap_, p50, p80 = pr_full(yte, clf.predict_proba(Xte)[:, 1])
            rows.append(dict(service=svc, feats=name, n_test=len(yte),
                             n_events=int(yte.sum()),
                             base=round(float(yte.mean()), 4),
                             pr_auc=round(ap_, 4), p_at_r50=p50, p_at_r80=p80))
            log(f"[{svc} {name}] ev={int(yte.sum())} PR={ap_:.3f} "
                f"P@R50={p50:.3f} P@R80={p80:.3f}")
    pd.DataFrame(rows).to_csv(f"{args.out_dir}/special.csv", index=False)
    log(f"wrote -> {args.out_dir}/special.csv")


if __name__ == "__main__":
    main()

