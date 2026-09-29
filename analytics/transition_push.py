#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import pandas as pd

HS = [6, 12]
DELTA_ABS = 0.15
DB_PAT = ("mongo", "mysql")


def log(msg):
    print(msg, flush=True)


LAG_COLS = ["cpu_utilization", "rps_total", "active_in", "active_for",
            "throttle_ratio", "p99_latency", "req_byte_rate", "vol_rps",
            "vol_cpu", "concurrency", "frontend_rps", "mesh_rps",
            "caller_sum", "rps_slope5"]
LAG_STEPS = [0, 3, 6, 12]


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


def analyze(csv_path, out_dir):
    from sklearn.ensemble import HistGradientBoostingClassifier

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    base_cols = [c for c in df.columns if c not in ("timestamp", "msname")]
    svcs = [s for s in sorted(df.msname.unique())
            if not any(p in s for p in DB_PAT) and s != "ts-voucher-service"]
    log(f"apps services: {len(svcs)}")
    per = {}
    for svc in svcs:
        g = df[df.msname == svc].reset_index(drop=True)
        alive = (g["replicas"].to_numpy(float) > 0)
        if alive.sum() < 500:
            continue
        per[svc] = (g, alive)
    log(f"kept {len(per)} services with live rows")

    rel_delta = {}
    for svc, (g, alive) in per.items():
        n = len(g)
        tr = g["cpu_utilization"].to_numpy(float)[:int(n * 0.70)]
        sd = float(np.nanstd(tr))
        rel_delta[svc] = max(DELTA_ABS, 2.0 * sd if np.isfinite(sd) else DELTA_ABS)
    log("rel_delta range: %.3f .. %.3f" % (min(rel_delta.values()),
                                           max(rel_delta.values())))

    BC = {}
    for svc, (g, alive) in per.items():
        cpu = g["cpu_utilization"].ffill().bfill().to_numpy(float)
        n = len(g)
        tr = cpu[:int(n * 0.70)]
        var = float((1.4826 * np.median(np.abs(np.diff(tr)))) ** 2) + 1e-10
        BC[svc] = {h: bocpd_mass(cpu, var, h) for h in HS}

    def build_matrix(svc_list, H, mode, train_span):
        Xs, ys, ms = [], [], []
        for svc in svc_list:
            g, alive = per[svc]
            n = len(g)
            F0 = g[base_cols].ffill(limit=12).bfill().to_numpy(float)
            cols = [F0]
            if mode in ("lag", "lagrecent", "ens"):
                blocks = [F0]
                for l in (3, 6, 12):
                    R = np.full_like(F0, np.nan)
                    R[l:] = F0[:-l]
                    keep = [base_cols.index(c) for c in LAG_COLS
                            if c in base_cols]
                    blocks.append(R[:, keep])
                F = np.concatenate(blocks, axis=1)
            else:
                F = F0
            ok = np.isfinite(F).all(axis=1) & alive
            cpu = g["cpu_utilization"].to_numpy(float)
            d = rel_delta[svc]
            lab_s = np.zeros(n, dtype=int)
            lab_d = np.zeros(n, dtype=int)
            lab_s[:-H] = (cpu[H:] - cpu[:-H] >= d).astype(int)
            lab_d[:-H] = (cpu[H:] - cpu[:-H] <= -d).astype(int)
            if train_span == "recent":
                trm = (np.arange(n) >= int(n * 0.40)) & (np.arange(n) < int(n * 0.70))
            else:
                trm = np.arange(n) < int(n * 0.70)
            tem = (np.arange(n) >= int(n * 0.80)) & (np.arange(n) + H < n)
            Xs.append((F, ok, trm, tem))
            ys.append((lab_s, lab_d))
            ms.append(svc)
        return Xs, ys, ms

    results = []
    import time
    t0 = time.time()
    for H in HS:
        for kind in ("spike", "drop"):
            for mode in ("base", "lag", "lagrecent", "ens"):
                train_span = "recent" if mode == "lagrecent" else "old"
                Xs, ys, ms = build_matrix(sorted(per), H, mode, train_span)
                Xtr_l, ytr_l, Xte_l, yte_l, bc_te = [], [], [], [], []
                for (F, ok, trm, tem), (ls, ld), svc in zip(Xs, ys, ms):
                    if train_span == "recent":
                        n = len(F)
                        trm = ((np.arange(n) >= int(n * 0.40))
                               & (np.arange(n) < int(n * 0.70)))
                    else:
                        trm = np.arange(len(F)) < int(len(F) * 0.70)
                    lab = ls if kind == "spike" else ld
                    Xtr_l.append(F[trm & ok])
                    ytr_l.append(lab[trm & ok])
                    Xte_l.append(F[tem & ok])
                    yte_l.append(lab[tem & ok])
                    bc_te.append(BC[svc][H][tem & ok])
                Xtr, ytr = np.concatenate(Xtr_l), np.concatenate(ytr_l)
                Xte, yte = np.concatenate(Xte_l), np.concatenate(yte_l)
                bcte = np.concatenate(bc_te)
                if yte.sum() < 10 or ytr.sum() < 10:
                    results.append(dict(H=H, kind=kind, mode=mode,
                                        n_test=len(yte),
                                        n_events=int(yte.sum()),
                                        note="too-few-events"))
                    continue
                clf = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
                    min_samples_leaf=100, class_weight="balanced",
                    random_state=42)
                clf.fit(Xtr, ytr)
                pr = clf.predict_proba(Xte)[:, 1]
                ap, p50, p80 = pr_full(yte, pr)
                row = dict(H=H, kind=kind, mode=mode, n_test=len(yte),
                           n_events=int(yte.sum()),
                           base=round(float(yte.mean()), 5),
                           pr_auc=round(ap, 4), p_at_r50=p50, p_at_r80=p80)
                if mode == "ens":
                    r1 = pd.Series(pr).rank(pct=True).to_numpy()
                    r2 = pd.Series(bcte).rank(pct=True).to_numpy()
                    ape, p5e, p8e = pr_full(yte, (r1 + r2) / 2)
                    row["ens_pr"] = round(ape, 4)
                    row["ens_p50"] = p5e
                    row["ens_p80"] = p8e
                results.append(row)
                log(f"[H={H} {kind:6} {mode:9}] ev={int(yte.sum())} "
                    f"PR={ap:.3f} P@R50={p50:.3f} P@R80={p80:.3f} "
                    f"({time.time()-t0:.0f}s)")
    pd.DataFrame(results).to_csv(f"{out_dir}/push.csv", index=False)
    log(f"wrote -> {out_dir}/push.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="analytics/data/push")
    args = ap.parse_args()
    analyze(args.csv, args.out_dir)


if __name__ == "__main__":
    main()

