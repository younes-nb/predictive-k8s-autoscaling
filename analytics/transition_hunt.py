#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import pandas as pd

HS = [3, 6, 12]
DELTA = 0.2
HI = 0.8
LO = 0.3


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
        mu2[1:][g] = (kappa[:-1][g] * mu[:-1][g] + xt) / (kappa[:-1][g] + 1)
        kap2[1:][g] = kappa[:-1][g] + 1
        R, mu, kappa = R2, mu2, kap2
        out[t] = float(np.exp(R[:h + 1][np.isfinite(R[:h + 1])]).sum())
    return out


def analyze(csv_path, out_dir):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.mixture import GaussianMixture

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    log(f"loaded {len(df)} rows x {df.shape[1]} cols, "
        f"services={sorted(df.msname.unique())}")
    log(f"window: {df.timestamp.min()} .. {df.timestamp.max()}")

    feat_all = [c for c in df.columns if c not in ("timestamp", "msname")]
    inv = []
    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True)
        cpu = g["cpu_utilization"].to_numpy(float)
        rps = g["rps_total"].to_numpy(float)
        rep = g["replicas"].to_numpy(float)
        j5 = cpu[5:] - cpu[:-5]
        inv.append(dict(
            service=svc, n=len(g),
            cpu_mean=round(float(np.mean(cpu)), 4),
            cpu_std=round(float(np.std(cpu)), 4),
            cpu_max=round(float(np.max(cpu)), 3),
            frac_gt_08=round(float((cpu > 0.8).mean()), 4),
            frac_gt_10=round(float((cpu > 1.0).mean()), 4),
            n_spike_02=0, n_drop_02=0, n_hi08=0, n_lo03=0,
            max_rps=round(float(np.max(rps)), 1),
            max_thr=round(float(np.max(g["throttle_ratio"].to_numpy(float))), 4),
            max_qin=round(float(np.max(g["queue_in"].to_numpy(float))), 3),
            max_actin=round(float(np.max(g["active_in"].to_numpy(float))), 2),
            err_sum=round(float(g["err_rate"].sum() * 10), 1),
            replicas=sorted(map(float, np.unique(rep))).copy(),
            n_scales=int((np.diff(rep) != 0).sum()),
        ))
        for h in HS:
            j = cpu[h:] - cpu[:-h]
            inv[-1][f"spike_h{h}"] = int((j >= DELTA).sum())
            inv[-1][f"drop_h{h}"] = int((j <= -DELTA).sum())
            inv[-1][f"hi08_h{h}"] = int((cpu[h:] >= HI).sum())
    pd.DataFrame(inv).to_csv(f"{out_dir}/inventory.csv", index=False)
    for r in inv:
        log(f"{r['service']:22} n={r['n']} cpu={r['cpu_mean']}±{r['cpu_std']} "
            f"max={r['cpu_max']} >0.8:{r['frac_gt_08']} "
            f"spk(d=.2,H5)={r['spike_h3'] + r['spike_h6'] + r['spike_h12']}//"
            f"hi08={r['hi08_h12']} scales={r['n_scales']} "
            f"maxrps={r['max_rps']} maxthr={r['max_thr']} "
            f"maxqin={r['max_qin']} maxact={r['max_actin']} rep={r['replicas']}")

    per_svc = {}
    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 300:
            continue
        per_svc[svc] = g
    ZF = {}
    for svc, g in per_svc.items():
        n = len(g)
        ZF[svc] = (np.arange(n) < int(n * 0.70),
                   np.arange(n) >= int(n * 0.80))

    log("BOCPD masses ...")
    BC = {}
    for svc, g in per_svc.items():
        cpu = g["cpu_utilization"].ffill().bfill().to_numpy(float)
        n = len(g)
        tr = cpu[:int(n * 0.70)]
        var = float((1.4826 * np.median(np.abs(np.diff(tr)))) ** 2) + 1e-10
        BC[svc] = {h: bocpd_mass(cpu, var, h) for h in HS}

    log("GMM regimes ...")
    GM = {}
    for svc, g in per_svc.items():
        trm, _ = ZF[svc]
        Z = np.stack([g[c].ffill().bfill().to_numpy(float)
                      for c in ("cpu_utilization", "active_in",
                                "throttle_ratio")], axis=1)
        mu, sd = Z[trm].mean(axis=0), Z[trm].std(axis=0) + 1e-12
        Zs = (Z - mu) / sd
        gm = GaussianMixture(n_components=3, covariance_type="diag",
                             random_state=42, n_init=3,
                             reg_covar=1e-3).fit(Zs[trm])
        GM[svc] = gm.predict_proba(Zs)

    FEATS = {
        "base": feat_all,
    }

    def labels(g, kind, H):
        cpu = g["cpu_utilization"].to_numpy(float)
        n = len(cpu)
        if kind == "spike":
            return (cpu[H:] - cpu[:-H] >= DELTA).astype(int)
        if kind == "drop":
            return (cpu[H:] - cpu[:-H] <= -DELTA).astype(int)
        if kind == "hi08":
            return (cpu[H:] >= HI).astype(int)
        if kind == "lo03":
            return (cpu[H:] <= LO).astype(int)
        raise ValueError(kind)

    results = []
    import time
    t0 = time.time()
    for scope, svcs in (("pooled", sorted(per_svc)),
                        ("frontend", ["frontend"] if "frontend" in per_svc else [])):
        if not svcs:
            continue
        for H in HS:
            for kind in ("spike", "drop", "hi08", "lo03"):
                for feat_mode in ("base", "+bocpd", "+gmm"):
                    Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
                    for svc in svcs:
                        g = per_svc[svc]
                        n = len(g)
                        trm, tem = ZF[svc]
                        F = g[FEATS["base"]].ffill(limit=12).bfill().to_numpy(float)
                        if feat_mode == "+bocpd":
                            F = np.concatenate([F, BC[svc][H][:, None]], axis=1)
                        elif feat_mode == "+gmm":
                            F = np.concatenate([F, GM[svc]], axis=1)
                        ok = np.isfinite(F).all(axis=1)
                        lab = labels(g, kind, H)
                        m = np.arange(n - H)
                        tri = m[trm[:n - H]]
                        tei = m[tem[:n - H]]
                        Xtr_l.append(F[tri][ok[tri]])
                        ytr_l.append(lab[tri][ok[tri]])
                        Xte_l.append(F[tei][ok[tei]])
                        yte_l.append(lab[tei][ok[tei]])
                    Xtr, ytr = np.concatenate(Xtr_l), np.concatenate(ytr_l)
                    Xte, yte = np.concatenate(Xte_l), np.concatenate(yte_l)
                    if len(yte) < 100 or yte.sum() < 10 or ytr.sum() < 10:
                        results.append(dict(scope=scope, H=H, kind=kind,
                                            feats=feat_mode, n_test=len(yte),
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
                    results.append(dict(scope=scope, H=H, kind=kind,
                                        feats=feat_mode, n_test=len(yte),
                                        n_events=int(yte.sum()),
                                        base=round(float(yte.mean()), 5),
                                        pr_auc=round(ap, 4),
                                        p_at_r50=p50, p_at_r80=p80))
                    log(f"[{scope} H={H} {kind:6} {feat_mode:6}] "
                        f"ev={int(yte.sum())} base={yte.mean():.4f} "
                        f"PR={ap:.3f} P@R50={p50:.3f} P@R80={p80:.3f} "
                        f"({time.time()-t0:.0f}s)")
    for H in HS:
        for kind in ("spike", "drop"):
            ys, ss, vs = [], [], []
            for svc in sorted(per_svc):
                g = per_svc[svc]
                n = len(g)
                tem = ZF[svc][1]
                m = np.arange(n - H)[tem[:n - H]]
                lab = labels(g, kind, H)[m]
                cpu = g["cpu_utilization"].to_numpy(float)
                vel = (cpu[m] - cpu[m - min(H, 12)]) / min(H, 12) * H
                ys.append(lab)
                ss.append(BC[svc][H][m])
                vs.append(vel)
            y = np.concatenate(ys)
            if y.sum() < 10:
                continue
            for nm, sc in (("bocpd", np.concatenate(ss)),
                           ("vel", np.concatenate(vs))):
                ap, p50, p80 = pr_full(y, sc)
                results.append(dict(scope="pooled", H=H, kind=kind,
                                    feats=nm, n_test=len(y),
                                    n_events=int(y.sum()),
                                    base=round(float(y.mean()), 5),
                                    pr_auc=round(ap, 4),
                                    p_at_r50=p50, p_at_r80=p80))
                log(f"[pooled H={H} {kind:6} {nm:6}] ev={int(y.sum())} "
                    f"PR={ap:.3f} P@R50={p50:.3f} P@R80={p80:.3f}")
    pd.DataFrame(results).to_csv(f"{out_dir}/hunt.csv", index=False)
    log(f"wrote -> {out_dir}/hunt.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="analytics/data/fulltest")
    args = ap.parse_args()
    analyze(args.csv, args.out_dir)


if __name__ == "__main__":
    main()

