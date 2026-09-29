#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd

STEP = 10
LAGS = [0, 1, 2, 3, 6, 12]
HORIZONS = [1, 3, 6, 12, 30, 60]

QUEUE = ["queue_in", "queue_for", "active_in", "active_for"]
RPSFAM = ["rps_total", "http_mcr", "providerrpc_mcr", "frontend_rps",
          "mesh_rps", "upstream_rps_sum", "caller_rps_max", "from_frontend",
          "n_callers", "root_rps"]
SAT = ["cpu_lim", "mem_lim", "throttle_ratio", "flag_frac", "err_frac",
       "restart_rate", "pgfault", "pgmajfault", "replicas", "desired_replicas",
       "unavailable"]
ENG = ["rps_slope5", "cpu_slope3", "mem_delta5", "rps_z30", "ewma_gap",
       "vol_rps", "vol_cpu", "concurrency", "scale_recency", "tod_sin",
       "tod_cos", "neigh_cpu_mean", "neigh_cpu_slope3", "neigh_rps_z30_mean",
       "neigh_rps_slope5_mean", "p99_latency", "req_byte_rate",
       "resp_byte_rate", "req_bytes_per_req", "resp_bytes_per_req", "net_rx"]
GROUPS = {"queue": QUEUE, "rps": RPSFAM, "sat": SAT, "eng": ENG}


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ccf_lead(x, y, max_lag=60):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 500 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    xz = (x - x.mean()) / x.std()
    yz = (y - y.mean()) / y.std()
    n = len(x)
    return {k: float(np.dot(xz[:n - k], yz[k:]) / (n - k))
            for k in range(0, max_lag + 1)}


def onsets_of(cpu, thresh=2.5, refractory=30, quiet=6):
    z = (cpu - np.nanmean(cpu)) / (np.nanstd(cpu) + 1e-12)
    sig = z > thresh
    out, i, last = [], 0, -10 ** 9
    while i < len(z):
        if sig[i] and i - last > refractory and not sig[max(0, i - quiet):i].any():
            out.append(i)
            last = i
        i += 1
    return out


def design(g, cols, h, lags=LAGS):
    cols = [c for c in cols if c in g.columns and g[c].notna().any()]
    if "cpu_utilization" not in cols:
        return None, None
    Xb = g[cols].ffill(limit=12).bfill().to_numpy(float)
    ok = np.isfinite(Xb).all(axis=1)
    n = len(Xb)
    F = np.concatenate([np.roll(Xb, l, axis=0) for l in lags], axis=1)
    lo, hi = max(lags), n - h
    if hi - lo < 800:
        return None, None
    m = ok[lo:hi] & ok[lo + h:hi + h]
    for l in lags:
        m = m & ok[lo - l:hi - l]
    if m.sum() < 800:
        return None, None
    return F[lo:hi][m], Xb[lo + h:hi + h, 0][m]


def analyze(csv_path, out_dir):
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    ccf_rows, hor_rows, pre_rows, clf_rows = [], [], [], []
    avail_groups = {k: [c for c in v if c in df.columns]
                    for k, v in GROUPS.items()}

    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True)
        cpu = g["cpu_utilization"].to_numpy(float)
        if np.nanstd(cpu) < 1e-3 or len(g) < 3000:
            continue
        log(f"[{svc}] n={len(g)} cpu={np.nanmean(cpu):.3f}±{np.nanstd(cpu):.3f} "
            f"max={np.nanmax(cpu):.3f}")
        cands = [c for grp in avail_groups.values() for c in grp
                 if g[c].std(skipna=True) > 0]
        for cand in cands:
            cc = ccf_lead(g[cand].to_numpy(float), cpu)
            if cc is None:
                continue
            kb = max(cc, key=lambda k: abs(cc[k]))
            row = dict(service=svc, candidate=cand, best_lead_steps=kb,
                       best_corr=round(cc[kb], 3), corr_lag0=round(cc[0], 3))
            for h in (1, 3, 6, 12, 30):
                row[f"lead_{h}"] = round(cc[h], 3)
            ccf_rows.append(row)
        ons = onsets_of(pd.Series(cpu).ffill().bfill().to_numpy())
        for cand in cands:
            xs = pd.Series(g[cand].to_numpy(float)).ffill().bfill()
            xz = ((xs - xs.mean()) / (xs.std() + 1e-12)).to_numpy()
            hits, leads = 0, []
            for o in ons:
                q0, q1 = max(0, o - 24), max(0, o - 12)
                if q1 <= q0:
                    continue
                if np.nanmax(np.abs(xz[q0:q1])) > 1.0:
                    continue
                over = np.nonzero(xz[q1:o] > 2.0)[0]
                if len(over):
                    hits += 1
                    leads.append(o - (q1 + over[0]))
            pre_rows.append(dict(
                service=svc, candidate=cand, n_events=len(ons),
                hit_rate=round(hits / max(1, len(ons)), 3),
                median_lead=round(float(np.median(leads)), 1) if leads else None))
        variants = {
            "persist": None,
            "cpu": ["cpu_utilization"],
            "cpu+rps": ["cpu_utilization"] + avail_groups.get("rps", []),
            "cpu+rps+eng": ["cpu_utilization"] + avail_groups.get("rps", [])
            + avail_groups.get("eng", []) + avail_groups.get("sat", []),
            "full": ["cpu_utilization"] + [c for grp in avail_groups.values()
                                           for c in grp],
        }
        for h in HORIZONS:
            res = dict(service=svc, horizon_s=h * STEP)
            for vname, cols in variants.items():
                if vname == "persist":
                    F, y = design(g, ["cpu_utilization"], h)
                    if F is None:
                        continue
                    cut = int(len(y) * 0.7)
                    p = F[cut:, 0]
                    yt = y[cut:]
                else:
                    F, y = design(g, cols, h)
                    if F is None:
                        continue
                    cut = int(len(y) * 0.7)
                    sc = StandardScaler().fit(F[:cut])
                    p = Ridge(alpha=1.0).fit(
                        sc.transform(F[:cut]), y[:cut]).predict(
                        sc.transform(F[cut:]))
                    yt = y[cut:]
                ss = ((yt - yt.mean()) ** 2).sum() + 1e-12
                res[f"r2_{vname}"] = round(
                    float(1 - ((yt - p) ** 2).sum() / ss), 4)
            res["n_test"] = cut
            hor_rows.append(res)
        if len(ons) < 5:
            continue
        qcols = [c for c in avail_groups.get("queue", [])
                 if c in g.columns and g[c].abs().max() > 0]
        feat_noq = ["cpu_utilization"] + [c for k, grp in avail_groups.items()
                                          if k != "queue" for c in grp]
        feat_noq = [c for c in feat_noq if c in g.columns and g[c].notna().any()]
        W0 = g[feat_noq].ffill(limit=12).bfill().to_numpy(float)
        ok0 = np.isfinite(W0).all(axis=1)
        sets = {"noqueue": (feat_noq, ok0)}
        if qcols:
            feat_q = feat_noq + qcols
            Wq = g[feat_q].ffill(limit=12).bfill().to_numpy(float)
            sets["withqueue"] = (feat_q, np.isfinite(Wq).all(axis=1))
        blk = np.arange(len(g)) // 200
        for sname, (cols, ok) in sets.items():
            W = g[cols].ffill(limit=12).bfill().to_numpy(float)
            trm = (blk % 2 == 0) & ok
            tem = (blk % 2 == 1) & ok
            if trm.sum() < 800 or tem.sum() < 200:
                continue
            for Hh in (6, 12):
                lab = np.zeros(len(g))
                for o in ons:
                    lab[max(0, o - Hh):o] = 1
                if lab[tem].sum() == 0 or lab[trm].sum() == 0:
                    continue
                sc = StandardScaler().fit(W[trm])
                clf = LogisticRegression(C=1.0, max_iter=2000,
                                         class_weight="balanced")
                clf.fit(sc.transform(W[trm]), lab[trm])
                pr = clf.predict_proba(sc.transform(W[tem]))[:, 1]
                yt = lab[tem]
                ap = float(average_precision_score(yt, pr))
                order = np.argsort(-pr)
                tp = np.cumsum(yt[order] == 1)
                rec = tp / max(1, yt.sum())
                prec = tp / (np.arange(len(yt)) + 1)
                pm = prec[rec >= 0.5]
                clf_rows.append(dict(
                    service=svc, features=sname, horizon_steps=Hh,
                    n_spikes=len(ons), base_rate=round(float(yt.mean()), 4),
                    pr_auc=round(ap, 4),
                    prec_at_rec50=round(float(pm.max()), 3) if len(pm) else 0.0))
        log(f"[{svc}] spikes={len(ons)} queue_active_cols={qcols}")

    pd.DataFrame(ccf_rows).to_csv(f"{out_dir}/ccf.csv", index=False)
    pd.DataFrame(hor_rows).to_csv(f"{out_dir}/horizons.csv", index=False)
    pd.DataFrame(pre_rows).to_csv(f"{out_dir}/precursors.csv", index=False)
    pd.DataFrame(clf_rows).to_csv(f"{out_dir}/spike_clf.csv", index=False)
    log(f"wrote -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="analytics/data/tier0_validate")
    args = ap.parse_args()
    analyze(args.csv, args.out_dir)


if __name__ == "__main__":
    main()

