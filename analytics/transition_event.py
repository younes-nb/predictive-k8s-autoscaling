#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import pandas as pd

H = 6
DELTA = 0.2

WORKLOAD = ["rps_total", "http_mcr", "providerrpc_mcr", "frontend_rps",
            "mesh_rps", "upstream_rps_sum", "caller_rps_max", "from_frontend",
            "n_callers", "root_rps"]
SAT = ["cpu_utilization", "cpu_lim", "mem_lim", "throttle_ratio", "flag_frac",
       "err_frac", "restart_rate", "pgfault", "replicas", "desired_replicas",
       "unavailable"]
QUEUE = ["queue_in", "queue_for", "active_in", "active_for"]
FLOW = ["caller_sum", "neigh_cpu_mean", "neigh_rps_z30_mean"]
DYN = ["rps_slope5", "cpu_slope3", "mem_delta5", "rps_z30", "ewma_gap",
       "vol_rps", "vol_cpu", "concurrency", "scale_recency",
       "p99_latency", "req_byte_rate", "resp_byte_rate", "net_rx",
       "tod_sin", "tod_cos"]
LAGS = ["rps_total", "active_in", "active_for", "throttle_ratio",
        "cpu_utilization", "mesh_rps", "caller_sum", "vol_rps"]
LAG_STEPS = [3, 6, 12]


def log(msg):
    print(msg, flush=True)


def onset_events(cpu, delta, H, refractory=30):
    j = cpu[H:] - cpu[:-H]
    sig = j >= delta
    out = []
    last = -10 ** 9
    for i in np.nonzero(sig)[0]:
        if i - last > refractory:
            out.append(i)
            last = i
    return out


def score_alarms(scores, onsets, n, H, step_s=10):
    onsets = np.asarray(sorted(onsets), dtype=int)
    E = len(onsets)
    hours = n * step_s / 3600.0
    res0 = {"n_events": E, "pr_auc": 0.0, "p_at_r50": 0.0, "p_at_r80": 0.0,
            "fa_per_h_at_r50": 0.0, "fa_per_h_at_r80": 0.0,
            "median_lead_steps": None}
    if E == 0:
        return res0
    scores = np.asarray(scores, dtype=float)
    thr = np.unique(scores[::max(1, len(scores) // 400)])
    rec, prec, falarm, leads = [], [], [], []
    for t in thr:
        alm = np.nonzero(scores >= t)[0]
        if len(alm) == 0:
            rec.append(0.0)
            prec.append(1.0)
            falarm.append(0.0)
            continue
        lo = np.searchsorted(alm, np.maximum(onsets - H, 0), side="left")
        hi = np.searchsorted(alm, onsets, side="right") - 1
        has = (lo <= np.minimum(hi, len(alm) - 1)) & (lo < len(alm))
        tp = int(has.sum())
        lo2 = np.searchsorted(onsets, alm, side="left")
        tp_alm = int(np.sum(lo2 < E))
        fp = len(alm) - tp_alm
        rec.append(tp / E)
        prec.append(tp / max(1, tp + fp))
        falarm.append(fp / hours)
        if t == thr[0]:
            for k in np.nonzero(has)[0]:
                o = onsets[k]
                cand = alm[(alm >= o - H) & (alm <= o)]
                if len(cand):
                    leads.append(o - cand.max())
    rec, prec, falarm = map(np.array, (rec, prec, falarm))
    idx = np.argsort(rec, kind="stable")
    ap = float(np.trapezoid(prec[idx], rec[idx]))
    out = dict(res0)
    out["pr_auc"] = round(ap, 4)
    for target in (0.5, 0.8):
        m = rec >= target
        if m.any():
            j = np.nonzero(m)[0][np.argmax(prec[m])]
            out[f"p_at_r{int(target*100)}"] = round(float(prec[j]), 4)
            out[f"fa_per_h_at_r{int(target*100)}"] = round(float(falarm[j]), 2)
    out["median_lead_steps"] = round(float(np.median(leads)), 1) if leads else None
    return out


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="analytics/data/event")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    feat4 = [c for c in WORKLOAD + SAT + QUEUE + FLOW + DYN if c in df.columns]
    log(f"4-family features: {len(feat4)}")

    Xtr_l, ytr_l, Xca_l, yca_l, Xte_l, meta = [], [], [], [], [], []
    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 1000:
            continue
        F0 = g[feat4].ffill(limit=12).bfill().to_numpy(float)
        blocks = [F0]
        for l in LAG_STEPS:
            R = np.full_like(F0, np.nan)
            R[l:] = F0[:-l]
            keep = [feat4.index(c) for c in LAGS if c in feat4]
            blocks.append(R[:, keep])
        F = np.concatenate(blocks, axis=1)
        ok = np.isfinite(F).all(axis=1)
        cpu = g["cpu_utilization"].to_numpy(float)
        lab = np.zeros(n, dtype=int)
        lab[:-H] = (cpu[H:] - cpu[:-H] >= DELTA).astype(int)
        itr = np.arange(n) < int(n * 0.70)
        ica = ((np.arange(n) >= int(n * 0.70)) & (np.arange(n) < int(n * 0.80)))
        ite = (np.arange(n) >= int(n * 0.80)) & (np.arange(n) + H < n)
        Xtr_l.append(F[itr & ok])
        ytr_l.append(lab[itr & ok])
        Xca_l.append(F[ica & ok])
        yca_l.append(lab[ica & ok])
        Xte_l.append(F[ite & ok])
        meta.append((svc, np.nonzero(ite & ok)[0],
                     onset_events(cpu, DELTA, H), n))
    Xtr, ytr = np.concatenate(Xtr_l), np.concatenate(ytr_l)
    Xca, yca = np.concatenate(Xca_l), np.concatenate(yca_l)
    Xte, _ = np.concatenate(Xte_l), None
    log(f"train {len(ytr)} (pos {ytr.sum()}), cal {len(yca)} (pos {yca.sum()})")

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=100, class_weight="balanced", random_state=42)
    clf.fit(Xtr, ytr)
    p_cal = clf.predict_proba(Xca)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, yca)
    p_raw = clf.predict_proba(Xte)[:, 1]
    p_calib = iso.predict(p_raw)

    res = {}
    for name, scores in (("raw", p_raw), ("calibrated", p_calib)):
        all_r, all_p, all_fa, all_l = [], [], [], []
        ev_total, detail = 0, {}
        off = 0
        for (svc, idx, ons, n) in meta:
            m = len(idx)
            s = scores[off:off + m]
            back = {r: k for k, r in enumerate(idx)}
            r = score_alarms(s, sorted(back[o] for o in ons if o in back),
                             m, H)
            off += m
            ev_total += r["n_events"]
            for k in ("pr_auc", "p_at_r50", "p_at_r80",
                      "fa_per_h_at_r50", "fa_per_h_at_r80"):
                all_r.append(r.get(k, 0) if "pr" in k or "p_at" in k else 0)
            detail[svc] = r
        res[name] = {"n_events": ev_total, "detail": detail}
    pooled = {}
    for name, scores in (("raw", p_raw), ("calibrated", p_calib)):
        off = 0
        y_all, s_all = [], []
        for (svc, idx, ons, n) in meta:
            m = len(idx)
            s = scores[off:off + m]
            off += m
            g = df[df.msname == svc].reset_index(drop=True)
            cpu = g["cpu_utilization"].to_numpy(float)
            lab = np.zeros(len(g), dtype=int)
            lab[:-H] = (cpu[H:] - cpu[:-H] >= DELTA).astype(int)
            y_all.append(lab[idx])
            s_all.append(s)
        y_all = np.concatenate(y_all)
        s_all = np.concatenate(s_all)
        po, pl = [], []
        off = 0
        for (svc, idx, ons, n) in meta:
            m = len(idx)
            back = {r: k for k, r in enumerate(idx)}
            po.extend([back[o] + off for o in ons if o in back])
            off += m
        r = score_alarms(s_all, sorted(po), len(s_all), H)
        pooled[name] = r
        log(f"[{name}] pooled events={r['n_events']} PR-AUC={r['pr_auc']} "
            f"P@R50={r['p_at_r50']} P@R80={r['p_at_r80']} "
            f"FA/h@R80={r['fa_per_h_at_r80']} lead={r['median_lead_steps']}")
    with open(f"{args.out_dir}/event_scores.json", "w") as f:
        json.dump({"pooled": pooled}, f, indent=2)
    log(f"wrote -> {args.out_dir}/event_scores.json")
    try:
        np.savez_compressed(
            f"{args.out_dir}/event_diag.npz",
            y_row=y_all, scores=s_all,
            onsets=np.asarray(sorted(po), dtype=int),
            H=np.asarray([H]))
        log("wrote event_diag.npz")
    except Exception as e:
        log(f"diag save skipped: {e}")


if __name__ == "__main__":
    main()

