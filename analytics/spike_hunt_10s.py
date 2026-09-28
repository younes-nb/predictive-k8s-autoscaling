#!/usr/bin/env python3
"""Spike-hunt v2: which features unlock CPU-spike prediction?

Reuses analytics/data/leadlag_10s/metrics_10s.csv (v1, 10s signals) and adds
a supplemental export over the same grid:
  - saturation proximity: cpu/limits, mem/limits, page-fault rates, crash
    restarts, non-OK istio response-flag fraction (UH/UO/UF/RL/...),
    throttle ratio (throttled/total periods)
  - call-graph pressure: per-edge RPS from reporter="source" (caller mix,
    frontend-as-root pressure, total mesh RPS)
  - node pressure: cluster node load + cpu steal (global exogenous regime)
Engineered per service: rps/cpu velocity+acceleration, rolling volatility,
Little's-law concurrency proxy (rps*p99), scale-event recency, time-of-day.

Tests:
  A. greedy forward-selection ablation of feature GROUPS on Ridge R2 at
     60s horizon (cpu-history baseline, then add the group with best gain)
  B. spike-ONSET classifier: P(CPU up-spike onset in next H steps),
     logistic regression, PR-AUC + precision@recall>=0.5, H in {6, 12}
  C. cross-service CCF: frontend RPS / caller-sum vs each backend CPU,
     lags 0..60 steps (call-graph propagation delay?)

Run from repo root:
    python analytics/spike_hunt_10s.py --start "2026-09-21 21:00:00" \\
        --v1-csv analytics/data/leadlag_10s/metrics_10s.csv \\
        --out-dir analytics/data/spike_hunt
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import requests

PROMETHEUS_URL = "http://localhost:9090"
NS = "online-boutique"
TEHRAN = pytz.timezone("Asia/Tehran")
STEP = 10
MAXPTS = 10000 * STEP

SUPP_QUERIES = {
    "cpu_lim": (
        'sum by (pod) (rate(container_cpu_usage_seconds_total{namespace='
        '"online-boutique", container="server"}[1m])) / sum by (pod) ('
        'kube_pod_container_resource_limits{resource="cpu", namespace='
        '"online-boutique", container="server"})',
        "pod",
    ),
    "mem_lim": (
        'sum by (pod) (container_memory_working_set_bytes{namespace='
        '"online-boutique", container="server"}) / sum by (pod) ('
        'kube_pod_container_resource_limits{resource="memory", namespace='
        '"online-boutique", container="server"})',
        "pod",
    ),
    "periods": (
        'sum by (pod) (rate(container_cpu_cfs_periods_total{namespace='
        '"online-boutique"}[2m]))',
        "pod",
    ),
    "pgfault": (
        'sum by (pod) (rate(container_memory_failures_total{namespace='
        '"online-boutique", container="server", failure_type="pgfault"}[2m]))',
        "pod",
    ),
    "pgmajfault": (
        'sum by (pod) (rate(container_memory_failures_total{namespace='
        '"online-boutique", container="server", failure_type="pgmajfault"}[2m]))',
        "pod",
    ),
    "restarts": (
        'max by (pod) (kube_pod_container_status_restarts_total{'
        'namespace="online-boutique"})',
        "pod",
    ),
    "flags": (
        'sum by (destination_workload, response_flags) (rate('
        'istio_requests_total{reporter="destination",'
        ' destination_workload_namespace="online-boutique",'
        ' response_flags!="-"}[1m]))',
        "destination_workload",
    ),
    "edges": (
        'sum by (source_workload, destination_workload) (rate('
        'istio_requests_total{reporter="source",'
        ' source_workload_namespace="online-boutique",'
        ' destination_workload_namespace="online-boutique"}[1m]))',
        "destination_workload",
    ),
    "node_load": ("avg(node_load1)", None),
    "steal": (
        'avg(rate(node_cpu_seconds_total{mode="steal"}[2m]))', None),
}

SUM_METRICS = {"periods", "pgfault", "pgmajfault"}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def pod_to_deployment(pod):
    parts = pod.split("-")
    return "-".join(parts[:-2]) if len(parts) >= 3 else pod


def fetch(name, query, start_ts, end_ts):
    log(f"querying {name} ...")
    merged = {}
    s = int(start_ts)
    while s <= int(end_ts):
        e = min(s + MAXPTS, int(end_ts))
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={"query": query, "start": s, "end": e,
                    "step": f"{STEP}s"},
            timeout=600,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"query {name} failed: {payload}")
        for series in payload["data"]["result"]:
            key = tuple(sorted(series["metric"].items()))
            slot = merged.setdefault(key, [series["metric"], {}])
            for t, v in series["values"]:
                slot[1][int(t)] = v
        s = e + STEP
    return [{"metric": m, "values": [[t, v] for t, v in sorted(vals.items())]}
            for m, vals in merged.values()]


def export_supp(start_ts, end_ts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    grid = np.arange(int(start_ts) // STEP * STEP, int(end_ts) + 1, STEP)
    idx = pd.DatetimeIndex(pd.to_datetime(grid, unit="s", utc=True))
    per_metric, flag_detail, edge_detail = {}, {}, {}
    for name, (query, label) in SUPP_QUERIES.items():
        result = fetch(name, query, grid[0], grid[-1])
        frames = {}
        for series in result:
            m = series["metric"]
            if label is None:
                ent = "GLOBAL"
            else:
                ent = m.get(label, "")
                if not ent:
                    continue
                if label == "pod":
                    ent = pod_to_deployment(ent)
            if name == "flags":
                flag_detail.setdefault(ent, []).append(series)
                continue
            if name == "edges":
                edge_detail.setdefault(ent, []).append(series)
                continue
            ts = np.array([int(v[0]) for v in series["values"]])
            try:
                vals = np.array([float(v[1]) for v in series["values"]])
            except ValueError:
                continue
            s = pd.Series(vals, index=pd.to_datetime(ts, unit="s", utc=True))
            s = s[~s.index.duplicated(keep="last")].reindex(idx)
            frames.setdefault(ent, []).append(s)
        if name in ("flags", "edges"):
            continue
        merged = {}
        for ent, lst in frames.items():
            cat = pd.concat(lst, axis=1)
            if name in SUM_METRICS:
                merged[ent] = cat.sum(axis=1, min_count=1)
            elif name == "restarts":
                merged[ent] = cat.max(axis=1)
            else:
                merged[ent] = cat.mean(axis=1)
        per_metric[name] = merged
        log(f"  {name}: {len(merged)} entities")

    entities = set()
    for frames in per_metric.values():
        entities.update(frames.keys())
    entities.discard("redis-cart")
    entities.discard("")
    entities.discard("unknown")
    entities.discard("GLOBAL")

    glob = {}
    for name in ("node_load", "steal"):
        for ent, s in per_metric.get(name, {}).items():
            glob[name] = s.reindex(idx).ffill(limit=6)

    out_frames = []
    for ent in sorted(entities):
        full = pd.DataFrame(index=idx)
        for name, frames in per_metric.items():
            if name in ("node_load", "steal"):
                continue
            if ent in frames:
                full[name] = frames[ent].values
        if "cpu_lim" in full:
            full["cpu_lim"] = full["cpu_lim"].ffill(limit=6)
        if "mem_lim" in full:
            full["mem_lim"] = full["mem_lim"].ffill(limit=6)
        for name in ("periods", "pgfault", "pgmajfault"):
            if name in full:
                full[name] = full[name].fillna(0.0)
        if "restarts" in full:
            full["restarts"] = full["restarts"].ffill(limit=30).bfill().fillna(0.0)
        if "periods" in full and full["periods"].notna().any():
            den = full["periods"].replace(0.0, np.nan)
            thr = full.get("throttle")
            full["throttle_ratio"] = np.nan
        for gname, gs in glob.items():
            full[gname] = gs.values
        # flags -> flag_frac of total per workload
        if ent in flag_detail:
            fsum = None
            for series in flag_detail[ent]:
                ts = np.array([int(v[0]) for v in series["values"]])
                vals = np.array([float(v[1]) for v in series["values"]])
                s = pd.Series(vals,
                              index=pd.to_datetime(ts, unit="s", utc=True))
                s = s[~s.index.duplicated(keep="last")].reindex(idx).fillna(0.0)
                fsum = s if fsum is None else fsum + s
            full["flag_rate"] = fsum.values if fsum is not None else 0.0
        else:
            full["flag_rate"] = 0.0
        # edges inbound to ent
        if ent in edge_detail:
            tot, mx, by_src = None, None, {}
            for series in edge_detail[ent]:
                src = series["metric"].get("source_workload", "?")
                ts = np.array([int(v[0]) for v in series["values"]])
                vals = np.array([float(v[1]) for v in series["values"]])
                s = pd.Series(vals,
                              index=pd.to_datetime(ts, unit="s", utc=True))
                s = s[~s.index.duplicated(keep="last")].reindex(idx).fillna(0.0)
                by_src[src] = s
                tot = s if tot is None else tot + s
                mx = s if mx is None else pd.concat([mx, s], axis=1).max(axis=1)
            full["caller_sum"] = tot.values if tot is not None else 0.0
            full["caller_max"] = mx.values if mx is not None else 0.0
            for src in ("frontend", "checkoutservice", "cartservice"):
                full[f"from_{src}"] = (by_src[src].values if src in by_src
                                       else 0.0)
        else:
            for c in ("caller_sum", "caller_max", "from_frontend",
                      "from_checkoutservice", "from_cartservice"):
                full[c] = 0.0
        full["timestamp"] = full.index.tz_convert(TEHRAN).strftime(
            "%Y-%m-%d %H:%M:%S")
        full["msname"] = ent
        out_frames.append(full)
        log(f"  [{ent}] {len(full)} rows")
    df = pd.concat(out_frames, ignore_index=True)
    df = df.sort_values(["msname", "timestamp"]).reset_index(drop=True)
    path = os.path.join(out_dir, "supp_10s.csv")
    df.to_csv(path, index=False)
    log(f"saved {len(df)} rows -> {path}")
    return path


def add_engineered(df):
    df = df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["msname", "ts"]).reset_index(drop=True)
    out = []
    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True).copy()
        for c in ("rps", "cpu"):
            if c in g:
                v = g[c].ffill().bfill().to_numpy(float)
                g[f"{c}_vel"] = np.concatenate([[0], np.diff(v)])
                g[f"{c}_acc"] = np.concatenate([[0, 0], np.diff(v, 2)])
        for c, ws in (("cpu", (6, 30)), ("rps", (6, 30))):
            if c in g:
                for w in ws:
                    g[f"vol_{c}_{w}"] = g[c].rolling(w, min_periods=2).std(
                        ).bfill().fillna(0.0).to_numpy()
        if "rps" in g and "p99" in g:
            g["concurrency"] = (g["rps"].fillna(0.0).to_numpy(float)
                                * g["p99"].ffill().bfill().fillna(0.0)
                                .to_numpy(float) / 1000.0)
        if "replicas" in g:
            rep = g["replicas"].ffill().bfill().to_numpy(float)
            chg = np.nonzero(np.concatenate([[False],
                                             rep[1:] != rep[:-1]]))[0]
            last = np.zeros(len(g))
            cur = -10 ** 9
            ci = 0
            for i in range(len(g)):
                while ci < len(chg) and chg[ci] <= i:
                    cur = chg[ci]
                    ci += 1
                last[i] = i - cur
            g["scale_recency"] = np.minimum(last, 360)
        m = g["ts"].dt.minute.to_numpy() + g["ts"].dt.hour.to_numpy() * 60
        g["tod_sin"] = np.sin(2 * np.pi * m / 1440.0)
        g["tod_cos"] = np.cos(2 * np.pi * m / 1440.0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_model_frame(g, feat_cols, h, lags=(0, 1, 2, 3, 6, 12)):
    cols = [c for c in feat_cols
            if c in g.columns and g[c].notna().any()]
    if not cols or "cpu" not in cols:
        return None, None
    work = g[cols].ffill(limit=12).bfill()
    Xb = work.to_numpy(float)
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


def analyze(v1_path, supp_path, out_dir):
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score

    os.makedirs(out_dir, exist_ok=True)
    v1 = pd.read_csv(v1_path, parse_dates=["timestamp"])
    sp = pd.read_csv(supp_path, parse_dates=["timestamp"])
    # throttle lives in v1; bring it + flag/throttle_ratio together
    keep_supp = [c for c in sp.columns if c not in ("timestamp", "msname")]
    df = v1.merge(sp, on=["msname", "timestamp"], how="left",
                  suffixes=("", "_s"))
    if "throttle" in df and "throttle_ratio" in df:
        pass
    # flag_frac of total rps
    df["flag_frac"] = (df["flag_rate"].fillna(0.0)
                       / df["rps"].replace(0.0, np.nan)).fillna(0.0)
    # throttle ratio needs periods + v1 throttle
    if "periods" in df and "throttle" in df:
        df["throttle_ratio"] = (df["throttle"].fillna(0.0)
                                / df["periods"].replace(0.0, np.nan))
        df["throttle_ratio"] = df["throttle_ratio"].fillna(0.0)
    df = add_engineered(df)
    # mesh-global cols broadcast to every service
    fe = df[df.msname == "frontend"][["timestamp", "rps"]].rename(
        columns={"rps": "frontend_rps"})
    df = df.merge(fe, on="timestamp", how="left")
    df["mesh_rps"] = df.groupby("timestamp")["rps"].transform("sum")

    GROUPS = {
        "rps": ["rps", "rps_http", "rps_grpc"],
        "lat": ["p50", "p90", "p99"],
        "bytes": ["req_bytes", "resp_bytes", "net_rx", "net_tx"],
        "throttle": ["throttle", "throttle_ratio"],
        "satur": ["cpu_lim", "mem_lim", "pgfault", "pgmajfault",
                  "flag_frac", "restarts"],
        "node": ["node_load", "steal"],
        "upstream": ["frontend_rps", "mesh_rps", "caller_sum", "caller_max",
                     "from_frontend", "from_checkoutservice", "from_cartservice"],
        "eng": ["rps_vel", "rps_acc", "cpu_vel", "vol_cpu_6", "vol_cpu_30",
                "vol_rps_6", "vol_rps_30", "concurrency", "scale_recency"],
        "tod": ["tod_sin", "tod_cos"],
    }
    GROUPS = {k: [c for c in v if c in df.columns]
              for k, v in GROUPS.items()}
    GROUPS = {k: v for k, v in GROUPS.items() if v}
    log(f"groups: { {k: len(v) for k, v in GROUPS.items()} }")

    H = 6  # selection horizon: 60s
    abl_rows, clf_rows, cc_rows = [], [], []

    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True)
        cpu = g["cpu"].to_numpy(float)
        if np.nanstd(cpu) < 1e-3 or len(g) < 3000:
            continue
        # ---- C: cross-service CCF (frontend root + caller_sum) ----
        for cand in ("frontend_rps", "caller_sum", "mesh_rps"):
            if cand not in g or g[cand].std(skipna=True) == 0:
                continue
            x = g[cand].ffill().bfill().to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(cpu)
            x, y = x[mask], cpu[mask]
            xz, yz = (x - x.mean()) / (x.std() + 1e-12), \
                (y - y.mean()) / (y.std() + 1e-12)
            n = len(x)
            cc = {}
            for k in range(0, 61):
                cc[k] = float(np.dot(xz[:n - k], yz[k:]) / (n - k))
            kb = max(cc, key=lambda k: abs(cc[k]))
            cc_rows.append(dict(service=svc, candidate=cand,
                               best_lead_steps=kb,
                               best_corr=round(cc[kb], 3),
                               corr_lag0=round(cc[0], 3),
                               lead_6=round(cc[6], 3),
                               lead_12=round(cc[12], 3)))
        # ---- A: greedy forward ablation at H=6 ----
        base = ["cpu"]
        F0, y0 = build_model_frame(g, base, H)
        if F0 is None:
            continue
        cut = int(len(y0) * 0.7)
        sc = StandardScaler().fit(F0[:cut])
        r2p = Ridge(alpha=1.0).fit(sc.transform(F0[:cut]),
                                   y0[:cut]).score(sc.transform(F0[cut:]),
                                                   y0[cut:])
        abl_rows.append(dict(service=svc, step=0, added="cpu_hist",
                             test_r2=round(float(r2p), 4), gain=None))
        cur = {"cpu_hist": r2p}
        selected = ["cpu"]
        remaining = dict(GROUPS)
        for step in range(1, 8):
            best_gain, best_g, best_r2 = 0, None, None
            for gname, cols in remaining.items():
                F, y = build_model_frame(g, selected + cols, H)
                if F is None:
                    continue
                cut = int(len(y) * 0.7)
                sc = StandardScaler().fit(F[:cut])
                r2 = Ridge(alpha=1.0).fit(
                    sc.transform(F[:cut]), y[:cut]).score(
                    sc.transform(F[cut:]), y[cut:])
                if r2 - r2p > best_gain:
                    best_gain, best_g, best_r2 = r2 - r2p, gname, r2
            if best_g is None or best_gain < 0.005:
                break
            r2p = best_r2
            selected += remaining.pop(best_g)
            abl_rows.append(dict(service=svc, step=step, added=best_g,
                                 test_r2=round(float(r2p), 4),
                                 gain=round(float(best_gain), 4)))
        # ---- B: spike-onset classifier ----
        z = (cpu - np.nanmean(cpu)) / (np.nanstd(cpu) + 1e-12)
        onsets = []
        i, last = 0, -10 ** 9
        sig = z > 2.5
        while i < len(z):
            if sig[i] and i - last > 30 and not sig[max(0, i - 6):i].any():
                onsets.append(i)
                last = i
            i += 1
        if len(onsets) < 5:
            log(f"[{svc}] SKIP onsets={len(onsets)}")
            continue
        log(f"[{svc}] onsets={len(onsets)} proceeding to classifier")
        feat_all = selected + [c for grp in remaining.values() for c in grp]
        feat_all = [c for c in feat_all
                    if c in g.columns and g[c].notna().any()]
        W = g[feat_all].ffill(limit=12).bfill().to_numpy(float)
        ok = np.isfinite(W).all(axis=1)
        # blocked split: spikes cluster in the load peak, so a last-30%
        # tail split leaves the test set with zero positives. Alternate
        # 200-step blocks between train/test instead.
        blk = np.arange(len(g)) // 200
        test_mask = (blk % 2 == 1) & ok
        train_mask = (blk % 2 == 0) & ok
        if test_mask.sum() < 200 or train_mask.sum() < 800:
            continue
        log(f"[{svc}] classifier spikes={len(onsets)}")
        for Hh in (6, 12):
            lab = np.zeros(len(g))
            for o in onsets:
                lab[max(0, o - Hh):o] = 1
            Wtr, ytr = W[train_mask], lab[train_mask]
            Wte, yte = W[test_mask], lab[test_mask]
            if yte.sum() == 0 or ytr.sum() == 0:
                log(f"[{svc}] H={Hh}: SKIP ytr={int(ytr.sum())} "
                    f"yte={int(yte.sum())} okfrac={float(ok.mean()):.3f}")
                continue
            sc = StandardScaler().fit(Wtr)
            clf = LogisticRegression(C=1.0, max_iter=2000,
                                     class_weight="balanced")
            clf.fit(sc.transform(Wtr), ytr)
            pr = clf.predict_proba(sc.transform(Wte))[:, 1]
            yt = yte
            ap = float(average_precision_score(yt, pr)) if yt.sum() else 0.0
            base_rate = float(yt.mean())
            # precision at recall>=0.5
            order = np.argsort(-pr)
            tp = np.cumsum(yt[order] == 1)
            rec = tp / max(1, yt.sum())
            prec = tp / (np.arange(len(yt)) + 1)
            pm = prec[rec >= 0.5]
            p_at_r = round(float(pm.max()), 3) if len(pm) else 0.0
            # best single-feature AP for reference
            best_single, best_nm = 0.0, ""
            for j, nm in enumerate(feat_all):
                f1 = Wte[:, j]
                if np.std(f1) < 1e-12:
                    continue
                try:
                    a = average_precision_score(yt, f1)
                    if a > best_single:
                        best_single, best_nm = float(a), nm
                except Exception:
                    pass
            # direction-aware single: also try -f1 (dips)
            for j, nm in enumerate(feat_all):
                f1 = -Wte[:, j]
                if np.std(f1) < 1e-12:
                    continue
                try:
                    a = average_precision_score(yt, f1)
                    if a > best_single:
                        best_single, best_nm = float(a), "-" + nm
                except Exception:
                    pass
            clf_rows.append(dict(
                service=svc, horizon_steps=Hh, n_spikes=len(onsets),
                base_rate=round(base_rate, 4),
                pr_auc=round(ap, 4),
                prec_at_rec50=p_at_r,
                best_single=best_nm,
                best_single_ap=round(best_single, 4)))
            log(f"[{svc}] H={Hh}: spikes={len(onsets)} base={base_rate:.3f} "
                f"PR-AUC={ap:.3f} (best single {best_nm}={best_single:.3f})")

    pd.DataFrame(abl_rows).to_csv(os.path.join(out_dir, "ablation.csv"),
                                  index=False)
    pd.DataFrame(clf_rows).to_csv(os.path.join(out_dir, "spike_clf.csv"),
                                  index=False)
    pd.DataFrame(cc_rows).to_csv(os.path.join(out_dir, "crossccf.csv"),
                                 index=False)
    log(f"wrote ablation/clf/crossccf -> {out_dir}")


def parse_time(s):
    try:
        return float(s)
    except ValueError:
        return TEHRAN.localize(
            datetime.strptime(s, "%Y-%m-%d %H:%M:%S")).timestamp()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-09-21 21:00:00")
    ap.add_argument("--end", default=None)
    ap.add_argument("--v1-csv",
                    default="analytics/data/leadlag_10s/metrics_10s.csv")
    ap.add_argument("--out-dir", default="analytics/data/spike_hunt")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--supp-csv", default=None)
    args = ap.parse_args()

    if not args.analyze_only:
        end_ts = time.time() if not args.end else parse_time(args.end)
        supp = export_supp(parse_time(args.start), end_ts, args.out_dir)
    else:
        supp = args.supp_csv or os.path.join(args.out_dir, "supp_10s.csv")
    if not args.export_only:
        analyze(args.v1_csv, supp, args.out_dir)


if __name__ == "__main__":
    main()
