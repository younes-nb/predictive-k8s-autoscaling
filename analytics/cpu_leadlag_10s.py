#!/usr/bin/env python3
"""10s-granularity CPU lead-lag analysis for the Online Boutique load test.

Part 1 (export): pulls per-service signals from Prometheus at 10s step over
    the load-test window. Extends analytics/export_hpa.py with sub-minute
    leading-indicator candidates: per-protocol RPS, 5xx fraction, istio
    p50/p90/p99 latency, request/response byte rates, CPU-throttle rate and
    pod network rx/tx rates.
Part 2 (analyze): for each service, measures
    (a) cross-correlation of every candidate vs CPU at lags -600s..+600s,
    (b) spike/dip precursor hit-rate (did a candidate move in the 12 steps
        / 2 min before a CPU transition?),
    (c) Ridge-regression R2 vs a persistence baseline at horizons
        1..60 steps (10s..10min), with a cpu-only ablation so the marginal
        value of MCR (rps) and friends is explicit.

Run from repo root:
    python analytics/cpu_leadlag_10s.py --start "2026-09-21 21:00:00" \\
        --out-dir analytics/data/leadlag_10s
    python analytics/cpu_leadlag_10s.py --analyze-only \\
        --csv analytics/data/leadlag_10s/metrics_10s.csv \\
        --out-dir analytics/data/leadlag_10s
"""

import argparse
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytz
import requests

PROMETHEUS_URL = "http://localhost:9090"
NAMESPACE = "online-boutique"
TEHRAN = pytz.timezone("Asia/Tehran")
STEP = 10

QUERIES = {
    "replicas": (
        'kube_deployment_status_replicas_available{namespace="online-boutique"}',
        "deployment",
    ),
    "rps_http": (
        'sum(rate(istio_requests_total{reporter="destination",'
        ' request_protocol="http",'
        ' destination_workload_namespace="online-boutique"}[1m]))'
        " by (destination_workload)",
        "destination_workload",
    ),
    "rps_grpc": (
        'sum(rate(istio_requests_total{reporter="destination",'
        ' request_protocol="grpc",'
        ' destination_workload_namespace="online-boutique"}[1m]))'
        " by (destination_workload)",
        "destination_workload",
    ),
    "rps_5xx": (
        'sum(rate(istio_requests_total{reporter="destination",'
        ' destination_workload_namespace="online-boutique",'
        ' response_code=~"5.*"}[1m])) by (destination_workload)',
        "destination_workload",
    ),
    "cpu": (
        'sum by (pod) (rate(container_cpu_usage_seconds_total{namespace='
        '"online-boutique", container="server"}[1m])) / sum by (pod) ('
        'kube_pod_container_resource_requests{resource="cpu", namespace='
        '"online-boutique", container="server"})',
        "pod",
    ),
    "memory": (
        'sum by (pod) (container_memory_working_set_bytes{namespace='
        '"online-boutique", container="server"}) / sum by (pod) ('
        'kube_pod_container_resource_requests{resource="memory", namespace='
        '"online-boutique", container="server"})',
        "pod",
    ),
    "throttle": (
        'sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{'
        'namespace="online-boutique"}[2m]))',
        "pod",
    ),
    "net_rx": (
        'sum by (pod) (rate(container_network_receive_bytes_total{namespace='
        '"online-boutique", pod!=""}[1m]))',
        "pod",
    ),
    "net_tx": (
        'sum by (pod) (rate(container_network_transmit_bytes_total{namespace='
        '"online-boutique", pod!=""}[1m]))',
        "pod",
    ),
}

for _q, _le in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99)):
    QUERIES[_q] = (
        f"histogram_quantile({_le}, sum by (destination_workload, le) (rate("
        'istio_request_duration_milliseconds_bucket{reporter="destination",'
        ' destination_workload_namespace="online-boutique"}[2m])))',
        "destination_workload",
    )

QUERIES["req_bytes"] = (
    'sum(rate(istio_request_bytes_sum{reporter="destination",'
    ' destination_workload_namespace="online-boutique"}[1m]))'
    " by (destination_workload)",
    "destination_workload",
)
QUERIES["resp_bytes"] = (
    'sum(rate(istio_response_bytes_sum{reporter="destination",'
    ' destination_workload_namespace="online-boutique"}[1m]))'
    " by (destination_workload)",
    "destination_workload",
)

POD_METRICS = {"cpu", "memory", "throttle", "net_rx", "net_tx"}
ZERO_FILL = {"rps_http", "rps_grpc", "rps_5xx", "throttle", "net_rx", "net_tx",
             "req_bytes", "resp_bytes"}

CANDIDATES = ["rps", "p50", "p90", "p99", "err_frac", "req_bytes", "resp_bytes",
              "throttle", "net_rx", "net_tx"]
LAGS = [0, 1, 2, 3, 6, 12]
HORIZONS = [1, 3, 6, 12, 30, 60]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def pod_to_deployment(pod):
    parts = pod.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:-2])
    return pod


def fetch(name, query, start_ts, end_ts):
    log(f"querying {name} ...")
    # Prometheus caps query_range at 11,000 points/series; chunk long windows.
    chunk = 10000 * STEP
    merged = {}
    s = int(start_ts)
    while s <= int(end_ts):
        e = min(s + chunk, int(end_ts))
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
    out = []
    for metric, vals in merged.values():
        out.append({"metric": metric,
                    "values": [[t, vals[t]] for t in sorted(vals)]})
    return out


def export_window(start_ts, end_ts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    grid = np.arange(int(start_ts) // STEP * STEP, int(end_ts) + 1, STEP)
    idx = pd.DatetimeIndex(pd.to_datetime(grid, unit="s", utc=True))
    per_metric = {}
    for name, (query, label) in QUERIES.items():
        result = fetch(name, query, grid[0], grid[-1])
        frames = {}
        for series in result:
            ent = series["metric"].get(label, "")
            if not ent:
                continue
            if name in POD_METRICS:
                ent = pod_to_deployment(ent)
            ts = np.array([int(v[0]) for v in series["values"]])
            try:
                vals = np.array([float(v[1]) for v in series["values"]])
            except ValueError:
                continue
            s = pd.Series(vals, index=pd.to_datetime(ts, unit="s", utc=True))
            s = s[~s.index.duplicated(keep="last")].reindex(idx)
            frames.setdefault(ent, []).append(s)
        per_metric[name] = frames
        log(f"  {name}: {len(frames)} entities")

    # merge multiple pod series per deployment with NaN-aware aggregation
    # (pods churn over 37h; plain (a+b)/2 would poison everything with NaN)
    SUM_METRICS = {"throttle", "net_rx", "net_tx"}
    merged = {}
    for name, frames in per_metric.items():
        m = {}
        for ent, series_list in frames.items():
            cat = pd.concat(series_list, axis=1)
            if name in SUM_METRICS:
                m[ent] = cat.sum(axis=1, min_count=1)
            else:
                m[ent] = cat.mean(axis=1)
        merged[name] = m
    per_metric = merged

    entities = set()
    for frames in per_metric.values():
        entities.update(frames.keys())
    entities.discard("redis-cart")
    entities.discard("")
    entities.discard("unknown")

    out_frames = []
    for ent in sorted(entities):
        full = pd.DataFrame(index=idx)
        for name, frames in per_metric.items():
            if ent in frames:
                full[name] = frames[ent].values
        if "cpu" not in full or full["cpu"].isna().all():
            log(f"  [{ent}] skipped (no cpu signal)")
            continue
        full["cpu"] = full["cpu"].ffill(limit=6)
        if "memory" in full:
            full["memory"] = full["memory"].ffill(limit=6)
        if "replicas" in full:
            full["replicas"] = full["replicas"].ffill(limit=30).bfill()
        else:
            full["replicas"] = 1.0
        for name in ZERO_FILL:
            if name in full:
                full[name] = full[name].fillna(0.0)
        for name in ("p50", "p90", "p99"):
            if name in full:
                full[name] = full[name].ffill(limit=12)
        def _col(name):
            if name in full:
                return full[name].fillna(0.0)
            return pd.Series(0.0, index=full.index)

        full["rps"] = _col("rps_http") + _col("rps_grpc")
        tot = full["rps"].replace(0.0, np.nan)
        full["err_frac"] = (_col("rps_5xx") / tot).fillna(0.0)
        full = full.dropna(subset=["cpu"])
        if full.empty:
            continue
        full["timestamp"] = full.index.tz_convert(TEHRAN).strftime("%Y-%m-%d %H:%M:%S")
        full["msname"] = ent
        out_frames.append(full)
        log(f"  [{ent}] {len(full)} rows")
    if not out_frames:
        raise SystemExit("no service frames assembled")
    df = pd.concat(out_frames, ignore_index=True)
    cols = (["timestamp", "msname", "cpu", "memory", "replicas", "rps_http",
             "rps_grpc", "rps", "rps_5xx", "err_frac", "p50", "p90", "p99",
             "req_bytes", "resp_bytes", "throttle", "net_rx", "net_tx"])
    cols = [c for c in cols if c in df.columns]
    df = df.sort_values(["msname", "timestamp"]).reset_index(drop=True)
    path = os.path.join(out_dir, "metrics_10s.csv")
    df.round({c: 5 for c in cols if c not in ("timestamp", "msname")}).to_csv(
        path, index=False)
    log(f"saved {len(df)} rows x {len(cols)} cols -> {path}")
    return path


def ccf(x, y, max_lag):
    """corr(x[t-k], y[t]) for k in [-max_lag, +max_lag]; k>0 => x leads."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 100 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    xz = (x - x.mean()) / x.std()
    yz = (y - y.mean()) / y.std()
    n = len(x)
    out = {}
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            a, b = xz[: n - k], yz[k:]
        else:
            a, b = xz[-k:], yz[: n + k]
        out[k] = float(np.dot(a, b) / len(a))
    return out


def excursions(s, thresh=2.5, refractory=30, direction="up"):
    """Onset indices of |z|-excursions with quiet period before onset."""
    z = (s - s.mean()) / (s.std() + 1e-12)
    sig = z > thresh if direction == "up" else z < -thresh
    onsets = []
    last = -refractory - 1
    i = 0
    while i < len(s):
        if sig[i] and i - last > refractory:
            if i == 0 or not sig[max(0, i - 6):i].any() or i - last > refractory:
                onsets.append(i)
                last = i
                i += refractory
                continue
        i += 1
    return onsets, z


def analyze(csv_path, out_dir):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    summary = []
    ccf_rows = []
    hor_rows = []
    spike_rows = []

    for svc, g in df.groupby("msname"):
        g = g.reset_index(drop=True)
        cpu = g["cpu"].to_numpy(float)
        n = len(cpu)
        stat = dict(service=svc, n=n, cpu_mean=float(np.mean(cpu)),
                    cpu_std=float(np.std(cpu)), cpu_min=float(np.min(cpu)),
                    cpu_max=float(np.max(cpu)))
        stat["acf_lag1"] = float(pd.Series(cpu).autocorr(lag=1) or 0.0)
        stat["acf_lag6"] = float(pd.Series(cpu).autocorr(lag=6) or 0.0)
        stat["acf_lag30"] = float(pd.Series(cpu).autocorr(lag=30) or 0.0)
        summary.append(stat)
        log(f"[{svc}] n={n} cpu={stat['cpu_mean']:.3f}±{stat['cpu_std']:.3f} "
            f"acf1={stat['acf_lag1']:.2f} acf6={stat['acf_lag6']:.2f}")

        avail = [c for c in CANDIDATES if c in g and np.isfinite(
            g[c].to_numpy(float)).sum() > 100 and g[c].std(skipna=True) > 0]
        for cand in avail:
            x = g[cand].to_numpy(float)
            cc = ccf(x, cpu, 60)
            if cc is None:
                continue
            leads = {k: v for k, v in cc.items() if k > 0}
            k_best = max(leads, key=lambda k: abs(leads[k]))
            row = dict(service=svc, candidate=cand,
                       best_lead_steps=k_best,
                       best_lead_corr=round(leads[k_best], 3),
                       corr_lag0=round(cc[0], 3))
            for h in (1, 3, 6, 12, 30, 60):
                row[f"lead_{h}"] = round(cc[h], 3)
            ccf_rows.append(row)

        for direction in ("up", "down"):
            onsets, _ = excursions(pd.Series(cpu), direction=direction)
            stat[f"n_{direction}"] = len(onsets)
            for cand in avail:
                xs = pd.Series(g[cand].to_numpy(float)).ffill().bfill()
                xz = (xs - xs.mean()) / (xs.std() + 1e-12)
                z = xz.to_numpy()
                hits, leads = 0, []
                for o in onsets:
                    # STRICT precedence: candidate quiet in [o-24, o-12),
                    # then crosses while CPU still quiet in [o-12, o).
                    q0, q1 = max(0, o - 24), max(0, o - 12)
                    quiet = np.abs(z[q0:q1])
                    if len(quiet) and np.nanmax(quiet) > 1.0:
                        continue  # already elevated: coincident, not leading
                    win = z[q1:o]
                    if len(win) == 0:
                        continue
                    if direction == "up":
                        over = np.nonzero(win > 2.0)[0]
                    else:
                        over = np.nonzero(win < -2.0)[0]
                    if len(over):
                        hits += 1
                        leads.append(o - (q1 + over[0]))
                denom = max(1, len(onsets))
                spike_rows.append(dict(
                    service=svc, direction=direction, candidate=cand,
                    n_events=len(onsets),
                    hit_rate_strict=round(hits / denom, 3),
                    median_lead_steps=round(float(np.median(leads)), 1)
                    if leads else None))

        # horizon sweep (needs variance + latency-free rows handled via ffill)
        if stat["cpu_std"] < 1e-3 or n < 2000:
            continue
        work = g[["cpu"] + [c for c in
                  ["rps", "p99", "err_frac", "req_bytes", "resp_bytes",
                   "throttle", "net_rx", "p50"] if c in g]].copy()
        work = work.ffill(limit=12).bfill()
        if work.isna().any().any():
            continue
        feat_cols = [c for c in work.columns if c != "cpu"]
        Xb = np.stack([work[c].to_numpy(float) for c in ["cpu"] + feat_cols],
                      axis=1)
        max_h = max(HORIZONS)
        ok = np.isfinite(Xb).all(axis=1)
        ok[:max(LAGS) + 1] = False
        Xb = Xb[ok]
        y_full = Xb[:, 0]
        nrows = len(Xb)
        cut = int(nrows * 0.7)
        for h in HORIZONS:
            F = np.concatenate(
                [np.roll(Xb, l, axis=0)[:, :] for l in LAGS], axis=1)
            F = F[max(LAGS): nrows - h]
            yt = y_full[max(LAGS) + h: nrows]
            if len(yt) < 500:
                continue
            cut_h = int(len(yt) * 0.7)
            scaler = StandardScaler().fit(F[:cut_h])
            Fs = scaler.transform(F)
            r_full = Ridge(alpha=1.0).fit(Fs[:cut_h], yt[:cut_h])
            p_full = r_full.predict(Fs[cut_h:])
            cpu_only = [i * Xb.shape[1] for i in range(len(LAGS))]
            r_cpu = Ridge(alpha=1.0).fit(Fs[:cut_h][:, cpu_only], yt[:cut_h])
            p_cpu = r_cpu.predict(Fs[cut_h:][:, cpu_only])
            p_pers = F[cut_h:, 0]
            y_t = yt[cut_h:]
            ss = ((y_t - y_t.mean()) ** 2).sum() + 1e-12
            r2 = lambda p: 1 - ((y_t - p) ** 2).sum() / ss
            mae = lambda p: float(np.abs(y_t - p).mean())
            hor_rows.append(dict(
                service=svc, horizon_steps=h, horizon_s=h * STEP,
                n_test=len(y_t),
                r2_persist=round(r2(p_pers), 4),
                r2_cpu_only=round(r2(p_cpu), 4),
                r2_full=round(r2(p_full), 4),
                mae_persist=round(mae(p_pers), 5),
                mae_full=round(mae(p_full), 5)))

    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "svc_stats.csv"),
                                 index=False)
    pd.DataFrame(ccf_rows).to_csv(os.path.join(out_dir, "ccf.csv"), index=False)
    pd.DataFrame(spike_rows).to_csv(os.path.join(out_dir, "spike_precursors.csv"),
                                    index=False)
    pd.DataFrame(hor_rows).to_csv(os.path.join(out_dir, "horizons.csv"),
                                  index=False)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for s in summary:
            f.write(repr(s) + "\n")
    log(f"wrote stats/ccf/precursors/horizons -> {out_dir}")
    plot_results(out_dir)
    return out_dir


def plot_results(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        hor = pd.read_csv(os.path.join(out_dir, "horizons.csv"))
    except Exception as e:
        log(f"plots skipped ({e})")
        return
    if hor.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for svc, g in hor.groupby("service"):
        g = g.sort_values("horizon_s")
        ax.plot(g["horizon_s"], g["r2_full"], "o-", label=f"{svc} full")
        ax.plot(g["horizon_s"], g["r2_persist"], "x--", alpha=0.5,
                label=f"{svc} persist")
    ax.set_xlabel("horizon (s)")
    ax.set_ylabel("test R2")
    ax.set_title("CPU predictability vs horizon (10s steps)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "r2_vs_horizon.png"), dpi=120)
    plt.close(fig)
    log("plot saved")


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
    ap.add_argument("--out-dir", default="analytics/data/leadlag_10s")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    if not args.analyze_only:
        if not args.end:
            end_ts = time.time()
        else:
            end_ts = parse_time(args.end)
        csv_path = export_window(parse_time(args.start), end_ts, args.out_dir)
    else:
        csv_path = args.csv or os.path.join(args.out_dir, "metrics_10s.csv")
    if not args.export_only:
        analyze(csv_path, args.out_dir)


if __name__ == "__main__":
    main()
