#!/usr/bin/env python3

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

NS = "train-ticket"
CF = 'container!="istio-proxy",container!="POD"'

Q_CPU_NUM = ('sum by (pod) (rate(container_cpu_usage_seconds_total'
             '{namespace="%s", %s}[15s]))' % (NS, CF))
Q_CPU_DEN = ('sum by (pod) (kube_pod_container_resource_requests'
             '{resource="cpu", namespace="%s", %s})' % (NS, CF))
Q_HTTP = ('sum(rate(istio_requests_total{reporter="destination", '
          'request_protocol="http", destination_workload_namespace="%s"}[15s])) '
          'by (destination_workload)' % NS)
Q_GRPC = ('sum(rate(istio_requests_total{reporter="destination", '
          'request_protocol="grpc", destination_workload_namespace="%s"}[15s])) '
          'by (destination_workload)' % NS)
Q_EDGES = ('sum by (source_workload, destination_workload) '
           '(rate(istio_requests_total{reporter="source", '
           'source_workload_namespace="%s", '
           'destination_workload_namespace="%s"}[15s]))' % (NS, NS))


def log(msg):
    print(msg, flush=True)


def pod_to_deployment(pod):
    parts = pod.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:-2])
    return pod


def fetch_range(prom, query, start_ts, end_ts, step="1s"):
    r = requests.get("%s/api/v1/query_range" % prom,
                     params={"query": query, "start": int(start_ts),
                             "end": int(end_ts), "step": step},
                     timeout=300)
    r.raise_for_status()
    p = r.json()
    if p.get("status") != "success":
        raise RuntimeError("query failed: %s" % p.get("error"))
    return p["data"]["result"]


def frame_by_label(results, label):
    out = {}
    for res in results:
        ent = res["metric"].get(label, "")
        pts = [(int(t), float(v)) for t, v in res["values"]
               if v not in ("NaN", "+Inf", "-Inf", "nan", "inf", "-inf")]
        if pts:
            s = pd.Series(dict(pts)).sort_index()
            out[ent] = s[~s.index.duplicated(keep="last")]
    return out


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--minutes", type=float, default=15)
    pa.add_argument("--start", type=float, default=None,
                    help="explicit window start (epoch s); with --end, backdated settled export")
    pa.add_argument("--end", type=float, default=None,
                    help="explicit window end (epoch s)")
    pa.add_argument("--prom", default="http://localhost:9090")
    pa.add_argument("--out", default=None)
    pa.add_argument("--half", type=int, default=120,
                    help="zoom half-width in seconds")
    pa.add_argument("--jump_win", type=int, default=10,
                    help="jump window in seconds for biggest-transition search")
    args = pa.parse_args()

    end = args.end if args.end else time.time()
    start = args.start if args.start else end - args.minutes * 60
    log("fetching 1s data: %.0fs window (%s..%s)" % (
        end - start,
        pd.to_datetime(int(start), unit="s"),
        pd.to_datetime(int(end), unit="s")))

    cpu_num = frame_by_label(fetch_range(args.prom, Q_CPU_NUM, start, end), "pod")
    cpu_den = frame_by_label(fetch_range(args.prom, Q_CPU_DEN, start, end), "pod")
    http = frame_by_label(fetch_range(args.prom, Q_HTTP, start, end), "destination_workload")
    grpc = frame_by_label(fetch_range(args.prom, Q_GRPC, start, end), "destination_workload")
    edges = fetch_range(args.prom, Q_EDGES, start, end)
    log("pods=%d http_wl=%d edges_series=%d" % (len(cpu_num), len(http), len(edges)))

    grid = np.arange(int(start), int(end) + 1)

    dep_cpu = {}
    for pod, num in cpu_num.items():
        den = cpu_den.get(pod)
        if den is None:
            continue
        dep = pod_to_deployment(pod)
        num_g = num.reindex(grid).ffill(limit=3)
        den_g = den.reindex(grid).ffill(limit=5).replace(0.0, np.nan)
        cpu = (num_g / den_g).clip(lower=0.0)
        dep_cpu.setdefault(dep, []).append(cpu)
    dep_cpu = {d: pd.concat(v, axis=1).mean(axis=1) for d, v in dep_cpu.items() if v}

    wl_rps = {}
    for wl in set(http) | set(grpc):
        h = http.get(wl, pd.Series(dtype=float)).reindex(grid).fillna(0.0)
        gr = grpc.get(wl, pd.Series(dtype=float)).reindex(grid).fillna(0.0)
        wl_rps[wl] = (h + gr).fillna(0.0)

    up = {}
    for res in edges:
        dst = res["metric"].get("destination_workload", "")
        s = pd.Series({int(t): float(v) for t, v in res["values"]
                       if v not in ("NaN", "+Inf", "-Inf")}).sort_index()
        s = s[~s.index.duplicated(keep="last")].reindex(grid).fillna(0.0)
        up[dst] = up.get(dst, 0.0) + s if isinstance(up.get(dst), float) else (
            up[dst] + s if dst in up else s)

    log("deployments cpu=%d rps=%d" % (len(dep_cpu), len(wl_rps)))
    overlap = sorted(set(dep_cpu) & set(wl_rps))
    log("cpu+rps overlap=%d e.g. %s" % (len(overlap), overlap[:5]))

    JW = args.jump_win
    best = (0.0, None, -1)
    for dep in overlap:
        c = dep_cpu[dep].bfill().ffill().to_numpy(float)
        if np.isnan(c).all():
            continue
        j = np.abs(c[JW:] - c[:-JW])
        k = int(np.nanargmax(j))
        if float(j[k]) > best[0]:
            best = (float(j[k]), dep, k)
    jump, svc, k = best
    if svc is None:
        raise SystemExit("no usable series")
    log("SELECTED %s biggest %ds jump=%.3f at +%ds" % (svc, JW, jump, k))

    c = dep_cpu[svc].bfill().ffill().to_numpy(float)
    r = wl_rps[svc].to_numpy(float)
    u = up.get(svc, pd.Series(0.0, index=grid)).reindex(grid).fillna(0.0).to_numpy(float)
    center = k + JW // 2
    lo, hi = max(0, center - args.half), min(len(grid), center + args.half)
    xx = pd.to_datetime(grid[lo:hi], unit="s")

    fig, axes = plt.subplots(3, 1, figsize=(20, 11), sharex=True)
    fig.suptitle("%s -- 1s BURST ZOOM (high load, biggest %ds jump=%.3f, centered +/-%ds, NO prediction)" % (
        svc, JW, jump, args.half), fontsize=13, fontweight="bold", y=0.99)
    ci = args.half
    ax = axes[0]
    ax.plot(xx, c[lo:hi], color="#1976D2", lw=1.1, label="Actual CPU (per deployment, 1s steps)")
    ax.axvline(xx[ci if ci < len(xx) else -1], color="red", ls="--", lw=1.2, alpha=0.7,
               label="transition center")
    ax.set_ylim(0, max(1.0, float(np.nanmax(c[lo:hi])) * 1.15))
    ax.set_ylabel("CPU util")
    ax.set_title("CPU utilization (1s resolution)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)
    axes[1].plot(xx, r[lo:hi], color="#2E7D32", lw=1.0,
                 label="Deployment MCR = http+grpc RPS (raw, 1s steps)")
    axes[1].axvline(xx[ci if ci < len(xx) else -1], color="red", ls="--", lw=1.2, alpha=0.7)
    axes[1].set_ylabel("RPS")
    axes[1].set_title("MCR of the deployment (NOT normalized, raw requests/s)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.2)
    axes[2].plot(xx, u[lo:hi], color="#6A1B9A", lw=1.0,
                 label="Upper-services MCR sum (raw, 1s steps)")
    axes[2].axvline(xx[ci if ci < len(xx) else -1], color="red", ls="--", lw=1.2, alpha=0.7)
    axes[2].set_ylabel("RPS")
    axes[2].set_title("Sum of MCR of upper (caller) services (NOT normalized, raw requests/s)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.2)
    axes[2].xaxis.set_major_locator(mdates.SecondLocator(interval=30))
    axes[2].xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    for tick in axes[2].get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    axes[2].grid(True, which="minor", alpha=0.12)
    fig.tight_layout()
    out = args.out or ("/tmp/opencode/report10s/plots/burst_1s_zoom_%s.png" % svc)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log("wrote %s" % out)


if __name__ == "__main__":
    main()

