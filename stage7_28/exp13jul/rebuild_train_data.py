#!/usr/bin/env python3
"""Collect the PBScaler training dataset AFTER a load test, from Mimir.

The collector does not need to be attached while k6 runs: Alloy scrapes the
cluster at 5 s into Mimir, and bucket series live ~15-20 h, so the whole
measured window can be collected post-hoc. This script re-issues every query
collect_training_data.py runs per tick as ONE range query (step = interval)
over the window and stitches the same train_data.csv / metrics.csv schema,
including the slo_reward label. It also recovers starved captures (mode A).

Modes:
  A) recovery of a starved capture (original meta.json defines the window):
     python3 rebuild_train_data.py --dataset <ds_dir>
     -> writes train_data_rebuilt.csv / metrics_rebuilt.csv / meta_rebuilt.json

  B) post-hoc collection after a finished load test:
     python3 rebuild_train_data.py --start <epoch> --duration <s> \
         --interval 5 --out <ds_dir> [--workload <csv>]
     -> writes train_data.csv / metrics.csv / meta.json
     service list is derived from the cluster (stateless deployments), same
     as collect_training_data.py.

slo_reward: 1 = no violation, 0 = violation (front P90 > SLO), -1 = no data.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

NS = "train-ticket"
FRONT = "ts-ui-dashboard"
GATEWAY = "train-ticket-gateway-istio"
SLO_DEFAULT = 200.0


def parse_iso(s):
    """fromisoformat on py3.10 doesn't take trailing Z."""
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def q(url, params, timeout=300):
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}/api/v1/query_range?{qs}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    if d.get("status") != "success":
        raise RuntimeError(f"query failed: {params.get('query')}")
    return d["data"]["result"]


def series_map(result):
    out = {}
    for res in result:
        metric = res.get("metric", {})
        vals = {}
        for ts, v in res.get("values", []):
            try:
                vals[int(ts)] = float(v)
            except (ValueError, TypeError):
                vals[int(ts)] = float("nan")
        key = tuple(sorted((k, v) for k, v in metric.items() if k != "__name__"))
        out[key] = vals
    return out


def by(tsmap, ts):
    v = tsmap.get(ts)
    return v if v is not None else float("nan")


def svc_from_pod(pod, svcs):
    best = ""
    for s in svcs:
        if pod.startswith(s + "-") and len(s) > len(best):
            best = s
    return best


def stateless_svcs_from_cluster(namespace=NS):
    """Same derivation as KubernetesClient.get_svcs_without_state."""
    out = subprocess.run(
        ["kubectl", "get", "deployments", "-n", namespace, "-o", "name"],
        capture_output=True, text=True, timeout=60).stdout
    state_svcs = ("redis", "rabbitmq", "mongo", "mysql")
    svcs = []
    for line in out.splitlines():
        name = line.split("/", 1)[-1].strip()
        if not name or name == "loadgenerator" or name == "pbscaler":
            continue
        if any(s in name for s in state_svcs):
            continue
        svcs.append(name)
    svcs.sort()
    return svcs


def collect_window(args, svcs, t0, duration, interval, slo, front, gateway):
    ticks = [t0 + k * interval for k in range(duration // interval)]
    start, end = t0, t0 + duration - 1
    print(f"window {datetime.fromtimestamp(t0, tz=timezone.utc).isoformat()} "
          f".. +{duration}s ({len(ticks)} ticks @ {interval}s)")

    P = {"start": start, "end": end, "step": interval}

    def run(query):
        return series_map(q(args.mimir_url, dict(P, query=query)))

    iso = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).isoformat()

    # 1) per-svc qps
    qps_by_svc = {}
    for key, tm in run('sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m])) '
                       'by (destination_workload)' % NS).items():
        svc = dict(key).get("destination_workload")
        if svc:
            qps_by_svc[svc] = tm

    # 2) counts (spec replicas preferred, status fallback)
    cnt_spec, cnt_status = {}, {}
    for key, tm in run('kube_deployment_spec_replicas{namespace="%s"}' % NS).items():
        d = dict(key).get("deployment")
        if d:
            cnt_spec[d] = tm
    for key, tm in run('kube_deployment_status_replicas{namespace="%s"}' % NS).items():
        d = dict(key).get("deployment")
        if d:
            cnt_status[d] = tm

    # 3) cpu + mem per pod
    pod_cpu = [(dict(key)["pod"], tm)
               for key, tm in run('(sum(rate(container_cpu_usage_seconds_total{namespace="%s",'
                                  'container!~"POD|istio-proxy|alloy"}[1m])) by (pod))' % NS).items()
               if "pod" in dict(key)]
    pod_mem = [(dict(key)["pod"], tm)
               for key, tm in run('(sum(container_memory_usage_bytes{namespace="%s",'
                                  'container!~"POD|istio-proxy|alloy"}) by (pod)) / (1024*1024)' % NS).items()
               if "pod" in dict(key)]

    # 4) success rate
    sr_by_svc = {}
    for key, tm in run('(sum(rate(istio_requests_total{reporter="destination", '
                       'response_code!~"5.*",namespace="%s"}[1m])) by (destination_workload) / '
                       'sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m])) '
                       'by (destination_workload))' % (NS, NS)).items():
        svc = dict(key).get("destination_workload")
        if svc:
            sr_by_svc[svc] = tm

    # 5) per-svc latency quantiles (irate — mirrors get_svc_latency)
    lat_by_svc = {}
    for kind, qq in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
        r = run('histogram_quantile(%.2f, sum(irate(istio_request_duration_milliseconds_bucket'
                '{reporter="destination", destination_workload_namespace="%s"}[1m])) '
                'by (destination_workload, le))' % (qq, NS))
        for key, tm in r.items():
            svc = dict(key).get("destination_workload")
            if svc:
                lat_by_svc.setdefault(svc, {})[kind] = tm

    # 6) front p90 + avg (gateway_edge, service fallback)
    fp90_edge = run('histogram_quantile(0.90, sum(rate(istio_request_duration_milliseconds_bucket'
                    '{reporter="destination", destination_workload="%s", source_workload="%s"}[1m])) '
                    'by (destination_workload, source_workload, le))' % (front, gateway))
    fp90_svc = run('histogram_quantile(0.90, sum(rate(istio_request_duration_milliseconds_bucket'
                   '{reporter="destination", destination_workload="%s"}[1m])) '
                   'by (destination_workload, le))' % front)
    favg_edge = run('sum(rate(istio_request_duration_milliseconds_sum{reporter="destination", '
                    'destination_workload="%s", source_workload="%s"}[1m])) by (destination_workload) / '
                    'sum(rate(istio_request_duration_milliseconds_count{reporter="destination", '
                    'destination_workload="%s", source_workload="%s"}[1m])) by (destination_workload)'
                    % (front, gateway, front, gateway))
    favg_svc = run('sum(rate(istio_request_duration_milliseconds_sum{reporter="destination", '
                   'destination_workload="%s"}[1m])) by (destination_workload) / '
                   'sum(rate(istio_request_duration_milliseconds_count{reporter="destination", '
                   'destination_workload="%s"}[1m])) by (destination_workload)' % (front, front))

    # 7) per-svc avg
    savg_by_svc = {}
    for key, tm in run('sum(rate(istio_request_duration_milliseconds_sum{reporter="destination", '
                       'destination_workload_namespace="%s"}[1m])) by (destination_workload) / '
                       'sum(rate(istio_request_duration_milliseconds_count{reporter="destination", '
                       'destination_workload_namespace="%s"}[1m])) by (destination_workload)'
                       % (NS, NS)).items():
        svc = dict(key).get("destination_workload")
        if svc:
            savg_by_svc[svc] = tm

    edge_key = (("destination_workload", front), ("source_workload", gateway))

    def first_val(maps, ts):
        for tm in maps:
            if ts in tm and tm[ts] == tm[ts]:
                return tm[ts]
        return float("nan")

    n_viol = n_noviol = n_nodata = front_seen = 0
    rows_train, rows_full = [], []
    for ts in ticks:
        rt = {"timestamp": iso(ts), "front_p90": float("nan"), "front_avg": float("nan"),
              "front_source": "", "slo_reward": -1, "total_qps": 0.0}
        rf = dict(rt)
        total = 0.0
        for s in svcs:
            qv = by(qps_by_svc.get(s, {}), ts)
            if qv != qv:
                qv = 0.0
            total += qv
            cv = by(cnt_spec.get(s, {}), ts)
            if cv != cv:
                cv = by(cnt_status.get(s, {}), ts)
            cv = int(round(cv)) if cv == cv else 0
            cpu = sum(by(tm, ts) for p, tm in pod_cpu if svc_from_pod(p, svcs) == s)
            mem = sum(by(tm, ts) for p, tm in pod_mem if svc_from_pod(p, svcs) == s)
            rt[s + "&qps"] = qv
            rt[s + "&count"] = cv
            rf[s + "&qps"] = qv
            rf[s + "&count"] = cv
            rf[s + "&cpu_usage"] = round(cpu, 9)
            rf[s + "&mem_usage"] = round(mem, 6)
            for kind in ("p50", "p90", "p99"):
                rf[s + "&" + kind] = by(lat_by_svc.get(s, {}).get(kind, {}), ts)
            rf[s + "&avg"] = by(savg_by_svc.get(s, {}), ts)
            rf[s + "&success_rate"] = by(sr_by_svc.get(s, {}), ts)
        rt["total_qps"] = rf["total_qps"] = total

        p90 = by(fp90_edge.get(edge_key, {}), ts)
        if p90 == p90:
            src = "gateway_edge"
        else:
            p90 = by(fp90_svc.get((("destination_workload", front),), {}), ts)
            src = "service" if p90 == p90 else ""
        favg = by(favg_edge.get((("destination_workload", front),), {}), ts)
        if favg != favg:
            favg = by(favg_svc.get((("destination_workload", front),), {}), ts)

        rt["front_p90"] = rf["front_p90"] = p90
        rt["front_avg"] = rf["front_avg"] = favg
        rt["front_source"] = rf["front_source"] = src
        if p90 == p90:
            front_seen += 1
            if rt.get(front + "&qps", 0.0) > 0:
                violated = p90 > slo
                rt["slo_reward"] = rf["slo_reward"] = 0 if violated else 1
                n_viol += int(violated)
                n_noviol += int(not violated)
        else:
            rt["front_p90"] = rf["front_p90"] = float("nan")
            n_nodata += 1
        rows_train.append(rt)
        rows_full.append(rf)

    return rows_train, rows_full, {
        "front_p90_samples": front_seen, "no_data_rows": n_nodata,
        "labels_violation": n_viol, "labels_no_violation": n_noviol,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="mode A: dataset dir with meta.json (recovery)")
    ap.add_argument("--start", type=int, help="mode B: window start epoch")
    ap.add_argument("--duration", type=int, help="mode B: window duration s")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--out", help="mode B: output dataset dir")
    ap.add_argument("--workload", default="", help="mode B: workload csv (recorded in meta)")
    ap.add_argument("--mimir-url", default=os.environ.get(
        "PBS_MIMIR_URL", "http://127.0.0.1:18080/prometheus"))
    ap.add_argument("--suffix", default="_rebuilt")
    args = ap.parse_args()

    if args.dataset:
        ds = args.dataset
        meta = json.load(open(os.path.join(ds, "meta.json")))
        svcs = meta["services"]
        t0 = int(parse_iso(meta["started_at"]).timestamp())
        duration = int(meta["duration_s"])
        interval = int(meta["interval_s"])
        slo = float(meta["SLO_ms"])
        front = meta["front_svc"]
        out_meta_name = f"meta{args.suffix}.json"
        out_train = os.path.join(ds, f"train_data{args.suffix}.csv")
        out_full = os.path.join(ds, f"metrics{args.suffix}.csv")
        prov = "mimir range replay of a starved capture (mode A)"
    elif args.start is not None and args.duration:
        os.makedirs(args.out, exist_ok=True)
        svcs = stateless_svcs_from_cluster()
        t0 = args.start
        duration = args.duration
        interval = args.interval
        slo = float(os.environ.get("PBS_SLO", SLO_DEFAULT))
        front = FRONT
        out_meta_name = "meta.json"
        out_train = os.path.join(args.out, "train_data.csv")
        out_full = os.path.join(args.out, "metrics.csv")
        prov = "post-hoc collection from Mimir after load test (mode B)"
        meta = {"started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
                "duration_s": duration, "interval_s": interval,
                "namespace": NS, "SLO_ms": slo, "front_svc": front,
                "services": svcs, "workload_csv": args.workload,
                "mimir_url": args.mimir_url,
                "note": "slo_reward: 1=no violation, 0=violation, -1=no front traffic"}
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    else:
        ap.error("need --dataset (mode A) or --start + --duration --out (mode B)")
        return 2

    rows_train, rows_full, stats = collect_window(
        args, svcs, t0, duration, interval, slo, front, GATEWAY)

    train_cols = (["timestamp", "front_p90", "front_avg", "front_source", "slo_reward",
                   "total_qps"] + [s + f"&{k}" for s in svcs for k in ("qps", "count")])
    full_cols = (["timestamp", "front_p90", "front_avg", "front_source", "slo_reward",
                  "total_qps"] + [s + f"&{k}" for s in svcs
                                  for k in ("qps", "count", "cpu_usage", "mem_usage",
                                            "p50", "avg", "p90", "p99", "success_rate")])
    with open(out_train, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=train_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_train)
    with open(out_full, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=full_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_full)

    cov = 100.0 * stats["front_p90_samples"] / len(rows_train) if rows_train else 0.0
    rmeta = {"provenance": prov, "started_at": meta["started_at"],
             "duration_s": duration, "interval_s": interval, "rows": len(rows_train),
             "front_p90_samples": stats["front_p90_samples"],
             "no_data_rows": stats["no_data_rows"],
             "coverage_pct": round(cov, 2),
             "labels_violation": stats["labels_violation"],
             "labels_no_violation": stats["labels_no_violation"],
             "SLO_ms": slo, "front_svc": front}
    if args.dataset:
        # mode A: keep the recovery report next to the originals
        with open(os.path.join(os.path.dirname(out_train), out_meta_name), "w") as f:
            json.dump(rmeta, f, indent=2)
    else:
        # mode B: merge into the collector-schema meta.json so downstream
        # validation tooling sees the usual fields
        meta.update(rmeta)
        meta["result"] = "COMPLETE" if cov >= 95 else "INCOMPLETE"
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    print(json.dumps(rmeta, indent=2))
    print(f"wrote {out_train}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
