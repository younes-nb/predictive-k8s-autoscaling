#!/usr/bin/env python3
"""Collect the PBScaler SLO-violation-predictor training dataset from the
live cluster while the k6 load test drives it.

Paper reference (Sec. III-C2 + IV-C2): Prometheus gathers each
microservice's workload (QPS) and P90 tail latency at regular intervals while
the K8s default autoscaler (HPA) performs elastic scaling; each interval is
labeled by comparing the *front-end* microservice's P90 tail latency with the
SLO:

    slo_reward = 1  if front P90 <= SLO (no violation)
    slo_reward = 0  if front P90 >  SLO (violation)
    slo_reward = -1 if no front traffic in that interval (unlabeled)

Outputs (in --out, default data/<runid> under pbscaler/):
    train_data.csv    predictor training schema — one row per interval:
                      timestamp, <svc>&qps, <svc>&count, front&p90,
                      gateway&qps (user arrival rate), slo_reward
    metrics.csv       everything the collector saw (adds p50/p90/p99 per
                      svc, cpu/mem usage per svc, success rate) for
                      post-hoc analysis / richer predictors
    meta.json         run parameters (SLO, interval, workload csv, svc list)

Read-only: it never touches replicas or HPAs — HPAs stay enabled exactly as
the paper's data generation (built on the K8s default autoscaler).

Usage:
  cd pbscaler && ./.venv/bin/python scripts/collect_training_data.py \
        --duration 900 --interval 10 \
        --workload /proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv
  # local (runner) via Mimir port-forward:
  PBS_MIMIR_URL=http://127.0.0.1:18080 ...
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.Config import Config, get_env_int               # noqa: E402
from util.KubernetesClient import KubernetesClient          # noqa: E402
from util.PrometheusClient import PrometheusClient          # noqa: E402

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import requests                                             # noqa: E402


def svc_from_pod(pod: str, svcs_sorted: list) -> str:
    """longest deployment-name prefix match for a pod name"""
    for svc in svcs_sorted:
        if pod.startswith(svc + "-"):
            return svc
    return None


def per_svc_resource(prom_util: PrometheusClient, svcs_sorted: list) -> dict:
    """per-deployment CPU (cores, 1m rate) and memory (MB) via one query each,
    aggregated pod->deployment client-side."""
    out = {}
    ns = prom_util.namespace
    cpu = prom_util.execute_prom(prom_util.prom_no_range_url,
                                 '(sum(rate(container_cpu_usage_seconds_total{namespace="%s",container!~"POD|istio-proxy|alloy"}[1m])) by (pod))' % ns)
    for res in cpu:
        pod = res['metric'].get('pod')
        if not pod:
            continue
        d = res.get('value', [0, 0])
        out.setdefault(svc_from_pod(pod, svcs_sorted), 0.0)
        out[svc_from_pod(pod, svcs_sorted)] += float(d[1]) if len(d) > 1 else 0.0
    mem = prom_util.execute_prom(prom_util.prom_no_range_url,
                                 '(sum(container_memory_usage_bytes{namespace="%s",container!~"POD|istio-proxy|alloy"}) by (pod)) / (1024*1024)' % ns)
    memout = {}
    for res in mem:
        pod = res['metric'].get('pod')
        if not pod:
            continue
        d = res.get('value', [0, 0])
        svc = svc_from_pod(pod, svcs_sorted)
        memout.setdefault(svc, 0.0)
        memout[svc] += float(d[1]) if len(d) > 1 else 0.0
    return out, memout


def success_rate(prom_util: PrometheusClient) -> dict:
    sql = ('(sum(rate(istio_requests_total{reporter="destination", response_code!~"5.*",namespace="%s"}[1m])) by (destination_workload) / '
           'sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m])) by (destination_workload))') % (
        prom_util.namespace, prom_util.namespace)
    out = {}
    for res in prom_util.execute_prom(prom_util.prom_no_range_url, sql):
        svc = res['metric'].get('destination_workload')
        if svc:
            out[svc] = prom_util._instant([res])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration', type=int, default=None,
                    help='seconds to collect (default: PBS_DURATION or 1800)')
    ap.add_argument('--interval', type=int, default=10,
                    help='collection interval seconds (paper: 5)')
    ap.add_argument('--out', default=None,
                    help='output dir (default pbscaler/data/traindata_<ts>)')
    ap.add_argument('--workload', default=None,
                    help='MCR csv the load test replays (recorded in meta.json only)')
    args = ap.parse_args()

    config = Config()
    duration = args.duration or get_env_int('PBS_DURATION', 1800)
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'traindata_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)

    # sanity: Mimir reachable
    try:
        requests.get(config.prom_no_range_url, params={'query': 'up'},
                     timeout=10).raise_for_status()
    except Exception as e:
        sys.exit(f'Prometheus/Mimir API unreachable at {config.prom_no_range_url}: {e}\n'
                 f'For local runs: kubectl -n monitoring port-forward svc/mimir 18080:8080 '
                 f'and PBS_MIMIR_URL=http://127.0.0.1:18080')

    prom_util = PrometheusClient(config)
    k8s_util = KubernetesClient(config)

    # services follow the deployments of the namespace (frozen at start)
    svcs = k8s_util.get_svcs_without_state()
    print(f'namespace={config.namespace} stateless services={len(svcs)}')
    print(f'SLO={config.SLO} ms front={config.front_svc} interval={args.interval}s duration={duration}s')
    print(f'out={out_dir}')
    front = config.front_svc

    meta = {
        'started_at': datetime.utcnow().isoformat() + 'Z',
        'duration_s': duration,
        'interval_s': args.interval,
        'namespace': config.namespace,
        'SLO_ms': config.SLO,
        'front_svc': front,
        'services': svcs,
        'workload_csv': args.workload,
        'mimir_url': config.prom_no_range_url,
        'note': 'slo_reward: 1=no violation, 0=violation, -1=no front traffic',
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    train_path = os.path.join(out_dir, 'train_data.csv')
    full_path = os.path.join(out_dir, 'metrics.csv')
    train_f = None
    train_w = None
    full_f = None
    full_w = None

    def cols_row(extra=None):
        row = {'timestamp': pd.Timestamp.now(tz='UTC').isoformat()}
        row['front_p90'] = np.nan
        row['front_avg'] = np.nan
        row['front_source'] = ''
        row['slo_reward'] = -1
        row['total_qps'] = 0.0
        for s in svcs:
            row[s + '&qps'] = 0.0
            row[s + '&count'] = 0
            row[s + '&cpu_usage'] = 0.0
            row[s + '&mem_usage'] = 0.0
            row[s + '&p50'] = np.nan
            row[s + '&avg'] = np.nan
            row[s + '&p90'] = np.nan
            row[s + '&p99'] = np.nan
            row[s + '&success_rate'] = np.nan
        if extra:
            row.update(extra)
        return row

    n_rows = 0
    n_violation = 0
    n_noviolation = 0
    front_seen = 0
    t0 = time.time()
    next_tick = t0
    print('\nCollecting — start the k6 load test now (load_testing/run_test.sh)\n')

    while time.time() - t0 < duration:
        next_tick += args.interval
        row = cols_row()
        ts = row['timestamp']
        try:
            # workloads + replica counts + resources + success rate
            qps = prom_util.get_svc_qps()
            counts = k8s_util.get_all_counts()
            cpu_mb, mem_mb = per_svc_resource(prom_util, svcs)
            sr = success_rate(prom_util)
            lat = prom_util.get_svc_latency()
            front_p90 = prom_util.get_front_p90(front, config.gateway_name)
            front_avg = prom_util.get_front_avg(front, config.gateway_name)
            avg_map = prom_util.get_svc_avg()

            total = 0.0
            for s in svcs:
                q = qps.get(s + '&qps', 0.0)
                row[s + '&qps'] = q
                total += q
                row[s + '&count'] = counts.get(s, 0)
                row[s + '&cpu_usage'] = round(cpu_mb.get(s, 0.0), 9)
                row[s + '&mem_usage'] = round(mem_mb.get(s, 0.0), 6)
                row[s + '&p50'] = lat.get(s + '&p50', np.nan)
                row[s + '&avg'] = avg_map.get(s, np.nan)
                row[s + '&p90'] = lat.get(s + '&p90', np.nan)
                row[s + '&p99'] = lat.get(s + '&p99', np.nan)
                row[s + '&success_rate'] = sr.get(s, np.nan)
            row['total_qps'] = total
            row['front_p90'] = front_p90.get('front_p90', np.nan)
            row['front_avg'] = front_avg
            row['front_source'] = front_p90.get('front_source', '')

            if row['front_p90'] == row['front_p90']:  # not nan
                front_seen += 1
                if row[front + '&qps'] > 0:
                    violated = row['front_p90'] > config.SLO
                    row['slo_reward'] = 0 if violated else 1
                    n_violation += int(violated)
                    n_noviolation += int(not violated)
            else:
                row['front_p90'] = np.nan

            # lazy header creation (service list frozen at start)
            if train_f is None:
                train_cols = ['timestamp', 'front_p90', 'front_avg', 'front_source', 'slo_reward',
                              'total_qps'] + [s + f'&{k}' for s in svcs for k in ('qps', 'count')]
                train_f = open(train_path, 'w')
                train_w = csv.DictWriter(train_f, fieldnames=train_cols, extrasaction='ignore')
                train_w.writeheader()
                full_cols = list(row.keys())
                full_f = open(full_path, 'w')
                full_w = csv.DictWriter(full_f, fieldnames=full_cols, extrasaction='ignore')
                full_w.writeheader()
            train_w.writerow(row)
            full_w.writerow(row)
            train_f.flush()
            full_f.flush()
            n_rows += 1
            ts_short = ts[11:19]
            print(f'[{ts_short}Z] row {n_rows}: front_p90={row["front_p90"]} '
                  f'qps_total={total:.2f} slo_reward={row["slo_reward"]}', flush=True)
        except Exception as e:
            print(f'[{ts[11:19]}Z] tick FAILED: {type(e).__name__}: {e}', flush=True)

        sleep = next_tick - time.time()
        if sleep > 0:
            time.sleep(sleep)

    for fh in (train_f, full_f):
        if fh:
            fh.close()

    # ---- verification summary -------------------------------------------
    print('\n================ collection summary ================')
    print(f'rows written            : {n_rows} (+ header) -> {train_path}')
    print(f'front P90 samples seen  : {front_seen}')
    print(f'slo labels: violation=0 : {n_violation}   no-violation=1: {n_noviolation}')
    ok = n_rows > 3 and front_seen > 0 and (n_violation + n_noviolation) > 0
    print('RESULT:', 'OK — dataset usable' if ok else
          'INCOMPLETE — check istio telemetry/traffic above (see pbscaler README)')
    meta['rows'] = n_rows
    meta['front_p90_samples'] = front_seen
    meta['labels_violation'] = n_violation
    meta['labels_no_violation'] = n_noviolation
    meta['ended_at'] = datetime.utcnow().isoformat() + 'Z'
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=2)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
