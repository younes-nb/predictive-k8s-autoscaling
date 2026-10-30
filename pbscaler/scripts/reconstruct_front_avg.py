#!/usr/bin/env python3
"""Reconstruct Run-A front_avg (and per-service avg) from Mimir blocks.

Run A (HPA 4h) started before the collector recorded front_avg/&avg columns
(those landed in v1.2.5-era collector). Same PromQL + completion semantics
(rate[1m] over istio duration sum/count) evaluated at 5 s over the run window;
output CSV joins the Run-A dataset dir. Run promptly (~18-20h retention).
"""
import argparse
import datetime as dt
import json
import sys
import time
import urllib.request

sys.path.insert(0, '/users/younes02/predictive-k8s-autoscaling/pbscaler')
from config.Config import Config  # noqa: E402

CFG = Config()


def query_range(sql: str, start_s: float, end_s: float, step: str):
    url = (CFG.prom_range_url + f"?query={urllib.parse.quote(sql)}"
           f"&start={start_s}&end={end_s}&step={step}")
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    if data.get('status') != 'success':
        raise RuntimeError(str(data)[:300])
    return data['data']['result']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Run A train_data.csv (for the time range)')
    ap.add_argument('--out', required=True, help='output CSV: timestamp,front_avg_ms')
    ap.add_argument('--step', default='15s')
    args = ap.parse_args()

    import csv
    ts = [r['timestamp'] for r in csv.DictReader(open(args.csv))]
    t_min = dt.datetime.fromisoformat(ts[0]).timestamp()
    t_max = dt.datetime.fromisoformat(ts[-1]).timestamp()
    front = CFG.front_svc
    gw = CFG.gateway_name
    mean_sql = (f'sum(rate(istio_request_duration_milliseconds_sum{{reporter="destination",'
                f'destination_workload="{front}", source_workload="{gw}"}}[1m]))'
                f' / sum(rate(istio_request_duration_milliseconds_count{{reporter="destination",'
                f'destination_workload="{front}", source_workload="{gw}"}}[1m]))')
    res = query_range(mean_sql, t_min - 3, t_max + 3, args.step)
    if not res:
        raise SystemExit('empty range result — did retention lapse?')
    pts = [(float(v[0]), float(v[1])) for v in res[0]['values']]  # epoch seconds, ms
    with open(args.out, 'w') as f:
        f.write('timestamp,front_avg_ms\n')
        for t, v in pts:
            f.write(f"{dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()},{v:.6f}\n")
    import statistics as st
    lab = [v for _, v in pts]
    print(f"front_avg reconstructed: n={len(lab)} med={st.median(lab):.1f} "
          f"p95={sorted(lab)[int(len(lab)*0.95)]:.1f} max={max(lab):.1f} ms")
    print(f"-> {args.out}")


if __name__ == '__main__':
    main()  # style: keeps import-time side effects minimal
