import argparse
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import requests

MIMIR_URL = "http://localhost:8080"
MIMIR_SERVICE = "svc/mimir"
MIMIR_NAMESPACE = "monitoring"
LOCAL_PORT = 8080
STEP_SECONDS = 60
OUTPUT_DIR = "/proj/k8sautoscaledl-PG0"
OUTPUT_NAME = "hpa_logs.csv"

NAMESPACE = "train-ticket"

HPA_TARGET = 0.80

def build_queries(ns, cf):
    return {
    "replicas": {
        "query": f'kube_deployment_status_replicas_available{{namespace="{ns}"}}',
        "labels": ["deployment"],
    },
    "replicas_desired": {
        "query": f'kube_deployment_status_replicas{{namespace="{ns}"}}',
        "labels": ["deployment"],
    },
    "RPS_HTTP": {
        "query": f'sum(rate(istio_requests_total{{reporter="destination", request_protocol="http", destination_workload_namespace="{ns}"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "RPS_GRPC": {
        "query": f'sum(rate(istio_requests_total{{reporter="destination", request_protocol="grpc", destination_workload_namespace="{ns}"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "RPS_5XX": {
        "query": f'sum(rate(istio_requests_total{{reporter="destination", destination_workload_namespace="{ns}", response_code=~"5.*"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "FLAGS": {
        "query": f'sum(rate(istio_requests_total{{reporter="destination", destination_workload_namespace="{ns}", response_flags!="-"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "CPU": {
        "query": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{ns}", {cf}}}[1m])) / sum by (pod) (kube_pod_container_resource_requests{{resource="cpu", namespace="{ns}", {cf}}})',
        "labels": ["pod"],
    },
    "CPU_RAW": {
        "query": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{ns}", {cf}}}[1m]))',
        "labels": ["pod"],
    },
    "LIMITS_CPU": {
        "query": f'sum by (pod) (kube_pod_container_resource_limits{{resource="cpu", namespace="{ns}", {cf}}})',
        "labels": ["pod"],
    },
    "Memory": {
        "query": f'sum by (pod) (container_memory_working_set_bytes{{namespace="{ns}", {cf}}}) / sum by (pod) (kube_pod_container_resource_requests{{resource="memory", namespace="{ns}", {cf}}})',
        "labels": ["pod"],
    },
    "MEM_RAW": {
        "query": f'sum by (pod) (container_memory_working_set_bytes{{namespace="{ns}", {cf}}})',
        "labels": ["pod"],
    },
    "LIMITS_MEM": {
        "query": f'sum by (pod) (kube_pod_container_resource_limits{{resource="memory", namespace="{ns}", {cf}}})',
        "labels": ["pod"],
    },
    "P99": {
        "query": f'histogram_quantile(0.99, sum by (destination_workload, le) (rate(istio_request_duration_milliseconds_bucket{{reporter="destination", destination_workload_namespace="{ns}"}}[2m])))',
        "labels": ["destination_workload"],
    },
    "REQ_BYTES": {
        "query": f'sum(rate(istio_request_bytes_sum{{reporter="destination", destination_workload_namespace="{ns}"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "RESP_BYTES": {
        "query": f'sum(rate(istio_response_bytes_sum{{reporter="destination", destination_workload_namespace="{ns}"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "THROTTLED": {
        "query": f'sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{{namespace="{ns}"}}[2m]))',
        "labels": ["pod"],
    },
    "PERIODS": {
        "query": f'sum by (pod) (rate(container_cpu_cfs_periods_total{{namespace="{ns}"}}[2m]))',
        "labels": ["pod"],
    },
    "NET_RX": {
        "query": f'sum by (pod) (rate(container_network_receive_bytes_total{{namespace="{ns}", pod!=""}}[1m]))',
        "labels": ["pod"],
    },
    "RESTARTS": {
        "query": f'max by (pod) (kube_pod_container_status_restarts_total{{namespace="{ns}"}})',
        "labels": ["pod"],
    },
    "PGFAULT": {
        "query": f'sum by (pod) (rate(container_memory_failures_total{{namespace="{ns}", {cf}, failure_type="pgfault"}}[2m]))',
        "labels": ["pod"],
    },
    "PGMAJFAULT": {
        "query": f'sum by (pod) (rate(container_memory_failures_total{{namespace="{ns}", {cf}, failure_type="pgmajfault"}}[2m]))',
        "labels": ["pod"],
    },
    "EDGES": {
        "query": f'sum by (source_workload, destination_workload) (rate(istio_requests_total{{reporter="source", source_workload_namespace="{ns}", destination_workload_namespace="{ns}"}}[1m]))',
        "labels": ["destination_workload"],
        "extra_labels": ["source_workload"],
    },
    "Q_ACTIVE_IN": {
        "query": 'sum(envoy_cluster_upstream_rq_active{cluster_name=~"inbound.*"}) by (pod)',
        "labels": ["pod"],
    },
    "Q_PENDING_IN": {
        "query": 'sum(envoy_cluster_upstream_rq_pending_active{cluster_name=~"inbound.*"}) by (pod)',
        "labels": ["pod"],
    },
    "Q_ACTIVE_OUT": {
        "query": 'sum(envoy_cluster_upstream_rq_active{cluster_name!~"inbound.*|xds-grpc"}) by (pod)',
        "labels": ["pod"],
    },
    "Q_PENDING_OUT": {
        "query": 'sum(envoy_cluster_upstream_rq_pending_active{cluster_name!~"inbound.*|xds-grpc"}) by (pod)',
        "labels": ["pod"],
    },
}

FINAL_COLUMNS = ["timestamp", "msname", "cpu_utilization", "memory_utilization",
                 "replicas", "desired_replicas", "unavailable", "restart_rate",
                 "http_mcr", "providerrpc_mcr", "rps_total", "caller_rps_max",
                 "n_callers", "upstream_rps_sum", "root_rps", "req_byte_rate",
                 "resp_byte_rate", "req_bytes_per_req", "resp_bytes_per_req",
                 "err_rate", "err_frac", "flag_rate", "flag_frac", "queue_in",
                 "queue_for", "active_in", "active_for", "p99_latency",
                 "throttle_ratio", "net_rx", "rps_z30", "rps_slope5",
                 "ewma_gap", "cpu_slope3", "mem_delta5", "tod_sin", "tod_cos",
                 "neigh_cpu_mean", "neigh_cpu_slope3", "neigh_rps_z30_mean",
                 "neigh_rps_slope5_mean", "from_frontend", "frontend_rps",
                 "mesh_rps", "concurrency", "scale_recency", "vol_rps",
                 "vol_cpu", "cpu_lim", "mem_lim", "pgfault", "pgmajfault"]


def is_reachable(url):
    for path in ("/ready", "/-/healthy"):
        try:
            with urllib.request.urlopen(f"{url}{path}", timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def start_port_forward():
    proc = subprocess.Popen(
        ["kubectl", "port-forward", MIMIR_SERVICE, f"{LOCAL_PORT}:8080", "-n", MIMIR_NAMESPACE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://localhost:{LOCAL_PORT}/ready", timeout=3) as resp:
                if resp.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("kubectl port-forward to Mimir did not become ready")


def pod_to_deployment(pod):
    parts = pod.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:-2])
    return pod


def fetch_range(mimir_url, query, start_ts, end_ts, step_sec):
    out = {}
    chunk = 10000 * step_sec
    s = int(start_ts)
    while s <= int(end_ts):
        e = min(s + chunk, int(end_ts))
        response = requests.get(
            f"{mimir_url}/prometheus/api/v1/query_range",
            params={"query": query, "start": s, "end": e,
                    "step": f"{step_sec}s"},
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"query failed: {payload.get('error')}")
        for result in payload["data"]["result"]:
            key = tuple(sorted(result["metric"].items()))
            slot = out.setdefault(key, [result["metric"], {}])
            for t, v in result["values"]:
                slot[1][int(t)] = v
        s = e + step_sec
    return [{"metric": m, "values": [[t, v] for t, v in sorted(vals.items())]}
            for m, vals in out.values()]


def fetch_metric_data(metric_name, query_info, start_ts, end_ts, prom_url):
    step_sec = STEP_SECONDS
    print(f"  -> Querying {metric_name}...")
    try:
        results = fetch_range(prom_url, query_info["query"], start_ts,
                              end_ts, step_sec)
        if not results:
            print(f"  No data returned for {metric_name}.")
            return pd.DataFrame(columns=["ts", "entity", metric_name])
        extra = query_info.get("extra_labels", [])

        rows = []
        for result in results:
            metric_labels = result["metric"]
            label_vals = {c: metric_labels.get(c, "") for c in query_info["labels"]}
            extra_vals = {c: metric_labels.get(c, "") for c in extra}
            for value in result["values"]:
                try:
                    val = float(value[1])
                except ValueError:
                    val = np.nan
                rows.append([int(value[0]), *label_vals.values(),
                             *extra_vals.values(), val])

        cols = ["ts"] + query_info["labels"] + extra + [metric_name]
        df = pd.DataFrame(rows, columns=cols)
        df = df.sort_values("ts").drop_duplicates(
            subset=["ts"] + query_info["labels"] + extra, keep="last")
        if "pod" in query_info["labels"]:
            df["deployment"] = df["pod"].map(pod_to_deployment)
            group_keys = ["ts", "deployment"] + extra
            df = (
                df.groupby(group_keys, as_index=False)[metric_name]
                .mean()
            )
            df = df.rename(columns={"deployment": "entity"})
        else:
            key_col = query_info["labels"][0]
            df = df.rename(columns={key_col: "entity"})
        keep = ["ts", "entity"] + extra + [metric_name]
        return df[keep]

    except Exception as e:
        print(f"Error fetching {metric_name}: {e}")
        return pd.DataFrame(columns=["ts", "entity", metric_name])


def fetch_and_process_data(start_ts, end_ts, prom_url, out_path,
                           namespace="train-ticket",
                           container_filter='container!="istio-proxy",container!="POD"'):
    print(f"Fetching metrics from {prom_url}...")
    print(f"Namespace: {namespace} | container filter: {container_filter}")
    QUERIES = build_queries(namespace, container_filter)
    tehran_tz = pytz.timezone("Asia/Tehran")
    step_sec = STEP_SECONDS

    proc = None
    if not is_reachable(prom_url):
        print(
            f"Mimir not reachable at {prom_url}, "
            "starting kubectl port-forward..."
        )
        proc = start_port_forward()

    try:
        raw = {}
        for metric_name, query_info in QUERIES.items():
            raw[metric_name] = fetch_metric_data(
                metric_name, query_info, start_ts, end_ts, prom_url
            )

        n_points = int((end_ts - start_ts) // step_sec) + 1
        grid = [start_ts + k * step_sec for k in range(n_points)]
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(grid), unit="s", utc=True))

        def rekey(df, col):
            s = df.rename(columns={"entity": "msname"}).set_index("ts")[col]
            s.index = pd.to_datetime(pd.Index(s.index), unit="s", utc=True)
            return s.reindex(idx)

        services = set()
        for _df in raw.values():
            if "entity" in _df.columns:
                services.update(_df["entity"].unique().tolist())
        services.discard("")

        def wins(sec):
            return max(3, int(round(sec / step_sec)))

        n_slope, n_z, n_vol, n_ewma = wins(300), wins(1800), wins(360), wins(600)

        def slope_last(s, n):
            v = s.to_numpy(float)
            out = np.full(len(v), np.nan)
            for i in range(len(v)):
                lo = max(0, i - n + 1)
                seg = v[lo:i + 1]
                seg = seg[np.isfinite(seg)]
                if len(seg) >= 3:
                    x = np.arange(len(seg))
                    out[i] = float(np.polyfit(x, seg, 1)[0])
            return pd.Series(out, index=s.index)

        def get(df, dep, col, fill=None):
            if dep in df["entity"].values:
                s = rekey(df[df["entity"] == dep], col)
                return s if fill is None else s.fillna(fill)
            if fill is None:
                return None
            return zeros() if not isinstance(fill, pd.Series) else fill

        def zeros():
            return pd.Series(0.0, index=idx)

        edge_map = {}
        if "source_workload" in raw["EDGES"].columns:
            for dep, gdep in raw["EDGES"].groupby("entity"):
                d = {}
                for src, gsrc in gdep.groupby("source_workload"):
                    s = gsrc.set_index("ts")[ "EDGES"]
                    s.index = pd.to_datetime(pd.Index(s.index), unit="s", utc=True)
                    d[src] = s.reindex(idx).fillna(0.0)
                edge_map[dep] = d

        base = {}
        for dep in sorted(services):
            full = pd.DataFrame(index=idx)
            cpu = get(raw["CPU"], dep, "CPU")
            mem = get(raw["Memory"], dep, "Memory")
            if cpu is None and mem is None:
                print(f"  [{dep}] skipped (no cpu/memory signal)")
                continue
            full["cpu_utilization"] = cpu
            full["memory_utilization"] = mem
            cpu_raw = get(raw["CPU_RAW"], dep, "CPU_RAW")
            lim_cpu = get(raw["LIMITS_CPU"], dep, "LIMITS_CPU")
            if cpu_raw is not None and lim_cpu is not None:
                full["cpu_lim"] = (cpu_raw / lim_cpu.replace(0.0, np.nan)).clip(upper=4.0)
            mem_raw = get(raw["MEM_RAW"], dep, "MEM_RAW")
            lim_mem = get(raw["LIMITS_MEM"], dep, "LIMITS_MEM")
            if mem_raw is not None and lim_mem is not None:
                full["mem_lim"] = (mem_raw / lim_mem.replace(0.0, np.nan)).clip(upper=4.0)
            rep = get(raw["replicas"], dep, "replicas")
            des = get(raw["replicas_desired"], dep, "replicas_desired")
            full["replicas"] = (rep.ffill().bfill() if rep is not None else 1)
            if des is not None:
                des = des.ffill().bfill()
                full["desired_replicas"] = des
                full["unavailable"] = (des - full["replicas"]).clip(lower=0.0)
            else:
                full["desired_replicas"] = full["replicas"]
                full["unavailable"] = 0.0
            rst = get(raw["RESTARTS"], dep, "RESTARTS")
            if rst is not None:
                rst = rst.ffill().bfill().fillna(0.0)
                full["restart_rate"] = rst.diff().clip(lower=0.0).fillna(0.0) / step_sec
            else:
                full["restart_rate"] = 0.0
            http = get(raw["RPS_HTTP"], dep, "RPS_HTTP", fill=0.0)
            grpc = get(raw["RPS_GRPC"], dep, "RPS_GRPC", fill=0.0)
            full["http_mcr"] = http
            full["providerrpc_mcr"] = grpc
            full["rps_total"] = http + grpc
            e5 = get(raw["RPS_5XX"], dep, "RPS_5XX", fill=0.0)
            full["err_rate"] = e5
            full["err_frac"] = (e5 / full["rps_total"].replace(0.0, np.nan)).fillna(0.0)
            fl = get(raw["FLAGS"], dep, "FLAGS", fill=0.0)
            full["flag_rate"] = fl
            full["flag_frac"] = (fl / full["rps_total"].replace(0.0, np.nan)).fillna(0.0)
            full["p99_latency"] = get(raw["P99"], dep, "P99")
            rb = get(raw["REQ_BYTES"], dep, "REQ_BYTES", fill=0.0)
            pb = get(raw["RESP_BYTES"], dep, "RESP_BYTES", fill=0.0)
            full["req_byte_rate"] = rb
            full["resp_byte_rate"] = pb
            tiny = full["rps_total"] < 0.1
            full["req_bytes_per_req"] = (rb / full["rps_total"].replace(0.0, np.nan)).mask(tiny, 0.0).fillna(0.0).clip(upper=1e6)
            full["resp_bytes_per_req"] = (pb / full["rps_total"].replace(0.0, np.nan)).mask(tiny, 0.0).fillna(0.0).clip(upper=1e6)
            th = get(raw["THROTTLED"], dep, "THROTTLED", fill=0.0)
            pe = get(raw["PERIODS"], dep, "PERIODS", fill=0.0)
            full["throttle_ratio"] = (th / pe.replace(0.0, np.nan)).fillna(0.0).clip(upper=1.0)
            full["net_rx"] = get(raw["NET_RX"], dep, "NET_RX", fill=0.0)
            full["pgfault"] = get(raw["PGFAULT"], dep, "PGFAULT", fill=0.0)
            full["pgmajfault"] = get(raw["PGMAJFAULT"], dep, "PGMAJFAULT", fill=0.0)
            full["queue_in"] = get(raw["Q_PENDING_IN"], dep, "Q_PENDING_IN", fill=0.0)
            full["active_in"] = get(raw["Q_ACTIVE_IN"], dep, "Q_ACTIVE_IN", fill=0.0)
            full["queue_for"] = get(raw["Q_PENDING_OUT"], dep, "Q_PENDING_OUT", fill=0.0)
            full["active_for"] = get(raw["Q_ACTIVE_OUT"], dep, "Q_ACTIVE_OUT", fill=0.0)
            inbound = edge_map.get(dep, {})
            if inbound:
                mat = pd.DataFrame(inbound).fillna(0.0)
                full["caller_sum"] = mat.sum(axis=1)
                full["caller_rps_max"] = mat.max(axis=1)
                full["upstream_rps_sum"] = full["caller_sum"]
                full["n_callers"] = (mat > 1e-9).sum(axis=1).astype(float)
                fe_cols = [c for c in ("frontend", "ts-ui-dashboard") if c in mat]
                full["from_frontend"] = mat[fe_cols].sum(axis=1) if fe_cols else 0.0
            else:
                full["caller_sum"] = 0.0
                full["caller_rps_max"] = 0.0
                full["upstream_rps_sum"] = 0.0
                full["n_callers"] = 0.0
                full["from_frontend"] = 0.0
            minute_of_day = (full.index.tz_convert(tehran_tz).hour * 60
                             + full.index.tz_convert(tehran_tz).minute).to_numpy()
            full["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
            full["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
            full["rps_slope5"] = slope_last(full["rps_total"], n_slope)
            full["cpu_slope3"] = slope_last(full["cpu_utilization"], n_slope)
            full["mem_delta5"] = full["memory_utilization"].diff(n_slope)
            rm = full["rps_total"].rolling(n_z, min_periods=max(3, n_z // 4)).mean()
            rs = full["rps_total"].rolling(n_z, min_periods=max(3, n_z // 4)).std()
            full["rps_z30"] = ((full["rps_total"] - rm) / rs.replace(0.0, np.nan)).fillna(0.0)
            full["ewma_gap"] = (full["rps_total"]
                                - full["rps_total"].ewm(span=n_ewma, min_periods=3).mean()).fillna(0.0)
            full["vol_rps"] = full["rps_total"].rolling(n_vol, min_periods=3).std().fillna(0.0)
            full["vol_cpu"] = full["cpu_utilization"].rolling(n_vol, min_periods=3).std().fillna(0.0)
            full["concurrency"] = (full["rps_total"].fillna(0.0)
                                   * full["p99_latency"].fillna(0.0) / 1000.0)
            rep_chg = full["replicas"].ne(full["replicas"].shift(1)).cumsum()
            full["scale_recency"] = (full.groupby(rep_chg).cumcount().astype(float)
                                     * step_sec).clip(upper=3600.0)
            base[dep] = full

        if not base:
            raise SystemExit("No service frames could be assembled.")

        mesh_rps = sum(f["rps_total"].fillna(0.0) for f in base.values())
        fe_name = next((c for c in ("frontend", "ts-ui-dashboard") if c in base), None)
        fe_rps = base[fe_name]["rps_total"].fillna(0.0) if fe_name else mesh_rps * 0.0
        frames = []
        for dep, full in base.items():
            full["frontend_rps"] = fe_rps.values
            full["mesh_rps"] = mesh_rps.values
            full["root_rps"] = fe_rps.values
            callers = [c for c in edge_map.get(dep, {}) if c in base and c != dep]
            if callers:
                ncpu = pd.DataFrame({c: base[c]["cpu_utilization"] for c in callers})
                full["neigh_cpu_mean"] = ncpu.mean(axis=1)
                full["neigh_cpu_slope3"] = slope_last(full["neigh_cpu_mean"], n_slope)
                nz = pd.DataFrame({c: base[c]["rps_z30"] for c in callers})
                full["neigh_rps_z30_mean"] = nz.mean(axis=1)
                nsl = pd.DataFrame({c: base[c]["rps_slope5"] for c in callers})
                full["neigh_rps_slope5_mean"] = nsl.mean(axis=1)
            else:
                full["neigh_cpu_mean"] = 0.0
                full["neigh_cpu_slope3"] = 0.0
                full["neigh_rps_z30_mean"] = 0.0
                full["neigh_rps_slope5_mean"] = 0.0
            for _nc in ("neigh_cpu_mean", "neigh_cpu_slope3",
                        "neigh_rps_z30_mean", "neigh_rps_slope5_mean"):
                full[_nc] = full[_nc].fillna(0.0)
            full["p99_latency"] = full["p99_latency"].ffill(limit=n_z).fillna(0.0)
            for _c in FINAL_COLUMNS:
                if _c not in ("timestamp", "msname") and _c not in full.columns:
                    full[_c] = 0.0
            hist_cols = ["rps_slope5", "cpu_slope3", "mem_delta5", "rps_z30",
                         "ewma_gap", "vol_rps", "vol_cpu"]
            first_valid = max(full[_c].first_valid_index() for _c in hist_cols)
            if first_valid is not None:
                full = full.loc[first_valid:]
            full = full.dropna(subset=["cpu_utilization", "memory_utilization"], how="all")
            if full.empty:
                print(f"  [{dep}] skipped (no cpu/memory signal)")
                continue
            full["timestamp"] = full.index.tz_convert(tehran_tz).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            full["msname"] = dep
            full["replicas"] = full["replicas"].round().astype(int)
            full["desired_replicas"] = full["desired_replicas"].round().astype(int)
            full["unavailable"] = full["unavailable"].round().astype(int)
            full["n_callers"] = full["n_callers"].round().astype(int)
            have = [c for c in FINAL_COLUMNS if c in full.columns]
            frames.append(full[have])
            print(f"  [{dep}] {len(full)} rows")

        if not frames:
            raise SystemExit("No service frames could be assembled.")

        final_df = pd.concat(frames, ignore_index=True)
        final_df = final_df.sort_values(["msname", "timestamp"]).reset_index(drop=True)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        final_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(final_df)} records x "
              f"{final_df.shape[1]} cols to {out_path}")
    finally:
        if proc is not None:
            proc.terminate()


def analyze_file(csv_path, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as ticker
    from tqdm import tqdm

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows x {df.shape[1]} cols from {csv_path}")
    cpu_col = "cpu_utilization" if "cpu_utilization" in df.columns else "cpu"
    mem_col = "memory_utilization" if "memory_utilization" in df.columns else "memory"
    print(f"Avg CPU: {df[cpu_col].mean():.4f}")
    print(f"Avg Memory: {df[mem_col].mean():.4f}")
    print(f"Avg Replicas: {df['replicas'].mean():.2f}")
    print(f"Avg Threshold (HPA target): {HPA_TARGET:.4f}")

    os.makedirs(plots_dir, exist_ok=True)
    services = sorted(df["msname"].unique())
    cols = 2
    rows = (len(services) + 1) // cols

    def grid():
        fig, axes = plt.subplots(rows, cols, figsize=(36, 6 * rows), sharex=False)
        axes = axes.flatten()
        for j in range(len(services), len(axes)):
            axes[j].axis("off")
        return fig, axes

    def style(ax):
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(True, alpha=0.3)
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc="upper left")

    def finish(fig, suffix):
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=0.6)
        out = os.path.join(plots_dir, f"hpa_{suffix}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {out}")

    fig, axes = grid()
    for i, dep in tqdm(list(enumerate(services)), desc="Plotting CPU", unit="svc"):
        ax = axes[i]
        g = df[df["msname"] == dep]
        ax.plot(g["timestamp"], g[cpu_col], label="CPU", color="blue", alpha=0.6)
        ax.axhline(y=HPA_TARGET, label="Threshold (0.80)", color="red", linestyle=":")
        ax.set_title(f"Service: {dep}", fontweight="bold")
        ax.set_ylabel("CPU Utilization")
        ax.set_ylim(-0.05, 1.05)
        style(ax)
    finish(fig, "cpu")

    fig, axes = grid()
    for i, dep in tqdm(list(enumerate(services)), desc="Plotting memory", unit="svc"):
        ax = axes[i]
        g = df[df["msname"] == dep]
        ax.plot(g["timestamp"], g[mem_col], label="Memory", color="blue", alpha=0.6)
        ax.axhline(y=HPA_TARGET, label="Threshold (0.80)", color="red", linestyle=":")
        ax.set_title(f"Service: {dep}", fontweight="bold")
        ax.set_ylabel("Memory Utilization")
        ax.set_ylim(-0.05, max(1.05, float(g[mem_col].max()) + 0.05))
        style(ax)
    finish(fig, "mem")

    fig, axes = grid()
    for i, dep in tqdm(list(enumerate(services)), desc="Plotting replicas", unit="svc"):
        ax = axes[i]
        g = df[df["msname"] == dep]
        ax.step(g["timestamp"], g["replicas"], label="Replicas", color="green",
                where="post", alpha=0.7)
        ax.set_title(f"Service: {dep}", fontweight="bold")
        ax.set_ylabel("Replicas")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        lo, hi = float(g["replicas"].min()), float(g["replicas"].max())
        ax.set_ylim(min(lo, HPA_TARGET) - 0.2, hi + 0.5)
        style(ax)
    finish(fig, "replicas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export per-service cpu/memory/replicas/incoming-mcr "
                    "plus Tier-0 spike features (saturation, call-graph "
                    "pressure, envoy queues, engineered dynamics). Column "
                    "names match build_windows._CSV_COLUMN_MAP so the CSV "
                    "feeds --csv_path directly."
    )
    parser.add_argument("--start", required=False, default=None,
                        help="Start time ('YYYY-MM-DD HH:MM:SS' Tehran or Unix ts)")
    parser.add_argument("--end", required=False, default=None,
                        help="End time ('YYYY-MM-DD HH:MM:SS' Tehran or Unix ts)")
    parser.add_argument("--step", type=int, default=STEP_SECONDS,
                        help="Grid step in seconds, >=10 (Prometheus scrape "
                             "cadence; default %(default)s)")
    parser.add_argument("--namespace", type=str, default="train-ticket",
                        help="Application namespace to export (default %(default)s)")
    parser.add_argument("--container-filter", type=str,
                        default='container!="istio-proxy",container!="POD"',
                        help="PromQL container selector for pod-level metrics "
                             "(default %(default)s)")
    parser.add_argument("--mimir-url", type=str, default=MIMIR_URL,
                        help="Mimir API base URL (default localhost:8080 via port-forward)")
    parser.add_argument("--out", type=str,
                        default=os.path.join(OUTPUT_DIR, OUTPUT_NAME),
                        help="Output CSV path (default %(default)s)")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Analyze an existing CSV (plots + averages) instead of exporting")

    args = parser.parse_args()
    if args.analyze:
        analyze_file(args.analyze, os.path.join(OUTPUT_DIR, "hpa_plots"))
        sys.exit(0)
    if not args.start or not args.end:
        parser.error("--start and --end are required for export mode")
    tehran_tz = pytz.timezone("Asia/Tehran")

    def parse_time_arg(time_str):
        try:
            return float(time_str)
        except ValueError:
            dt_aware = tehran_tz.localize(
                datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            )
            return dt_aware.timestamp()

    start_timestamp = parse_time_arg(args.start)
    end_timestamp = parse_time_arg(args.end)

    if args.step < 10:
        parser.error("--step must be >= 10 (Prometheus scrape cadence)")
    STEP_SECONDS = args.step

    fetch_and_process_data(start_timestamp, end_timestamp, args.mimir_url, args.out,
                           args.namespace, args.container_filter)

