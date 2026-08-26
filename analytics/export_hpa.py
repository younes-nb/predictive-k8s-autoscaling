import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
import argparse
import pytz
import subprocess
import time
import urllib.request

PROMETHEUS_URL = "http://localhost:9090"
PROM_SERVICE = "svc/prometheus-stack-kube-prom-prometheus"
PROM_NAMESPACE = "monitoring"
LOCAL_PORT = 9090
STEP_SECONDS = 60
OUTPUT_DIR = "/proj/k8sautoscaledl-PG0"
OUTPUT_NAME = "hpa_graph_logs.csv"

NAMESPACE = "online-boutique"

# Spike-detection lead indicators fetched alongside the base HPA metrics:
#   P99_LATENCY  request duration p99 per workload (queueing builds here
#                before utilization pegs)
#   THROTTLE     CFS throttling ratio per pod (CPU quota saturation)
#   NET_RX       network receive bytes per pod (traffic proxy)
# Plus the istio call graph (EDGE_QUERY) which yields root/caller/upstream
# load features: entry-point traffic leads every downstream service.
QUERIES = {
    "replicas": {
        "query": f'kube_horizontalpodautoscaler_status_current_replicas{{namespace="{NAMESPACE}"}}',
        "labels": ["horizontalpodautoscaler", "hpa"],
    },
    "RPS_HTTP": {
        "query": f'sum(rate(istio_requests_total{{reporter="destination", request_protocol="http", destination_workload_namespace="{NAMESPACE}"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "RPS_GRPC": {
        "query": f'sum(rate(istio_requests_total{{reporter="destination", request_protocol="grpc", destination_workload_namespace="{NAMESPACE}"}}[1m])) by (destination_workload)',
        "labels": ["destination_workload"],
    },
    "P99_LATENCY": {
        "query": f'histogram_quantile(0.99, sum(rate(istio_request_duration_milliseconds_bucket{{reporter="destination", destination_workload_namespace="{NAMESPACE}"}}[1m])) by (le, destination_workload))',
        "labels": ["destination_workload"],
    },
    "THROTTLE": {
        "query": f'sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{{namespace="{NAMESPACE}", container="server"}}[1m])) / sum by (pod) (rate(container_cpu_cfs_periods_total{{namespace="{NAMESPACE}", container="server"}}[1m]))',
        "labels": ["pod"],
    },
    "NET_RX": {
        "query": f'sum by (pod) (rate(container_network_receive_bytes_total{{namespace="{NAMESPACE}"}}[1m]))',
        "labels": ["pod"],
    },
    "CPU": {
        "query": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}", container="server"}}[1m])) / sum by (pod) (kube_pod_container_resource_requests{{resource="cpu", namespace="{NAMESPACE}", container="server"}})',
        "labels": ["pod"],
    },
    "Memory": {
        "query": f'sum by (pod) (container_memory_working_set_bytes{{namespace="{NAMESPACE}", container="server"}}) / sum by (pod) (kube_pod_container_resource_requests{{resource="memory", namespace="{NAMESPACE}", container="server"}})',
        "labels": ["pod"],
    },
}

EDGE_QUERY = (
    f'sum(rate(istio_requests_total{{reporter="destination", '
    f'destination_workload_namespace="{NAMESPACE}"}}[1m])) '
    f'by (source_workload, destination_workload)'
)

FINAL_COLUMNS = [
    "timestamp", "msname", "replicas",
    "http_mcr", "providerrpc_mcr",
    "rps_total", "caller_rps_max", "n_callers", "upstream_rps_sum", "root_rps",
    "p99_latency", "throttle_ratio", "net_rx",
    "rps_z30", "rps_slope5", "ewma_gap", "cpu_slope3", "mem_delta5",
    "tod_sin", "tod_cos",
    "neigh_cpu_mean", "neigh_cpu_slope3",
    "neigh_rps_z30_mean", "neigh_rps_slope5_mean",
    "cpu_utilization", "memory_utilization",
]

EPS = 1e-9


def is_reachable(url):
    try:
        with urllib.request.urlopen(f"{url}/-/healthy", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_port_forward():
    proc = subprocess.Popen(
        ["kubectl", "port-forward", PROM_SERVICE, f"{LOCAL_PORT}:9090", "-n", PROM_NAMESPACE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if is_reachable(PROMETHEUS_URL):
            return proc
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("kubectl port-forward to Prometheus did not become ready")


def fetch_metric_data(metric_name, query_info, start_ts, end_ts, prom_url):
    params = {
        "query": query_info["query"],
        "start": start_ts,
        "end": end_ts,
        "step": f"{STEP_SECONDS}s",
    }
    print(f"  -> Querying {metric_name}...")
    try:
        response = requests.get(
            f"{prom_url}/api/v1/query_range",
            params=params,
            timeout=120,
        )
        response.raise_for_status()
        results = response.json().get("data", {}).get("result", [])
        if not results:
            print(f"  No data returned for {metric_name}.")
            return pd.DataFrame(columns=["ts"] + query_info["labels"])

        rows = []
        for result in results:
            metric_labels = result["metric"]
            label_vals = {c: metric_labels.get(c, "") for c in query_info["labels"]}
            for value in result["values"]:
                try:
                    val = float(value[1])
                except ValueError:
                    val = np.nan
                rows.append([int(value[0]), *label_vals.values(), val])

        cols = ["ts"] + query_info["labels"] + [metric_name]
        df = pd.DataFrame(rows, columns=cols)
        if "pod" in query_info["labels"]:
            # Pods come and go with autoscaling; collapse to deployment level.
            df["deployment"] = df["pod"].map(pod_to_deployment)
            df = (
                df.groupby(["ts", "deployment"], as_index=False)[metric_name]
                .mean()
            )
            df = df.rename(columns={"deployment": "entity"})
        else:
            key_col = query_info["labels"][0]
            df = df.rename(columns={key_col: "entity"})
        # HPA objects are named "<deployment>-hpa" on this cluster.
        df["entity"] = df["entity"].str.replace(r"-hpa$", "", regex=True)
        return df[["ts", "entity", metric_name]]

    except Exception as e:
        print(f"Error fetching {metric_name}: {e}")
        return pd.DataFrame(columns=["ts", "entity", metric_name])


def pod_to_deployment(pod):
    parts = pod.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:-2])
    return pod


def build_edges(start_ts, end_ts, prom_url):
    print("  -> Querying EDGE_RPS (istio call graph)...")
    info = {"query": EDGE_QUERY}
    df = fetch_metric_data_edges(start_ts, end_ts, prom_url)
    if df.empty:
        raise SystemExit(
            "No istio edge data returned; cannot build graph features."
        )
    return df


def fetch_metric_data_edges(start_ts, end_ts, prom_url):
    params = {
        "query": EDGE_QUERY,
        "start": start_ts,
        "end": end_ts,
        "step": f"{STEP_SECONDS}s",
    }
    try:
        response = requests.get(
            f"{prom_url}/api/v1/query_range", params=params, timeout=120
        )
        response.raise_for_status()
        results = response.json().get("data", {}).get("result", [])
        rows = []
        for r in results:
            src = r["metric"].get("source_workload", "")
            dst = r["metric"].get("destination_workload", "")
            for v in r["values"]:
                try:
                    val = float(v[1])
                except ValueError:
                    val = np.nan
                rows.append([int(v[0]), src, dst, val])
        return pd.DataFrame(rows, columns=["ts", "src", "dst", "rps"])
    except Exception as e:
        print(f"Error fetching edge data: {e}")
        return pd.DataFrame(columns=["ts", "src", "dst", "rps"])


def graph_features(edges_df, grid):
    """Per-timestamp call-graph indicators from the edge list.

    root_rps          total traffic injected by entry-point sources (sources
                      that never appear as a destination): leads every service
    caller_rps_max    hottest single inbound edge per service
    upstream_rps_sum  total inbound load of each caller (how busy callers are)
    n_callers         active inbound edges (smoothed)
    """
    e = edges_df[edges_df["rps"].notna()].copy()

    src_totals = e.groupby(["ts", "src"], as_index=False)["rps"].sum()
    dsts = set(e["dst"].unique())
    roots = sorted(set(src_totals["src"].unique()) - dsts)
    print(f"  Graph entry points (roots): {roots}")

    root_rps = (
        src_totals[src_totals["src"].isin(roots)]
        .groupby("ts", as_index=False)["rps"].sum()
        .rename(columns={"rps": "root_rps"})
    )

    caller_agg = e.groupby(["ts", "dst"]).agg(
        caller_rps_max=("rps", "max"),
        n_callers_raw=("rps", lambda v: int((v > 1e-6).sum())),
    ).reset_index()
    edge_pairs = e[["ts", "src", "dst"]].drop_duplicates()
    up = (
        src_totals.rename(columns={"src": "caller", "rps": "caller_total"})
        .merge(edge_pairs.rename(columns={"src": "caller"}), on=["ts", "caller"])
        .groupby(["ts", "dst"], as_index=False)["caller_total"].sum()
        .rename(columns={"caller_total": "upstream_rps_sum"})
    )
    graph = caller_agg.merge(up, on=["ts", "dst"], how="outer")

    grid_df = pd.DataFrame({"ts": grid})
    graph = graph.merge(grid_df, on="ts", how="right")
    graph["n_callers_raw"] = graph["n_callers_raw"].fillna(0)
    graph = graph.sort_values(["dst", "ts"])
    graph["n_callers"] = (
        graph.groupby("dst")["n_callers_raw"]
        .transform(lambda s: s.rolling(5, min_periods=1).median())
    )
    graph = graph.drop(columns=["n_callers_raw"])
    return graph, root_rps.set_index("ts")["root_rps"]


def add_causal_features(full):
    """Strictly backward-looking dynamics so the model can learn momentum
    instead of copying the current level."""
    eps = EPS
    rps_total = full["rps_total"].fillna(0.0)
    http = full["http_mcr"].fillna(0.0)
    grpc = full["providerrpc_mcr"].fillna(0.0)

    base_sig = http.add(grpc)
    base_mean = base_sig.rolling(31, min_periods=6).mean()
    base_std = base_sig.rolling(31, min_periods=6).std(ddof=0)
    floor = np.maximum(0.15 * base_mean.abs(), 0.05)
    denom = np.maximum(base_std, floor) + eps
    full["rps_z30"] = np.tanh(((base_sig - base_mean) / denom) / 4.0)

    denom_s = rps_total.shift(1).rolling(5, min_periods=3).mean().abs().clip(lower=0.05)
    rel_slope = ((rps_total - rps_total.shift(5)) / denom_s).replace([np.inf, -np.inf], 0)
    full["rps_slope5"] = np.tanh(1.5 * rel_slope.fillna(0.0))

    fast = rps_total.ewm(span=3, adjust=False).mean()
    slow = rps_total.ewm(span=15, adjust=False).mean()
    gap = (fast - slow) / np.maximum(slow.abs(), 0.2)
    full["ewma_gap"] = np.tanh(gap / 2.0)

    cpu = full["cpu_utilization"]
    mem = full["memory_utilization"]
    full["cpu_slope3"] = np.tanh(6.0 * (cpu - cpu.shift(3)))
    full["mem_delta5"] = np.tanh(8.0 * (mem - mem.shift(5)))

    tod = full.index.tz_convert("Asia/Tehran")
    ang = 2 * np.pi * (tod.hour * 60 + tod.minute) / 1440.0
    full["tod_sin"] = np.sin(ang)
    full["tod_cos"] = np.cos(ang)
    return full


def finalize_service_frame(full, dep, tehran_tz):
    """Causal dynamics + bounded transforms + final column layout."""
    full = add_causal_features(full)
    full = full.dropna(how="any")
    if len(full) < 60:
        return None

    out = full.reset_index().rename(columns={"index": "ts"})
    out["timestamp"] = out["ts"].dt.tz_convert(tehran_tz).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["msname"] = dep

    # Count-rate features are sqrt-compressed so global min-max scaling in
    # build_windows keeps regular variation visible next to extreme spikes.
    for col in ("http_mcr", "providerrpc_mcr", "rps_total",
                "caller_rps_max", "upstream_rps_sum", "root_rps"):
        out[col] = np.sqrt(out[col].clip(lower=0))
    out["n_callers"] = (full["n_callers"].to_numpy() / 5.0).clip(0, 1)
    out["p99_latency"] = (
        np.log1p(full["p99_latency"].to_numpy().clip(min=0)) / np.log1p(20000.0)
    ).clip(0, 1)
    out["net_rx"] = (np.log1p(full["net_rx"].to_numpy().clip(min=0)) / 18.0).clip(0, 1)
    out["throttle_ratio"] = np.clip(full["throttle_ratio"].to_numpy(), 0, 1)

    out = out.round({
        "cpu_utilization": 4, "memory_utilization": 4,
        "http_mcr": 3, "providerrpc_mcr": 3,
        "rps_total": 3, "caller_rps_max": 3, "upstream_rps_sum": 3, "root_rps": 3,
        "p99_latency": 4, "throttle_ratio": 4, "net_rx": 4,
        "rps_z30": 4, "rps_slope5": 4, "ewma_gap": 4,
        "cpu_slope3": 4, "mem_delta5": 4,
        "neigh_cpu_mean": 4, "neigh_cpu_slope3": 4,
        "neigh_rps_z30_mean": 4, "neigh_rps_slope5_mean": 4,
        "tod_sin": 4, "tod_cos": 4,
    })
    out["replicas"] = out["replicas"].round().astype(int)
    return out[FINAL_COLUMNS]


def add_neighbor_features(raw_frames, edges_df, grid_ts):
    """One-hop message passing over the call graph, computed statically.

    For each service, callers' states are aggregated with rps-weighted
    averaging (weights = trailing 30-min mean edge rps, causal). Workload
    signals (rps_z30 / rps_slope5 of callers) propagate BEFORE the callee's
    CPU reacts, so they are the leading indicators; caller CPU level is the
    contemporaneous companion. Services without callers fall back to their
    own values (self-loop). Returns {dep: DataFrame[4 neighbor cols]}.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(grid_ts), unit="s", utc=True))
    e = edges_df[edges_df["rps"].notna()]
    if e.empty:
        return {}

    # per-edge causal weight: trailing 30-min mean of edge rps
    w_series = {}
    for (src, dst), g in e.groupby(["src", "dst"]):
        s = g.set_index("ts")["rps"]
        s.index = pd.to_datetime(pd.Index(s.index), unit="s", utc=True)
        s = s.reindex(idx).ffill().fillna(0.0)
        w_series[(src, dst)] = s.rolling(30, min_periods=1).mean()

    def wide(col):
        return pd.DataFrame(
            {dep: f[col] for dep, f in raw_frames.items() if col in f.columns}
        ).reindex(idx)

    cpu_wide = wide("cpu_utilization")
    z_wide = wide("rps_z30")
    slope_wide = wide("rps_slope5")

    out = {}
    dsts = {dst for (_, dst) in w_series}
    for dep, frame in raw_frames.items():
        if dep not in dsts:
            continue
        num_c = pd.Series(0.0, index=idx); den_c = pd.Series(0.0, index=idx)
        num_z = pd.Series(0.0, index=idx); den_z = pd.Series(0.0, index=idx)
        num_s = pd.Series(0.0, index=idx); den_s = pd.Series(0.0, index=idx)
        for (src, dst), w in w_series.items():
            if dst != dep:
                continue
            if src in cpu_wide.columns:
                num_c = num_c + w * cpu_wide[src].fillna(0.0)
                den_c = den_c + w
            if src in z_wide.columns:
                num_z = num_z + w * z_wide[src].fillna(0.0)
                den_z = den_z + w
            if src in slope_wide.columns:
                num_s = num_s + w * slope_wide[src].fillna(0.0)
                den_s = den_s + w

        own_cpu = cpu_wide[dep].fillna(0.0) if dep in cpu_wide.columns else 0.0
        nm = pd.Series(
            np.where(den_c > 1e-9, num_c / den_c.replace(0.0, np.nan), own_cpu),
            index=idx,
        ).ffill().bfill().clip(0, 1)
        nz = pd.Series(
            np.where(den_z > 1e-9, num_z / den_z.replace(0.0, np.nan), 0.0),
            index=idx,
        ).ffill().bfill().clip(-1, 1)
        ns = pd.Series(
            np.where(den_s > 1e-9, num_s / den_s.replace(0.0, np.nan), 0.0),
            index=idx,
        ).ffill().bfill().clip(-1, 1)

        f = pd.DataFrame(index=frame.index)
        f["neigh_cpu_mean"] = nm.reindex(frame.index)
        f["neigh_cpu_slope3"] = np.tanh(
            6.0 * (f["neigh_cpu_mean"] - f["neigh_cpu_mean"].shift(3))
        )
        f["neigh_rps_z30_mean"] = nz.reindex(frame.index)
        f["neigh_rps_slope5_mean"] = ns.reindex(frame.index)
        out[dep] = f
    return out


def fetch_and_process_data(start_ts, end_ts, prom_url, out_path):
    print(f"Fetching metrics from {prom_url}...")
    tehran_tz = pytz.timezone("Asia/Tehran")
    step_sec = STEP_SECONDS

    proc = None
    if not is_reachable(prom_url):
        print(
            f"Prometheus not reachable at {prom_url}, "
            "starting kubectl port-forward..."
        )
        proc = start_port_forward()

    try:
        raw = {}
        for metric_name, query_info in QUERIES.items():
            raw[metric_name] = fetch_metric_data(
                metric_name, query_info, start_ts, end_ts, prom_url
            )

        edges = build_edges(start_ts, end_ts, prom_url)

        n_points = int((end_ts - start_ts) // step_sec) + 1
        grid = [start_ts + k * step_sec for k in range(n_points)]

        graph, root_series = graph_features(edges, grid)
        # Total inbound rate per service straight off the call graph.
        rps_total = (
            edges.groupby(["ts", "dst"], as_index=False)["rps"].sum()
            .rename(columns={"dst": "msname", "rps": "rps_total"})
        )
        graph = graph.rename(columns={"dst": "msname"}).merge(
            rps_total, on=["ts", "msname"], how="outer"
        )

        # Metric name -> CSV/feature column name.
        METRIC_TO_COL = {
            "replicas": "replicas",
            "RPS_HTTP": "http_mcr",
            "RPS_GRPC": "providerrpc_mcr",
            "P99_LATENCY": "p99_latency",
            "CPU": "cpu_utilization",
            "Memory": "memory_utilization",
            "THROTTLE": "throttle_ratio",
            "NET_RX": "net_rx",
        }
        for metric_name, col in METRIC_TO_COL.items():
            if not raw[metric_name].empty:
                raw[metric_name] = raw[metric_name].rename(columns={metric_name: col})

        core = raw["replicas"].rename(columns={"entity": "msname"})
        for m in ("RPS_HTTP", "RPS_GRPC", "P99_LATENCY"):
            core = core.merge(
                raw[m].rename(columns={"entity": "msname"}),
                on=["ts", "msname"], how="outer",
            )
        pod_all = raw["CPU"].rename(columns={"entity": "msname"})
        for m in ("Memory", "THROTTLE", "NET_RX"):
            pod_all = pod_all.merge(
                raw[m].rename(columns={"entity": "msname"}),
                on=["ts", "msname"], how="outer",
            )
        core = core.merge(pod_all, on=["ts", "msname"], how="outer")
        core = core.merge(graph, on=["ts", "msname"], how="outer")
        core = core[core["msname"] != "redis-cart"]

        print(f"\nAssembling {core['msname'].nunique()} service frames "
              f"on a strict {step_sec}s grid...")

        def assemble(dep, g):
            idx = pd.DatetimeIndex(
                pd.to_datetime(pd.Series(grid), unit="s", utc=True)
            )

            def rekey(col_src, col):
                s = col_src.set_index("ts")[col]
                s.index = pd.to_datetime(pd.Index(s.index), unit="s", utc=True)
                return s.reindex(idx)

            full = pd.DataFrame(index=idx)
            for col in FINAL_COLUMNS[2:]:
                if col in ("root_rps", "neigh_cpu_mean", "neigh_cpu_slope3",
                           "neigh_rps_z30_mean", "neigh_rps_slope5_mean"):
                    continue
                if col in g.columns:
                    full[col] = rekey(g, col)
            if not root_series.empty:
                rs = root_series.copy()
                rs.index = pd.to_datetime(pd.Index(rs.index), unit="s", utc=True)
                full["root_rps"] = rs.reindex(idx)

            all_nan = [c for c in full.columns if full[c].notna().sum() == 0]
            for c in all_nan:
                full[c] = 0.0

            # Interior gaps up to 3 minutes are interpolated. Remaining holes
            # are semantically-zero minutes (e.g. histogram_quantile is NaN
            # when a service received no requests), so traffic/latency-derived
            # features collapse to 0 instead of splitting the series; replica
            # counts carry forward. Only missing CPU/memory can still cut rows.
            full = full.interpolate(method="time", limit=3)
            zero_fill = [
                "http_mcr", "providerrpc_mcr", "p99_latency", "throttle_ratio",
                "net_rx", "rps_total", "caller_rps_max", "n_callers",
                "upstream_rps_sum", "root_rps",
            ]
            full = full.fillna({c: 0.0 for c in zero_fill if c in full.columns})
            if "replicas" in full.columns:
                full["replicas"] = (
                    full["replicas"].ffill().bfill().fillna(1)
                )
            full = full.dropna(how="any")
            diffs = full.index.to_series().diff().dt.total_seconds()
            if len(full):
                seg = (diffs != step_sec).cumsum()
                full = full[seg == seg.value_counts().idxmax()]
            return full if len(full) >= 60 else None

        raw_frames = {}
        for dep, g in core.groupby("msname"):
            full = assemble(dep, g)
            if full is None:
                print(f"  [{dep}] skipped (clean series too short)")
                continue
            raw_frames[dep] = full

        # Dynamics first so neighbor aggregation can use leading workload
        # signals (rps_z30 / rps_slope5), not just contemporaneous CPU.
        for full in raw_frames.values():
            add_causal_features(full)

        neigh = add_neighbor_features(raw_frames, edges, grid)
        if neigh:
            print(f"  neighbor features computed for {len(neigh)} services")

        frames = []
        for dep, full in raw_frames.items():
            if dep in neigh:
                full = full.join(neigh[dep], how="left")
            for col, fb in (
                ("neigh_cpu_mean", full["cpu_utilization"]),
                ("neigh_rps_z30_mean", 0.0),
                ("neigh_rps_slope5_mean", 0.0),
            ):
                if col not in full.columns:
                    full[col] = fb
            if "neigh_cpu_slope3" not in full.columns:
                full["neigh_cpu_slope3"] = np.tanh(
                    6.0 * (full["neigh_cpu_mean"] - full["neigh_cpu_mean"].shift(3))
                )
            out = finalize_service_frame(full, dep, tehran_tz)
            if out is not None:
                frames.append(out)
                print(f"  [{dep}] {len(out)} rows "
                      f"({out['timestamp'].iloc[0]} .. {out['timestamp'].iloc[-1]})")

        if not frames:
            raise SystemExit("No service frames could be assembled.")

        final_df = pd.concat(frames, ignore_index=True)
        final_df = final_df.sort_values(["msname", "timestamp"]).reset_index(drop=True)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        final_df.to_csv(out_path, index=False)
        print(f"\nSuccessfully saved {len(final_df)} records x "
              f"{final_df.shape[1]} cols to {out_path}\n")

        print("=" * 40)
        print("GLOBAL DATASET METRICS")
        print("=" * 40)
        print(f"Services:             {final_df['msname'].nunique()}")
        print(f"Total Data Points:    {len(final_df)}")
        print(f"Avg Replicas:         {final_df['replicas'].mean():.2f}")
        print(f"Avg CPU:              {final_df['cpu_utilization'].mean():.2%}")
        print(f"Avg Memory:           {final_df['memory_utilization'].mean():.2%}")
        for tgt in ("cpu_utilization", "memory_utilization"):
            for svc, gsvc in final_df.groupby("msname"):
                x = gsvc[tgt].to_numpy()
                moves = np.abs(np.diff(x)) / np.maximum(np.abs(x[:-1]), 0.02)
                big = moves > 0.40
                print(f"  {tgt:>20s} {svc:<24s} |delta|>40%: {int(big.sum()):>5d} "
                      f"({100 * big.mean():.2f}% of steps)")
        print("=" * 40)
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch HPA/resource/call-graph data from Prometheus and emit a "
                    "spike-aware training CSV (strict 60s grid per service)."
    )
    parser.add_argument("--start", required=True,
                        help="Start time ('YYYY-MM-DD HH:MM:SS' Tehran or Unix ts)")
    parser.add_argument("--end", required=True,
                        help="End time ('YYYY-MM-DD HH:MM:SS' Tehran or Unix ts)")
    parser.add_argument("--prometheus-url", type=str, default=PROMETHEUS_URL,
                        help="Prometheus API base URL (default localhost:9090 via port-forward)")
    parser.add_argument("--out", type=str,
                        default=os.path.join(OUTPUT_DIR, OUTPUT_NAME),
                        help="Output CSV path (default %(default)s)")

    args = parser.parse_args()
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

    fetch_and_process_data(start_timestamp, end_timestamp, args.prometheus_url, args.out)
