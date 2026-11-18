#!/usr/bin/env python3
"""Stage 12K HPA experiment — live monitor (v2).

Differences from exp13jul/monitor.py (v1), all documented in
fixes_12k/DEVIATIONS_12k.md:

1. Deployment replica columns (auth_rep, gw_rep, ui_rep, rep_status_sum,
   rep_avail_sum) are sampled from `kubectl get deploy -o json` every 30 s
   instead of Mimir's kube-state-metrics series — the ksm replica series are
   exactly 2x duplicated on this cluster (two ksm instances), which made the
   v1 columns double-count. kubectl is the single source of truth. The cached
   sample (<=30 s old) is reused for the 15 s live.csv rows. A full snapshot
   of every deployment is appended to dep_history.jsonl on each 30 s sample.
   kubectl_deploy_ok=0 marks ticks where the kubectl sample failed (columns
   are then NULL, never Mimir-backed).

2. Tuned incident thresholds: the v1 steady-state voucher 500s (~4.5/s) and
   ui verifycode 404s (~4.6/s) were both script/manifest defects that are
   fixed for this run, so the low 5xx thresholds are raised (BURST 5->8/s,
   RATE 0.5->1.5/s sustained 30 s); anything below is still recorded in
   live.csv five_xx_rate every tick but no longer floods incidents.jsonl.

3. New incident types for the 12k operating point:
   - AUTH_AT_MAX: ts-auth-service pinned at its HPA maxReplicas with real
     traffic for >=2 min (saturation signal, replaces guessing from p90).
   - VU_HEAP_HIGH / VU_HEAP_EXTREME: k6 VU pile-up approaching the 3000 pool
     cap (>=2250 / >=2700 active).
   - ITER_DROPS: k6 interrupted-iterations delta >= 15 within one tick.
   AUTH_SLOWDOWN now needs a 2-sample streak like the other latency rules.

4. hpa_history.jsonl rows additionally carry minReplicas/maxReplicas so the
   saturation analysis does not need a second query.

Everything else (sampling periods, live.csv schema prefix, traffic-start
detection, trace_idx/offered axis, read-only posture) is identical to v1.
"""
import argparse, csv, json, os, re, subprocess, time, urllib.parse, urllib.request

KUBECTL = os.environ.get("KUBECTL", "/usr/local/bin/kubectl")
MIMIR = os.environ.get("PBS_MIMIR_URL", "http://127.0.0.1:18080") + "/prometheus"
# Paths updated for the 2026-11-16 move (runner == loadgen node); was a
# hardcoded /users/younes02/cluster-repo/admin.conf.
KUBECONFIG = os.environ.get("KUBECONFIG",
                            "/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf")
POOL = int(os.environ.get("USER_POOL", "3000"))

def q(expr, timeout=10):
    url = MIMIR + "/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)["data"]["result"]

def val(expr, default=None):
    try:
        r = q(expr)
        if r:
            return float(r[0]["value"][1])
    except Exception:
        pass
    return default

def append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def kubectl_json(args, timeout=25):
    try:
        out = subprocess.run([KUBECTL] + args, capture_output=True, text=True,
                             timeout=timeout,
                             env={**os.environ, "KUBECONFIG": KUBECONFIG})
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except Exception:
        return None

def sample_deployments():
    """{name: {spec, status, available, ready, unavailable}} via kubectl."""
    d = kubectl_json(["get", "deploy", "-n", "train-ticket", "-o", "json"])
    if d is None:
        return None
    out = {}
    for it in d.get("items", []):
        st = it.get("status", {}) or {}
        out[it["metadata"]["name"]] = {
            "spec": it.get("spec", {}).get("replicas") or 0,
            "status": st.get("replicas") or 0,
            "available": st.get("availableReplicas") or 0,
            "ready": st.get("readyReplicas") or 0,
            "unavailable": st.get("unavailableReplicas") or 0,
        }
    return out

def load_mcr(path, max_requests):
    """MCR[i] = round(raw/peak*MAXR); offered rps = MCR[i]/60 (COMPRESS=1)."""
    vals = []
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        ci = header.index("http_mcr") if "http_mcr" in header else -1
        for row in rdr:
            if not row or ci >= len(row):
                continue
            try:
                vals.append(float(row[ci]))
            except ValueError:
                pass
    peak = max(vals) or 1.0
    return [round(v / peak * max_requests) for v in vals]

def parse_k6_tail(tail):
    """k6 non-TTY progress: 'running (0m45.0s), 0142/0300 VUs, 12/30 shared iters, 345 complete and 2 interrupted iterations'"""
    out = {}
    m = re.search(r"running \(([^)]+)\),\s*(\d+)/(\d+) VUs[^\n]*?(\d+) complete and (\d+) interrupted", tail)
    if m:
        out["elapsed"], out["vus_active"], out["vus_alloc"], out["iters_complete"], out["iters_interrupted"] = \
            m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    else:
        m2 = re.search(r"(\d+) complete and (\d+) interrupted", tail)
        if m2:
            out["iters_complete"], out["iters_interrupted"] = int(m2.group(1)), int(m2.group(2))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--t0", type=float, required=True, help="k6 traffic-start epoch (monitor detects)")
    ap.add_argument("--end", type=float, required=True, help="stop epoch")
    ap.add_argument("--mcr", required=True, help="jul13 slice csv (mcr_jul13_slice.csv)")
    ap.add_argument("--max-requests", type=int, default=12000)
    ap.add_argument("--start-min", type=int, default=529)
    a = ap.parse_args()
    rd = a.run_dir
    os.makedirs(rd, exist_ok=True)
    live = os.path.join(rd, "live.csv")
    inc = os.path.join(rd, "incidents.jsonl")
    hpah = os.path.join(rd, "hpa_history.jsonl")
    deph = os.path.join(rd, "dep_history.jsonl")
    evh = os.path.join(rd, "events_history.jsonl")
    ndh = os.path.join(rd, "node_history.jsonl")
    k6log = os.path.join(rd, "k6_console.log")
    if not os.path.exists(live):
        with open(live, "w") as f:
            f.write("ts,front_p90,front_avg,five_xx_rate,auth_qps,auth_p90,auth_rep,auth_avail,"
                    "gw_rep,ui_rep,rep_status_sum,rep_avail_sum,restarts_sum,nodes_ready,"
                    "vus_active,iters_complete,iters_interrupted,trace_idx,offered_rps,"
                    "kubectl_deploy_ok\n")
    mcr = load_mcr(a.mcr, a.max_requests)
    nlen = len(mcr)

    last_inc = {}
    COOLDOWN = 300
    restarts_prev = None
    p90_high_streak = 0
    five50_streak = 0
    auth_slow_streak = 0
    auth_max_streak = 0
    iters_int_prev = None
    t_first_traffic = None
    dep_cache = None          # last kubectl deployment sample (<=30 s old)
    n = 0
    append(inc, {"utc": now(), "type": "MONITOR_START",
                 "detail": {"t0": a.t0, "end": a.end, "mcr_len": nlen,
                            "start_min": a.start_min, "max_requests": a.max_requests,
                            "monitor": "v2", "pool": POOL}})

    def emit(itype, detail, force=False):
        t = time.time()
        if not force and t - last_inc.get(itype, 0) < COOLDOWN:
            return
        last_inc[itype] = t
        append(inc, {"utc": now(), "epoch": t, "type": itype, "detail": detail})

    while time.time() < a.end:
        t = time.time()
        n += 1
        front_p90 = val('histogram_quantile(0.9, sum(rate(istio_request_duration_milliseconds_bucket{reporter="destination",destination_workload="ts-ui-dashboard"}[1m])) by (le))')
        front_avg = None
        try:
            s = q('sum(rate(istio_request_duration_milliseconds_sum{reporter="destination",destination_workload="ts-ui-dashboard"}[1m]))')
            c = q('sum(rate(istio_request_duration_milliseconds_count{reporter="destination",destination_workload="ts-ui-dashboard"}[1m]))')
            if s and c and float(c[0]["value"][1]) > 0:
                front_avg = float(s[0]["value"][1]) / float(c[0]["value"][1]) * 1000.0
        except Exception:
            pass
        five_xx = val('sum(rate(istio_requests_total{reporter="destination",namespace="train-ticket",response_code=~"5.."}[1m]))', 0.0) or 0.0
        auth_qps = val('sum(rate(istio_requests_total{reporter="destination",destination_workload="ts-auth-service"}[1m]))', 0.0) or 0.0
        auth_p90 = val('histogram_quantile(0.9, sum(rate(istio_request_duration_milliseconds_bucket{reporter="destination",destination_workload="ts-auth-service"}[1m])) by (le))')
        restarts_sum = val('sum(kube_pod_container_status_restarts_total{namespace="train-ticket"})', 0.0) or 0.0
        nodes_ready = val('count(kube_node_status_condition{condition="Ready",status="true"})', 0) or 0

        # replica columns: kubectl only (ksm series are 2x duplicated here)
        auth_rep = auth_avail = gw_rep = ui_rep = rep_sum = rep_avail = None
        kubectl_ok = 0
        if dep_cache is not None:
            kubectl_ok = 1
            ad = dep_cache.get("ts-auth-service", {})
            auth_rep, auth_avail = ad.get("status"), ad.get("available")
            gd = dep_cache.get("train-ticket-gateway-istio", {})
            gw_rep = gd.get("status")
            ud = dep_cache.get("ts-ui-dashboard", {})
            ui_rep = ud.get("status")
            rep_sum = sum(v["status"] for v in dep_cache.values())
            rep_avail = sum(v["available"] for v in dep_cache.values())

        k6stat, trace_idx, offered = {}, None, None
        try:
            if os.path.exists(k6log):
                with open(k6log, errors="replace") as f:
                    f.seek(max(0, os.path.getsize(k6log) - 20000))
                    k6stat = parse_k6_tail(f.read())
        except Exception:
            pass
        if t_first_traffic is None:
            uiqps = val('sum(rate(istio_requests_total{reporter="destination",destination_workload="ts-ui-dashboard"}[30s]))', 0.0) or 0.0
            if uiqps > 0.2:
                t_first_traffic = t
                emit("K6_TRAFFIC_START", {"first_traffic_utc": now(), "ui_rps_30s": uiqps}, force=True)
        if t_first_traffic is not None:
            elapsed_min = (t - t_first_traffic) / 60.0
            trace_idx = (a.start_min + int(elapsed_min)) % nlen
            offered = mcr[trace_idx] / 60.0
        append(live, {"ts": now(), "front_p90": front_p90, "front_avg": front_avg,
                      "five_xx_rate": five_xx, "auth_qps": auth_qps, "auth_p90": auth_p90,
                      "auth_rep": auth_rep, "auth_avail": auth_avail, "gw_rep": gw_rep,
                      "ui_rep": ui_rep, "rep_status_sum": rep_sum, "rep_avail_sum": rep_avail,
                      "restarts_sum": restarts_sum, "nodes_ready": nodes_ready,
                      "vus_active": k6stat.get("vus_active"), "iters_complete": k6stat.get("iters_complete"),
                      "iters_interrupted": k6stat.get("iters_interrupted"),
                      "trace_idx": trace_idx, "offered_rps": offered,
                      "kubectl_deploy_ok": kubectl_ok})

        # ---- incident candidates ----
        if front_p90 is not None and front_p90 > 1000:
            p90_high_streak += 1
            emit("P90_EXTREME", {"front_p90": front_p90, "streak": p90_high_streak})
        elif front_p90 is not None and front_p90 > 400:
            p90_high_streak += 1
            if p90_high_streak >= 2:
                emit("P90_SPIKE_HIGH", {"front_p90": front_p90, "streak": p90_high_streak})
        else:
            p90_high_streak = 0
        if five_xx > 8.0:
            five50_streak += 1
            emit("HTTP_5XX_BURST", {"rate_1m": five_xx, "streak": five50_streak})
        elif five_xx > 1.5:
            five50_streak += 1
            if five50_streak >= 2:
                emit("HTTP_5XX_RATE", {"rate_1m": five_xx, "streak": five50_streak})
        else:
            five50_streak = 0
        if restarts_prev is not None and restarts_sum > restarts_prev:
            emit("POD_RESTART", {"restarts_sum": restarts_sum, "prev": restarts_prev})
        restarts_prev = restarts_sum
        if nodes_ready < 6:
            emit("NODE_NOT_READY", {"nodes_ready": nodes_ready}, force=True)
        if auth_p90 is not None and auth_p90 > 5000 and auth_qps > 0.1:
            auth_slow_streak += 1
            if auth_slow_streak >= 2:
                emit("AUTH_SLOWDOWN", {"auth_p90": auth_p90, "auth_qps": auth_qps,
                                       "streak": auth_slow_streak})
        else:
            auth_slow_streak = 0
        if rep_avail is not None and rep_sum is not None and rep_sum - rep_avail > 5:
            emit("MANY_UNREADY_PODS", {"status": rep_sum, "available": rep_avail})

        # 12k-specific signals
        vus = k6stat.get("vus_active")
        if vus is not None and t_first_traffic is not None:
            if vus >= int(0.9 * POOL):
                emit("VU_HEAP_EXTREME", {"vus_active": vus, "pool": POOL})
            elif vus >= int(0.75 * POOL):
                emit("VU_HEAP_HIGH", {"vus_active": vus, "pool": POOL})
        ints = k6stat.get("iters_interrupted")
        if ints is not None and iters_int_prev is not None:
            delta = ints - iters_int_prev
            if delta >= 15:
                emit("ITER_DROPS", {"delta": delta, "total": ints})
        if ints is not None:
            iters_int_prev = ints

        if n % 2 == 0:  # HPA + deployments every 30 s
            d = kubectl_json(["get", "hpa", "-n", "train-ticket", "-o", "json"])
            auth_current = auth_hpa_max = None
            if d:
                for h in d.get("items", []):
                    st = h.get("status", {})
                    sp = h.get("spec", {})
                    cur = st.get("currentReplicas")
                    if h["metadata"]["name"] == "ts-auth-service-hpa":
                        auth_current = cur
                        auth_hpa_max = sp.get("maxReplicas")
                    append(hpah, {"utc": now(), "name": h["metadata"]["name"],
                                  "desired": st.get("desiredReplicas"),
                                  "current": cur,
                                  "min": sp.get("minReplicas"),
                                  "max": sp.get("maxReplicas"),
                                  "lastScale": st.get("lastScaleTime")})
            dep_cache = sample_deployments()
            if dep_cache is not None:
                append(deph, {"utc": now(), "deployments": dep_cache})
            # AUTH_AT_MAX: pinned at HPA ceiling with real traffic for >=2 min
            if auth_current is not None and auth_hpa_max is not None:
                amax = auth_hpa_max
                if auth_current >= amax and auth_qps > 1.0:
                    auth_max_streak += 1
                    if auth_max_streak >= 4:
                        emit("AUTH_AT_MAX", {"current": auth_current, "max": amax,
                                             "streak": auth_max_streak,
                                             "auth_qps": auth_qps, "auth_p90": auth_p90})
                else:
                    auth_max_streak = 0
        if n % 4 == 0:  # nodes every 60 s
            nd = kubectl_json(["get", "nodes", "-o", "json"])
            if nd:
                rows = []
                for it in nd.get("items", []):
                    conds = {c["type"]: c["status"] for c in it.get("status", {}).get("conditions", [])}
                    rows.append({"node": it["metadata"]["name"], "ready": conds.get("Ready"),
                                 "mem": conds.get("MemoryPressure"), "disk": conds.get("DiskPressure")})
                append(ndh, {"utc": now(), "nodes": rows})
        if n % 8 == 0:  # events every 120 s
            ev = kubectl_json(["get", "events", "-n", "train-ticket", "-o", "json", "--sort-by=.lastTimestamp"])
            if ev:
                keep = [{"ts": (it.get("lastTimestamp") or it.get("firstTimestamp")),
                         "obj": it.get("involvedObject", {}).get("name"),
                         "reason": it.get("reason"), "msg": (it.get("message") or "")[:220],
                         "count": it.get("count")} for it in ev.get("items", [])]
                append(evh, {"utc": now(), "events": keep})
        if n % 4 == 0 and front_p90 is None and t_first_traffic is not None and (t - t_first_traffic) > 120:
            emit("FRONT_TELEMETRY_EMPTY", {"since_traffic_s": t - t_first_traffic}, force=True)
        time.sleep(max(0, 15 - (time.time() - t)))
    append(inc, {"utc": now(), "type": "MONITOR_STOPPED", "detail": {"reason": "end reached"}})

if __name__ == "__main__":
    main()
