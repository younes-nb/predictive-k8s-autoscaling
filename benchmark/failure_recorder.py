#!/usr/bin/env python3
"""Failure-driven event recorder for the 25k HPA rerun (full-capture).

Runs on the master, polls Mimir every ~10 s and writes one JSONL record per
failing (service, code) observation above threshold, with:

  * hop attribution  — which upstream services feed 5xx into it (source-reporter
                       rate incl. response_flags), so gateway/envoy-generated
                       failures are distinguishable from app responses;
  * per-pod state    — app-container CPU (container!=""-corrected), sidecar CPU,
                       pod working set, CFS throttle ratio, net rx/tx, restarts,
                       ready/available, HPA current/desired/max;
  * app exceptions   — exact exception classes + message lines, primary from
                       Loki (per-service line-filtered range queries over the
                       app containers), fallback to the log_streamer per-pod
                       files (last ~2 min, tagged by pod);
  * classification   — heuristic root-cause tags per event;
  * cluster context  — k6 VU heap/iters (from the shipped console), front p90,
                       recent k8s scale events for the failing service.

Records: RUN_DIR/failures.jsonl (shipped live to loadgen by the orchestrator's
shipper). Heartbeat every 5 min keeps the timeline queryable even when nothing
fails. Thresholds via env (5XX_RATE_MIN, 4XX_RATE_MIN, EVENT_COOLDOWN_S).

Read-only: Mimir queries + kubectl get only. Exits at --end.
"""
import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

MIMIR = os.environ.get("PBS_MIMIR_URL", "http://127.0.0.1:18080") + "/prometheus"
LOKI = os.environ.get("LOKI_URL", "http://127.0.0.1:3100")
# Paths updated for the 2026-11-16 move (runner == loadgen node): kubectl and
# the kubeconfig no longer live under /users/younes02.
KUBECTL = os.environ.get("KUBECTL", "/usr/local/bin/kubectl")
KUBECONFIG = os.environ.get("KUBECONFIG",
                            "/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf")
NS = "train-ticket"

FIVE_MIN = float(os.environ.get("5XX_RATE_MIN", "0.03"))
FOUR_MIN = float(os.environ.get("4XX_RATE_MIN", "0.15"))
COOLDOWN = float(os.environ.get("EVENT_COOLDOWN_S", "60"))

APP_CPU_Q = ('sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="%s",'
             'pod=~"%s",container!="",container!="POD",container!="istio-proxy"}[2m]))')
PROXY_CPU_Q = ('sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="%s",'
               'pod=~"%s",container="istio-proxy"}[2m]))')
MEM_Q = ('container_memory_working_set_bytes{namespace="%s",pod=~"%s",container=""}')
THR_Q = ('sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{namespace="%s",pod=~"%s"}[2m]))'
         ' / sum by (pod) (rate(container_cpu_cfs_periods_total{namespace="%s",pod=~"%s"}[2m]))')
NET_Q = ('sum by (pod) (rate(container_network_{d}_bytes_total{namespace="%s",pod=~"%s"}[2m]))')
DEST_Q = ('sum by (destination_workload,response_code) (rate(istio_requests_total{namespace="%s",'
          'response_code=~"[45]..",reporter="destination"}[1m]))')
HOP_Q = ('sum by (source_workload,destination_workload,response_code,response_flags) '
         '(rate(istio_requests_total{namespace="%s",response_code=~"5..",reporter="source"}[1m]))')
CUM_Q = ('sum by (destination_workload,response_code) (increase(istio_requests_total{namespace="%s",'
         'response_code=~"[45]..",reporter="destination"}[%ds]))')
FLAGS_Q = ('sum by (response_flags) (rate(istio_requests_total{namespace="%s",reporter="destination",'
           'destination_workload="%s",response_code="%s"}[1m]))')
FRONT_P90_Q = ('histogram_quantile(0.9, sum(rate(istio_request_duration_milliseconds_bucket'
               '{reporter="destination",destination_workload="ts-ui-dashboard"}[1m])) by (le))')
RESTARTS_Q = 'sum(kube_pod_container_status_restarts_total{namespace="%s"})'
NODE_CPU_Q = 'max(100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m]))))'


_last_q_err = [0.0]


def q(expr, timeout=12):
    url = MIMIR + "/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)["data"]["result"]
    except Exception as e:
        # Never swallow silently: a bad selector used to zero out every per-pod
        # metric with no trace anywhere. Throttle to one line per 60 s.
        now = time.time()
        if now - _last_q_err[0] > 60:
            _last_q_err[0] = now
            sys.stderr.write("WARN mimir query failed: %s: %s\n" % (type(e).__name__, str(e)[:200]))
            sys.stderr.flush()
        return []


def qmap(expr):
    out = {}
    for s in q(expr):
        out[json.dumps(s["metric"], sort_keys=True)] = float(s["value"][1])
    return out


def kubectl_json(args, timeout=30):
    try:
        out = subprocess.run([KUBECTL] + args, capture_output=True, text=True,
                             timeout=timeout,
                             env={**os.environ, "KUBECONFIG": KUBECONFIG})
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except Exception:
        return None


def kubectl_scale_events(service, since_epoch):
    try:
        out = subprocess.run([KUBECTL, "-n", NS, "get", "events", "-o", "json",
                              "--sort-by=.lastTimestamp"], capture_output=True,
                             text=True, timeout=30,
                             env={**os.environ, "KUBECONFIG": KUBECONFIG})
        if out.returncode != 0:
            return []
        ev = json.loads(out.stdout)
        rows = []
        for it in ev.get("items", []):
            obj = it.get("involvedObject", {}).get("name", "")
            if service not in obj:
                continue
            ts = it.get("lastTimestamp") or it.get("firstTimestamp")
            if not ts:
                continue
            try:
                e = time.mktime(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
            except Exception:
                continue
            if e >= since_epoch - 120:
                rows.append({"ts": ts, "obj": obj, "reason": it.get("reason"),
                             "msg": (it.get("message") or "")[:180]})
        return rows[-8:]
    except Exception:
        return []


def svc_pods_regex(service):
    # _re2_escape, not re.escape: Python's re.escape escapes '-' as '\-',
    # which Mimir/RE2 rejects with "unknown escape sequence U+002D" and every
    # per-pod metric silently degrades to 0.
    if service.startswith("ts-"):
        return "ts-%s-.*" % _re2_escape(service[3:])
    return _re2_escape(service) + "-.*"


EXC_RE = re.compile(r"\w+(?:\.\w+)*\.(?:\w+)?Exception[\w$]*")
# app log lines carry a local-time clock ("2026-11-14 17:09:33.839 ..." = UTC+8
# on the train-ticket pods); T or space separator both accepted.
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})")
LOG_TZ_OFF_H = float(os.environ.get("LOG_TZ_OFFSET_H", "8"))
# LogQL line filter (RE2) — mirrors the file-tail filter used by the fallback.
LOKI_FILTER = r"ERROR|Exception|SEVERE|FATAL|WARN.*(refus|timeout|reset)"
# The ingress gateway is Envoy: 5xx/4xx show up in access lines, not ERROR lines
# (matches `..." 500 - via_upstream ...`).
LOKI_GATEWAY_FILTER = r'\" [45][0-9][0-9] '
# App-pod Envoy (istio-proxy) access lines. A request that never reached the
# application leaves no Java exception anywhere; it is only visible here, as a
# 4xx/5xx status and/or an upstream/downstream-failure response flag. The
# `via_upstream` 200 path is not matched.
#   UF upstream connection failure        UH no healthy upstream
#   UO upstream overflow (circuit break)  URX upstream retry limit exceeded
#   UC upstream connection termination    UT upstream request timeout
#   NR no route configured                DC downstream (client) disconnected
#   SI stream idle timeout
# DC matters because a k6-side timeout or cancel surfaces only as a downstream
# disconnect on the sidecar, with the application never seeing a completed call.
SIDECAR_UPSTREAM_FLAGS = "UF|UH|UO|URX|UC|UT|NR|SI"
SIDECAR_DOWNSTREAM_FLAGS = "DC"
SIDECAR_FLAG_ALT = SIDECAR_UPSTREAM_FLAGS + "|" + SIDECAR_DOWNSTREAM_FLAGS


def sidecar_filter(flag_alt):
    return r'\" [45][0-9][0-9] |\" [0-9][0-9][0-9] (%s) ' % flag_alt


LOKI_SIDECAR_FILTER = sidecar_filter(SIDECAR_FLAG_ALT)
# Envoy's X-Request-ID, consistent across the gateway and the app sidecar hops
# (verified 2026-11-16: same UUID at both hops of one request).
REQID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# Response flag out of an Envoy access line (`... "GET /x HTTP/1.1" 503 UF ...`).
SIDECAR_FLAG_RE = re.compile(r'" [0-9][0-9][0-9] (UF|UH|UO|URX|UC|UT|NR|SI|DC) ')
# Envoy access lines carry a UTC RFC3339 timestamp in brackets:
#   [2026-11-16T15:13:12.908Z] "GET /path HTTP/1.1" 200 - ...
# log_streamer.sh writes `kubectl logs --timestamps` output, which prefixes that
# with its own timestamp (`2026-11-16T18:41:34.511079640+03:30 [2026-11-16T...]`),
# so the bracketed Envoy timestamp must be searched for, not anchored to the start.
ENVOY_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})")

loki_state = {"up": None}  # None = untested, else True/False


def loki_query_range(expr, start_ns, end_ns, limit=60, timeout=12):
    url = LOKI + "/loki/api/v1/query_range?" + urllib.parse.urlencode({
        "query": expr, "start": str(int(start_ns)), "end": str(int(end_ns)),
        "limit": str(limit), "direction": "backward"})
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r).get("data", {}).get("result", [])


def _re2_escape(s):
    """Escape only chars RE2 treats specially (re.escape escapes '-' as '\\-',
    which Loki's RE2 parser rejects with HTTP 400)."""
    return re.sub(r"([.+*?(){}\[\]^$|\\])", r"\\\1", s)


def loki_pod_selector(service):
    """Label matcher for log streams of a service (Loki `instance` is
    ns/pod:container). App services: app containers only (negative matcher
    drops istio sidecar streams). The ingress gateway pods run ONLY the
    istio-proxy (Envoy) container, so there the exclusion is dropped."""
    if "gateway" in service:
        # Classic istio-ingressgateway lives in istio-system (2026-11-21).
        return 'instance=~"istio-system/istio-ingressgateway-.*"'
    if service == "ts-ui-dashboard":
        return 'instance=~"train-ticket/ts-ui-dashboard-.*"'
    if service.startswith("ts-"):
        pod = "ts-%s-.*" % _re2_escape(service[3:])
    else:
        pod = _re2_escape(service) + "-.*"
    return 'instance=~"train-ticket/%s", instance!~".*istio-proxy"' % pod


def loki_exceptions(service, since_epoch, now_s, limit=40):
    """Exception lines for a service from Loki (primary source). Scans are kept
    cheap: window clamped to the last ~7 min and line-filtered server-side.
    Returns (rows, classes) with rows {"ts", "pod", "line"}."""
    start_ns = int(max(since_epoch - 120, now_s - 420) * 1e9)
    end_ns = int((now_s + 2) * 1e9)
    filt = LOKI_GATEWAY_FILTER if ("gateway" in service
                                   or service == "ts-ui-dashboard") else LOKI_FILTER
    expr = '{%s} |~ "%s"' % (loki_pod_selector(service), filt)
    res = loki_query_range(expr, start_ns, end_ns, limit=limit)
    rows, classes = [], collections.Counter()
    for stream in res:
        pod = (stream.get("stream") or {}).get("instance", "?").split("/")[-1]
        for ts_ns, line in stream.get("values", []):
            rows.append({"ts": round(int(ts_ns) / 1e9, 3), "pod": pod,
                         "line": line[:340]})
            mc = EXC_RE.search(line)
            if mc:
                classes[mc.group(0).split(".")[-1]] += 1
    rows.sort(key=lambda r: r["ts"])
    return rows[-limit:], dict(classes)


def loki_sidecar_selector(service):
    """Label matcher for the app pod's istio-proxy (Envoy) stream.

    App pods run istio-proxy as a NATIVE sidecar (spec.initContainers,
    restartPolicy=Always), so their streams are `train-ticket/<pod>:istio-proxy`
    and were previously dropped by the negative matcher in
    `loki_pod_selector`. Returns None for the gateway/ts-ui-dashboard, whose only
    container already is istio-proxy and is covered by `loki_pod_selector`."""
    if "gateway" in service or service == "ts-ui-dashboard":
        return None
    if service.startswith("ts-"):
        pod = "ts-%s-.*" % _re2_escape(service[3:])
    else:
        pod = _re2_escape(service) + "-.*"
    return 'instance=~"train-ticket/%s:istio-proxy"' % pod


def loki_sidecar_lines(service, since_epoch, now_s, limit=20):
    """Envoy access lines for a service's app pods (primary sidecar source).

    Returns rows {"ts", "pod", "req_id", "line"}. Raises on Loki errors; the
    caller treats that like any other Loki outage.

    Two passes: the upstream-failure flags are queried first, and DC
    (downstream/client disconnect) only if that returns nothing. Under load DC
    is emitted by every client-side cancel and would otherwise crowd out the
    UF/UC/UT lines that actually explain a 503."""
    sel = loki_sidecar_selector(service)
    if not sel:
        return []
    start_ns = int(max(since_epoch - 120, now_s - 420) * 1e9)
    end_ns = int((now_s + 2) * 1e9)
    for flags in (SIDECAR_UPSTREAM_FLAGS, SIDECAR_FLAG_ALT):
        expr = '{%s} |~ "%s"' % (sel, sidecar_filter(flags))
        rows = []
        for stream in loki_query_range(expr, start_ns, end_ns, limit=limit):
            pod = (stream.get("stream") or {}).get("instance", "?").split("/")[-1]
            for ts_ns, line in stream.get("values", []):
                m = REQID_RE.search(line)
                rows.append({"ts": round(int(ts_ns) / 1e9, 3), "pod": pod,
                             "req_id": m.group(0) if m else None,
                             "line": line[:340]})
        if rows:
            rows.sort(key=lambda r: r["ts"])
            return rows[-limit:]
    return []


def extract_sidecar_files(logd, service, since_epoch, limit=8):
    """Fallback sidecar rows from the log_streamer files (`<pod>__istio-proxy.log`).

    Used when Loki is down. Envoy stamps a UTC timestamp in brackets on every
    access line; the file lines are additionally prefixed by `kubectl logs
    --timestamps`, so the bracketed timestamp is searched for. Same two-pass
    upstream-first precedence as `loki_sidecar_lines`."""
    pats = []
    if service.startswith("ts-"):
        pats = glob.glob(os.path.join(logd, "ts-%s-*__istio-proxy.log" % service[3:]))
    elif "gateway" in service:
        pats = glob.glob(os.path.join(logd, "train-ticket-gateway*__istio-proxy.log"))
    rows_all = []
    for f in sorted(pats):
        pod = os.path.basename(f).split("__")[0]
        try:
            with open(f, errors="replace") as fh:
                fh.seek(max(0, os.path.getsize(f) - 400000))
                tail = fh.read()
        except Exception:
            continue
        for ln in tail.splitlines():
            m = ENVOY_TS_RE.search(ln)
            if not m:
                continue
            try:
                e = time.mktime(time.strptime(m.group(1) + "T" + m.group(2),
                                              "%Y-%m-%dT%H:%M:%S")) - time.timezone
            except Exception:
                continue
            if e < since_epoch - 120:
                continue
            rid = REQID_RE.search(ln)
            rows_all.append({"ts": e, "pod": pod,
                             "req_id": rid.group(0) if rid else None,
                             "line": ln[:340]})
    for flags in (SIDECAR_UPSTREAM_FLAGS, SIDECAR_FLAG_ALT):
        rx = re.compile(sidecar_filter(flags))
        rows = [r for r in rows_all if rx.search(r["line"])]
        if rows:
            rows.sort(key=lambda r: r["ts"])
            return rows[-limit:]
    return []


def sidecar_lines_for(service, since_epoch, now_s, limit=20):
    """Loki first (primary), log_streamer files as fallback. Returns (rows, source)."""
    if loki_state["up"] is not False:
        try:
            rows = loki_sidecar_lines(service, since_epoch, now_s, limit)
            loki_state["up"] = True
            return rows, "loki"
        except Exception:
            if loki_state["up"] is None:
                loki_state["up"] = False
    return extract_sidecar_files(logd, service, since_epoch, limit), "files"


def exceptions_for(service, since_epoch, now_s, limit=40):
    """Loki first; falls back to the log_streamer tail files if Loki is down
    or empty. Annotates which source produced the rows."""
    if loki_state["up"] is not False:
        try:
            rows, classes = loki_exceptions(service, since_epoch, now_s, limit)
            loki_state["up"] = True
            return rows, classes, "loki"
        except Exception:
            if loki_state["up"] is None:
                loki_state["up"] = False
    rows, classes = extract_exceptions(logd, service, since_epoch, limit)
    return rows, classes, "files"


def extract_exceptions(logd, service, since_epoch, limit=8):
    pats = []
    if service.startswith("ts-"):
        pats = glob.glob(os.path.join(logd, "ts-%s-*__*.log" % service[3:]))
    elif "gateway" in service:
        pats = glob.glob(os.path.join(logd, "train-ticket-gateway*__*.log"))
    rows, classes = [], collections.Counter()
    for f in sorted(pats):
        pod = os.path.basename(f).split("__")[0]
        try:
            with open(f, errors="replace") as fh:
                fh.seek(max(0, os.path.getsize(f) - 600000))
                tail = fh.read()
        except Exception:
            continue
        hits = []
        for ln in tail.splitlines():
            if not re.search(r"ERROR|Exception|SEVERE|FATAL|WARN.*(refus|timeout|reset)", ln, re.I):
                continue
            m = TS_RE.match(ln)
            if not m:
                continue
            try:
                e = time.mktime(time.strptime(m.group(1) + "T" + m.group(2)[:15],
                                              "%Y-%m-%dT%H:%M:%S")) \
                    - time.timezone - LOG_TZ_OFF_H * 3600
            except Exception:
                continue
            if e >= since_epoch - 120:
                hits.append((e, ln[:340]))
        for _, ln in sorted(hits)[-limit:]:
            rows.append({"pod": pod, "line": ln})
            mc = EXC_RE.search(ln)
            if mc:
                classes[mc.group(0).split(".")[-1]] += 1
    return rows, dict(classes)


def sidecar_tags(sidecar_lines):
    """Mesh-side tags from Envoy access lines.

    This is the point of capturing the sidecar: it separates a 503 the
    application produced from one the mesh produced (no healthy upstream,
    connection failure, upstream timeout, circuit breaker, no route), which is
    invisible in the Java exception stream."""
    flags = set()
    for r in sidecar_lines or ():
        m = SIDECAR_FLAG_RE.search(r.get("line") or "")
        if m:
            flags.add(m.group(1))
    out = []
    if "UH" in flags:
        out.append("MESH_NO_HEALTHY_UPSTREAM")
    if flags & {"UF", "UC"}:
        out.append("MESH_UPSTREAM_CONN_FAILURE")
    if flags & {"UT", "URX"}:
        out.append("MESH_UPSTREAM_TIMEOUT")
    if "UO" in flags:
        out.append("MESH_CIRCUIT_BREAKER")
    if "NR" in flags:
        out.append("MESH_NO_ROUTE")
    if "DC" in flags:
        out.append("CLIENT_DISCONNECT")
    return out


def classify(kind, code, classes, unready, scale_events, sidecar=()):
    tags = []
    txt = " ".join(classes.keys()).lower()
    if kind == "5xx":
        if "mongosocket" in txt or "socketexception" in txt and "mongo" in txt:
            tags.append("DB_POOL_STALE")
        if "httpservererrorexception" in txt:
            tags.append("DOWNSTREAM_HTTP_5XX")
        if "sockettimeout" in txt or "readtimeout" in txt or "timed out" in txt:
            tags.append("TIMEOUT")
        if "nullpointer" in txt or "illegalstate" in txt or "numberformat" in txt:
            tags.append("APP_LOGIC")
        if code == "503":
            if unready > 0 or scale_events:
                tags.append("SCALE_CHURN_BOOT")
            tags.extend(sidecar_tags(sidecar))
            if not tags:
                tags.append("503_APP_OR_MESH")
    else:
        if code == "401":
            tags.append("AUTH_REJECT_TRACE_PATTERN")
        elif code == "404":
            tags.append("ROUTE_OR_STATIC")
        elif code == "403":
            tags.append("FORBIDDEN")
        else:
            tags.append("HTTP_4XX")
    return tags or ["UNCLASSIFIED"]


K6_TAIL_RE = re.compile(
    r"running \(([^)]+)\),\s*(\d+)/(\d+) VUs[^\n]*?(\d+) complete and (\d+) interrupted")


def k6_stats(run_dir):
    f = os.path.join(run_dir, "k6_console.log")
    try:
        with open(f, errors="replace") as fh:
            fh.seek(max(0, os.path.getsize(f) - 20000))
            m = K6_TAIL_RE.search(fh.read())
        if m:
            return {"elapsed": m.group(1), "vus_active": int(m.group(2)),
                    "vus_alloc": int(m.group(3)), "iters_complete": int(m.group(4)),
                    "iters_interrupted": int(m.group(5))}
    except Exception:
        pass
    return {}


def append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


logd = "."  # set in main(); used by the file fallback path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--runid", required=True)
    a = ap.parse_args()
    rd = a.run_dir
    global logd
    logd = os.path.join(rd, "logs")
    outf = os.path.join(rd, "failures.jsonl")
    t_traffic = None          # detected via ui rps
    last_emit = {}            # (service, codeclass) -> epoch
    n = 0
    append(outf, {"utc": now(), "type": "RECORDER_START", "runid": a.runid,
                  "detail": {"five_min": FIVE_MIN, "four_min": FOUR_MIN,
                             "cooldown": COOLDOWN, "end": a.end}})

    while time.time() < a.end:
        t0 = time.time()
        n += 1
        if t_traffic is None:
            r = q('sum(rate(istio_requests_total{reporter="destination",'
                  'destination_workload="ts-ui-dashboard"}[30s]))')
            if r and float(r[0]["value"][1]) > 0.2:
                t_traffic = t0
                append(outf, {"utc": now(), "type": "TRAFFIC_START", "epoch": t_traffic})
        cum = {}
        if t_traffic is not None:
            elapsed = int(min(5700, max(60, t0 - t_traffic + 60)))
            for k, v in qmap(CUM_Q % (NS, elapsed)).items():
                m = json.loads(k)
                cum[(m.get("destination_workload"), m.get("response_code"))] = v
        rates = {}
        for k, v in qmap(DEST_Q % NS).items():
            m = json.loads(k)
            rates[(m.get("destination_workload"), m.get("response_code"))] = v
        hops = []
        for k, v in qmap(HOP_Q % NS).items():
            m = json.loads(k)
            if v > 0.01:
                hops.append({"src": m.get("source_workload"), "dst": m.get("destination_workload"),
                             "code": m.get("response_code"), "flags": m.get("response_flags", ""),
                             "rate_1m": round(v, 3)})
        failing = sorted({(s, c) for (s, c), v in rates.items()
                          if v > (FIVE_MIN if c.startswith("5") else FOUR_MIN)})
        ctx = {"front_p90": (q(FRONT_P90_Q) or [{"value": [0, None]}])[0]["value"][1],
               "k6": k6_stats(rd)}
        if failing:
            regex = "|".join(svc_pods_regex(s) for s, _ in failing)
            pods_q = kubectl_json(["get", "pods", "-n", NS, "-o", "json"]) or {}
            hpa_q = kubectl_json(["get", "hpa", "-n", NS, "-o", "json"]) or {}
            hpa = {h["metadata"]["name"]: h for h in hpa_q.get("items", [])}
            appcpu, proxycpu, mem, thr, netr, netx = {}, {}, {}, {}, {}, {}
            for name, res in (("app", qmap(APP_CPU_Q % (NS, regex))),
                              ("proxy", qmap(PROXY_CPU_Q % (NS, regex))),
                              ("mem", qmap(MEM_Q % (NS, regex))),
                              ("netr", qmap(NET_Q.replace("{d}", "receive") % (NS, regex))),
                              ("netx", qmap(NET_Q.replace("{d}", "transmit") % (NS, regex)))):
                for k, v in res.items():
                    pod = json.loads(k).get("pod")
                    {"app": appcpu, "proxy": proxycpu, "mem": mem,
                     "netr": netr, "netx": netx}[name].setdefault(pod, []).append(v)
            for k, v in qmap(THR_Q % (NS, regex, NS, regex)).items():
                pod = json.loads(k).get("pod")
                thr.setdefault(pod, []).append(v)
            for s, code in failing:
                key = (s, "5xx" if code.startswith("5") else code)
                if t0 - last_emit.get(key, 0) < COOLDOWN:
                    continue
                last_emit[key] = t0
                since = t_traffic or (t0 - 300)
                pods, scale_ev = [], []
                pl = pods_q.get("items", []) if isinstance(pods_q, dict) else []
                dep_unready = 0
                for p in pl:
                    if not p["metadata"]["name"].startswith(
                            "ts-%s-" % s[3:] if s.startswith("ts-") else s):
                        continue
                    cs = p.get("status", {}).get("containerStatuses") or []
                    pod_cpu = {}
                    for c in cs:
                        pod_cpu[c["name"]] = {"ready": c.get("ready"),
                                              "restarts": (c.get("restartCount", 0))}
                    pn = p["metadata"]["name"]
                    ws = max(mem.get(pn, [0])) / 1048576
                    pods.append({
                        "pod": pn,
                        "phase": p.get("status", {}).get("phase"),
                        "ready": all(c.get("ready") for c in cs) if cs else None,
                        "restarts": sum(c.get("restartCount", 0) for c in cs),
                        "app_cpu": round(max(appcpu.get(pn, [0]) or [0]), 3),
                        "proxy_cpu": round(max(proxycpu.get(pn, [0]) or [0]), 3),
                        "pod_ws_mb": round(ws, 1),
                        "thr_max": round(max(thr.get(pn, [0]) or [0]), 3),
                        "net_rx_kbps": round(max(netr.get(pn, [0]) or [0]) / 1024, 1),
                        "net_tx_kbps": round(max(netx.get(pn, [0]) or [0]) / 1024, 1)})
                    if not (cs and all(c.get("ready") for c in cs)):
                        dep_unready += 1
                hname = "%s-hpa" % s
                h = hpa.get(hname, {})
                hst = h.get("status", {}) if isinstance(h, dict) else {}
                hsp = h.get("spec", {}) if isinstance(h, dict) else {}
                classes = {}
                if code.startswith("5"):
                    exrows, classes, exc_src = exceptions_for(s, since, t0, 40)
                    scale_ev = kubectl_scale_events(s, since)
                else:
                    exrows, classes, exc_src = exceptions_for(s, since, t0, 4)
                    scale_ev = []
                # Envoy access lines for this service's app pods. This is the
                # only place a request that never reached the application shows
                # up (upstream failure / connection reset / 503 with no Java
                # exception), and it carries the X-Request-ID shared with the
                # gateway hop.
                sidecar_rows, sidecar_src = sidecar_lines_for(s, since, t0, 20)
                rec = {"utc": now(), "epoch": t0, "type": "FAILURE_EVENT",
                       "kind": "5xx" if code.startswith("5") else "4xx",
                       "service": s, "code": code,
                       "rate_1m": round(rates[(s, code)], 3),
                       "cum_since_traffic": round(cum.get((s, code), 0), 1),
                       "pods": pods, "unready_pods": dep_unready,
                       "hpa": {"current": hst.get("currentReplicas"),
                               "desired": hst.get("desiredReplicas"),
                               "min": hsp.get("minReplicas"), "max": hsp.get("maxReplicas")},
                       "upstream_5xx_hops_1m": [h2 for h2 in hops
                                                if h2["dst"] == s and h2["code"] == code][:12],
                       "app_exceptions": exrows,
                       "exc_source": exc_src,
                       "exception_classes": classes,
                       "sidecar_lines": sidecar_rows,
                       "sidecar_source": sidecar_src,
                       "sidecar_request_ids": [r["req_id"] for r in sidecar_rows
                                               if r.get("req_id")],
                       "scale_events": scale_ev,
                       "classify": classify("5xx" if code.startswith("5") else "4xx",
                                            code, classes, dep_unready, scale_ev,
                                            sidecar_rows),
                       "cluster_ctx": ctx}
                append(outf, rec)
        if n % 30 == 0:  # heartbeat every ~5 min
            top = sorted(rates.items(), key=lambda kv: -kv[1])[:10]
            append(outf, {"utc": now(), "type": "HEARTBEAT",
                          "restarts_sum": (q(RESTARTS_Q % NS) or [{"value": [0, None]}])[0]["value"][1],
                          "node_cpu_max_pct": (q(NODE_CPU_Q) or [{"value": [0, None]}])[0]["value"][1],
                          "loki_up": loki_state["up"],
                          "top_error_rates": [{"service": s, "code": c, "rate_1m": round(v, 3)}
                                              for (s, c), v in top],
                          "cluster_ctx": ctx})
        time.sleep(max(0, 10 - (time.time() - t0)))
    append(outf, {"utc": now(), "type": "RECORDER_STOP", "events": n})


if __name__ == "__main__":
    main()
