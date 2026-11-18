#!/usr/bin/env python3
"""Build the one-file post-run report for a TrainTicket load test.

One invocation -> ONE timestamped Markdown file. Nothing else is printed except
the path of the file that was written.

    python3 build_run_report.py <run_dir> [<run_dir> ...] [options]
    python3 build_run_report.py --runs-dir <dir> [--target 10-20]

Options
    --output FILE     write here (default: <run_dir>/REPORT_<runid>_<ts>.md for a
                      single run, <reports>/loadtest_report_<ts>.md for several)
    --mimir URL       Mimir base (default http://127.0.0.1:18080)
    --no-mimir        skip the Mimir supplements (node level, KSM, istio)
    --baseline N      1-based index of the run used for deltas (default 1)
    --target PCT      SLO violation band, e.g. "10-20" (default 10-20)

Report layout
    0  AGGREGATED RESULTS  (headline table, deltas vs baseline, auto verdicts)
    1  Load profile: offered vs delivered, latency
    2  SLO (headline, per-service latency, longest violation run)
    3  Reliability (failure events, 5xx, restarts)
    4  Resource utilisation (per service CPU/mem/network, node level)
    5  Autoscaler behaviour (events, reversals, replica-minutes, per-HPA peaks)
    6  Derived comparison (efficiency, stability, provisioning cost)
    7  Data quality and coverage
    A  Per-run detail

Sources, per run: params.json, k6_summary.json, live.csv, failures.jsonl,
dataset_*/{train_data.csv,metrics.csv,meta.json}, hpa_history.jsonl,
collector_exit_codes.txt, run_status.json -- plus Mimir for node-level,
kube-state-metrics and istio series that are not stored on disk.
"""

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import statistics
import sys
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def utc(ts=None):
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s):
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        pass
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def pct(part, whole):
    return 100.0 * part / whole if whole else float("nan")


def fmt(v, nd=1, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return dash
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def quantile(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats(vals):
    vals = [v for v in vals
            if v is not None and isinstance(v, (int, float))
            and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals), "mean": statistics.mean(vals),
        "p50": quantile(vals, 0.50), "p90": quantile(vals, 0.90),
        "p95": quantile(vals, 0.95), "p99": quantile(vals, 0.99),
        "max": max(vals), "min": min(vals),
    }


def fnum(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


class Mimir:
    def __init__(self, base, enabled=True):
        self.base = base.rstrip("/")
        self.enabled = enabled
        self.calls = 0
        self.failures = 0

    def _get(self, path, params):
        url = f"{self.base}{path}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                self.calls += 1
                return json.load(r)
        except Exception:
            self.failures += 1
            return {}

    def range(self, query, start, end, step=60):
        """-> {series_json: [(ts, value), ...]}"""
        d = self._get("/prometheus/api/v1/query_range", {
            "query": query, "start": int(start), "end": int(end), "step": step})
        out = {}
        for s in d.get("data", {}).get("result", []):
            out[json.dumps(s.get("metric", {}), sort_keys=True)] = [
                (float(t), float(v)) for t, v in s["values"]]
        return out

    def value(self, query, ts=None):
        p = {"query": query}
        if ts:
            p["time"] = int(ts)
        d = self._get("/prometheus/api/v1/query", p)
        r = d.get("data", {}).get("result", [])
        return float(r[0]["value"][1]) if r else float("nan")


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #

SVC_KEYS = ("qps", "count", "cpu_usage", "mem_usage", "p50", "avg", "p90", "p99",
            "success_rate")


class Run:
    def __init__(self, path, mimir):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path.rstrip("/"))
        self.mimir = mimir
        self.params = {}
        self.status = {}
        self.summary = {}
        self.live = []
        self.failures = []
        self.hpa = []
        self.metrics = []
        self.meta = {}
        self.train = []
        self.exit_codes = {}
        self.dataset_dir = None
        self._load_disk()
        self.window = self._window()
        self.svc = self._per_service()
        self._load_mimir()

    def _p(self, *a):
        return os.path.join(self.path, *a)

    def _load_disk(self):
        for attr, fn in (("params", "params.json"), ("status", "run_status.json")):
            p = self._p(fn)
            if os.path.exists(p):
                try:
                    setattr(self, attr, json.load(open(p)))
                except Exception:
                    pass
        p = self._p("k6_summary.json")
        if os.path.exists(p):
            try:
                self.summary = json.load(open(p))
            except Exception:
                pass
        self.live = read_jsonl(self._p("live.csv"))
        self.failures = read_jsonl(self._p("failures.jsonl"))
        self.hpa = read_jsonl(self._p("hpa_history.jsonl"))
        ec = self._p("collector_exit_codes.txt")
        if os.path.exists(ec):
            for line in open(ec, errors="replace"):
                parts = line.split()
                if len(parts) == 2:
                    self.exit_codes[parts[0]] = parts[1]

        ds = sorted(glob.glob(self._p("dataset_*")))
        if ds:
            self.dataset_dir = ds[-1]
            mp = os.path.join(ds[-1], "meta.json")
            if os.path.exists(mp):
                try:
                    self.meta = json.load(open(mp))
                except Exception:
                    pass
            tp = os.path.join(ds[-1], "train_data.csv")
            if os.path.exists(tp):
                with open(tp, newline="", errors="replace") as fh:
                    self.train = list(csv.DictReader(fh))
            mp2 = os.path.join(ds[-1], "metrics.csv")
            if os.path.exists(mp2):
                with open(mp2, newline="", errors="replace") as fh:
                    self.metrics = list(csv.DictReader(fh))

    def _window(self):
        launch = self.params.get("launch_epoch") or self.status.get("launch_epoch")
        end = self.params.get("end_epoch") or self.status.get("end_epoch")
        if launch and end:
            return int(launch), int(end)
        for src in (self.metrics, self.train, self.live):
            ts = [parse_ts(r.get("timestamp") or r.get("ts")) for r in src]
            ts = [t for t in ts if t]
            if ts:
                return int(min(ts)) - 60, int(max(ts)) + 60
        return 0, 0

    def _per_service(self):
        """{service: {metric: [values]}} from metrics.csv"""
        out = {}
        if not self.metrics:
            return out
        cols = list(self.metrics[0].keys())
        for col in cols:
            if "&" not in col:
                continue
            svc, key = col.split("&", 1)
            if key not in SVC_KEYS:
                continue
            vals = [fnum(r.get(col)) for r in self.metrics]
            out.setdefault(svc, {})[key] = vals
        return out

    # -- Mimir supplements (optional) ---------------------------------------- #
    def _load_mimir(self):
        self.node_cpu = {}
        self.node_mem = {}
        self.pending = {}
        self.oom = {}
        self.istio_5xx = {}
        if not self.mimir.enabled or not self.window[1]:
            return
        s, e = self.window
        self.node_cpu = self._agg(self.mimir.range(
            'sum by (node) (1 - rate(node_cpu_seconds_total{mode="idle"}[1m]))', s, e))
        self.node_mem = self._agg(self.mimir.range(
            'sum by (node) (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)', s, e))
        self.pending = self._agg(self.mimir.range(
            'count by (node) (kube_pod_status_phase{namespace="train-ticket",phase="Pending"} == 1)', s, e))
        self.oom = self._agg(self.mimir.range(
            'sum by (pod) (kube_pod_container_status_last_terminated_reason'
            '{namespace="train-ticket",reason="OOMKilled"})', s, e))
        self.istio_5xx = self._agg(self.mimir.range(
            'sum by (destination_service) (rate(istio_requests_total'
            '{reporter="destination",response_code=~"5.."}[1m]))', s, e))

    @staticmethod
    def _agg(series):
        out = {}
        for key, pts in series.items():
            try:
                label = json.loads(key)
            except Exception:
                label = {}
            name = (label.get("node") or label.get("pod")
                    or label.get("destination_service") or key)
            vals = [v for _, v in pts]
            if vals:
                out.setdefault(name, []).extend(vals)
        return {k: stats(v) for k, v in out.items()}

    # -- derived -------------------------------------------------------------- #
    def autoscaler(self):
        return self.params.get("autoscaler") or ("pbscaler" if "pbs" in self.name else "hpa")

    def k6(self, key, default=float("nan")):
        m = self.summary.get("metrics", {})
        if key not in m:
            return default
        v = m[key]
        if isinstance(v, dict):
            return v.get("value", v.get("count", default))
        return v

    def slo(self):
        c = {"0": 0, "1": 0, "-1": 0}
        for r in self.train:
            k = r.get("slo_reward", "-1")
            c[k] = c.get(k, 0) + 1
        v, ok = c.get("0", 0), c.get("1", 0)
        return {"viol": v, "ok": ok, "unlab": c.get("-1", 0),
                "rate": pct(v, v + ok)}

    def front_p90(self):
        return stats([fnum(r.get("front_p90")) for r in self.train])

    def violation_streak(self):
        best = cur = 0
        best_start = cur_start = None
        for r in self.train:
            if r.get("slo_reward") == "0":
                if cur == 0:
                    cur_start = r.get("timestamp")
                cur += 1
                if cur > best:
                    best, best_start = cur, cur_start
            else:
                cur = 0
        return best, best_start

    def hpa_events(self):
        """-> (events, reversals, peak_observed_per_service)"""
        last = {}
        events = []
        for row in self.hpa:
            name = row.get("name")
            cur = row.get("current")
            if name is None or cur is None:
                continue
            prev = last.get(name)
            if prev is not None and cur != prev:
                events.append({"svc": name, "from": prev, "to": cur,
                               "utc": row.get("utc")})
            last[name] = cur
        reversals = 0
        hist = {}
        for ev in events:
            h = hist.setdefault(ev["svc"], [])
            if len(h) >= 2 and (h[-1] - h[-2]) * (ev["to"] - ev["from"]) < 0:
                reversals += 1
            h.append(ev["to"])
        peak = {}
        for row in self.hpa:
            n = row.get("name")
            c = row.get("current")
            if n is None or c is None:
                continue
            peak[n] = max(peak.get(n, 0), c)
        return events, reversals, peak

    def replica_minutes(self):
        total = 0.0
        prev_t = None
        for row in self.hpa:
            t = parse_ts(row.get("utc"))
            cur = row.get("current")
            if t is None or cur is None:
                continue
            if prev_t:
                total += cur * (t - prev_t) / 60.0
            prev_t = t
        return total

    def delivered(self):
        sch = self.k6("scheduled_iterations_expected", self.k6("iterations", 0))
        done = self.k6("iterations", 0)
        drop = self.k6("dropped_iterations", 0)
        return sch, done, drop, pct(done, sch)

    def req_failed_pct(self):
        v = self.k6("http_req_failed")
        # k6's http_req_failed.value IS the failure ratio (1 = failed)
        return 100.0 * v if v == v else float("nan")

    def slo_verdict(self, target):
        rate = self.slo()["rate"]
        if rate != rate:
            return "no SLO labels"
        lo, hi = target
        if rate < lo:
            return "under target"
        if rate > hi:
            return "over target"
        return "in target"

    def svc_stats(self):
        """{svc: {metric: stats}}"""
        out = {}
        for svc, d in self.svc.items():
            out[svc] = {k: stats(v) for k, v in d.items() if k in SVC_KEYS}
        return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


class Report:
    def __init__(self, runs, baseline=1, target=(10.0, 20.0), mimir_url=""):
        self.runs = runs
        self.baseline = baseline
        self.target = target
        self.mimir_url = mimir_url
        self.L = []

    def w(self, s=""):
        self.L.append(s)

    def table(self, headers, rows, aligns=None):
        self.w("| " + " | ".join(headers) + " |")
        self.w("| " + " | ".join(aligns or ["---"] * len(headers)) + " |")
        for r in rows:
            self.w("| " + " | ".join(str(c) for c in r) + " |")
        self.w()

    def note(self, text):
        self.w(f"_{text}_")
        self.w()

    def header(self):
        ts = utc()
        self.w(f"# Load-test report — {ts}")
        self.w()
        self.w("Generated automatically by `stage7_28/exp13jul/build_run_report.py`.")
        self.w()
        self.w(f"- Runs: {', '.join('`%s`' % r.name for r in self.runs)}")
        self.w(f"- Baseline for deltas: run {self.baseline} (`{self.runs[self.baseline-1].name}`)")
        self.w(f"- SLO target band: {self.target[0]:.0f}–{self.target[1]:.0f} % violations")
        if self.mimir_url:
            self.w(f"- Mimir: `{self.mimir_url}` (node-level / KSM / istio supplements)")
        else:
            self.w("- Mimir supplements: disabled")
        self.w()

    def section_0(self):
        self.w("## 0. Aggregated results")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            sch, done, drop, dl = r.delivered()
            slo = r.slo()
            fp = r.front_p90()
            ev, rev, peak = r.hpa_events()
            rows.append([
                f"{i}", f"`{r.name}`", r.autoscaler(),
                fmt(r.params.get("max_requests"), 0),
                fmt(dl, 1) + " %",
                fmt(slo["rate"], 1) + " %",
                fmt(fp.get("p95"), 0) + " ms",
                fmt(r.req_failed_pct(), 2) + " %",
                fmt(drop, 0),
                fmt(len(ev), 0),
                fmt(max(peak.values()) if peak else float("nan"), 0),
                r.slo_verdict(self.target),
            ])
        self.table(
            ["#", "run", "autoscaler", "offered/min", "delivered", "SLO viol",
             "front p95", "req failed", "dropped", "scale evts", "peak rep", "SLO"],
            rows,
            ["---:", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:",
             "---:", "---:", "---"])
        self.section_0b()
        self.section_0c()

    def section_0b(self):
        self.w("### 0.1 Deltas vs the baseline run")
        self.w()
        base = self.runs[self.baseline - 1]
        b_slo = base.slo()["rate"]
        b_fp = base.front_p90().get("p95")
        b_dl = base.delivered()[3]
        b_cpu = sum((s.get("cpu_usage", {}).get("mean") or 0) * 1000
                    for s in base.svc_stats().values())
        rows = []
        for i, r in enumerate(self.runs, 1):
            if i == self.baseline:
                continue
            s = r.slo()["rate"]
            f = r.front_p90().get("p95")
            d = r.delivered()[3]
            c = sum((x.get("cpu_usage", {}).get("mean") or 0) * 1000
                    for x in r.svc_stats().values())
            rows.append([f"{i}", f"`{r.name}`",
                         f"{s - b_slo:+.1f} pp" if s == s and b_slo == b_slo else "—",
                         f"{f - b_fp:+.0f} ms" if f == f and b_fp == f and b_fp == b_fp else "—",
                         f"{d - b_dl:+.1f} pp" if d == d and b_dl == b_dl else "—",
                         f"{c - b_cpu:+.0f} m" if c == c and b_cpu == b_cpu else "—"])
        if rows:
            self.table(["#", "run", "Δ SLO viol", "Δ front p95", "Δ delivered",
                        "Δ CPU (sum of per-svc means)"], rows)
        else:
            self.note("Only the baseline run.")

    def section_0c(self):
        self.w("### 0.2 Automatic verdicts")
        self.w()
        for i, r in enumerate(self.runs, 1):
            sch, done, drop, dl = r.delivered()
            slo = r.slo()
            fp = r.front_p90()
            ev, rev, peak = r.hpa_events()
            bits = []
            rate = slo["rate"]
            bits.append(f"SLO {rate:.1f} % ({r.slo_verdict(self.target)})"
                        if rate == rate else "no SLO labels")
            bits.append(f"delivery {dl:.1f} %" if dl == dl else "delivery n/a")
            p95 = fp.get("p95", float("nan"))
            bits.append(f"front p95 {p95:.0f} ms" if p95 == p95 else "front p95 n/a")
            if drop:
                bits.append(f"{drop:,} dropped iterations")
            if ev:
                bits.append(f"{len(ev)} scale events, {rev} reversals, "
                            f"peak {max(peak.values())} replicas")
            bad = [f"{k}={v}" for k, v in r.exit_codes.items() if v != "0"]
            if bad:
                bits.append("collectors " + ", ".join(bad))
            st = r.status.get("status")
            if st and st != "complete":
                bits.append(f"run status `{st}`"
                            + (f" ({'; '.join(r.status.get('reasons', []))})"
                               if r.status.get("reasons") else ""))
            self.w(f"{i}. **{r.autoscaler()}** — " + "; ".join(bits) + ".")
        self.w()

    def section_1(self):
        self.w("## 1. Load profile — offered vs delivered")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            sch, done, drop, dl = r.delivered()
            rows.append([f"{i}", fmt(sch), fmt(done), fmt(drop),
                         fmt(dl, 1) + " %" if dl == dl else "—",
                         fmt(r.params.get("user_pool"), 0),
                         fmt(r.params.get("max_vus"), 0),
                         fmt(r.params.get("trace_window_peak_per_min"), 0)])
        self.table(["#", "scheduled", "completed", "dropped", "delivered", "pool",
                    "maxVUs", "trace peak/min"], rows)
        self.w("Latency from the k6 summary:")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            m = r.summary.get("metrics", {})
            d = m.get("http_req_duration", {})
            it = m.get("iteration_duration", {})
            rows.append([f"{i}",
                         fmt(d.get("med"), 0), fmt(d.get("p(90)"), 0),
                         fmt(d.get("p(95)"), 0), fmt(d.get("max"), 0),
                         fmt(it.get("med"), 0), fmt(it.get("p(95)"), 0),
                         fmt(m.get("dropped_iterations", {}).get("count"), 0)])
        self.table(["#", "http p50", "http p90", "http p95", "http max",
                    "iter p50", "iter p95", "dropped"], rows)

    def section_2(self):
        self.w("## 2. SLO")
        self.w()
        slo_ms = self.runs[0].meta.get("SLO_ms", 200)
        self.w(f"Front p90 > **{slo_ms:.0f} ms** on a 5 s tick "
               f"(`histogram_quantile(0.90, sum(rate(istio_request_duration_milliseconds_bucket"
               f"{{destination_workload=\"ts-ui-dashboard\"}}[1m])) by (le))`).")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            s = r.slo()
            fp = r.front_p90()
            best, start = r.violation_streak()
            rows.append([f"{i}", fmt(s["viol"]), fmt(s["ok"]), fmt(s["unlab"]),
                         fmt(s["rate"], 1) + " %",
                         fmt(fp.get("p50"), 0), fmt(fp.get("p95"), 0),
                         fmt(fp.get("max"), 0),
                         f"{best * 5 / 60:.1f} min" if best else "—"])
        self.table(["#", "viol", "ok", "unlabeled", "rate", "p50 ms", "p95 ms",
                    "max ms", "longest run"], rows)

        self.w("Per-service latency (from `metrics.csv`, 5 s samples):")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            ss = r.svc_stats()
            items = [(k, v) for k, v in ss.items() if v.get("p90", {}).get("n")]
            items.sort(key=lambda x: -x[1]["p90"]["p90"])
            for svc, v in items[:15]:
                rows.append([f"{i}", svc, fmt(v["p50"]["p50"], 0),
                             fmt(v["p90"]["p90"], 0), fmt(v["p99"]["p99"], 0),
                             fmt(v["avg"]["mean"], 0)])
        if rows:
            self.table(["#", "service", "p50 ms", "p90 ms", "p99 ms", "avg ms"], rows)
        else:
            self.note("No per-service latency data in this run.")

    def section_3(self):
        self.w("## 3. Reliability")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            ev = [e for e in r.failures if e.get("type") == "FAILURE_EVENT"]
            codes = {}
            for e in ev:
                codes[e.get("code")] = codes.get(e.get("code"), 0) + 1
            top = ", ".join(f"{k}×{v}" for k, v in
                            sorted(codes.items(), key=lambda x: -x[1])[:5])
            rows.append([f"{i}", fmt(len(ev)), top or "—",
                         fmt(r.k6("dropped_iterations"), 0)])
        self.table(["#", "failure events", "codes", "dropped iters"], rows)

        self.w("Per-service availability (from `metrics.csv`, `success_rate`):")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            ss = r.svc_stats()
            items = [(k, v) for k, v in ss.items() if v.get("success_rate", {}).get("n")]
            items.sort(key=lambda x: (x[1]["success_rate"]["mean"]
                                      if x[1]["success_rate"]["mean"] == x[1]["success_rate"]["mean"]
                                      else 2))
            for svc, v in items[:15]:
                sr = v["success_rate"]["mean"]
                rows.append([f"{i}", svc, fmt(100 * sr, 2) + " %",
                             fmt(v["qps"]["mean"], 1)])
        if rows:
            self.table(["#", "service", "success rate", "avg qps"], rows)
        else:
            self.note("No per-service success_rate in this run.")

        if any(r.istio_5xx for r in self.runs):
            self.w("Per-service 5xx rate (istio, destination reporter):")
            self.w()
            rows = []
            for i, r in enumerate(self.runs, 1):
                for svc, st in sorted(r.istio_5xx.items(),
                                      key=lambda x: -x[1]["mean"])[:10]:
                    rows.append([f"{i}", svc.replace(".train-ticket.svc.cluster.local", ""),
                                 fmt(st["mean"], 2) + " /s", fmt(st["max"], 2) + " /s"])
            if rows:
                self.table(["#", "service", "avg 5xx/s", "peak 5xx/s"], rows)

        self.w("Restarts and OOM (from `live.csv` and KSM):")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            rs = [fnum(x.get("restarts_sum")) for x in r.live]
            rs = [x for x in rs if x == x]
            oom = sum(st["max"] for st in r.oom.values()) if r.oom else 0
            rows.append([f"{i}", fmt(rs[0], 0) if rs else "—",
                         fmt(rs[-1], 0) if rs else "—",
                         fmt(max(rs) if rs else float("nan"), 0), fmt(oom, 0)])
        self.table(["#", "restarts start", "restarts end", "restarts peak", "OOM"], rows)

    def section_4(self):
        self.w("## 4. Resource utilisation")
        self.w()
        self.w("Per-service CPU and memory (from `metrics.csv`, `cpu_usage` in cores "
               "and `mem_usage` in MiB):")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            ss = r.svc_stats()
            items = [(k, v) for k, v in ss.items() if v.get("cpu_usage", {}).get("n")]
            items.sort(key=lambda x: -x[1]["cpu_usage"]["mean"])
            for svc, v in items[:15]:
                rows.append([f"{i}", svc,
                             fmt(v["cpu_usage"]["mean"] * 1000, 0) + " m",
                             fmt(v["cpu_usage"]["max"] * 1000, 0) + " m",
                             fmt(v["mem_usage"]["mean"], 0) + " Mi",
                             fmt(v["mem_usage"]["max"], 0) + " Mi"])
        if rows:
            self.table(["#", "service", "cpu avg", "cpu max", "mem avg", "mem max"], rows)
        else:
            self.note("No per-service resource data in this run.")

        if any(r.node_cpu for r in self.runs):
            self.w("Per-node utilisation (Mimir / node_exporter):")
            self.w()
            rows = []
            for i, r in enumerate(self.runs, 1):
                for node, st in sorted(r.node_cpu.items()):
                    rows.append([f"{i}", node, fmt(st["mean"] * 100, 0) + " %",
                                 fmt(st["max"] * 100, 0) + " %"])
            if rows:
                self.table(["#", "node", "cpu avg", "cpu max"], rows)

        if any(r.pending for r in self.runs):
            self.w("Pending pods and OOM kills (capacity signals):")
            self.w()
            rows = []
            for i, r in enumerate(self.runs, 1):
                pend = max([st["max"] for st in r.pending.values()] or [0])
                oom = sum(st["max"] for st in r.oom.values())
                rows.append([f"{i}", fmt(pend, 0), fmt(oom, 0)])
            self.table(["#", "peak pending pods", "OOM events"], rows)

    def section_5(self):
        self.w("## 5. Autoscaler behaviour")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            ev, rev, peak = r.hpa_events()
            ups = sum(1 for e in ev if e["to"] > e["from"])
            downs = sum(1 for e in ev if e["to"] < e["from"])
            rows.append([f"{i}", r.autoscaler(), fmt(len(ev)), fmt(ups), fmt(downs),
                         fmt(rev),
                         fmt(max(peak.values()) if peak else float("nan"), 0),
                         fmt(r.replica_minutes(), 0)])
        self.table(["#", "autoscaler", "events", "up", "down", "reversals",
                    "peak rep", "replica-min"], rows)

        self.w("Peak replicas per autoscaler target:")
        self.w()
        for i, r in enumerate(self.runs, 1):
            _, _, peak = r.hpa_events()
            if not peak:
                continue
            top = sorted(peak.items(), key=lambda x: -x[1])[:10]
            self.w(f"{i}. " + ", ".join(f"`{k.replace('-hpa','')}` {v}" for k, v in top))
        self.w()

        self.w("Scale events in time order (first 40):")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            for e in r.hpa_events()[0][:40]:
                rows.append([f"{i}", e.get("utc"), e["svc"].replace("-hpa", ""),
                             e["from"], e["to"],
                             "up" if e["to"] > e["from"] else "down"])
        if rows:
            self.table(["#", "utc", "target", "from", "to", "dir"], rows)
        else:
            self.note("No replica changes recorded.")

    def section_6(self):
        self.w("## 6. Derived comparison — efficiency, stability, provisioning cost")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            sch, done, drop, dl = r.delivered()
            ss = r.svc_stats()
            cpu_vals = [v["cpu_usage"]["mean"] for v in ss.values()
                        if v.get("cpu_usage", {}).get("n")]
            mem_vals = [v["mem_usage"]["mean"] for v in ss.values()
                        if v.get("mem_usage", {}).get("n")]
            cpu_mean = sum(cpu_vals) if cpu_vals else float("nan")
            mem_mean = sum(mem_vals) if mem_vals else float("nan")
            rep_min = r.replica_minutes()
            per1k_cpu = cpu_mean / (done / 1000) if done and cpu_vals else float("nan")
            per1k_mem = mem_mean / (done / 1000) if done and mem_vals else float("nan")
            per1k_rep = rep_min / (done / 1000) if done else float("nan")
            rows.append([f"{i}", fmt(per1k_cpu, 3), fmt(per1k_mem, 1),
                         fmt(per1k_rep, 3), fmt(rep_min, 0)])
        self.table(["#", "cpu-cores / 1k iters", "MiB / 1k iters",
                    "replica-min / 1k iters", "replica-min"], rows)
        self.note(
            "Lower CPU-cores, MiB and replica-minutes per 1k delivered iterations is "
            "more efficient. Compare against SLO compliance: an autoscaler that uses "
            "fewer resources by violating the SLO is not better.")

    def section_7(self):
        self.w("## 7. Data quality and coverage")
        self.w()
        rows = []
        for i, r in enumerate(self.runs, 1):
            m = r.meta
            rows.append([f"{i}",
                         str(m.get("result", "—")),
                         fmt(m.get("coverage_pct"), 1) + " %"
                         if m.get("coverage_pct") is not None else "—",
                         fmt(m.get("no_data_rows"), 0)
                         if m.get("no_data_rows") is not None else "—",
                         fmt(m.get("rows"), 0) if m.get("rows") is not None else "—",
                         fmt(len(r.train)), fmt(r.slo()["unlab"])])
        self.table(["#", "result", "coverage", "no_data", "rows", "train rows",
                    "unlabeled"], rows)

        self.w("Collectors:")
        self.w()
        for i, r in enumerate(self.runs, 1):
            if not r.exit_codes:
                self.w(f"{i}. no collector_exit_codes.txt")
                continue
            bad = [f"{k}={v}" for k, v in r.exit_codes.items() if v != "0"]
            self.w(f"{i}. " + (", ".join(bad) if bad else "all exit 0"))
        self.w()

        if self.mimir_url:
            self.w(f"Mimir query failures while building this report: "
                   f"{sum(1 for r in self.runs if r.mimir.failures)} run(s) with failures.")
            self.w()

    def appendix(self):
        self.w("## A. Per-run detail")
        self.w()
        for i, r in enumerate(self.runs, 1):
            self.w(f"### A.{i} `{r.name}`")
            self.w()
            p = r.params
            self.w(f"- autoscaler `{r.autoscaler()}`; offered/min {p.get('max_requests')}; "
                   f"pool {p.get('user_pool')}; maxVUs {p.get('max_vus')}; "
                   f"window {p.get('test_min')} min; warmup {p.get('warmup_s')} s")
            self.w(f"- trace {p.get('trace_day')} {p.get('trace_duration_min')} min "
                   f"({p.get('trace_window_start_utc')} .. {p.get('trace_window_end_utc')}), "
                   f"slice sha256 `{(p.get('trace_slice_sha256') or '')[:16]}…`")
            if r.window[1]:
                self.w(f"- window {dt.datetime.utcfromtimestamp(r.window[0]).isoformat()}Z .. "
                       f"{dt.datetime.utcfromtimestamp(r.window[1]).isoformat()}Z")
            self.w(f"- status `{r.status.get('status', '?')}`"
                   + (f" — {'; '.join(r.status.get('reasons', []))}"
                      if r.status.get("reasons") else ""))
            self.w()

    def render(self):
        self.header()
        self.section_0()
        self.section_1()
        self.section_2()
        self.section_3()
        self.section_4()
        self.section_5()
        self.section_6()
        self.section_7()
        self.appendix()
        return "\n".join(self.L) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def discover(runs_dir):
    out = []
    for name in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, name)
        if os.path.isdir(p) and (os.path.exists(os.path.join(p, "params.json"))
                                 or os.path.exists(os.path.join(p, "k6_summary.json"))):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Build the one-file post-run report (aggregated results first).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories")
    ap.add_argument("--runs-dir", help="discover run directories here")
    ap.add_argument("--output", help="output file")
    ap.add_argument("--mimir", default=os.environ.get("MIMIR_URL", "http://127.0.0.1:18080"))
    ap.add_argument("--no-mimir", action="store_true")
    ap.add_argument("--baseline", type=int, default=1)
    ap.add_argument("--target", default="10-20")
    args = ap.parse_args()

    paths = list(args.runs)
    if args.runs_dir:
        paths += discover(args.runs_dir)
    paths = [p for p in paths if os.path.isdir(p)]
    if not paths:
        print("no run directories given or found", file=sys.stderr)
        return 2

    try:
        lo, hi = [float(x) for x in args.target.split("-")]
    except ValueError:
        lo, hi = 10.0, 20.0

    mimir = Mimir(args.mimir, enabled=not args.no_mimir)
    runs = []
    for p in paths:
        try:
            runs.append(Run(p, mimir))
        except Exception as e:
            print(f"warning: could not read {p}: {e}", file=sys.stderr)
    if not runs:
        print("no readable runs", file=sys.stderr)
        return 2

    baseline = min(max(args.baseline, 1), len(runs))
    rep = Report(runs, baseline=baseline, target=(lo, hi),
                 mimir_url="" if args.no_mimir else args.mimir)
    text = rep.render()

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output:
        out = args.output
    elif len(runs) == 1:
        out = os.path.join(runs[0].path, f"REPORT_{runs[0].name}_{ts}.md")
    else:
        base = "/proj/k8sautoscaledl-PG0/reports/stage7_28/exp13jul"
        out = os.path.join(base, f"loadtest_report_{ts}.md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(text)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())