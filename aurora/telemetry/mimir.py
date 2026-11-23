"""Mimir client.

Two jobs:

  * ``fetch`` -- run every feature query once over a time range and return
    ``{service: {feature: [(ts, value)]}}``. One query per family, not one per
    service: with 42 services a naive implementation would issue ~800 queries
    per cycle and become the slowest part of the control path.
  * ``nav()`` -- small helpers used by the lifecycle reconciler.

Every query is a range query so the window manager receives the original 5 s
resolution instead of a 15 s average.
"""

import json
import time
import urllib.parse
import urllib.request

from features.groups import queries


class MimirError(RuntimeError):
    pass


def _merge_sum(dst, key, pts):
    cur = dict(dst.get(key, []))
    for ts, v in pts:
        cur[ts] = cur.get(ts, 0.0) + v
    dst[key] = sorted(cur.items())


def _merge_max(dst, key, pts):
    cur = dict(dst.get(key, []))
    for ts, v in pts:
        cur[ts] = max(cur.get(ts, float("-inf")), v)
    dst[key] = sorted(cur.items())


def _resolve_pod(pod, services):
    """Map a pod name to its service by longest matching prefix.

    Pods are ``<deployment>-<replicaset-hash>-<pod-hash>``; cAdvisor emits some
    series per pod with no container label, so the service has to be recovered
    from the name.
    """
    if not pod:
        return None
    best = None
    for svc in services:
        if pod == svc or pod.startswith(svc + "-"):
            if best is None or len(svc) > len(best):
                best = svc
    return best


class MimirClient:
    def __init__(self, config):
        self.cfg = config
        self._cache = {}

    # ------------------------------------------------------------------
    def _get(self, url, params, timeout):
        data = urllib.parse.urlencode(params).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return json.load(fh)
        except Exception as exc:                      # noqa: BLE001
            raise MimirError(f"{type(exc).__name__}: {exc}") from exc

    def range(self, expr, start, end, step=None, timeout=10):
        step = step or self.cfg.telemetry_step_s
        payload = self._get(self.cfg.range_url,
                            {"query": expr, "start": int(start),
                             "end": int(end), "step": int(step)}, timeout)
        if payload.get("status") != "success":
            raise MimirError(payload.get("error", "bad status"))
        return payload.get("data", {}).get("result", [])

    def instant(self, expr, timeout=10):
        payload = self._get(self.cfg.query_url, {"query": expr}, timeout)
        if payload.get("status") != "success":
            raise MimirError(payload.get("error", "bad status"))
        return payload.get("data", {}).get("result", [])

    # ------------------------------------------------------------------
    def fetch(self, services, start, end, step=None, workers=1):
        """Return {service: {feature: [(ts, value)]}} for the requested window.

        ``services`` is the iterable of service names to keep. Queries that fail
        are recorded as absent rather than raising: a missing optional metric is
        normal, and the availability mask carries that through to the model.

        Pod-scoped series are mapped to their owning Deployment by prefix, so a
        query can group by ``pod`` without the caller needing container names to
        be unique (they are not -- several Deployments use the same container
        name).
        """
        out = {s: {} for s in services}
        self._cache = {}

        def _one(mq):
            t0 = time.time()
            try:
                return mq, self.range(mq.promql, start, end, step), time.time() - t0
            except MimirError:
                return mq, None, time.time() - t0

        qs = queries(self.cfg.namespace)
        results = []
        if workers and workers > 1:
            # Bounded parallelism: a cycle's families are independent queries.
            # The deadline in the control loop is the real bound; this only
            # stops serial latency from consuming the whole interval.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                results = list(pool.map(_one, qs))
        else:
            results = [_one(q) for q in qs]

        for mq, series, dt in results:
            self._family_s = getattr(self, "_family_s", {})
            self._family_s[mq.name] = dt
            if series is None:
                if mq.required:
                    raise MimirError(f"required query failed: {mq.name}")
                continue
            key = mq.name
            for s in series:
                labels = s.get("metric", {})
                svc = labels.get(mq.label_key)
                if svc is None:
                    # Fall back to the other identity labels this query may
                    # carry (destination_workload for Istio, deployment for
                    # kube-state-metrics, container for legacy series).
                    svc = (labels.get("destination_workload")
                           or labels.get("deployment")
                           or labels.get("container"))
                if svc not in out:
                    svc = _resolve_pod(svc, out)
                if svc not in out:
                    continue
                pts = []
                for ts, val in s.get("values", []):
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    if v != v or v in (float("inf"), float("-inf")):
                        continue
                    pts.append((int(ts), v))
                if not pts:
                    continue
                # A pod-scoped query returns one series per pod. Several pods
                # belong to one service, so collapse them per timestamp instead
                # of letting the last series overwrite the others (which
                # under-counted throughput and made pod_start_time arbitrary).
                if mq.label_key == "pod" and key != "pod_start_time":
                    _merge_sum(out[svc], key, pts)
                elif key == "pod_start_time":
                    # max(): the newest start time is the current process; the
                    # pod identity is not what this feature is about.
                    _merge_max(out[svc], key, pts)
                else:
                    out[svc][key] = pts
            self._cache[key] = len(series)
        return out

    def cache_summary(self):
        return dict(self._cache)

    # ------------------------------------------------------------------
    def services(self):
        """Deployment names in the namespace that are HPA-managed / scalable."""
        res = self.instant(
            f'kube_deployment_spec_replicas{{namespace="{self.cfg.namespace}"}}')
        return sorted({r["metric"].get("deployment", "") for r in res} - {""})

# ---------------------------------------------------------------------------
# Directed invocation edges
# ---------------------------------------------------------------------------
EDGE_RATE = ('sum by (source_workload, destination_workload) '
             '(rate(istio_requests_total{{reporter="destination",'
             'destination_workload_namespace="{ns}"}}[1m]))')
EDGE_ERR = ('sum by (source_workload, destination_workload) '
            '(rate(istio_requests_total{{reporter="destination",'
            'destination_workload_namespace="{ns}",response_code=~"5.."}}[1m]))')


def edge_query(expr, ns, quantile=None):
    if quantile is None:
        return expr.format(ns=ns)
    return ('histogram_quantile({q}, sum by (source_workload, destination_workload, le) '
            '(rate(istio_request_duration_milliseconds_bucket{{reporter="destination",'
            'destination_workload_namespace="{ns}"}}[1m])))').format(q=quantile, ns=ns)


class EdgeClient:
    """Directed source -> destination edge observations."""

    def __init__(self, config):
        self.cfg = config
        self.mimir = MimirClient(config)

    def fetch(self, start, end, step=None):
        """Return {edge_key: {feature: [(ts, value)]}}."""
        step = step or self.cfg.telemetry_step_s
        out = {}
        specs = [
            ("rate", edge_query(EDGE_RATE, self.cfg.namespace)),
            ("err", edge_query(EDGE_ERR, self.cfg.namespace)),
            ("p50", edge_query(None, self.cfg.namespace, 0.50)),
            ("p90", edge_query(None, self.cfg.namespace, 0.90)),
            ("p99", edge_query(None, self.cfg.namespace, 0.99)),
        ]
        for name, expr in specs:
            try:
                series = self.mimir.range(expr, start, end, step)
            except MimirError:
                continue
            for s in series:
                m = s.get("metric", {})
                src = m.get("source_workload") or "external"
                dst = m.get("destination_workload")
                if not dst:
                    continue
                # Edge identity includes BOTH namespaces: a workload name can
                # exist in more than one namespace, and merging those would
                # create a dependency edge that does not exist.
                key = (src, dst)
                d = out.setdefault(key, {})
                pts = []
                for ts, val in s.get("values", []):
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    if v != v:
                        continue
                    pts.append((int(ts), v))
                if not pts:
                    continue
                # Several series can share a (src, dst) pair when more than one
                # namespace is involved; merge them per timestamp.
                if name in d:
                    _merge_sum(d, name, pts)
                else:
                    d[name] = pts
        return out
