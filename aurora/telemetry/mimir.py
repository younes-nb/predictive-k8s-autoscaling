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
import urllib.parse
import urllib.request

from features.groups import queries


class MimirError(RuntimeError):
    pass


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
    def fetch(self, services, start, end, step=None):
        """Return {service: {feature: [(ts, value)]}} for the requested window.

        ``services`` is the iterable of service names to keep. Queries that fail
        are recorded as absent rather than raising: a missing optional metric is
        normal, and the availability mask carries that through to the model.
        """
        out = {s: {} for s in services}
        self._cache = {}
        for mq in queries(self.cfg.namespace):
            try:
                series = self.range(mq.promql, start, end, step)
            except MimirError:
                if mq.required:
                    raise
                continue
            key = mq.name
            for s in series:
                labels = s.get("metric", {})
                svc = labels.get(mq.label_key) or labels.get("container") \
                    or labels.get("destination_workload")
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
                if pts:
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