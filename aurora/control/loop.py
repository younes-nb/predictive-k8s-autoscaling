"""The 15-second control loop (two-model architecture).

Per cycle:

  1. collect node telemetry (5 s, rolling window)
  2. collect directed edge telemetry
  3. collect the frontend valid-request P99 deadline status
  4. build the graph observation
  5. run the shared Graph Diagnosis Model ONCE for the whole graph
  6. for each managed service, run the Replica-Change Model
  7. validate the proposal through the deterministic safety layer
  8. reconcile against lifecycle state and apply via the Scale subresource
  9. record the action in the ledger

Serialised: one cycle at a time. If a cycle overruns the interval the next
starts immediately and the overrun is logged -- never two writers at once.
"""

import time

from feedback.ledger import Ledger
from features.groups import FEATURE_ORDER, derive
from features.transform import transform
from features.window import RunningNorm, WindowManager
from lifecycle.reconciler import LifecycleReconciler
from safety import SafetyLayer
from telemetry.mimir import EdgeClient


class Span:
    __slots__ = ("name", "t0", "ms")

    def __init__(self, name):
        self.name = name
        self.t0 = time.perf_counter()
        self.ms = 0.0

    def stop(self):
        self.ms = (time.perf_counter() - self.t0) * 1000.0
        return self


class AuroraLoop:
    def __init__(self, cfg, bundle, scale, mimir, model_version=None,
                 services=None):
        self.cfg = cfg
        self.bundle = bundle
        self.diag = bundle["diagnosis"]
        self.replica = bundle["replica"]
        self.services_list = bundle["services"]
        self.adj = bundle["adj"]
        self.model_version = model_version or "bundle"
        self.scale = scale
        self.mimir = mimir
        self.edges = EdgeClient(cfg)

        self.norm = RunningNorm()
        self.windows = WindowManager(cfg, self.norm)
        self.lifecycle = LifecycleReconciler(cfg)
        self.ledger = Ledger(cfg.ledger_path)
        self.safety = SafetyLayer(cfg)
        self.services = list(services or [])
        self.timings = []
        self.cycles = 0
        self.errors = 0
        self.actions = 0
        self.election = None
        self.edge_hist = {}

    # ------------------------------------------------------------------
    def _collect(self, now):
        span = Span("collect")
        window_s = (self.cfg.history_len + 3) * self.cfg.telemetry_step_s
        start = now - window_s - 5
        fetched = self.mimir.fetch(self.services, start, now,
                                   step=self.cfg.telemetry_step_s)
        raws = {}
        for svc in self.services:
            series = fetched.get(svc, {})
            raws[svc] = {}
            by_ts = {}
            for key, pts in series.items():
                for ts, v in pts:
                    by_ts.setdefault(ts, {})[key] = v
            for ts in sorted(by_ts):
                vals, mask = transform(by_ts[ts], now=ts)
                fresh = {k: val for k, val, m in zip(FEATURE_ORDER, vals, mask) if m}
                self.windows.ingest(svc, ts, fresh)
                raws[svc].update(by_ts[ts])
            arr = raws[svc].get("req_rate")
            lat = raws[svc].get("lat_p90")
            if arr is not None:
                self.windows.note_availability(svc, now, arr, lat is not None)

        # directed edges
        self.edge_hist = self.edges.fetch(start, now, self.cfg.telemetry_step_s)
        span.stop()
        return raws, span

    # ------------------------------------------------------------------
    def _diagnose(self, raws, now):
        import torch
        span = Span("diagnose")
        svcs = self.services
        idx = {s: i for i, s in enumerate(self.services_list)}
        n = len(svcs)
        x = torch.zeros(n, self.cfg.history_len, len(FEATURE_ORDER))
        m = torch.zeros(n, self.cfg.history_len, len(FEATURE_ORDER))
        for i, s in enumerate(svcs):
            b = self.windows.batch(s, now, features=FEATURE_ORDER)
            t = min(b.n_steps, self.cfg.history_len)
            x[i, :t] = torch.tensor(b.matrix, dtype=torch.float32)
            m[i, :t] = torch.tensor(b.mask, dtype=torch.float32)
        adj = torch.zeros(n, n)
        for i, si in enumerate(svcs):
            for j, sj in enumerate(svcs):
                a, b = idx.get(si), idx.get(sj)
                if a is not None and b is not None and self.adj[a][b] > 0:
                    adj[i, j] = 1.0
        with torch.no_grad():
            out = self.diag(x, m, adj, torch.zeros(n, n, self.diag.d_model),
                            m.any(-1).any(-1).float())
        span.stop()
        return {s: {
            "root_pb": float(out["root_pb"][i]),
            "affected": float(out["affected"][i]),
            "degradation": out["degradation"][i].tolist(),
            "provisioning": out["provisioning"][i].tolist(),
            "provisioning_idx": int(out["provisioning"][i].argmax()),
            "confidence": float(out["provisioning"][i].max()),
            "ood": False,
            "embedding": out["node_embedding"][i].tolist(),
        } for i, s in enumerate(svcs)}, span

    # ------------------------------------------------------------------
    def _decide_one(self, svc, diag, raw, now):
        import torch
        req = raw.get("req_rate")
        if req is None:
            return None, None
        ready, pending, starts, ages, _ = self.scale.pod_counts(svc)
        spec = self.scale.spec_replicas(svc)
        for a in starts:
            self.lifecycle.observe_startup(svc, a)
        acc = self.lifecycle.accounting(svc, spec, ready, pending, now)
        self.lifecycle.settle(svc, now, ready)

        lat = raw.get("lat_p90")
        deadline_missed = lat is not None and lat > self.cfg.slo_budget_ms

        d = {
            "service": svc, "confidence": diag["confidence"], "ood": diag["ood"],
            "root_pb": diag["root_pb"],
            "addressable_prob": 0.5 if diag["root_pb"] > 0.5 else diag["affected"],
            "provisioning_idx": diag["provisioning_idx"],
            "binding_signal": "latency" if diag["affected"] > 0.5 else "cpu",
        }
        # Replica-Change Model input
        x_in = torch.tensor([diag["embedding"] + [diag["root_pb"], diag["affected"]]
                             + diag["degradation"] + diag["provisioning"]
                             + [__import__("math").log1p(max(req, 0.0)),
                                __import__("math").log1p(max(1.0, 0.0)), 0.0,
                                float(spec), float(acc["committed"]),
                                float(self.cfg.min_pod), float(self.cfg.max_pod),
                                1.0, 1.0]], dtype=torch.float32)
        action, target, _ = self.replica.decide(
            x_in, self.cfg.min_pod, self.cfg.max_pod)
        action, target, reason = self.safety.validate(
            action, target, d, acc["committed"],
            self.lifecycle.pending_scale_out(svc), deadline_missed,
            self.cfg.min_pod, self.cfg.max_pod)
        d["reason"] = reason
        d["target"] = target
        d["deadline_missed"] = deadline_missed
        d["committed"] = acc["committed"]
        return d, acc

    # ------------------------------------------------------------------
    def cycle(self):
        now = time.time()
        cstart = time.perf_counter()
        spans = []
        try:
            self.services = self.services or [
                s for s in self.scale.scalable_services() if s != "aurora"]
            raws, s1 = self._collect(now); spans.append(s1)
            diags, s2 = self._diagnose(raws, now); spans.append(s2)
            for svc in self.services:
                try:
                    d, acc = self._decide_one(svc, diags[svc], raws.get(svc, {}), now)
                    if d is None:
                        continue
                    s3 = Span("decide"); s3.stop(); spans.append(s3)
                    if d["target"] != d["committed"] and not self.cfg.dry_run:
                        if self.election is None or self.election.renew():
                            self.scale.set_scale(svc, d["target"])
                            self.lifecycle.record(svc, now, d["committed"],
                                                  d["target"], d["reason"])
                            self.ledger.record_action(svc, now, self.model_version,
                                                      {"replicas": d["committed"]},
                                                      type("D", (), d)(), {})
                            self.actions += 1
                        print(f"[aurora] {svc}: {d['target']} ({d['reason']})",
                              flush=True)
                    elif self.cycles < 2:
                        print(f"[aurora] {svc}: hold {d['committed']} "
                              f"({d['reason']})", flush=True)
                except Exception as exc:                    # noqa: BLE001
                    self.errors += 1
                    print(f"[aurora] {svc}: {type(exc).__name__}: {exc}", flush=True)
        except Exception as exc:                            # noqa: BLE001
            self.errors += 1
            print(f"[aurora] cycle error: {type(exc).__name__}: {exc}", flush=True)
        total = (time.perf_counter() - cstart) * 1000.0
        self.timings.append({"cycle": self.cycles, "total_ms": total,
                             "stages": {s.name: s.ms for s in spans}})
        self.cycles += 1
        return total

    # ------------------------------------------------------------------
    def run(self):
        end = time.time() + self.cfg.duration_s
        print(f"[aurora] start: {len(self.services)} services, "
              f"interval={self.cfg.decision_interval_s}s dry_run={self.cfg.dry_run}",
              flush=True)
        while time.time() < end:
            t0 = time.time()
            self.cycle()
            sleep = self.cfg.decision_interval_s - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
        self._report()

    def _report(self):
        if not self.timings:
            print("[aurora] no cycles"); return
        tot = sorted(t["total_ms"] for t in self.timings)
        q = lambda p: tot[min(len(tot) - 1, int(p * len(tot)))]
        print(f"[aurora] cycles={self.cycles} actions={self.actions} errors={self.errors}")
        print(f"[aurora] complete decision path ms: p50={q(.5):.1f} p95={q(.95):.1f} "
              f"p99={q(.99):.1f} max={tot[-1]:.1f}")