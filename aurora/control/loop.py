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
                 services=None, bounds=None, norm=None):
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
        if norm and norm.get("mean"):
            self.norm.set_prior(dict(zip(FEATURE_ORDER, norm["mean"])),
                                dict(zip(FEATURE_ORDER, norm["var"])))
            self.norm.freeze()
        self.windows = WindowManager(cfg, self.norm)
        self.lifecycle = LifecycleReconciler(cfg)
        self.ledger = Ledger(cfg.ledger_path)
        self.safety = SafetyLayer(cfg)
        self.services = list(services or [])
        self.bounds = dict(bounds or {})
        self.timings = []
        self.cycles = 0
        self.errors = 0
        self.actions = 0
        self.election = None
        self.edge_hist = {}
        self.pending_actions = []
        self.ood_centroid = bundle.get("ood_centroid")
        self.ood_threshold = bundle.get("ood_threshold")
        self.last_scale_in = {}

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
    def _frontend_slo(self):
        """User-facing deadline state: met / violated / unknown.

        Measured at the frontend workload over a 1 min rate, at the configured
        percentile. Missing data is "unknown", never "met" -- an unmeasured
        SLO must not authorise a scale-in.
        """
        pct = self.cfg.slo_percentile / 100.0
        expr = ('histogram_quantile(%s, sum(rate(istio_request_duration_milliseconds_bucket'
                '{reporter="destination",destination_workload="%s"}[1m])) by (le))'
                % (pct, self.cfg.front_svc))
        try:
            res = self.mimir.instant(expr)
        except Exception:
            return "unknown", None
        if not res:
            return "unknown", None
        try:
            val = float(res[0]["value"][1])
        except (TypeError, ValueError, KeyError):
            return "unknown", None
        return ("violated" if val > self.cfg.slo_budget_ms else "met"), val

    # ------------------------------------------------------------------
    def _diagnose(self, raws, now):
        import torch
        span = Span("diagnose")
        svcs = self.services
        # Indices are into the *runtime* node list (only managed services), not
        # the bundle's full service vocabulary.
        idx = {s: i for i, s in enumerate(svcs)}
        n = len(svcs)
        x = torch.zeros(n, self.cfg.history_len, len(FEATURE_ORDER))
        m = torch.zeros(n, self.cfg.history_len, len(FEATURE_ORDER))
        for i, s in enumerate(svcs):
            b = self.windows.batch(s, now, features=FEATURE_ORDER)
            t = min(b.n_steps, self.cfg.history_len)
            x[i, :t] = torch.tensor(b.matrix, dtype=torch.float32)
            m[i, :t] = torch.tensor(b.mask, dtype=torch.float32)
        # Sparse edge list: only observed caller->callee pairs, with their
        # latest 5-feature vector [rate, err, p50, p90, p99].
        import math
        src, dst, feats = [], [], []
        for si, sj in self.edge_hist:
            a, b = idx.get(si), idx.get(sj)
            if a is None or b is None:
                continue
            e = self.edge_hist[(si, sj)]
            latest = {k: e[k][-1][1] for k in ("rate", "err", "p50", "p90", "p99")
                      if e.get(k)}
            if not latest:
                continue
            src.append(a); dst.append(b)
            feats.append([math.log1p(max(latest.get(k, 0.0), 0.0))
                          for k in ("rate", "err", "p50", "p90", "p99")])
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else None
        edge_feat = torch.tensor(feats, dtype=torch.float32) if feats else None
        with torch.no_grad():
            out = self.diag(x, m, edge_index, edge_feat,
                            m.any(-1).any(-1).float())
        span.stop()
        import math
        res = {}
        for i, s in enumerate(svcs):
            # .tolist() rather than .numpy(): the container's torch is built
            # against NumPy 1.x and the image ships 2.x, so .numpy() raises.
            emb = out["node_embedding"][i].tolist()
            ood = False
            if self.ood_centroid is not None and self.ood_threshold:
                d = math.sqrt(sum((x - c) ** 2 for x, c in
                                  zip(emb, self.ood_centroid)))
                ood = d > self.ood_threshold
            rp = float(out["root_pb"][i]); af = float(out["affected"][i])
            deg = out["degradation"][i].tolist()
            prov = out["provisioning"][i].tolist()
            # C17: per-output confidence, combined conservatively
            conf = min(max(rp, 1.0 - rp), max(af, 1.0 - af),
                       max(prov) if prov else 0.0, max(deg) if deg else 0.0)
            res[s] = {
                "root_pb": rp,
                "affected": af,
                "degradation": deg,
                "provisioning": prov,
                "provisioning_idx": max(range(len(prov)), key=lambda j: prov[j]),
                "confidence": conf,
                "ood": ood,
                "embedding": emb,
            }
        return res, span

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

        slo_state, slo_value = self._frontend_slo()

        d = {
            "service": svc, "confidence": diag["confidence"], "ood": diag["ood"],
            "root_pb": diag["root_pb"], "affected": diag["affected"],
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
        min_pods, max_pods = self.bounds.get(
            svc, (self.cfg.min_pod, self.cfg.max_pod))
        action, target, reason = self.safety.validate(
            action, target, d, acc["committed"],
            self.lifecycle.pending_scale_out(svc), slo_state,
            min_pods, max_pods)
        # L05: lifecycle reconcile before anything is applied
        target, why = self.lifecycle.reconcile(svc, target, acc,
                                               min_pods, max_pods)
        if why.startswith("hold"):
            action, reason = "hold", f"{reason}; {why}"
        # C18: adaptive scale-in stabilization. A removal is only allowed after
        # the service has been stable for the configured window; otherwise hold.
        if action == "in":
            since = now - self.last_scale_in.get(svc, 0.0)
            if 0 < since < self.cfg.scale_in_stabilization_s:
                action, target, reason = "hold", acc["committed"], (
                    f"scale-in stabilization ({since:.0f}s < "
                    f"{self.cfg.scale_in_stabilization_s}s)")
        action, target, reason2 = self.safety.enforce_direction(
            action, target, acc["committed"])
        if reason2 != "ok":
            reason = f"{reason}; {reason2}"
        d["reason"] = reason
        d["target"] = target
        d["slo_state"] = slo_state
        d["slo_value_ms"] = slo_value
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
                        if self.election is not None and self.election.is_leader \
                                and self.election.renew():
                            self.scale.set_scale(svc, d["target"])
                            self.lifecycle.record(svc, now, d["committed"],
                                                  d["target"], d["reason"])
                            self.ledger.record_action(svc, now, self.model_version,
                                                      {"replicas": d["committed"]},
                                                      type("D", (), d)(), {})
                            self.actions += 1
                            if d["target"] < d["committed"]:
                                self.last_scale_in[svc] = now
                            self.pending_actions.append((svc, now, d["target"]))
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
        # F02: settle outcomes for actions whose pods are ready and enough
        # post-action time has passed. Unsettled actions are never training data.
        for pend in list(self.pending_actions):
            psvc, pts, ptgt = pend
            try:
                ready = self.scale.pod_counts(psvc)[0]
            except Exception:
                continue
            if now - pts >= 60 and ready >= ptgt:
                self.ledger.settle(psvc, pts, {"ready": ready,
                                               "target": ptgt})
                self.pending_actions.remove(pend)

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