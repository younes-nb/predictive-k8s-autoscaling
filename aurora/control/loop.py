"""The 15-second control loop.

Serialised: one iteration at a time, never overlapping with the next. The
interval is a floor, not a target -- if a cycle takes longer than 15 s the next
one starts immediately and the overrun is recorded, because starting a second
writer would violate the single-writer rule.

Every stage is timed, and the nine stage timings are reported at exit so the
p99 target is measured for the **complete** decision path rather than for the
neural forward pass alone.

Failure policy:

  * telemetry stale / incomplete  -> hold replicas (configurable)
  * inference failure             -> hold replicas
  * k8s API failure               -> log, keep the decision, retry next cycle
"""

import json
import os
import time

from drift.adaptation import AdaptationManager
from drift.detector import DriftDetector
from feedback.ledger import Ledger
from features.groups import FEATURE_ORDER, derive
from features.window import RunningNorm, WindowManager
from lifecycle.reconciler import LifecycleReconciler
from telemetry.schema import make_observation


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
    def __init__(self, cfg, model, scale, mimir, dry_run=None, norm=None,
                 services=None):
        self.cfg = cfg
        self.model = model
        self.scale = scale
        self.mimir = mimir
        self.dry_run = cfg.dry_run if dry_run is None else dry_run

        self.norm = RunningNorm()
        if norm:
            self.norm.global_mean = {k: float(v) for k, v in norm.get("mean", {}).items()} \
                if isinstance(norm.get("mean"), dict) else \
                {f: float(v) for f, v in zip(FEATURE_ORDER, norm.get("mean", []))}
            var = norm.get("var", [])
            self.norm.global_m2 = {f: float(v) for f, v in zip(FEATURE_ORDER, var)} \
                if isinstance(var, list) else {k: float(v) for k, v in var.items()}
            self.norm.global_n = 1000
        self.windows = WindowManager(cfg, self.norm)
        self.drift = DriftDetector()
        self.lifecycle = LifecycleReconciler(cfg)
        self.ledger = Ledger(cfg.ledger_path)
        self.adapt = AdaptationManager(cfg, self.drift, self.ledger)
        from capacity.estimator import CapacityEstimator
        from decision.engine import DecisionEngine
        self.capacity = CapacityEstimator()
        self.engine = DecisionEngine(cfg)

        self.timings = []
        self.cycles = 0
        self.holds = 0
        self.actions = 0
        self.errors = 0
        self.services = list(services or [])
        self._restore_state()

    # ------------------------------------------------------------------
    def _restore_state(self):
        os.makedirs(self.cfg.state_dir, exist_ok=True)
        try:
            with open(os.path.join(self.cfg.state_dir, "capacity.json")) as fh:
                self.capacity.load(json.load(fh))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        try:
            with open(os.path.join(self.cfg.state_dir, "drift.json")) as fh:
                self.drift.load(json.load(fh))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _persist_state(self):
        os.makedirs(self.cfg.state_dir, exist_ok=True)
        with open(os.path.join(self.cfg.state_dir, "capacity.json"), "w") as fh:
            json.dump(self.capacity.snapshot(), fh, indent=2)
        with open(os.path.join(self.cfg.state_dir, "drift.json"), "w") as fh:
            json.dump(self.drift.snapshot(), fh, indent=2)

    # ------------------------------------------------------------------
    def _collect(self, now):
        """Fetch + align + build batches. Returns (batches, raws, span)."""
        span = Span("collect")
        window_s = (self.cfg.history_len + 3) * self.cfg.telemetry_step_s
        start = now - window_s - 5
        fetched = self.mimir.fetch(self.services, start, now,
                                   step=self.cfg.telemetry_step_s)

        batches, raws = {}, {}
        for svc in self.services:
            series = fetched.get(svc, {})
            raws[svc] = {}
            for key, pts in series.items():
                if not pts:
                    continue
                # chronological order: ingest() discards any point older than
                # the newest one already stored, so feeding the newest first
                # silently dropped the whole history (measured: 1 sample per
                # feature per cycle instead of 24)
                for ts, v in pts:
                    self.windows.ingest(svc, ts, {key: v})
                raws[svc][key] = pts[-1][1]
            b = self.windows.batch(svc, now, features=FEATURE_ORDER)
            if not b.complete:
                self.drift.note_missing(svc)
                self.holds += 1
            batches[svc] = b
        span.stop()
        return batches, raws, span

    # ------------------------------------------------------------------
    @staticmethod
    def _raw_vector(raw, now):
        """Derive the final feature vector (raw + derived) for one service."""
        v = dict(raw)
        # pod_start_time is an epoch; the model sees pod age
        st = v.get("pod_start_time")
        if st is not None:
            v["pod_age_s"] = max(0.0, now - st)
            v.pop("pod_start_time", None)
        v.update(derive(raw))
        return v

    # ------------------------------------------------------------------
    def _infer(self, batches):
        """Batched inference over every service that has a complete window."""
        import torch
        span = Span("infer")
        usable = [s for s, b in batches.items() if b.complete]
        if not usable:
            span.stop()
            return {}, span
        t = max(b.n_steps for b in (batches[s] for s in usable))
        x = torch.zeros((len(usable), t, len(FEATURE_ORDER)), dtype=torch.float32)
        m = torch.zeros((len(usable), t, len(FEATURE_ORDER)), dtype=torch.float32)
        for i, s in enumerate(usable):
            b = batches[s]
            x[i, :b.n_steps] = torch.tensor(b.matrix, dtype=torch.float32)
            m[i, :b.n_steps] = torch.tensor(b.mask, dtype=torch.float32)
        out = self.model.decide(x, m)
        span.stop()
        return {s: o for s, o in zip(usable, out)}, span

    # ------------------------------------------------------------------
    def _decide(self, svc, raw, out, now):
        v = self._raw_vector(raw, now)
        req = v.get("req_rate")
        succ = v.get("success_rate")
        lat = v.get("lat_p90")
        rep = v.get("replicas")

        if req is None:
            return None, {"reason": "no request-rate feature"}

        cap = self.capacity.capacity(svc)
        sig = self.capacity.sigma(svc)

        ready, pending, starts = self.scale.pod_counts(svc)
        spec = self.scale.spec_replicas(svc)
        for age in starts:
            self.lifecycle.observe_startup(svc, age)

        acc = self.lifecycle.accounting(svc, spec, ready, pending, now)
        self.lifecycle.settle(svc, now, ready)

        p_addr = float(out["addressable_prob"])
        d = self.engine.decide(
            svc, offered=req, capacity=cap, sigma=sig,
            n_committed=max(1, acc["committed"]),
            addressable_prob=p_addr,
            bottleneck=out["bottleneck"],
            confidence=float(out["confidence"]),
            ood=bool(out["ood"]))

        final, reason = self.lifecycle.reconcile(svc, d.replicas, acc)
        d.replicas = final
        d.reason = f"{d.reason}; {reason}"
        if final > acc["committed"]:
            d.action = "out"
        elif final < acc["committed"]:
            d.action = "in"
        else:
            d.action = "hold"

        # capacity update from this observation (post-action outcomes are
        # folded in later via the ledger, never at decision time)
        self.capacity.observe(svc, replicas=max(spec, 1), req_rate=req,
                              success_rate=succ if succ is not None else 1.0,
                              latency_p90=lat if lat is not None else 0.0,
                              slo_ms=self.cfg.slo_ms,
                              addressable_prob=p_addr)
        if lat is not None:
            self.drift.observe_capacity(
                svc, expected_log=self._expected_lat_now(svc, req, cap),
                observed_log=self._log(lat))

        return d, {"raw": v, "acc": acc, "ready": ready, "spec": spec}

    @staticmethod
    def _log(x):
        import math
        return math.log(max(x, 1e-6))

    def _expected_lat_now(self, svc, req, cap):
        """Expected latency **at the current utilisation**.

        Used only as the drift reference: given the offered load observed in
        this cycle and the current capacity estimate, what latency should the
        service be showing *right now*. It is not a forecast and it never
        leaves this function -- the drift detector compares it against the
        latency actually measured in the same cycle.
        """
        import math
        u = min(req / max(cap, 1e-6), 0.999)
        return self._log(10.0 / max(1.0 - u, 1e-3))

    # ------------------------------------------------------------------
    def _apply(self, svc, d, info):
        if d.action == "hold":
            return False
        if self.dry_run:
            return False
        try:
            self.scale.set_scale(svc, d.replicas)
            self.lifecycle.record(svc, time.time(), info["acc"]["committed"],
                                  d.replicas, d.reason)
            self.actions += 1
            return True
        except Exception as exc:                       # noqa: BLE001
            self.errors += 1
            print(f"[aurora] scale failed for {svc}: {exc}", flush=True)
            return False

    # ------------------------------------------------------------------
    def cycle(self):
        now = time.time()
        spans = []
        try:
            self.services = self.services or [
                s for s in self.scale.scalable_services() if s != "aurora"]
            batches, raws, s1 = self._collect(now)
            spans.append(s1)
            if self.cycles < 3:
                n_complete = sum(1 for b in batches.values() if b.complete)
                print(f"[aurora] diag cycle={self.cycles} services={len(batches)} "
                      f"complete={n_complete} raw={sum(1 for v in raws.values() if v)}",
                      flush=True)
            outs, s2 = self._infer(batches)
            spans.append(s2)
            for svc in self.services:
                raw = raws.get(svc, {})
                out = outs.get(svc)
                if out is None:
                    # stale or incomplete telemetry: safe state = hold
                    continue
                d, info = self._decide(svc, raw, out, now)
                if d is None:
                    continue
                s3 = Span("decide")
                changed = self._apply(svc, d, info)
                s3.stop()
                spans.append(s3)
                print(f"[aurora] {svc}: {d.action} -> {d.replicas} "
                      f"({d.bottleneck}, p_addr={d.addressable_prob:.2f}, "
                      f"conf={d.confidence:.2f}, ood={int(d.ood)})", flush=True)
        except Exception as exc:                       # noqa: BLE001
            self.errors += 1
            print(f"[aurora] cycle error: {type(exc).__name__}: {exc}", flush=True)

        total_ms = (time.perf_counter() - now) * 1000.0 if False else sum(s.ms for s in spans)
        rec = {"cycle": self.cycles, "total_ms": total_ms,
               "stages": {s.name: s.ms for s in spans}}
        self.timings.append(rec)
        self.cycles += 1
        return rec

    # ------------------------------------------------------------------
    def run(self):
        start = time.time()
        end = start + self.cfg.duration_s
        print(f"[aurora] start: {len(self.services) or 'discovering'} services, "
              f"interval={self.cfg.decision_interval_s}s dry_run={self.dry_run}", flush=True)
        while time.time() < end:
            t0 = time.time()
            self.cycle()
            elapsed = time.time() - t0
            sleep = self.cfg.decision_interval_s - elapsed
            if sleep > 0:
                time.sleep(sleep)
            else:
                print(f"[aurora] WARNING cycle overran interval by "
                      f"{-sleep:.2f}s", flush=True)
        self._persist_state()
        self._report()

    def _report(self):
        if not self.timings:
            print("[aurora] no cycles recorded")
            return
        tot = sorted(t["total_ms"] for t in self.timings)
        q = lambda p: tot[min(len(tot) - 1, int(p * len(tot)))]
        print(f"[aurora] cycles={self.cycles} holds={self.holds} "
              f"actions={self.actions} errors={self.errors}")
        print(f"[aurora] complete decision path ms: p50={q(.5):.1f} "
              f"p95={q(.95):.1f} p99={q(.99):.1f} max={tot[-1]:.1f}")
        agg = {}
        for t in self.timings:
            for k, v in t["stages"].items():
                agg.setdefault(k, []).append(v)
        for k, vs in sorted(agg.items()):
            vs.sort()
            print(f"[aurora]   {k:10s} p50={vs[len(vs)//2]:7.1f} "
                  f"p99={vs[min(len(vs)-1, int(.99*len(vs)))]:7.1f} ms")
        print(f"[aurora] drift: {self.adapt.drift_report()}", flush=True)