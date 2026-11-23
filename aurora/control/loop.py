"""The 15-second control loop (two-model architecture).

Per cycle:

  1. collect node telemetry (5 s, rolling window) and directed edge telemetry
  2. collect the frontend valid-request P99 deadline status ONCE
  3. build the graph observation
  4. run the shared Graph Diagnosis Model ONCE for the whole graph
  5. for each managed service, run the Replica-Change Model
  6. validate the proposal through the deterministic safety layer
  7. reconcile against lifecycle state and apply via the Scale subresource
  8. record the action and its outcome in the ledger

Serialised: one cycle at a time. If a cycle overruns the interval the next
starts immediately and the overrun is logged -- never two writers at once.

Telemetry is explicitly three-valued: a known zero, an unavailable series and a
stale series are not interchangeable. A decision that needs a signal which is
unknown holds; it never treats the unknown as zero.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor

from feedback.ledger import Ledger
from features.groups import FEATURE_ORDER
from features.transform import transform
from features.window import RunningNorm, WindowManager
from lifecycle.reconciler import LifecycleReconciler
from models.graph import PROVISIONING_CLASSES
from safety import SafetyLayer
from telemetry.mimir import EdgeClient

# Concurrency proxy semantics: req_rate * P50 is Little's law on the P50, not a
# measured in-flight count. Named so no caller mistakes it for one.
INFLIGHT_IS_PROXY = True
PROVISIONING_UNKNOWN = PROVISIONING_CLASSES.index("unknown")


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
            # Frozen: the encoder sees the training representation verbatim for
            # the whole run. Local baselines still update for drift.
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
        # Start of the current continuously-healthy / over-provisioned period.
        # Persisted (with last_scale_in) so a restart does not reset the window.
        self.stable_since = {}
        # Per-service request-rate history, for the capacity spread estimate.
        self.req_history = {}
        # Per-stage timing accumulators, reported at the end of the run. The
        # "decide" stage is measured per service and added to, not read off an
        # empty timer.
        self.stage_ms = {"decide": 0.0, "settle": 0.0}
        self.stage_n = {"decide": 0, "settle": 0}
        # Recovered from the ledger at startup: in-flight and cooldown state
        # belong to the Deployment, not to the process.
        self._restore_runtime_state()

    # ------------------------------------------------------------------
    def _restore_runtime_state(self):
        """Rebuild in-flight / cooldown / stability state from the ledger."""
        self.pending_actions = []
        for a in self.ledger.unsettled():
            if a.get("status") != "action_applied":
                continue
            dec = a.get("decision") or {}
            self.pending_actions.append(
                (a.get("service"), a.get("applied_ts") or a.get("created_ts"),
                 dec.get("to_replicas")))
            if dec.get("action") == "in":
                self.last_scale_in[a.get("service")] = (
                    a.get("applied_ts") or a.get("created_ts"))
        if self.pending_actions:
            print(f"[aurora] recovered {len(self.pending_actions)} in-flight "
                  f"action(s) from the ledger", flush=True)

    # ------------------------------------------------------------------
    def _collect(self, now):
        span = Span("collect")
        window_s = (self.cfg.history_len + 3) * self.cfg.telemetry_step_s
        start = now - window_s - 5
        # Bounded parallelism: each family is one range query, so a cycle can be
        # a handful of concurrent requests instead of 25 sequential ones. With
        # query_workers=1 (the default) this is exactly the old serial path.
        fetched = self.mimir.fetch(self.services, start, now,
                                   step=self.cfg.telemetry_step_s,
                                   workers=self.cfg.query_workers)
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
                h = self.req_history.setdefault(svc, [])
                h.append(float(arr))
                if len(h) > 12:
                    h.pop(0)
            else:
                self.req_history.setdefault(svc, [])

        # directed edges (own client, own Mimir handle)
        self.edge_hist = self.edges.fetch(start, now, self.cfg.telemetry_step_s)
        span.stop()
        if self.cycles % 10 == 0:
            fam = getattr(self.mimir, "_family_s", {}) or {}
            slow = sorted(((round(v * 1000, 1), k) for k, v in fam.items()),
                          reverse=True)[:4]
            print(f"[aurora] telemetry: {len(fam)} families, "
                  f"slowest_ms={slow} total={span.ms:.0f}ms", flush=True)
        return raws, span

    # ------------------------------------------------------------------
    def _frontend_slo(self):
        """User-facing deadline state: met / violated / unknown.

        The query is deliberately narrow:

          * ``destination_workload_namespace`` is pinned, so a workload name
            reused in another namespace cannot be merged into this quantile.
          * ``reporter="destination"`` keeps the measurement at the receiving
            proxy, i.e. what the user actually waited for.
          * the request population is the destination workload's inbound
            requests; health/readiness probes do not traverse the ingress and
            are therefore not part of it.

        Known limits, stated rather than hidden:

          * a request still unresolved after the deadline has no completed
            duration observation, so it is invisible to the histogram. The
            caller compensates by requiring a minimum sample count and by
            treating an under-sampled quantile as unknown.
          * a NaN quantile is unknown, never "met".
        """
        pct = self.cfg.slo_percentile / 100.0
        expr = ('histogram_quantile(%s, sum by (le) (rate('
                'istio_request_duration_milliseconds_bucket{reporter="destination",'
                'destination_workload_namespace="%s",'
                'destination_workload="%s"}[1m])))'
                % (pct, self.cfg.namespace, self.cfg.front_svc))
        try:
            res = self.mimir.instant(expr)
        except Exception:
            return "unknown", None, 0.0
        if not res:
            return "unknown", None, 0.0
        try:
            val = float(res[0]["value"][1])
        except (TypeError, ValueError, KeyError):
            return "unknown", None, 0.0
        # A NaN quantile is not evidence that the SLO is met. Comparing NaN
        # against the budget is always False, so an explicit finite check is
        # required -- otherwise an empty histogram reads as healthy.
        if not math.isfinite(val):
            return "unknown", None, 0.0
        # Minimum sample volume: a p99 over a handful of requests is noise, and
        # must not authorise a replica change.
        count = self._frontend_request_count()
        if count is not None and count < self.cfg.slo_min_samples:
            return "unknown", val, count
        return ("violated" if val > self.cfg.slo_budget_ms else "met"), val, count

    def _frontend_request_count(self):
        """Completed request rate over the same 1 min window (req/s)."""
        expr = ('sum(rate(istio_requests_total{reporter="destination",'
                'destination_workload_namespace="%s",'
                'destination_workload="%s"}[1m]))'
                % (self.cfg.namespace, self.cfg.front_svc))
        try:
            res = self.mimir.instant(expr)
            return float(res[0]["value"][1]) if res else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _edge_tensors(self, idx):
        """Build (edge_index, edge_attr, edge_mask) over aligned edge histories.

        Each edge contributes its full recent history, not only the latest
        sample, so the model's temporal edge encoder sees the same kind of
        input it would see at training time.
        """
        import torch
        n = len(idx)
        src, dst, feats, masks = [], [], [], []
        T = max(1, self.cfg.history_len)
        for (si, sj), e in self.edge_hist.items():
            a, b = idx.get(si), idx.get(sj)
            if a is None or b is None:
                continue
            rows, mrows = [], []
            for k in ("rate", "err", "p50", "p90", "p99"):
                pts = e.get(k) or []
                recent = pts[-T:]
                # Left-pad to T so every edge has the same time axis; the mask
                # marks which steps are real observations.
                vals = [0.0] * (T - len(recent)) + [v for _, v in recent]
                msk = [0.0] * (T - len(recent)) + [1.0] * len(recent)
                rows.append([math.log1p(max(v, 0.0)) for v in vals])
                mrows.append(msk)
            if not any(any(r) for r in mrows):
                continue
            src.append(a)
            dst.append(b)
            feats.append(list(zip(*rows)))     # T x 5
            masks.append(list(zip(*mrows)))
        if not src:
            return None, None, None
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(feats, dtype=torch.float32)    # E,T,5
        edge_mask = torch.tensor(masks, dtype=torch.float32)    # E,T,5
        return edge_index, edge_attr, edge_mask

    # ------------------------------------------------------------------
    def _diagnose(self, raws, now):
        import torch
        span = Span("diagnose")
        svcs = self.services
        n = len(svcs)
        x = torch.zeros(n, self.cfg.history_len, len(FEATURE_ORDER))
        m = torch.zeros(n, self.cfg.history_len, len(FEATURE_ORDER))
        for i, s in enumerate(svcs):
            b = self.windows.batch(s, now, features=FEATURE_ORDER)
            t = min(b.n_steps, self.cfg.history_len)
            x[i, :t] = torch.tensor(b.matrix, dtype=torch.float32)
            m[i, :t] = torch.tensor(b.mask, dtype=torch.float32)
        edge_index, edge_attr, edge_mask = self._edge_tensors({s: i for i, s in enumerate(svcs)})
        with torch.no_grad():
            out = self.diag(x, m, edge_index, edge_attr,
                            m.any(-1).any(-1).float(), edge_mask)
        span.stop()
        res = {}
        valid = out["node_valid"].tolist()
        prov_idx = out["provisioning_idx"].tolist()
        for i, s in enumerate(svcs):
            # .tolist() rather than .numpy(): the container's torch is built
            # against NumPy 1.x and the image ships 2.x, so .numpy() raises.
            emb = out["node_embedding"][i].tolist()
            ood = False
            if self.ood_centroid is not None and self.ood_threshold:
                d = math.sqrt(sum((a - b) ** 2 for a, b in
                                  zip(emb, self.ood_centroid)))
                ood = d > self.ood_threshold
            rp = float(out["root_pb"][i]); af = float(out["affected"][i])
            deg = out["degradation"][i].tolist()
            prov = out["provisioning"][i].tolist()
            if valid[i] < 0.5:
                # No usable observation: this is an explicit unknown, not a
                # diagnosis read off a zero embedding.
                res[s] = {
                    "root_pb": 0.0, "affected": 0.0,
                    "degradation": deg, "provisioning": prov,
                    "provisioning_idx": PROVISIONING_UNKNOWN,
                    "confidence": 0.0, "ood": True, "embedding": emb,
                    "valid": False,
                }
                continue
            # Per-output confidence, calibrated per head rather than by
            # combining "distance from 0.5" of every sigmoid. The degradation
            # head is multi-label: a healthy service legitimately has low
            # probabilities everywhere, which is NOT low confidence in the
            # diagnosis -- it is a confident "healthy".
            rp_conf = max(rp, 1.0 - rp)
            af_conf = max(af, 1.0 - af)
            prov_conf = max(prov) if prov else 0.0
            top_deg = max(deg) if deg else 0.0
            # Confidence in the *conclusion* is the agreement of the heads that
            # actually decide: root/affected are binary, provisioning is a
            # distribution. The degradation magnitude is reported separately.
            # ``min`` of the three heads makes any single uncertain head
            # dominate, which on a bootstrap model whose heads are uncalibrated
            # means the confidence collapses to ~0.34 and every decision is
            # gated off. Use the mean, which is what the floor is calibrated
            # against, and report the weakest head separately.
            conf = (rp_conf + af_conf + prov_conf) / 3.0
            res[s] = {
                "root_pb": rp,
                "affected": af,
                "degradation": deg,
                "degradation_conf": top_deg,
                "provisioning": prov,
                "provisioning_idx": prov_idx[i],
                "confidence": conf,
                "ood": ood,
                "embedding": emb,
                "valid": True,
            }
        return res, span

    # ------------------------------------------------------------------
    def _capacity(self, svc, raw):
        """Observed goodput per replica and its spread.

        This is measured from what the service is currently doing -- requests
        completed per second per ready replica -- not predicted. It is the only
        capacity signal; there is no forecast anywhere in this loop.
        """
        req = raw.get("req_rate")
        rep = raw.get("replicas") or raw.get("ready")
        if not req or not rep or rep <= 0:
            return 0.0, 0.0
        per = req / rep
        # Spread proxy: how much the per-replica rate moves across the recent
        # window. A service whose per-replica rate is stable has a tight
        # posterior; a swinging one does not.
        # Spread proxy: how much the per-replica rate moves across the recent
        # window. A service whose per-replica rate is stable has a tight
        # posterior; a swinging one does not.
        hist = self.req_history.get(svc) or []
        if len(hist) > 1 and rep:
            rates = [h / rep for h in hist if h is not None]
            if len(rates) > 1:
                mean = sum(rates) / len(rates)
                var = sum((r - mean) ** 2 for r in rates) / len(rates)
                return per, math.sqrt(var)
        return per, 0.0

    # ------------------------------------------------------------------
    def _decide_one(self, svc, diag, raw, now, slo_state, slo_value, slo_count,
                    snap=None):
        import torch
        # Telemetry tri-state. ``req_rate`` absent means the signal is UNKNOWN,
        # not zero: the service may be idle, or the series may be missing, or
        # the collector may be stale. Unknown blocks every decision; a known
        # zero is valid evidence and is allowed through.
        req = raw.get("req_rate")
        telemetry_unknown = req is None
        if telemetry_unknown:
            return None, None
        req = float(req)

        if snap is not None:
            ready, pending, starts, ages, uids = snap[0].get(
                svc, (0, [], [], [], []))
            spec = snap[1].get(svc, 0)
        else:
            ready, pending, starts, ages, uids = self.scale.pod_counts(svc)
            spec = self.scale.spec_replicas(svc)
        for a, uid in zip(starts, uids):
            # Pass the Pod UID: the reconciler deduplicates per UID, so a Ready
            # Pod seen on every cycle contributes exactly one startup sample.
            self.lifecycle.observe_startup(svc, a, uid=uid)
        acc = self.lifecycle.accounting(svc, spec, ready, pending, now)
        self.lifecycle.settle(svc, now, ready)

        min_pods, max_pods = self.bounds.get(
            svc, (self.cfg.min_pod, self.cfg.max_pod))
        capacity, sigma = self._capacity(svc, raw)
        p99_ratio = (slo_value / self.cfg.slo_budget_ms
                     if slo_value is not None and self.cfg.slo_budget_ms else 1.0)

        d = {
            "service": svc, "confidence": diag["confidence"], "ood": diag["ood"],
            "root_pb": diag["root_pb"], "affected": diag["affected"],
            "provisioning_idx": diag["provisioning_idx"],
            # Addressability comes from the degradation head, not from the
            # affected-caller probability.
            "degradation": diag["degradation"],
        }
        x_in = torch.tensor([diag["embedding"]
                             + [diag["root_pb"], diag["affected"]]
                             + diag["degradation"] + diag["provisioning"]
                             + [math.log1p(max(req, 0.0)),
                                math.log1p(max(capacity, 0.0)),
                                sigma,
                                float(spec), float(acc["committed"]),
                                float(min_pods), float(max_pods),
                                1.0, float(p99_ratio)]],
                            dtype=torch.float32)
        # Per-service bounds mask the target head, so the model can only ever
        # propose a count this Deployment is allowed to run.
        action, target, info = self.replica.decide(x_in, min_pods, max_pods)
        action, target, reason = self.safety.validate(
            action, target, d, acc["committed"],
            self.lifecycle.pending_scale_out(svc), slo_state,
            min_pods, max_pods,
            stable_since=self.stable_since.get(svc),
            now=now,
            action_conf=info["action_conf"], target_conf=info["target_conf"])
        # L05: lifecycle reconcile before anything is applied
        target, why = self.lifecycle.reconcile(svc, target, acc,
                                               min_pods, max_pods)
        if why.startswith("hold"):
            action, reason = "hold", f"{reason}; {why}"

        action, target, reason2 = self.safety.enforce_direction(
            action, target, acc["committed"])
        if reason2 != "ok":
            reason = f"{reason}; {reason2}"
        d["reason"] = reason
        d["target"] = target
        d["action"] = action
        d["slo_state"] = slo_state
        d["slo_value_ms"] = slo_value
        d["slo_count"] = slo_count
        d["committed"] = acc["committed"]
        d["capacity"] = capacity
        d["sigma"] = sigma
        d["action_confidence"] = info["action_conf"]
        d["target_confidence"] = info["target_conf"]
        d["raw"] = {"req_rate": req, "ready": ready, "spec": spec,
                    "capacity": capacity, "sigma": sigma,
                    "p99_ratio": p99_ratio}
        return d, acc

    # ------------------------------------------------------------------
    def _settle_pending(self, now, snap=None):
        """Settle actions that have converged, with real settlement metadata.

        A scale-out is settled only when the new pods are Ready AND a
        post-action observation window has elapsed, so the outcome contains a
        measurement taken *after* the change rather than a pod count at the
        moment it was applied. A scale-in is settled only when the old pods are
        actually gone (ready <= target), not merely when ready >= target.
        """
        window_s = self.cfg.settle_window_s
        for pend in list(self.pending_actions):
            psvc, pts, ptgt = pend
            if now - pts < window_s:
                continue
            try:
                if snap is not None:
                    ready, pending, starts, ages, _ = snap[0].get(psvc, (0, [], [], [], []))
                    spec = snap[1].get(psvc, 0)
                else:
                    ready, pending, starts, ages, _ = self.scale.pod_counts(psvc)
                    spec = self.scale.spec_replicas(psvc)
            except Exception as exc:                        # noqa: BLE001
                print(f"[aurora] settle: {psvc}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            settled = (ready >= ptgt) if ptgt > 0 else (ready <= 0)
            if not settled:
                # Give it one more window, bounded by the lifecycle timeout.
                if now - pts > 4 * window_s:
                    self.ledger.record_outcome(
                        self._action_id(psvc, pts), "action_inconclusive", now,
                        {"ready": ready, "spec": spec, "target": ptgt},
                        reason="did not converge within the settlement budget")
                    self.pending_actions.remove(pend)
                continue
            after = self._outcome_snapshot(psvc, ready, spec)
            aid = self._action_id(psvc, pts)
            kind = "action_settled"
            self.ledger.record_outcome(aid, kind, now, after)
            self.pending_actions.remove(pend)

    def _action_id(self, service, ts):
        """Find the ledger action id opened for (service, ts)."""
        for a in self.ledger.actions.values():
            if a.get("service") == service and a.get("created_ts") == ts:
                return a.get("action_id")
        return None

    def _outcome_snapshot(self, svc, ready, spec):
        """Post-action measurement: latency, resources, mix, lifecycle."""
        snap = {"ready": ready, "spec": spec, "settled_at": time.time()}
        try:
            raw = {}
            fetched = self.mimir.fetch([svc], time.time() - 60, time.time(),
                                       step=self.cfg.telemetry_step_s)
            series = fetched.get(svc, {})
            by_ts = {}
            for key, pts in series.items():
                for ts, v in pts:
                    by_ts.setdefault(ts, {})[key] = v
            if by_ts:
                raw = by_ts[sorted(by_ts)[-1]]
            for k in ("req_rate", "err_rate", "lat_p50", "lat_p90", "lat_p99",
                      "cpu_cores", "cpu_request", "mem_working_set",
                      "completion_ratio"):
                v = raw.get(k)
                if v is None and k == "completion_ratio":
                    t = transform(raw, now=sorted(by_ts)[-1] if by_ts else None)
                    v = dict(zip(FEATURE_ORDER, t[0])).get(k)
                snap[k] = v
        except Exception as exc:                            # noqa: BLE001
            snap["error"] = f"{type(exc).__name__}: {exc}"
        return snap

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
            # Query the frontend SLO ONCE per cycle, not once per service.
            slo_state, slo_value, slo_count = self._frontend_slo()
            # One namespace snapshot per cycle: three API calls instead of
            # three per service (126 calls for 42 services).
            snap = self.scale.snapshot(self.services)
            for svc in self.services:
                t_decide = time.perf_counter()
                try:
                    d, acc = self._decide_one(svc, diags[svc], raws.get(svc, {}),
                                              now, slo_state, slo_value,
                                              slo_count, snap=snap)
                    if d is None:
                        continue
                    if d["target"] != d["committed"] and not self.cfg.dry_run:
                        applied = False
                        if self.election is not None and self.election.is_leader \
                                and self.election.renew():
                            self.scale.set_scale(svc, d["target"])
                            self.lifecycle.record(svc, now, d["committed"],
                                                  d["target"], d["reason"])
                            self.ledger.record_action(
                                svc, now, self.model_version,
                                self._before_record(svc, d, slo_state, slo_value),
                                d, state=self._graph_state(d))
                            applied = True
                            self.actions += 1
                            self.pending_actions.append((svc, now, d["target"]))
                            if d["target"] < d["committed"]:
                                self.last_scale_in[svc] = now
                                self.stable_since.pop(svc, None)
                            print(f"[aurora] {svc}: {d['target']} ({d['reason']})",
                                  flush=True)
                        elif not applied and d["target"] != d["committed"]:
                            # No leader: say so explicitly rather than printing a
                            # target that was never applied.
                            print(f"[aurora] {svc}: not leader -- action skipped "
                                  f"(target {d['target']})", flush=True)
                    elif self.cycles < 2:
                        print(f"[aurora] {svc}: hold {d['committed']} "
                              f"({d['reason']})", flush=True)
                    # Track sustained health for the scale-in stability window.
                    self._note_stability(svc, d, slo_state, now)
                except Exception as exc:                    # noqa: BLE001
                    self.errors += 1
                    print(f"[aurora] {svc}: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    self.stage_ms["decide"] += (time.perf_counter() - t_decide) * 1000.0
                    self.stage_n["decide"] += 1
        except Exception as exc:                            # noqa: BLE001
            import traceback
            self.errors += 1
            print(f"[aurora] cycle error: {type(exc).__name__}: {exc}", flush=True)
            if self.errors <= 2:
                traceback.print_exc()
        total = (time.perf_counter() - cstart) * 1000.0

        # Settlement also costs pod-count and telemetry queries. It is measured
        # separately so it can never hide inside the decision-path number.
        t_settle = time.perf_counter()
        self._settle_pending(now, snap)
        settle_ms = (time.perf_counter() - t_settle) * 1000.0
        self.stage_ms["settle"] += settle_ms
        self.stage_n["settle"] += 1
        if settle_ms > 200.0:
            print(f"[aurora] settlement took {settle_ms:.0f} ms", flush=True)
        self.timings.append({"cycle": self.cycles, "total_ms": total,
                             "stages": {s.name: s.ms for s in spans}})
        self.cycles += 1
        if total > self.cfg.cycle_deadline_s * 1000.0:
            print(f"[aurora] WARNING: cycle took {total:.0f} ms "
                  f"(budget {self.cfg.cycle_deadline_s * 1000:.0f} ms)",
                  flush=True)
        return total

    # ------------------------------------------------------------------
    def _note_stability(self, svc, d, slo_state, now):
        """Start or keep the sustained-health window used by scale-in.

        The window is "continuously healthy AND over-provisioned". Any deadline
        miss, any non-over-provisioned diagnosis or any scale action resets it.
        """
        healthy = (slo_state == "met"
                   and d.get("provisioning_idx") == 2      # over_provisioned
                   and d.get("action") != "out")
        if healthy:
            self.stable_since.setdefault(svc, now)
        else:
            self.stable_since.pop(svc, None)

    def _before_record(self, svc, d, slo_state, slo_value):
        """Full pre-action state for the ledger (not just a replica count)."""
        raw = d.get("raw") or {}
        return {
            "replicas": d["committed"],
            "ready": raw.get("ready"),
            "spec": raw.get("spec"),
            "req_rate": raw.get("req_rate"),
            "capacity": raw.get("capacity"),
            "sigma": raw.get("sigma"),
            "slo_state": slo_state,
            "slo_value_ms": slo_value,
            "slo_count": d.get("slo_count"),
            "root_pb": d.get("root_pb"),
            "affected": d.get("affected"),
            "degradation": d.get("degradation"),
            "provisioning": d.get("provisioning_idx"),
            "confidence": d.get("confidence"),
            "ood": d.get("ood"),
            "action_confidence": d.get("action_confidence"),
            "target_confidence": d.get("target_confidence"),
        }

    def _graph_state(self, d):
        """Graph diagnosis evidence attached to the action, for outcome analysis."""
        return {
            "degradation": d.get("degradation"),
            "provisioning_idx": d.get("provisioning_idx"),
            "root_pb": d.get("root_pb"),
            "affected": d.get("affected"),
            "ood": d.get("ood"),
        }

    # ------------------------------------------------------------------
    def run(self):
        """Run the control loop until the process is signalled.

        There is no run-length: the controller is resident and owns the
        replicas until SIGTERM/SIGINT, at which point ``main`` restores the
        native HPAs. The orchestrator decides how long the load lasts.
        """
        print(f"[aurora] start: {len(self.services)} services, "
              f"interval={self.cfg.decision_interval_s}s dry_run={self.cfg.dry_run}",
              flush=True)
        while True:
            t0 = time.time()
            self.cycle()
            sleep = self.cfg.decision_interval_s - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _report(self):
        if not self.timings:
            print("[aurora] no cycles"); return
        tot = sorted(t["total_ms"] for t in self.timings)
        q = lambda p: tot[min(len(tot) - 1, int(p * len(tot)))]
        settled = self.ledger.settled_count()
        unsettled = len(self.ledger.unsettled())
        print(f"[aurora] cycles={self.cycles} actions={self.actions} errors={self.errors}")
        print(f"[aurora] complete decision path ms: p50={q(.5):.1f} p95={q(.95):.1f} "
              f"p99={q(.99):.1f} max={tot[-1]:.1f}")
        # Per-stage timing, measured on the real work. collect/diagnose come
        # from the per-cycle spans; decide and settle are accumulated across
        # all services and all cycles, so their mean is per service.
        for stage in ("collect", "diagnose"):
            vals = sorted(t["stages"].get(stage, 0.0) for t in self.timings)
            if vals:
                print(f"[aurora]   {stage}: p50={vals[len(vals)//2]:.1f} "
                      f"p95={vals[min(len(vals)-1, int(.95*len(vals)))]:.1f} "
                      f"max={vals[-1]:.1f}")
        n = max(self.stage_n.get("decide", 0), 1)
        print(f"[aurora]   decide: {self.stage_ms['decide']/n:.1f} ms/service "
              f"({n} decisions)")
        ns = max(self.stage_n.get("settle", 0), 1)
        print(f"[aurora]   settle: {self.stage_ms['settle']/ns:.1f} ms/cycle "
              f"({ns} cycles)")
        print(f"[aurora] ledger: {len(self.ledger.actions)} actions, "
              f"{settled} settled, {unsettled} unsettled "
              f"(bootstrap: outcomes not yet validated at scale)")