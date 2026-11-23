"""Replica decision engine.

The model says what is happening; this decides what to do.

No SLO value enters here. The SLO is an evaluation metric, not a control input:
it is blind to outages, and a controller that optimises it directly will happily
shed load to protect a percentile. Instead the engine uses the service's
measured state:

  * the binding capacity across CPU / memory / latency / throttling / queue
  * how strongly the capacity signals agree (``spread``)
  * the completion ratio -- requests that actually finished
  * the blindness ratio -- how much of the latency signal is missing
  * the bottleneck identity and whether horizontal scaling can address it
  * the lifecycle state (committed replicas)
  * operational bounds

Each candidate replica count is scored by expected utility:

    U(N) =  w_perf    * P(binding*N >= offered_now)
          - w_cost    * N / max_replicas
          - w_risk    * sigma_binding
          - w_spread  * (spread - 1)
          - w_change  * |N - N_committed| / max_replicas
          + w_complete* completion_ratio

A hold band prevents oscillation when two counts are nearly equal.
"""

import math

EPS = 1e-9


class Decision:
    def __init__(self, replicas, action, reason, utility, basis,
                 addressable_prob, bottleneck, confidence, ood,
                 bottleneck_signal=None, completion_ratio=None,
                 blindness_ratio=None):
        self.replicas = replicas
        self.action = action          # hold | out | in
        self.reason = reason
        self.utility = utility
        self.basis = basis
        self.addressable_prob = addressable_prob
        self.bottleneck = bottleneck
        self.confidence = confidence
        self.ood = ood
        self.bottleneck_signal = bottleneck_signal
        self.completion_ratio = completion_ratio
        self.blindness_ratio = blindness_ratio

    def __repr__(self):
        return (f"Decision(N={self.replicas} {self.action}, u={self.utility:.4f}, "
                f"{self.reason})")


# Structural prior used only when the learned head is out of distribution or
# unsure. It encodes which bottlenecks more replicas can plausibly relieve. It
# never overrides a confident model -- it fills in when the model says "I do not
# know", which is exactly what the OOD output is for.
STRUCTURAL_PRIOR = {
    "cpu": 1.0,
    "memory": 0.8,
    "throttle": 1.0,
    "queue": 1.0,
    "latency": 0.5,
}


class DecisionEngine:
    def __init__(self, cfg, w_perf=1.0, w_cost=0.35, w_risk=0.50,
                 w_change=0.4, w_spread=0.3, w_complete=0.2,
                 hold_band=0.02, completion_floor=0.5):
        self.cfg = cfg
        self.w_perf = w_perf
        self.w_cost = w_cost
        self.w_risk = w_risk
        self.w_change = w_change
        self.w_spread = w_spread
        self.w_complete = w_complete
        self.hold_band = hold_band
        self.completion_floor = completion_floor

    # ------------------------------------------------------------------
    @staticmethod
    def _p_serves_current(capacity, sigma, n, offered):
        """P(N * capacity >= offered) for lognormal capacity.

        ``offered`` is the workload observed in the current cycle. This is a
        statement about whether ``n`` replicas can serve the load *right now*;
        no future demand or future resource value enters the calculation.
        """
        if n <= 0:
            return 0.0
        if capacity <= EPS:
            return 0.0
        mu = math.log(n) + math.log(capacity)
        if offered <= 0:
            return 1.0
        z = (mu - math.log(offered)) / max(sigma, EPS)
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # ------------------------------------------------------------------
    def decide(self, service, offered, binding, n_committed,
               addressable_prob, bottleneck, confidence, ood,
               completion_ratio=1.0, blindness_ratio=0.0,
               min_pods=None, max_pods=None):
        min_pods = self.cfg.min_pod if min_pods is None else min_pods
        max_pods = self.cfg.max_pod if max_pods is None else max_pods
        n_committed = max(min_pods, n_committed)

        # No usable capacity evidence -> hold. Treating that as zero capacity
        # produced an infinite requirement and scaled to maxReplicas.
        if not binding.get("valid", True):
            return Decision(n_committed, "hold",
                            "no usable capacity signal", 0.0, {},
                            addressable_prob, bottleneck, confidence, ood,
                            bottleneck_signal=None,
                            completion_ratio=completion_ratio,
                            blindness_ratio=blindness_ratio)

        cap = binding.get("capacity", 0.0)
        sigma = binding.get("sigma", 1.0)
        spread = binding.get("spread", 1.0)
        signal = binding.get("signal")
        p_addr = max(0.0, min(1.0, addressable_prob))
        # If the learned head is out of distribution or unsure, fall back to the
        # structural prior for the binding signal instead of letting a guess
        # veto a clear capacity signal.
        if (ood or confidence < 0.5) and signal in STRUCTURAL_PRIOR:
            p_addr = max(p_addr, STRUCTURAL_PRIOR[signal])
        # Dependency-bound guard, applied AFTER the OOD fallback so an
        # unfamiliar observation cannot re-enable scaling for a service that is
        # waiting on a downstream dependency. A service whose latency is high
        # while its own CPU is NOT under pressure is blocked on something else;
        # adding replicas only adds concurrent callers to the same dependency.
        # Measured 2026-11-22: AURORA gave ts-preserve 9.6 replicas (HPA 1.6)
        # and its p99 went 3.4x WORSE; same for ts-admin-order and ts-security.
        clp = binding.get("cpu_limit_pct")
        if signal == "latency" and clp is not None and clp < 0.35:
            p_addr = min(p_addr, 0.15)
        completion = max(0.0, min(1.0, completion_ratio))

        # Blindness inflates uncertainty on the latency-derived capacity: we
        # do not know how slow the service really is, so we must not act boldly.
        sigma_eff = sigma * (1.0 + max(0.0, blindness_ratio))

        # A service that is dropping requests is not helped by replicas unless
        # the binding signal is one that more replicas can relieve.
        scaling_indicated = p_addr
        if completion < self.completion_floor and signal in ("queue",):
            scaling_indicated *= 0.5

        # Effective capacity after the uncertainty haircut. Using a monotone
        # served-fraction term (rather than a probability) matters: when no N
        # can serve the load, P(...) is ~0 for every candidate and the cost term
        # would make the SMALLEST N win -- the engine would scale a saturated
          # service DOWN (measured: the front scaled to 1 with cap 0.08 and
        # 122 req/s offered).
        cap_eff = cap * math.exp(-self.w_risk * sigma_eff)
        best, scores = None, {}
        for n in range(min_pods, max_pods + 1):
            p_now = self._p_serves_current(cap, sigma_eff, n, offered)
            served = 1.0 if offered <= 0 else min(1.0, (n * cap_eff) / offered)
            util = (self.w_perf * served
                    - self.w_cost * (n / max(max_pods, 1))
                    - self.w_risk * sigma_eff
                    - self.w_spread * max(0.0, spread - 1.0)
                    - self.w_change * abs(n - n_committed) / max(max_pods, 1)
                    + self.w_complete * completion)
            util -= (1.0 - scaling_indicated) * 0.3 * (n / max(max_pods, 1))
            scores[n] = util
            if best is None or util > best[1]:
                best = (n, util)

        target, best_util = best

        # Best-effort floor: when the requirement exceeds the ceiling, no N can
        # serve the load, the served-fraction differences are negligible and the
        # cost term would otherwise pick the smallest N. Saturate at the bound
        # instead of scaling a saturated service down.
        required = (offered / cap_eff) if cap_eff > EPS else float("inf")
        # Only saturate at the ceiling when scaling is actually indicated: if
        # the model confidently says more replicas will not help, going to max
        # is pure waste (and it is what pinned 35/42 deployments at 10).
        floored = (required > max_pods and scaling_indicated >= 0.5
                   and binding.get("valid", True))
        if floored:
            target, best_util = max_pods, scores.get(max_pods, best_util)

        # The hold band must not undo a best-effort floor: scores[n_committed]
        # is higher than the floored score by construction, and reverting to it
        # would scale a saturated service back down (measured: the front stayed
        # at 1 replica with 122 req/s offered).
        if not floored:
            cur = scores.get(n_committed, best_util - 1.0)
            if cur >= best_util - self.hold_band:
                target, best_util = n_committed, cur

        if target > n_committed:
            action, reason = "out", f"N {n_committed}->{target} (u={best_util:.3f}, {signal})"
        elif target < n_committed:
            action, reason = "in", f"N {n_committed}->{target} (u={best_util:.3f}, {signal})"
        else:
            action, reason = "hold", f"N={n_committed} (u={best_util:.3f}, {signal})"

        basis = {
            "capacity": cap, "sigma": sigma, "spread": spread,
            "signal": signal, "signals": binding.get("signals", {}),
            "p_serves_current": self._p_serves_current(cap, sigma_eff, target, offered),
            "scores": scores,
        }
        return Decision(target, action, reason, best_util, basis,
                        p_addr, bottleneck, confidence, ood,
                        bottleneck_signal=signal,
                        completion_ratio=completion,
                        blindness_ratio=blindness_ratio)