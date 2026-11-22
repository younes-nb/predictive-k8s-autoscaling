"""Replica decision engine.

Separate from the neural model. The model says what is happening; this decides
what to do about it.

Each candidate replica count is scored by expected utility:

    U(N) =  w_perf   * P(capacity*N >= offered)
          - w_cost   * N / max_replicas
          - w_risk   * sigma_capacity
          - w_change * |N - N_committed| / max_replicas

The probability term uses the capacity posterior, so uncertainty automatically
makes the engine prefer larger N (and makes it conservative about scaling in).
A hold band prevents oscillation when two candidate counts are within a small
utility difference.
"""

import math

EPS = 1e-9


class Decision:
    def __init__(self, replicas, action, reason, utility, basis,
                 addressable_prob, bottleneck, confidence, ood):
        self.replicas = replicas
        self.action = action          # hold | out | in
        self.reason = reason
        self.utility = utility
        self.basis = basis
        self.addressable_prob = addressable_prob
        self.bottleneck = bottleneck
        self.confidence = confidence
        self.ood = ood

    def __repr__(self):
        return (f"Decision(N={self.replicas} {self.action}, "
                f"u={self.utility:.4f}, {self.reason})")


class DecisionEngine:
    def __init__(self, cfg, w_perf=1.0, w_cost=0.35, w_risk=0.5,
                 w_change=0.4, hold_band=0.02):
        self.cfg = cfg
        self.w_perf = w_perf
        self.w_cost = w_cost
        self.w_risk = w_risk
        self.w_change = w_change
        self.hold_band = hold_band

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
        # log(N*capacity) ~ Normal(log N + mu, sigma^2)
        mu = math.log(n) + math.log(capacity)
        if offered <= 0:
            return 1.0
        z = (mu - math.log(offered)) / max(sigma, EPS)
        # normal CDF via erf
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # ------------------------------------------------------------------
    def decide(self, service, offered, capacity, sigma, n_committed,
               addressable_prob, bottleneck, confidence, ood,
               min_pods=None, max_pods=None):
        min_pods = self.cfg.min_pod if min_pods is None else min_pods
        max_pods = self.cfg.max_pod if max_pods is None else max_pods
        n_committed = max(min_pods, n_committed)

        # A service whose bottleneck is not horizontally addressable is not
        # scaled for it; the model's addressability output has the final say,
        # the class only sets the prior.
        p_addr = max(0.0, min(1.0, addressable_prob))
        effective_offered = offered

        best = None
        scores = {}
        for n in range(min_pods, max_pods + 1):
            p_now = self._p_serves_current(capacity, sigma, n, effective_offered)
            # uncertainty makes the engine pay for headroom
            util = (self.w_perf * p_now
                    - self.w_cost * (n / max(max_pods, 1))
                    - self.w_risk * sigma
                    - self.w_change * abs(n - n_committed) / max(max_pods, 1))
            # do not chase capacity for a non-addressable bottleneck
            util -= (1.0 - p_addr) * 0.3 * (n / max(max_pods, 1))
            scores[n] = util
            if best is None or util > best[1]:
                best = (n, util)

        target, best_util = best
        # hold band: keep the current count if it is nearly as good
        cur = scores.get(n_committed, best_util - 1.0)
        if cur >= best_util - self.hold_band:
            target = n_committed
            best_util = cur

        if target > n_committed:
            action, reason = "out", f"N {n_committed}->{target} (u={best_util:.3f})"
        elif target < n_committed:
            action, reason = "in", f"N {n_committed}->{target} (u={best_util:.3f})"
        else:
            action, reason = "hold", f"N={n_committed} (u={best_util:.3f})"

        basis = {
            "capacity": capacity, "sigma": sigma,
            "p_serves_current": self._p_serves_current(
                capacity, sigma, target, effective_offered),
            "scores": scores,
        }
        return Decision(target, action, reason, best_util, basis,
                        p_addr, bottleneck, confidence, ood)