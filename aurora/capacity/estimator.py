"""Hierarchical (partial-pooled) capacity estimator.

Estimates **sustainable goodput per replica** for each service from that
service's own measured behaviour, shrunk toward a global prior when the service
has few observations.

The update is a conjugate normal update on ``log(capacity)``, which keeps the
estimate positive and makes the posterior variance available to the decision
engine -- the decision engine needs uncertainty, not just a point estimate.
"""

import math

EPS = 1e-9


class CapacityEstimator:
    def __init__(self, global_capacity=1.0, global_var=1.0, prior_strength=5.0):
        self.global_capacity = global_capacity
        self.global_var = global_var
        self.prior_strength = prior_strength
        self.mean = {}      # service -> log capacity
        self.var = {}       # service -> variance of log capacity
        self.n = {}         # service -> observation count

    # ------------------------------------------------------------------
    def observe(self, service, replicas, req_rate, success_rate, latency_p90,
                slo_ms, addressable_prob=1.0):
        """Fold one observation into the posterior.

        Only observations that look like the service was actually serving
        (``replicas>0``, ``req_rate>0``) carry information about capacity. A
        service that is slow because a dependency is slow must not teach the
        estimator that its own per-replica capacity is low, so the update is
        weighted by ``addressable_prob``.
        """
        if replicas <= 0 or req_rate <= 0:
            return
        goodput = req_rate * max(0.0, min(1.0, success_rate))
        per_replica = goodput / replicas
        if per_replica <= 0:
            return
        w = max(0.0, min(1.0, addressable_prob))
        if w <= 0:
            return
        obs = math.log(per_replica)

        # observation noise: relative, wider when the sample is less trustworthy
        obs_var = 0.25 + 0.5 * (1.0 - w)
        # slow service => capacity estimate is uncertain rather than simply low
        if latency_p90 > slo_ms:
            obs_var += 0.25

        n = self.n.get(service, 0)
        prior_n = self.prior_strength
        pm = (n * self.mean.get(service, math.log(self.global_capacity))
              + prior_n * math.log(self.global_capacity)) / (n + prior_n)
        pv = 1.0 / (1.0 / max(self.var.get(service, self.global_var), EPS)
                    + 1.0 / max(self.global_var, EPS))

        post_n = n + w
        post_var = 1.0 / (1.0 / max(pv, EPS) + w / obs_var)
        post_mean = (pv * pm + w * obs) / (pv + w) if (pv + w) > 0 else pm

        self.n[service] = post_n
        self.var[service] = post_var
        self.mean[service] = post_mean

    # ------------------------------------------------------------------
    def capacity(self, service):
        """Posterior mean capacity per replica (goodput units)."""
        return math.exp(self.mean.get(service, math.log(self.global_capacity)))

    def sigma(self, service):
        """Posterior std of log-capacity -> relative uncertainty."""
        return math.sqrt(max(self.var.get(service, self.global_var), EPS))

    # ------------------------------------------------------------------
    def replicas_for_current_load(self, service, offered, slo_ms=200.0,
                                  safety_z=1.0, max_replicas=10):
        """Replica count needed to serve the **current** offered workload.

        ``offered`` is the request rate observed in the current cycle, so this
        is the ``N_required = ceil(current_offered_workload /
        safe_capacity_per_replica)`` relation from the design, evaluated with a
        posterior-derived safety margin instead of a fixed 85 % headroom
        constant. Nothing here extrapolates demand or resources into the future.

        This is a helper for analysis and reporting; the control loop does not
        call it (the loop scores every candidate count through the decision
        engine instead).
        """
        cap = self.capacity(service)
        sig = self.sigma(service)
        # conservative capacity: mean less z*sigma on the log scale
        conservative = cap * math.exp(-safety_z * sig)
        if conservative <= EPS:
            return max_replicas
        n = int(math.ceil(offered / conservative))
        return max(1, min(max_replicas, n))

    # ------------------------------------------------------------------
    def snapshot(self):
        return {
            "global_capacity": self.global_capacity,
            "services": {
                s: {"mean_log": self.mean.get(s), "var": self.var.get(s),
                    "n": self.n.get(s, 0), "capacity": self.capacity(s)}
                for s in self.mean
            },
        }

    def load(self, snap):
        self.global_capacity = snap.get("global_capacity", self.global_capacity)
        self.mean = {k: v["mean_log"] for k, v in snap.get("services", {}).items()
                     if v.get("mean_log") is not None}
        self.var = {k: v["var"] for k, v in snap.get("services", {}).items()
                    if v.get("var") is not None}
        self.n = {k: v.get("n", 0) for k, v in snap.get("services", {}).items()}