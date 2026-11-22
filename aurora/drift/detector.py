"""Behaviour-drift detection.

Four independent signals, each answering a different question:

  * input drift        -- MMD between recent encoder embeddings and a reference
                          sample from the service's own history
  * capacity drift     -- CUSUM on the log residual between predicted and
                          observed goodput per replica
  * configuration drift-- hashes of resource limits, replica bounds, image tag
  * topology drift     -- change in the dependency edge set

Missing telemetry is *not* drift: the detector only runs on windows whose
availability mask is complete, and a gap increments a separate counter.
"""

import hashlib
import json

EPS = 1e-9


def _mmd2(x, y, sigma=None):
    """Unbiased MMD^2 with a Gaussian kernel."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    n, m = len(x), len(y)
    if sigma is None:
        # median heuristic on a subsample
        allv = x[: min(n, 64)] + y[: min(m, 64)]
        d = []
        for i in range(len(allv)):
            for j in range(i + 1, len(allv)):
                d.append(sum((a - b) ** 2 for a, b in zip(allv[i], allv[j])))
        sigma = (sorted(d)[len(d) // 2] / 2.0) ** 0.5 if d else 1.0
        sigma = max(sigma, 1e-6)
    gamma = 1.0 / (2.0 * sigma * sigma)

    def k(a, b):
        return pow(2.718281828459045, -gamma * sum((p - q) ** 2 for p, q in zip(a, b)))

    xx = sum(k(x[i], x[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    yy = sum(k(y[i], y[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    xy = sum(k(x[i], y[j]) for i in range(n) for j in range(m)) / (n * m)
    return float(xx + yy - 2 * xy)


class DriftDetector:
    def __init__(self, mmd_threshold=0.05, cusum_threshold=5.0, cusum_drift=0.5):
        self.reference = {}        # service -> list of embedding vectors
        self.mmd_threshold = mmd_threshold
        self.cusum_threshold = cusum_threshold
        self.k = cusum_drift
        self.cusum_pos = {}        # service -> CUSUM accumulator
        self.cusum_neg = {}
        self.config_hash = None
        self.topology_hash = None
        self.missing_ticks = {}

    # ------------------------------------------------------------------
    def set_reference(self, service, vectors, cap=200):
        self.reference[service] = list(vectors)[-cap:]

    def input_drift(self, service, vectors):
        ref = self.reference.get(service)
        if not ref or not vectors:
            return 0.0
        return _mmd2(list(vectors), ref[: min(len(ref), 64)])

    # ------------------------------------------------------------------
    def observe_capacity(self, service, expected_log, observed_log):
        """CUSUM on the residual.

        ``expected_log`` is what the current capacity estimate implies for the
        observation being scored -- not a prediction of a future value. A
        positive residual means the service came in worse than its own current
        estimate, which is capacity drift, not forecast error.
        """
        r = observed_log - expected_log
        cp = max(0.0, self.cusum_pos.get(service, 0.0) + r - self.k)
        cn = min(0.0, self.cusum_neg.get(service, 0.0) + r + self.k)
        self.cusum_pos[service] = cp
        self.cusum_neg[service] = cn
        return cp, cn

    def capacity_drift(self, service):
        """True when capacity moved materially below *or* above expectation."""
        return (self.cusum_pos.get(service, 0.0) >= self.cusum_threshold
                or self.cusum_neg.get(service, 0.0) <= -self.cusum_threshold)

    # ------------------------------------------------------------------
    @staticmethod
    def config_fingerprint(obj):
        blob = json.dumps(obj, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def note_config(self, service, limits, replicas, image=None):
        fp = self.config_fingerprint({"limits": limits, "replicas": replicas,
                                      "image": image})
        changed = fp != self.config_hash
        self.config_hash = fp
        return changed

    # ------------------------------------------------------------------
    def note_topology(self, edges):
        fp = self.config_fingerprint(sorted(tuple(sorted(e)) for e in edges))
        changed = fp != self.topology_hash
        self.topology_hash = fp
        return changed

    # ------------------------------------------------------------------
    def note_missing(self, service):
        self.missing_ticks[service] = self.missing_ticks.get(service, 0) + 1

    def snapshot(self):
        return {
            "mmd_threshold": self.mmd_threshold,
            "cusum_threshold": self.cusum_threshold,
            "config_hash": self.config_hash,
            "topology_hash": self.topology_hash,
            "missing_ticks": self.missing_ticks,
            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
        }

    def load(self, snap):
        self.config_hash = snap.get("config_hash")
        self.topology_hash = snap.get("topology_hash")
        self.missing_ticks = snap.get("missing_ticks", {})
        self.cusum_pos = snap.get("cusum_pos", {})
        self.cusum_neg = snap.get("cusum_neg", {})