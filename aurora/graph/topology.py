"""Lightweight service-dependency context.

Edges are derived from observed Istio traffic (``source_workload`` ->
``destination_workload``), not from a hand-written topology file, so a new
deployment is discovered rather than declared.

The graph is used **only** for attribution and diagnosis. It never throttles,
caps or reroutes the caller's requests: if Food calls Travel and Travel is hot,
Food's own replicas are still decided from Food's own bottleneck evidence.
"""

import json


class Topology:
    def __init__(self, namespace):
        self.namespace = namespace
        self.edges = {}          # (src, dst) -> request rate
        self.dependents = {}     # dst -> set(src)
        self.dependencies = {}   # src -> set(dst)

    def update(self, triplets):
        """triplets: iterable of (src, dst, rate)."""
        for src, dst, rate in triplets:
            if not src or not dst or src == dst:
                continue
            self.edges[(src, dst)] = float(rate)
        self._rebuild()

    def _rebuild(self):
        self.dependents = {}
        self.dependencies = {}
        for (src, dst) in self.edges:
            self.dependents.setdefault(dst, set()).add(src)
            self.dependencies.setdefault(src, set()).add(dst)

    # ------------------------------------------------------------------
    def upstream_of(self, service):
        """Services that call ``service``."""
        return self.dependents.get(service, set())

    def downstream_of(self, service):
        """Services ``service`` calls."""
        return self.dependencies.get(service, set())

    def max_upstream_rate(self, service):
        return max([r for (s, d), r in self.edges.items()
                    if d == service], default=0.0)

    # ------------------------------------------------------------------
    def context_features(self, service):
        """Graph features for the model (kept small; a GNN is not required)."""
        ups = self.upstream_of(service)
        downs = self.downstream_of(service)
        up_rates = [r for (s, d), r in self.edges.items() if d == service]
        down_rates = [r for (s, d), r in self.edges.items() if s == service]
        return {
            "n_upstream": float(len(ups)),
            "n_downstream": float(len(downs)),
            "upstream_rate": float(sum(up_rates)),
            "downstream_rate": float(sum(down_rates)),
            "max_upstream_rate": float(max(up_rates) if up_rates else 0.0),
        }

    def fingerprint(self):
        return sorted(tuple(sorted(e)) for e in self.edges)

    def to_dict(self):
        return {"namespace": self.namespace,
                "edges": [{"src": s, "dst": d, "rate": r}
                          for (s, d), r in sorted(self.edges.items())]}

    @classmethod
    def from_dict(cls, d):
        t = cls(d.get("namespace", "train-ticket"))
        t.update([(e["src"], e["dst"], e["rate"]) for e in d.get("edges", [])])
        return t

    def dumps(self):
        return json.dumps(self.to_dict(), sort_keys=True)