"""Feedback and action ledger.

One JSON object per line, appended and fsync'd, so a controller restart cannot
erase learning. Each record captures the state before the action, the model and
capacity estimates that produced it, and a placeholder for the measured outcome.

Outcomes are filled in later by ``settle`` once the new pods are ready and
enough post-action observations exist -- the model never grades its own homework.
"""

import json
import os
import time


def _get(d, key, default=None):
    """Read a field from either a dict or an object."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _decision_record(decision):
    """Normalise whatever the caller passes (dict or object) into a record."""
    basis = _get(decision, "basis", {}) or {}
    return {
        "from_replicas": _get(decision, "from_replicas",
                              _get(decision, "committed")),
        "to_replicas": _get(decision, "replicas", _get(decision, "target")),
        "action": _get(decision, "action"),
        "reason": _get(decision, "reason"),
        "utility": _get(decision, "utility"),
        "bottleneck": _get(decision, "bottleneck"),
        "root_pb": _get(decision, "root_pb"),
        "slo_state": _get(decision, "slo_state"),
        "slo_value_ms": _get(decision, "slo_value_ms"),
        "confidence": _get(decision, "confidence"),
        "ood": _get(decision, "ood"),
        "capacity": basis.get("capacity") if isinstance(basis, dict) else None,
        "sigma": basis.get("sigma") if isinstance(basis, dict) else None,
        "signal": basis.get("signal") if isinstance(basis, dict) else None,
    }


class Ledger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.rows = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def _append(self, rec):
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.rows.append(rec)

    # ------------------------------------------------------------------
    def record_action(self, service, ts, model_version, before, decision,
                      state):
        rec = {
            "kind": "action",
            "ts": ts,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            "service": service,
            "model_version": model_version,
            "before": before,
            "decision": _decision_record(decision),
            "outcome": None,
        }
        self._append(rec)
        return rec

    # ------------------------------------------------------------------
    def settle(self, service, ts, after):
        """Attach measured post-action outcomes to the newest unsettled action."""
        for rec in reversed(self.rows):
            if rec.get("kind") != "action" or rec.get("outcome") is not None:
                continue
            if rec.get("service") != service:
                continue
            rec["outcome"] = after
            rec["settled_ts"] = ts
            self._append({"kind": "outcome_ref", "ts": ts,
                          "service": service,
                          "from_replicas": rec["decision"]["from_replicas"],
                          "to_replicas": rec["decision"]["to_replicas"]})
            return rec
        return None

    def by_service(self, service, kind=None):
        return [r for r in self.rows
                if r.get("service") == service and (kind is None or r.get("kind") == kind)]

    def __len__(self):
        return len(self.rows)