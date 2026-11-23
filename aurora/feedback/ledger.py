"""Feedback and action ledger.

One JSON object per line, appended and fsync'd, so a controller restart cannot
erase learning.

The log is **event-sourced**: every state change is its own record
(``action_created``, ``action_applied``, ``action_settled``, ``action_failed``,
``action_inconclusive``) and the current state is reconstructed by replaying
those events. The previous design mutated an ``action`` record in place and
appended a separate ``outcome_ref``; on restart the action came back with
``outcome=None`` because the outcome itself was never appended.

The ledger is also the durable input for learning which scaling actions actually
helped. Until enough settled outcomes exist, the Replica-Change Model is a
bootstrap model and its recommendations must not be read as validated
scaling-benefit decisions.
"""

import json
import os
import time

EVENT_KINDS = ("action_created", "action_applied", "action_settled",
               "action_failed", "action_inconclusive")
TERMINAL_KINDS = ("action_settled", "action_failed", "action_inconclusive")


def _get(d, key, default=None):
    """Read a field from either a dict or an object."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _decision_record(decision):
    """Normalise whatever the caller passes (dict or object) into a record."""
    basis = _get(decision, "basis", {}) or {}
    if not isinstance(basis, dict):
        basis = {}
    return {
        "action": _get(decision, "action"),
        "from_replicas": _get(decision, "from_replicas",
                              _get(decision, "committed")),
        "to_replicas": _get(decision, "replicas", _get(decision, "target")),
        "target": _get(decision, "target"),
        "reason": _get(decision, "reason"),
        "root_pb": _get(decision, "root_pb"),
        "affected": _get(decision, "affected"),
        "binding_signal": _get(decision, "binding_signal"),
        "provisioning_idx": _get(decision, "provisioning_idx"),
        "provisioning": _get(decision, "provisioning"),
        "degradation": _get(decision, "degradation"),
        "slo_state": _get(decision, "slo_state"),
        "slo_value_ms": _get(decision, "slo_value_ms"),
        "confidence": _get(decision, "confidence"),
        "action_confidence": _get(decision, "action_confidence"),
        "target_confidence": _get(decision, "target_confidence"),
        "ood": _get(decision, "ood"),
        "capacity": basis.get("capacity"),
        "sigma": basis.get("sigma"),
        "signal": basis.get("signal"),
    }


class Ledger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.rows = []
        self.actions = {}
        self._load()

    # ------------------------------------------------------------------
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
        self._replay()

    def _replay(self):
        """Rebuild action state from the event log.

        ``action_created`` opens a record, ``action_applied`` attaches the
        pre-action state, and a terminal event closes it with the outcome. Any
        terminal event is authoritative regardless of process restarts, so a
        settled action does not become unsettled again.
        """
        self.actions = {}
        for rec in self.rows:
            kind = rec.get("kind")
            if kind not in EVENT_KINDS:
                continue
            aid = rec.get("action_id")
            if not aid:
                continue
            a = self.actions.setdefault(aid, {
                "action_id": aid, "service": rec.get("service"),
                "model_version": rec.get("model_version"),
                "created_ts": None, "applied_ts": None, "settled_ts": None,
                "before": None, "decision": None, "outcome": None,
                "status": "created", "failure": None,
            })
            if kind == "action_created":
                a["created_ts"] = rec.get("ts")
                a["service"] = rec.get("service", a["service"])
                a["model_version"] = rec.get("model_version", a["model_version"])
                if a["status"] == "created":
                    a["status"] = "created"
            elif kind == "action_applied":
                a["applied_ts"] = rec.get("ts")
                a["before"] = rec.get("before", a["before"])
                a["decision"] = rec.get("decision", a["decision"])
                if a["status"] == "created":
                    a["status"] = "applied"
            else:
                a["settled_ts"] = rec.get("ts")
                a["outcome"] = rec.get("outcome")
                a["status"] = kind
                if kind == "action_failed":
                    a["failure"] = rec.get("reason")

    def _append(self, rec):
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.rows.append(rec)

    @staticmethod
    def _new_action_id(service, ts, seq):
        return f"{int(ts)}-{service}-{seq}"

    # ------------------------------------------------------------------
    def record_action(self, service, ts, model_version, before, decision,
                      state=None, apply=True):
        """Open an action and, when ``apply`` is true, mark it applied.

        ``before`` must describe the full pre-action state (replicas, ready,
        pending, SLO, diagnosis, capacity) -- not just a replica count.
        """
        seq = len([r for r in self.rows if r.get("kind") == "action_created"])
        aid = self._new_action_id(service, ts, seq)
        created = {
            "kind": "action_created", "ts": ts,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            "action_id": aid, "service": service,
            "model_version": model_version,
            "before": before, "state": state or {},
            "decision": _decision_record(decision),
        }
        self._append(created)
        self.actions[aid] = {
            "action_id": aid, "service": service, "model_version": model_version,
            "created_ts": ts, "applied_ts": None, "settled_ts": None,
            "before": before, "decision": created["decision"], "outcome": None,
            "status": "created", "failure": None,
        }
        if apply:
            self.record_applied(aid, ts, before, decision)
        return aid

    def record_applied(self, action_id, ts, before, decision):
        rec = {"kind": "action_applied", "ts": ts,
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
               "action_id": action_id, "before": before,
               "decision": _decision_record(decision)}
        self._append(rec)
        a = self.actions.get(action_id)
        if a is not None and a.get("status") == "created":
            a["applied_ts"] = ts
            a["status"] = "applied"
        return rec

    def record_outcome(self, action_id, kind, ts, outcome, reason=None):
        """Append a terminal event for one action.

        ``kind`` must be one of ``action_settled`` / ``action_failed`` /
        ``action_inconclusive``. The complete outcome is stored on the event,
        so replaying the log reconstructs the settled result without needing
        the in-memory record to have survived.
        """
        if kind not in TERMINAL_KINDS:
            raise ValueError(f"not a terminal event: {kind}")
        rec = {"kind": kind, "ts": ts,
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
               "action_id": action_id, "outcome": outcome}
        if reason is not None:
            rec["reason"] = reason
        self._append(rec)
        a = self.actions.get(action_id)
        if a is not None:
            a["status"] = kind
            a["settled_ts"] = ts
            a["outcome"] = outcome
            a["failure"] = reason
        return rec

    # ------------------------------------------------------------------
    def settle(self, service, ts, after):
        """Settle the newest unsettled action for ``service``.

        ``ts`` is the *settlement* time, not the action time. Kept for callers
        that still use the positional form.
        """
        for aid in reversed(list(self.actions)):
            a = self.actions[aid]
            if a.get("service") != service:
                continue
            if a.get("status") in TERMINAL_KINDS or a.get("status") == "created":
                continue
            return self.record_outcome(aid, "action_settled", ts, after)
        return None

    def unsettled(self, service=None):
        return [a for a in self.actions.values()
                if a.get("status") not in TERMINAL_KINDS
                and (service is None or a.get("service") == service)]

    def settled_count(self, service=None):
        return sum(1 for a in self.actions.values()
                   if a.get("status") == "action_settled"
                   and (service is None or a.get("service") == service))

    def by_service(self, service, kind=None):
        return [r for r in self.rows
                if r.get("service") == service and (kind is None or r.get("kind") == kind)]

    def __len__(self):
        return len(self.rows)