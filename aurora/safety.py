"""Deterministic safety layer.

The Replica-Change Model proposes; this validates. Nothing reaches the Scale API
without passing through here.

Scale-out is allowed when the diagnosis says the service is under-provisioned (or
shows a root bottleneck that more replicas can relieve) AND the user-facing
deadline is currently missed AND the model is confident and in-distribution.

Scale-in is stricter: it requires an over-provisioned diagnosis, a healthy
deadline, no unresolved scale-out, and moves incrementally.

The layer is deliberately conservative: every ambiguous case resolves to hold.
"""

from models.graph import PROVISIONING_CLASSES

UNDER = PROVISIONING_CLASSES.index("under_provisioned")
ADEQUATE = PROVISIONING_CLASSES.index("adequate")
OVER = PROVISIONING_CLASSES.index("over_provisioned")
UNKNOWN = PROVISIONING_CLASSES.index("unknown")

# Signals that extra replicas can plausibly relieve.
ADDRESSABLE_SIGNALS = {"cpu", "throttle", "queue", "memory"}


class SafetyLayer:
    def __init__(self, cfg, min_confidence=0.55, max_ood_distance=1.0,
                 scale_in_step=3):
        self.cfg = cfg
        self.scale_in_step = scale_in_step
        self.min_confidence = min_confidence
        self.max_ood_distance = max_ood_distance

    # ------------------------------------------------------------------
    def validate(self, action, target, diag, committed, in_flight,
                 slo_state, min_pods, max_pods):
        """Return (action, target, reason)."""
        conf = diag.get("confidence", 0.0)
        ood = diag.get("ood", False)
        prov = diag.get("provisioning_idx", 3)
        signal = diag.get("binding_signal")
        root = diag.get("root_pb", 0.0)

        target = max(min_pods, min(max_pods, int(target)))
        confident = conf >= self.min_confidence and not ood
        # Tri-state: "unknown" is NOT "met". Missing latency must never
        # authorize a scale-in.
        deadline_missed = (slo_state == "violated")
        slo_unknown = (slo_state == "unknown")

        if action == "out":
            under = prov == UNDER
            # No fabricated addressability: a confident root PB on a signal
            # that replicas can relieve is the evidence.
            root_addr = root > 0.5 and signal in ADDRESSABLE_SIGNALS
            if in_flight:
                return "hold", committed, "scale-out in flight"
            if not confident:
                return "hold", committed, f"low confidence ({conf:.2f}) or OOD"
            if not (under or (root_addr and signal in ADDRESSABLE_SIGNALS)):
                return "hold", committed, f"not under-provisioned (prov={PROVISIONING_CLASSES[prov]})"
            if not deadline_missed and root <= 0.5:
                return "hold", committed, "no deadline miss and no root bottleneck"
            return "out", target, f"scale-out to {target} ({PROVISIONING_CLASSES[prov]})"

        if action == "in":
            if in_flight:
                return "hold", committed, "scale-out unresolved"
            if not confident:
                return "hold", committed, f"low confidence ({conf:.2f}) or OOD"
            if prov != OVER:
                return "hold", committed, f"not over-provisioned (prov={PROVISIONING_CLASSES[prov]})"
            if slo_unknown:
                return "hold", committed, "frontend SLO unknown"
            if deadline_missed:
                return "hold", committed, "deadline currently missed"
            cur = committed
            step = min(self.scale_in_step, max(1, cur - min_pods))
            target = max(min_pods, cur - step)
            return "in", target, f"incremental scale-in {cur}->{target}"

        return "hold", committed, "model chose hold"

    # ------------------------------------------------------------------
    @staticmethod
    def enforce_direction(action, target, committed):
        """The action label and the target count must agree.

        A model that says "out" but returns a lower count is a contradiction;
        the target is what the reconciler applies, so the label cannot be
        trusted and the action degrades to hold.
        """
        if action == "out" and target <= committed:
            return "hold", committed, f"direction mismatch (out, {committed}->{target})"
        if action == "in" and target >= committed:
            return "hold", committed, f"direction mismatch (in, {committed}->{target})"
        if action == "hold" and target != committed:
            return "hold", committed, "hold with a different target"
        return action, target, "ok"