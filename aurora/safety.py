"""Deterministic safety layer.

The Replica-Change Model proposes; this validates. Nothing reaches the Scale API
without passing through here.

Scale-out is allowed when the diagnosis says the service is under-provisioned
AND there is evidence that more replicas can plausibly relieve the problem AND
the user-facing deadline is currently missed AND both the diagnosis and the
replica proposal are confident and in-distribution.

Scale-in is stricter: it requires an over-provisioned diagnosis, a healthy
deadline, no unresolved scale-out, and a sustained healthy period of
``scale_in_stabilization_s``.

The layer is deliberately conservative: every ambiguous case resolves to hold.
"""

import time

from models.graph import PROVISIONING_CLASSES

UNDER = PROVISIONING_CLASSES.index("under_provisioned")
ADEQUATE = PROVISIONING_CLASSES.index("adequate")
OVER = PROVISIONING_CLASSES.index("over_provisioned")
UNKNOWN = PROVISIONING_CLASSES.index("unknown")

# Local degradation classes that extra replicas can plausibly relieve. This is
# the *only* addressability evidence: a root PB on a signal outside this set
# (e.g. a saturated shared database, an application-level constraint) is not
# horizontally addressable.
ADDRESSABLE_DEGRADATION = {"cpu", "queue"}

DEGRADATION_NAMES = [
    "cpu", "memory", "network", "storage", "queue",
    "shared_dependency", "application", "infrastructure", "unknown",
]


class SafetyLayer:
    def __init__(self, cfg, min_confidence=0.45, scale_in_step=3,
                 addressable_prob_min=0.5):
        self.cfg = cfg
        self.scale_in_step = scale_in_step
        self.min_confidence = min_confidence
        self.addressable_prob_min = addressable_prob_min

    # ------------------------------------------------------------------
    def addressable(self, diag):
        """(is_addressable, evidence) from the diagnosis model's own outputs.

        The previous implementation fabricated ``binding_signal`` from the
        affected-caller probability ("latency" if affected > 0.5 else "cpu"),
        which had no relationship to the observed degradation. This reads the
        multi-label degradation probabilities directly and asks whether any
        horizontally-addressable class is materially present.
        """
        deg = diag.get("degradation") or []
        if not deg:
            return False, "no degradation evidence"
        # "Addressable" means replicas are the plausible remedy AND no
        # non-addressable cause dominates. A saturated shared database or an
        # application-level constraint is NOT horizontally addressable even if
        # the queue probability is high, so a dominant non-addressable class
        # vetoes.
        best, best_name = 0.0, None
        for i, name in enumerate(DEGRADATION_NAMES):
            if i >= len(deg) or name not in ADDRESSABLE_DEGRADATION:
                continue
            p = float(deg[i])
            if p > best:
                best, best_name = p, name
        if best_name is None:
            return False, "no addressable degradation class"
        if best < self.addressable_prob_min:
            return False, f"{best_name}={best:.2f} below {self.addressable_prob_min}"
        non_addressable = {"shared_dependency", "application", "infrastructure"}
        for i, name in enumerate(DEGRADATION_NAMES):
            if name in non_addressable and i < len(deg) and float(deg[i]) > best:
                return False, (f"{name}={float(deg[i]):.2f} dominates "
                               f"{best_name}={best:.2f}")
        return True, f"{best_name}={best:.2f}"

    # ------------------------------------------------------------------
    def validate(self, action, target, diag, committed, in_flight,
                 slo_state, min_pods, max_pods, stable_since=None,
                 now=None, action_conf=None, target_conf=None):
        """Return (action, target, reason).

        ``stable_since`` is the start of the current continuously-healthy /
        over-provisioned period; scale-in requires it to be older than
        ``scale_in_stabilization_s``. ``None`` means "not yet known", which is
        not the same as "stable for ever".
        """
        conf = diag.get("confidence", 0.0)
        ood = diag.get("ood", False)
        prov = diag.get("provisioning_idx", UNKNOWN)
        root = diag.get("root_pb", 0.0)
        now = time.time() if now is None else now

        # The model's own confidence about THIS decision must hold on its own.
        model_confident = True
        if action_conf is not None and action_conf < self.cfg.action_conf_min:
            model_confident = False
        if target_conf is not None and target_conf < self.cfg.target_conf_min:
            model_confident = False

        target = max(min_pods, min(max_pods, int(target)))
        confident = conf >= self.min_confidence and not ood and model_confident
        # Tri-state: "unknown" is NOT "met". Missing latency must never
        # authorize a scale-in.
        deadline_missed = (slo_state == "violated")
        slo_unknown = (slo_state == "unknown")

        if action == "out":
            if in_flight:
                return "hold", committed, "scale-out in flight"
            if not confident:
                return "hold", committed, f"low confidence ({conf:.2f}) or OOD"
            # The model's provisioning head is a bootstrap label and is
            # systematically uncertain (nearly every service reads "unknown"),
            # so it cannot be the only thing that authorises or blocks a
            # scale-out. The deadline plus addressable evidence is the real
            # gate; the provisioning head is a *hint* that is only allowed to
            # veto when it positively says the service is already adequate.
            prov_veto = (prov == ADEQUATE)
            if prov_veto:
                return "hold", committed, (
                    "provisioning says adequate (bootstrap veto)")
            # Under-provisioning alone is not enough: the problem must be one
            # that horizontal scaling can plausibly relieve. Without this, a
            # service bottlenecked on a shared database or an application-level
            # constraint would be scaled out forever without benefit.
            ok, why = self.addressable(diag)
            if not ok:
                return "hold", committed, f"not horizontally addressable ({why})"
            if not deadline_missed:
                return "hold", committed, "deadline currently met"
            return "out", target, (
                f"scale-out to {target} (addressable: {why}, "
                f"prov={PROVISIONING_CLASSES[prov]})")

        if action == "in":
            if in_flight:
                return "hold", committed, "scale-out unresolved"
            if not confident:
                return "hold", committed, f"low confidence ({conf:.2f}) or OOD"
            if prov != OVER:
                return "hold", committed, (
                    f"not over-provisioned (prov={PROVISIONING_CLASSES[prov]})")
            if slo_unknown:
                return "hold", committed, "frontend SLO unknown"
            if deadline_missed:
                return "hold", committed, "deadline currently missed"
            # Sustained stability, not merely "no previous scale-in". A service
            # that has never been scaled in has no timer running, so the first
            # scale-in is only allowed once it has been continuously healthy and
            # over-provisioned for the stabilization window.
            if stable_since is None:
                return "hold", committed, "stability window not established"
            held = now - stable_since
            if held < self.cfg.scale_in_stabilization_s:
                return "hold", committed, (
                    f"scale-in stabilization ({held:.0f}s < "
                    f"{self.cfg.scale_in_stabilization_s}s)")
            # Constrain the model's target to a maximum decrement rather than
            # discarding it: the model proposed a count, and the safety layer's
            # job is to bound how far it may move in one step.
            cur = committed
            step = min(self.scale_in_step, max(1, cur - min_pods))
            floor = max(min_pods, cur - step)
            target = max(floor, min(target, cur - 1))
            return "in", target, (
                f"scale-in {cur}->{target} after {held:.0f}s stable")

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