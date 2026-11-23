"""Lifecycle-aware replica reconciler.

Tracks the pod lifecycle so that a scale-out that is still in progress is never
mistaken for spare capacity, and so that pending pods are never counted as
serving capacity.

    N_committed = N_serving + N_viable_pending

A pending pod counts as viable only while it is young enough to plausibly become
ready (age < the service's observed startup p95) and is not stuck in a state
that will never resolve.
"""

import os
import time

STUCK_REASONS = {"CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull",
                 "CreateContainerConfigError", "CreateContainerError",
                 "RunContainerError", "InvalidImageName"}


class Action:
    def __init__(self, service, ts, from_n, to_n, reason):
        self.service = service
        self.ts = ts
        self.from_n = from_n
        self.to_n = to_n
        self.reason = reason
        self.completed = False

    def __repr__(self):
        return f"Action({self.service} {self.from_n}->{self.to_n})"


class LifecycleReconciler:
    def __init__(self, cfg, startup_p95_s=None):
        self.cfg = cfg
        self.startup_p95 = {}          # service -> observed startup p95
        self.startup_samples = {}
        self.in_flight = {}            # service -> Action
        self.actions = []
        # A booting JVM burns CPU and throttles heavily for minutes.
        # 120 s was too short: the first decisions after it fired on
        # startup throttling and pinned cold services at max.
        self._allow_fast_scale_in = False
        # Maximum replicas removed by one scale-in decision. Three is enough for
        # an 8->5 reduction (the acceptance case) while still preventing a
            # single decision from draining a service.
        self.scale_in_step = 3
        self._seen_uids = {}           # service -> set(pod uid)
        self.last_scale_in = {}        # service -> ts
        self.default_startup_p95 = (
            startup_p95_s if startup_p95_s is not None
            else int(os.environ.get('AURORA_STARTUP_P95_S', '300')))

    # ------------------------------------------------------------------
    def observe_startup(self, service, duration_s, uid=None):
        """Record one Pod's startup duration, at most once per Pod.

        The same Ready pod is seen on every 15 s cycle; counting it each time
        made the distribution depend on controller uptime rather than on the
        number of Pods that actually started.
        """
        if duration_s is None or duration_s <= 0:
            return
        if uid is not None:
            seen = self._seen_uids.setdefault(service, set())
            if uid in seen:
                return
            seen.add(uid)
        s = self.startup_samples.setdefault(service, [])
        s.append(float(duration_s))
        if len(s) > 200:
            s.pop(0)
        xs = sorted(s)
        if xs:
            idx = max(0, int(0.95 * (len(xs) - 1)))
            self.startup_p95[service] = xs[idx]

    def startup_p95_for(self, service):
        return self.startup_p95.get(service, self.default_startup_p95)

    # ------------------------------------------------------------------
    def accounting(self, service, spec_replicas, ready, pending_pods, now=None):
        """Split replicas into serving / viable-pending / stuck.

        ``pending_pods`` is a list of dicts: {"age_s": float, "reason": str|None}
        """
        now = now if now is not None else time.time()
        viable, stuck = 0, 0
        for p in pending_pods:
            reason = (p.get("reason") or "")
            age = p.get("age_s")
            if reason in STUCK_REASONS:
                stuck += 1
                continue
            limit = self.startup_p95_for(service)
            if age is None or age <= limit:
                viable += 1
            else:
                # older than the startup p95 and still not ready: not usable
                stuck += 1
        n_serving = int(ready)
        n_viable = min(viable, max(0, int(spec_replicas) - n_serving))
        # Replicas in the Deployment spec that no Pod represents yet: the Scale
        # API has accepted them but the ReplicaSet has not created Pods. They
        # must stay visible so a scale-out is not repeated while they appear.
        represented = n_serving + len(pending_pods)
        requested_pending = max(0, int(spec_replicas) - represented)
        return {
            "spec": int(spec_replicas),
            "serving": n_serving,
            "viable_pending": n_viable,
            "stuck": stuck,
            "requested_pending": requested_pending,
            "committed": n_serving + n_viable + requested_pending,
        }

    # ------------------------------------------------------------------
    def record(self, service, ts, from_n, to_n, reason):
        act = Action(service, ts, from_n, to_n, reason)
        self.in_flight[service] = act
        self.actions.append(act)
        return act

    def settle(self, service, now, ready):
        """Clear an in-flight action once the new pods are ready (or it timed out)."""
        act = self.in_flight.get(service)
        if not act:
            return True
        if ready >= act.to_n:
            self.in_flight.pop(service, None)
            act.completed = True
            return True
        # a scale-out older than 2x the observed startup p95 is abandoned
        if now - act.ts > 2.0 * self.startup_p95_for(service):
            self.in_flight.pop(service, None)
            return False
        return False

    def pending_scale_out(self, service):
        act = self.in_flight.get(service)
        return act is not None and act.to_n > act.from_n

    # ------------------------------------------------------------------
    def reconcile(self, service, desired, acc, min_pods=None, max_pods=None):
        """Clamp ``desired`` against lifecycle reality.

        Returns (final_target, reason). Never issues a second scale-out while
        viable pods from the previous one are still starting.
        """
        min_pods = self.cfg.min_pod if min_pods is None else min_pods
        max_pods = self.cfg.max_pod if max_pods is None else max_pods
        committed = acc["committed"]
        serving = acc["serving"]

        if desired > committed and self.pending_scale_out(service):
            # previous scale-out still converging: do not stack another one
            return committed, f"hold: scale-out in flight ({committed} committed)"
        if desired > committed and acc.get("requested_pending", 0) > 0:
            # Scale accepted but Pods not created yet: wait, do not re-issue.
            return committed, (f"hold: {acc['requested_pending']} replica(s) "
                               f"requested but no Pod yet")

        if desired > committed:
            # allow the increase but never by more than the shortfall
            target = min(desired, max_pods)
        elif desired < committed:
            # Scale-in. The current Ready count is NOT a floor: those pods are
            # exactly the ones Kubernetes may remove. Safety comes from the
            # engine's capacity/uncertainty terms, from the pending-scale-out
            # guard above, and from moving incrementally.
            if self.pending_scale_out(service):
                return committed, "hold: scale-out in flight"
            target = max(desired, min_pods)
            # incremental: never remove more than one "step" below committed
            # unless the caller explicitly asks for a larger reduction
            if committed - target > self.scale_in_step and not self._allow_fast_scale_in:
                target = max(target, committed - self.scale_in_step)
        else:
            target = committed

        target = max(min_pods, min(max_pods, target))
        return target, f"target {target} (committed={committed}, serving={serving})"