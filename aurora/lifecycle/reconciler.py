"""Lifecycle-aware replica reconciler.

Tracks the pod lifecycle so that a scale-out that is still in progress is never
mistaken for spare capacity, and so that pending pods are never counted as
serving capacity.

    N_committed = N_serving + N_viable_pending

A pending pod counts as viable only while it is young enough to plausibly become
ready (age < the service's observed startup p95) and is not stuck in a state
that will never resolve.
"""

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
    def __init__(self, cfg, startup_p95_s=120.0):
        self.cfg = cfg
        self.startup_p95 = {}          # service -> observed startup p95
        self.startup_samples = {}
        self.in_flight = {}            # service -> Action
        self.actions = []
        self.default_startup_p95 = startup_p95_s

    # ------------------------------------------------------------------
    def observe_startup(self, service, duration_s):
        if duration_s is None or duration_s <= 0:
            return
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
        return {
            "spec": int(spec_replicas),
            "serving": n_serving,
            "viable_pending": n_viable,
            "stuck": stuck,
            "committed": n_serving + n_viable,
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

        if desired > committed:
            # allow the increase but never by more than the shortfall
            target = min(desired, max_pods)
        elif desired < committed:
            # scale-in: never below the number actually serving, so removing
            # pods can never take away capacity that is in use
            target = max(desired, min_pods)
            if serving > 0:
                target = max(target, min(serving, acc['spec']))
        else:
            target = committed

        target = max(min_pods, min(max_pods, target))
        return target, f"target {target} (committed={committed}, serving={serving})"