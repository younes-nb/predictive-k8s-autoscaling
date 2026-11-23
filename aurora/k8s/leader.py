"""Kubernetes Lease-based leader election.

Serial execution inside one process does not stop two AURORA replicas from
writing the same Deployment. A Lease gives cluster-level exclusivity:

  * acquire the Lease before touching any replica count;
  * renew it while leading;
  * the moment it is lost, stop writing (fail closed to hold).

A standby instance never patches replicas.
"""

import datetime
import os
import socket
import time

from kubernetes import client
from kubernetes.client.rest import ApiException

# RFC3339 timestamps come in several shapes:
#   2026-11-23T12:00:00Z          (no fraction, Z)
#   2026-11-23T12:00:00.123456Z   (microseconds, Z)
#   2026-11-23T12:00:00+00:00     (numeric offset, with or without fraction)
# The previous parser required a fraction and a "+0000" offset, so a plain
# ``...:00Z`` raised ValueError and the caller treated the Lease as stale.
_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
)


def parse_rfc3339(text):
    """Parse an RFC3339 timestamp into an aware datetime, or return ``None``.

    Returning ``None`` (rather than raising) lets callers distinguish "the
    timestamp is unparseable" from "the Lease is old" -- the two must not be
    conflated, because an unparseable timestamp is *not* evidence that the
    holder is gone.
    """
    if text is None:
        return None
    if isinstance(text, datetime.datetime):
        return text if text.tzinfo else text.replace(tzinfo=datetime.timezone.utc)
    s = str(text).strip()
    if not s:
        return None
    # ``+00:00`` is accepted by %z on Python 3.7+; ``Z`` is not. Normalise a
    # trailing Z to +00:00 and let the numeric-offset formats handle the rest.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    for fmt in _FORMATS:
        try:
            dt = datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    # Last resort: fromisoformat (Python 3.10 accepts the common RFC3339 forms)
    try:
        dt = datetime.datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)


class LeaderElection:
    def __init__(self, config, name="aurora-leader", lease_duration_s=30,
                 renew_s=10):
        self.cfg = config
        self.name = name
        self.lease_duration_s = lease_duration_s
        self.renew_s = renew_s
        self.identity = os.environ.get("HOSTNAME") or socket.gethostname()
        # Dynamic API: the typed client coerces the body into a model and
        # reformats timestamps without microseconds, which the apiserver
        # rejects. A dict passes through unchanged.
        self.api = client.CustomObjectsApi()
        self.is_leader = False
        self.last_renew = 0.0
        self.reason = "not started"
        # Original acquisition time of the *current* leadership spell. It is
        # preserved across renewals and reset only when another holder takes
        # over; the previous implementation copied renewTime into acquireTime
        # on every write.
        self.acquire_time = None
        self._last_holder = None

    # ------------------------------------------------------------------
    @staticmethod
    def _now():
        # Kubernetes requires RFC3339 with microseconds; a plain
        # "%Y-%m-%dT%H:%M:%SZ" is rejected with HTTP 400.
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".000000Z"

    def _body(self, holder, transitions=0):
        now = self._now()
        # ``acquireTime`` must be a string too: passing the datetime object the
        # API returns makes the client serialise it as "+00:00", which the
        # apiserver rejects with HTTP 400. When we are continuing to lead, the
        # original acquisition timestamp is kept -- acquireTime records when the
        # current holder acquired the Lease, not when it last renewed it.
        # ``renew`` (the previous renewTime) was copied into acquireTime by the
        # old implementation; that is exactly the bug.
        if holder == self.identity and self.acquire_time:
            acq = self.acquire_time
        else:
            acq = now
        self.acquire_time = acq
        return {
            "apiVersion": "coordination.k8s.io/v1",
            "kind": "Lease",
            "metadata": {"name": self.name, "namespace": self.cfg.namespace},
            "spec": {
                "holderIdentity": holder,
                "leaseDurationSeconds": self.lease_duration_seconds(),
                "acquireTime": acq,
                "renewTime": now,
                "leaseTransitions": int(transitions),
            },
        }

    def lease_duration_seconds(self):
        return int(self.lease_duration_s)

    def _get(self):
        try:
            return self.api.get_namespaced_custom_object(
                group="coordination.k8s.io", version="v1",
                namespace=self.cfg.namespace, plural="leases", name=self.name)
        except ApiException as exc:
            if getattr(exc, "status", None) == 404:
                return None
            raise

    # ------------------------------------------------------------------
    def _expired(self, lease):
        """Return (expired, reason).

        ``expired`` is True only when the Lease is *provably* older than the
        lease duration. A missing or unparseable renewTime is not proof of
        expiry: it is reported as unknown and the Lease is treated as still
        valid, so a format variation can never make a live leader look stale.
        """
        if not lease:
            return False, "absent"
        spec = lease.get("spec") or {}
        renew = spec.get("renewTime")
        if not renew:
            return False, "no renewTime -- treating as valid"
        dt = parse_rfc3339(renew)
        if dt is None:
            return False, f"unparseable renewTime {renew!r} -- treating as valid"
        age = time.time() - dt.timestamp()
        return age > self.lease_duration_s, f"age={age:.1f}s"

    # ------------------------------------------------------------------
    def try_acquire(self):
        """Attempt to become (or remain) leader. Returns True if leader."""
        now = time.time()
        try:
            lease = self._get()
        except ApiException as exc:
            self.reason = f"lease read failed: {exc}"
            self.is_leader = False
            return False

        spec = (lease or {}).get("spec") or {}
        holder = spec.get("holderIdentity")
        expired, why = self._expired(lease)

        if holder and holder != self.identity and not expired:
            self.reason = f"held by {holder} ({why})"
            self.is_leader = False
            self.acquire_time = None
            return False

        # A new holder (or an empty Lease) starts a new acquisition spell.
        if holder != self.identity:
            self.acquire_time = self._now()

        transitions = int(spec.get("leaseTransitions") or 0)
        if holder and holder != self.identity:
            transitions += 1

        body = self._body(self.identity, transitions)
        try:
            if lease is None:
                self.api.create_namespaced_custom_object(
                    group="coordination.k8s.io", version="v1",
                    namespace=self.cfg.namespace, plural="leases", body=body)
            else:
                rv = (lease.get("metadata") or {}).get("resourceVersion")
                if rv:
                    body["metadata"]["resourceVersion"] = rv
                self.api.replace_namespaced_custom_object(
                    group="coordination.k8s.io", version="v1",
                    namespace=self.cfg.namespace, plural="leases",
                    name=self.name, body=body)
        except ApiException as exc:
            self.reason = f"lease write failed: {exc}"
            self.is_leader = False
            return False

        self.is_leader = True
        self.last_renew = now
        self.reason = "leader"
        return True

    def renew(self):
        if not self.is_leader:
            return False
        if time.time() - self.last_renew < self.renew_s:
            return True
        return self.try_acquire()

    def release(self):
        """Drop leadership so another instance can take over promptly."""
        if not self.is_leader:
            return
        try:
            lease = self._get()
            if lease is not None:
                body = self._body("", 0)
                body["metadata"]["resourceVersion"] = lease["metadata"]["resourceVersion"]
                body["spec"]["leaseDurationSeconds"] = 1
                body["spec"]["holderIdentity"] = ""
                self.api.replace_namespaced_custom_object(
                    group="coordination.k8s.io", version="v1",
                    namespace=self.cfg.namespace, plural="leases",
                    name=self.name, body=body)
        except ApiException:
            pass
        self.is_leader = False
        self.acquire_time = None
        self.reason = "released"