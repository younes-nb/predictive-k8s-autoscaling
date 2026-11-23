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

    # ------------------------------------------------------------------
    @staticmethod
    def _now():
        # Kubernetes requires RFC3339 with microseconds; a plain
        # "%Y-%m-%dT%H:%M:%SZ" is rejected with HTTP 400.
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".000000Z"

    def _body(self, holder, renew=None):
        now = self._now()
        # ``acquireTime`` must be a string too: passing the datetime object the
        # API returns makes the client serialise it as "+00:00", which the
        # apiserver rejects with HTTP 400.
        acq = renew if isinstance(renew, str) else self._now()
        return {
            "apiVersion": "coordination.k8s.io/v1",
            "kind": "Lease",
            "metadata": {"name": self.name, "namespace": self.cfg.namespace},
            "spec": {
                "holderIdentity": holder,
                "leaseDurationSeconds": self.lease_duration_s,
                "acquireTime": acq,
                "renewTime": now,
                "leaseTransitions": 0,
            },
        }

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
        renew = spec.get("renewTime")
        stale = True
        if renew:
            try:
                renew_dt = datetime.datetime.strptime(
                    renew.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z")
                stale = (now - renew_dt.timestamp()) > self.lease_duration_s
            except ValueError:
                stale = True

        if holder and holder != self.identity and not stale:
            self.reason = f"held by {holder}"
            self.is_leader = False
            return False

        body = self._body(self.identity, renew)
        try:
            if lease is None:
                self.api.create_namespaced_custom_object(
                    group="coordination.k8s.io", version="v1",
                    namespace=self.cfg.namespace, plural="leases", body=body)
            else:
                body["metadata"]["resourceVersion"] = lease["metadata"]["resourceVersion"]
                if spec.get("leaseTransitions"):
                    body["spec"]["leaseTransitions"] = spec["leaseTransitions"] + (
                        0 if holder == self.identity else 1)
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
                body = self._body("", self._now())
                body["metadata"]["resourceVersion"] = lease["metadata"]["resourceVersion"]
                body["spec"]["leaseDurationSeconds"] = 1
                self.api.replace_namespaced_custom_object(
                    group="coordination.k8s.io", version="v1",
                    namespace=self.cfg.namespace, plural="leases",
                    name=self.name, body=body)
        except ApiException:
            pass
        self.is_leader = False
        self.reason = "released"