"""Deployment and service profiles, persisted to disk.

Two levels:

  * DeploymentProfile -- identity, service inventory, topology, resource
    configuration, feature availability, drift state. Built once per deployment
    by discovery; never copied from another deployment.
  * ServiceProfile -- per-service baselines, capacity posterior, prototype
    calibration state, startup observations, drift history.

Both survive a controller restart (``profiles.json`` on the ``aurora-state``
PVC), so learning is not lost when the pod restarts.
"""

import json
import os
import time


class ServiceProfile:
    def __init__(self, service):
        self.service = service
        self.created = time.time()
        self.updated = self.created
        self.n_observations = 0
        self.capacity = None
        self.capacity_sigma = None
        self.bottleneck_counts = {}
        self.addressable_ema = None
        self.startup_samples = []
        self.startup_p95 = None
        self.available_features = []
        self.drift = {"input": 0.0, "capacity": 0.0, "config": False,
                      "topology": False, "missing_ticks": 0}

    def touch(self):
        self.updated = time.time()

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        s = cls(d.get("service", ""))
        s.__dict__.update(d)
        return s


class DeploymentProfile:
    def __init__(self, namespace, deployment_id=None):
        self.namespace = namespace
        self.deployment_id = deployment_id or namespace
        self.created = time.time()
        self.updated = self.created
        self.services = {}
        self.topology = None
        self.available_metrics = []
        self.resource_config = {}
        self.image_versions = {}
        self.drift = {"input": False, "capacity": False, "config": False,
                      "topology": False, "schema": False}
        self.calibration = {}      # service -> {classes: {n: int}, done: bool}

    # ------------------------------------------------------------------
    def service(self, name):
        if name not in self.services:
            self.services[name] = ServiceProfile(name)
        return self.services[name]

    def touch(self):
        self.updated = time.time()

    def to_dict(self):
        return {
            "namespace": self.namespace,
            "deployment_id": self.deployment_id,
            "created": self.created,
            "updated": self.updated,
            "services": {k: v.to_dict() for k, v in self.services.items()},
            "topology": json.loads(self.topology.dumps()) if self.topology else None,
            "available_metrics": self.available_metrics,
            "resource_config": self.resource_config,
            "image_versions": self.image_versions,
            "drift": self.drift,
            "calibration": self.calibration,
        }

    @classmethod
    def from_dict(cls, d):
        p = cls(d.get("namespace", ""), d.get("deployment_id"))
        p.__dict__.update({k: v for k, v in d.items() if k != "services"})
        p.services = {k: ServiceProfile.from_dict(v)
                      for k, v in d.get("services", {}).items()}
        if d.get("topology"):
            from graph.topology import Topology
            p.topology = Topology.from_dict(d["topology"])
        return p


class ProfileStore:
    """Load/save the deployment profile atomically."""

    def __init__(self, path):
        self.path = path
        self.profile = None

    def exists(self):
        return os.path.exists(self.path)

    def load(self):
        with open(self.path) as fh:
            self.profile = DeploymentProfile.from_dict(json.load(fh))
        return self.profile

    def save(self, profile):
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(profile.to_dict(), fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
        self.profile = profile