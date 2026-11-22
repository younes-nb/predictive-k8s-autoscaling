"""Kubernetes integration: the Scale subresource, and HPA suspend/restore.

AURORA-HPA is the **single replica writer** for every workload it manages. The
native HPAs are suspended for the duration so the two cannot fight over
``.spec.replicas``; their manifests are saved to disk and a ConfigMap so a
controller crash still leaves a recoverable state.
"""

import json
import os
import time

from kubernetes import client, config as k8s_config
from kubernetes.client.rest import ApiException


class ScaleClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self._load_config()
        self.apps = client.AppsV1Api()
        self.core = client.CoreV1Api()
        self.autoscaling = client.AutoscalingV1Api()
        self.saved_hpas = []

    def _load_config(self):
        try:
            k8s_config.load_incluster_config()
            self.in_cluster = True
        except Exception:
            k8s_config.load_kube_config(config_file=self.cfg.k8s_config)
            self.in_cluster = False

    # ------------------------------------------------------------------
    def deployments(self):
        return self.apps.list_namespaced_deployment(self.cfg.namespace)

    def scalable_services(self):
        """Deployments that have an HPA (i.e. the autoscaler's managed set)."""
        names = set()
        try:
            for h in self.autoscaling.list_namespaced_horizontal_pod_autoscaler(
                    self.cfg.namespace).items:
                ref = h.spec.scale_target_ref
                if ref.kind == "Deployment":
                    names.add(ref.name)
        except ApiException:
            pass
        if not names:
            names = {d.metadata.name for d in self.deployments().items}
        return sorted(names)

    def get_scale(self, service):
        r = self.apps.read_namespaced_deployment_scale(service, self.cfg.namespace)
        return r.spec.replicas or 0

    def set_scale(self, service, replicas):
        body = {"spec": {"replicas": int(replicas)}}
        return self.apps.patch_namespaced_deployment_scale(
            service, self.cfg.namespace, body)

    # ------------------------------------------------------------------
    def pod_counts(self, service):
        """Return (ready, pending list of {age_s, reason}, spec, start_times)."""
        pods = self.core.list_namespaced_pod(
            self.cfg.namespace,
            label_selector=f"app={service}").items
        # fall back to a name prefix if the label is not "app"
        if not pods:
            pods = [p for p in self.core.list_namespaced_pod(self.cfg.namespace).items
                    if p.metadata.name.startswith(service)]
        now = time.time()
        ready = 0
        pending = []
        starts = []
        for p in pods:
            if not p.metadata.deletion_timestamp:
                age = now - p.metadata.creation_timestamp.timestamp() \
                    if p.metadata.creation_timestamp else None
                if p.status.phase == "Running":
                    cs = p.status.container_statuses or []
                    if cs and all(c.ready for c in cs):
                        ready += 1
                        starts.append(age)
                        continue
                reason = None
                for cs in (p.status.container_statuses or []):
                    st = cs.state
                    if st and st.waiting:
                        reason = st.waiting.reason
                        break
                pending.append({"age_s": age, "reason": reason})
        return ready, pending, starts

    def spec_replicas(self, service):
        d = self.apps.read_namespaced_deployment(service, self.cfg.namespace)
        return d.spec.replicas or 0

    # ------------------------------------------------------------------
    # HPA suspend / restore
    # ------------------------------------------------------------------
    def _save_hpa_state(self, items):
        os.makedirs(os.path.dirname(self.cfg.hpa_state_file) or ".", exist_ok=True)
        tmp = self.cfg.hpa_state_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(items, fh, indent=2)
        os.replace(tmp, self.cfg.hpa_state_file)
        self._mirror_state_to_configmap(items)

    def _mirror_state_to_configmap(self, items):
        cm_name = "aurora-hpa-state"
        body = {"apiVersion": "v1", "kind": "ConfigMap",
                "metadata": {"name": cm_name, "namespace": self.cfg.namespace},
                "data": {"hpa_state.json": json.dumps(items)}}
        try:
            self.core.create_namespaced_config_map(self.cfg.namespace, body)
        except ApiException as exc:
            if exc.status == 409:
                self.core.replace_namespaced_config_map(cm_name, self.cfg.namespace, body)

    def suspend_hpas(self):
        if not self.cfg.suspend_hpas:
            return []
        items = []
        for h in self.autoscaling.list_namespaced_horizontal_pod_autoscaler(
                self.cfg.namespace).items:
            items.append({
                "name": h.metadata.name,
                "namespace": h.metadata.namespace,
                "min": h.spec.min_replicas,
                "max": h.spec.max_replicas,
                "target": h.spec.scale_target_ref.name,
            })
        self.saved_hpas = items
        self._save_hpa_state(items)
        for it in items:
            try:
                self.autoscaling.delete_namespaced_horizontal_pod_autoscaler(
                    it["name"], it["namespace"])
            except ApiException as exc:
                if exc.status != 404:
                    raise
        return items

    def restore_hpas(self):
        if not self.cfg.restore_hpas:
            return 0
        n = 0
        for it in self.saved_hpas:
            body = {
                "apiVersion": "autoscaling/v1",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {"name": it["name"], "namespace": it["namespace"]},
                "spec": {
                    "scaleTargetRef": {"apiVersion": "apps/v1",
                                       "kind": "Deployment",
                                       "name": it["target"]},
                    "minReplicas": it.get("min") or 1,
                    "maxReplicas": it.get("max") or 1,
                },
            }
            try:
                self.autoscaling.create_namespaced_horizontal_pod_autoscaler(
                    it["namespace"], body)
                n += 1
            except ApiException as exc:
                if exc.status != 409:
                    raise
        self.saved_hpas = []
        return n