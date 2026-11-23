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


def _ready_duration(pod):
    """Seconds from pod creation to its first Ready transition."""
    created = pod.metadata.creation_timestamp
    if created is None:
        return None
    for cond in (pod.status.conditions or []):
        if cond.type == "Ready" and cond.last_transition_time is not None:
            return max(0.0, (cond.last_transition_time - created).total_seconds())
    return None


class ScaleClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self._load_config()
        self.apps = client.AppsV1Api()
        self.core = client.CoreV1Api()
        self.autoscaling = client.AutoscalingV1Api()
        # Dynamic access so the *full* HPA object (apiVersion, metrics,
        # behavior) is preserved verbatim, not just min/max/target.
        self.custom = client.CustomObjectsApi()
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
            # Do NOT silently fall back to every Deployment: an RBAC or API
            # error would silently expand the managed set. Fail closed and let
            # the caller decide.
            raise RuntimeError(
                "could not list HPAs in namespace "
                f"{self.cfg.namespace!r}; refusing to guess the managed set")
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
        # Use the Deployment's own selector: container names and the "app" label
        # convention are not generic (many charts use app.kubernetes.io/name).
        selector = None
        try:
            dep = self.apps.read_namespaced_deployment(service, self.cfg.namespace)
            match = (dep.spec.selector or {}).match_labels or {}
            if match:
                selector = ",".join(f"{k}={v}" for k, v in match.items())
        except ApiException:
            pass
        if selector:
            pods = self.core.list_namespaced_pod(
                self.cfg.namespace, label_selector=selector).items
        else:
            pods = [p for p in self.core.list_namespaced_pod(self.cfg.namespace).items
                    if any(o.kind == "ReplicaSet" for o in (p.metadata.owner_references or []))
                    and p.metadata.name.startswith(service)]
        now = time.time()
        ready = 0
        pending = []
        starts = []
        ready_ages = []
        ready_uids = []
        for p in pods:
            if not p.metadata.deletion_timestamp:
                age = now - p.metadata.creation_timestamp.timestamp() \
                    if p.metadata.creation_timestamp else None
                if p.status.phase == "Running":
                    cs = p.status.container_statuses or []
                    if cs and all(c.ready for c in cs):
                        ready += 1
                        # Startup duration is creation -> first Ready, not the
                        # pod's current age. Using age reported a 3 h old pod as
                        # a 3 h "startup" and inflated the p95 without bound.
                        starts.append(_ready_duration(p))
                        ready_ages.append(age)
                        ready_uids.append(p.metadata.uid)
                        continue
                reason = None
                for cs in (p.status.container_statuses or []):
                    st = cs.state
                    if st and st.waiting:
                        reason = st.waiting.reason
                        break
                pending.append({"age_s": age, "reason": reason})
        return ready, pending, starts, ready_ages, ready_uids

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
        """Delete the namespace HPAs, preserving each complete object."""
        if not self.cfg.suspend_hpas:
            return []
        items = []
        try:
            raw = self.custom.list_namespaced_custom_object(
                group="autoscaling", version="v2", namespace=self.cfg.namespace,
                plural="horizontalpodautoscalers")
        except ApiException as exc:
            raise RuntimeError(
                f"cannot list HPAs in {self.cfg.namespace!r}: {exc}") from exc
        for h in raw.get("items", []):
            meta = h.get("metadata", {})
            items.append({
                "name": meta.get("name"),
                "namespace": meta.get("namespace") or self.cfg.namespace,
                "target": ((h.get("spec", {}).get("scaleTargetRef") or {})
                           .get("name")),
                # the COMPLETE object: apiVersion, spec (metrics, targets,
                # behavior), labels and annotations
                "manifest": h,
            })
        self.saved_hpas = items
        self._save_hpa_state(items)
        for it in items:
            try:
                self.custom.delete_namespaced_custom_object(
                    group="autoscaling", version="v2",
                    namespace=it["namespace"],
                    plural="horizontalpodautoscalers", name=it["name"])
            except ApiException as exc:
                if getattr(exc, "status", None) != 404:
                    raise
        return items

    def load_saved_state(self):
        """Load HPA manifests from the state file / ConfigMap.

        Needed because ``self.saved_hpas`` is empty after a process restart;
        restoration must not depend on the original process being alive.
        """
        if self.saved_hpas:
            return self.saved_hpas
        raw = None
        try:
            with open(self.cfg.hpa_state_file) as fh:
                raw = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            try:
                cm = self.core.read_namespaced_config_map(
                    "aurora-hpa-state", self.cfg.namespace)
                raw = json.loads((cm.data or {}).get("hpa_state.json", "null"))
            except ApiException:
                raw = None
        if not raw:
            self.saved_hpas = []
            return []
        self.saved_hpas = raw
        return raw

    def restore_hpas(self):
        """Recreate each HPA from its complete saved object.

        Uses the same path whether called by the controller at shutdown or by
        the external recovery tool after a crash.
        """
        if not self.cfg.restore_hpas:
            return 0
        items = self.load_saved_state()
        for it in items:
            body = it.get("manifest")
            if not body:
                continue
            body = json.loads(json.dumps(body))
            meta = body.setdefault("metadata", {})
            for k in ("resourceVersion", "uid", "creationTimestamp",
                      "managedFields", "generation"):
                meta.pop(k, None)
            meta["name"] = it["name"]
            meta["namespace"] = it["namespace"]
            try:
                self.custom.create_namespaced_custom_object(
                    group="autoscaling", version="v2",
                    namespace=it["namespace"],
                    plural="horizontalpodautoscalers", body=body)
                n += 1
            except ApiException as exc:
                if getattr(exc, "status", None) != 409:
                    raise
        self.saved_hpas = []
        return n
