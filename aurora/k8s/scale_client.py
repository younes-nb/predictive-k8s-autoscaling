"""Kubernetes integration: the Scale subresource, and HPA suspend/restore.

AURORA-HPA is the **single replica writer** for every workload it manages. The
native HPAs are suspended for the duration so the two cannot fight over
``.spec.replicas``; their manifests are saved to disk and a ConfigMap so a
controller crash still leaves a recoverable state.

Suspension is transactional: if any deletion fails part-way, the HPAs already
deleted are restored immediately and the saved state is kept so a later process
can finish the job. A restart that finds the HPAs absent does not clear the
restoration obligation -- it inherits it.
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
        # Set by suspend_hpas(): the HPAs this process deleted, so a partial
        # failure can be rolled back without touching anyone else's.
        self._deleted_by_us = []

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
        if names:
            return sorted(names)
        # The HPAs are normally absent while AURORA is running (it suspends
        # them), so a restart must recover the managed set from the same
        # ConfigMap it uses for HPA state -- not guess it.
        try:
            if os.path.exists(self.cfg.hpa_state_file):
                disk = json.load(open(self.cfg.hpa_state_file))
                rec = {i.get("target") for i in disk if i.get("target")}
                if rec:
                    print(f"[aurora] managed set recovered from disk: {len(rec)}")
                    return sorted(rec)
        except Exception:
            pass
        try:
            cm = self.core.read_namespaced_config_map(
                "aurora-hpa-state", self.cfg.namespace)
            saved = json.loads((cm.data or {}).get("hpa_state.json", "null"))
            if saved:
                recovered = {i.get("target") for i in saved if i.get("target")}
                if recovered:
                    print(f"[aurora] managed set recovered from ConfigMap: "
                          f"{len(recovered)}")
                    return sorted(recovered)
        except Exception:
            pass
        raise RuntimeError(
            "no HPAs and no saved state in namespace "
            f"{self.cfg.namespace!r}; refusing to guess the managed set")

    def get_scale(self, service):
        r = self.apps.read_namespaced_deployment_scale(service, self.cfg.namespace)
        return r.spec.replicas or 0

    def set_scale(self, service, replicas):
        body = {"spec": {"replicas": int(replicas)}}
        return self.apps.patch_namespaced_deployment_scale(
            service, self.cfg.namespace, body)

    # ------------------------------------------------------------------
    def pod_counts(self, service):
        """Return (ready, pending list of {age_s, reason}, spec, start_times,
        ready_uids)."""
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

    def deployment_image(self, service):
        try:
            d = self.apps.read_namespaced_deployment(service, self.cfg.namespace)
            return d.spec.template.spec.containers[0].image
        except (ApiException, IndexError):
            return None

    # ------------------------------------------------------------------
    # HPA suspend / restore
    # ------------------------------------------------------------------
    def _save_hpa_state(self, items):
        # Never overwrite a good saved state with an empty one: a restart that
        # finds no HPAs (because a previous run suspended them) must not erase
        # the record of what to restore.
        if not items:
            print("[aurora] no HPAs to save; keeping the existing state")
            return
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
                # Carry the current resourceVersion: an update without it is
                # rejected, and a blind create would orphan the old object.
                cur = self.core.read_namespaced_config_map(cm_name, self.cfg.namespace)
                body["metadata"]["resourceVersion"] = cur.metadata.resource_version
                self.core.replace_namespaced_config_map(cm_name, self.cfg.namespace, body)

    def _delete_hpa(self, namespace, name):
        try:
            self.custom.delete_namespaced_custom_object(
                group="autoscaling", version="v2", namespace=namespace,
                plural="horizontalpodautoscalers", name=name)
            return True
        except ApiException as exc:
            if getattr(exc, "status", None) == 404:
                return True          # already gone
            raise

    def _create_hpa(self, namespace, manifest):
        body = json.loads(json.dumps(manifest))
        meta = body.setdefault("metadata", {})
        for k in ("resourceVersion", "uid", "creationTimestamp",
                  "managedFields", "generation"):
            meta.pop(k, None)
        meta["namespace"] = namespace
        self.custom.create_namespaced_custom_object(
            group="autoscaling", version="v2", namespace=namespace,
            plural="horizontalpodautoscalers", body=body)

    def suspend_hpas(self, managed_targets=None):
        """Suspend only the HPAs whose scaleTargetRef is a managed Deployment.

        HPAs for StatefulSets, other applications or unmanaged workloads are
        left untouched -- deleting them would break workloads AURORA does not
        own.

        The operation is transactional. If a deletion fails after some HPAs
        were removed, those HPAs are recreated from their saved manifests and
        the saved state is left in place so a later process can retry. Either
        the namespace is fully suspended or it is back as it was.
        """
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
        managed = set(managed_targets or [])
        for h in raw.get("items", []):
            meta = h.get("metadata", {})
            ref = ((h.get("spec", {}).get("scaleTargetRef") or {}))
            if ref.get("kind") not in (None, "Deployment"):
                continue
            if managed and ref.get("name") not in managed:
                continue
            spec = h.get("spec", {}) or {}
            items.append({
                "name": meta.get("name"),
                "namespace": meta.get("namespace") or self.cfg.namespace,
                "target": ((spec.get("scaleTargetRef") or {}).get("name")),
                # original operational bounds -- AURORA must not exceed these
                "min": spec.get("minReplicas"),
                "max": spec.get("maxReplicas"),
                # the COMPLETE object: apiVersion, spec (metrics, targets,
                # behavior), labels and annotations
                "manifest": h,
            })
        if not items:
            return []
        self.saved_hpas = items
        # Persist BEFORE the first deletion: a crash between the save and the
        # delete must still leave a recoverable record.
        self._save_hpa_state(items)
        self._deleted_by_us = []
        try:
            for it in items:
                self._delete_hpa(it["namespace"], it["name"])
                self._deleted_by_us.append(it)
        except Exception:
            # Roll back whatever we already removed, then re-raise. The saved
            # state stays on disk: if the rollback itself is incomplete, a later
            # process still knows what the original HPAs were.
            restored, failed = self._restore_items(self._deleted_by_us)
            print(f"[aurora] suspend aborted: rolled back {restored} HPA(s)"
                  + (f", {len(failed)} could not be restored" if failed else ""))
            if failed:
                print("[aurora] recovery state preserved: "
                      f"{[f['name'] for f in failed]}")
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

    def _restore_items(self, items):
        """Recreate each item from its complete saved object.

        Returns ``(restored, failed)``. Idempotent: an HPA that already exists
        counts as restored.
        """
        restored, failed = 0, []
        for it in items:
            body = it.get("manifest")
            if not body:
                failed.append(it)
                continue
            try:
                self.custom.get_namespaced_custom_object(
                    group="autoscaling", version="v2",
                    namespace=it["namespace"],
                    plural="horizontalpodautoscalers", name=it["name"])
                restored += 1
                continue
            except ApiException as exc:
                if getattr(exc, "status", None) != 404:
                    failed.append(it)
                    continue
            try:
                self._create_hpa(it["namespace"], body)
                restored += 1
            except ApiException as exc:
                if getattr(exc, "status", None) == 409:
                    restored += 1
                else:
                    failed.append(it)
        return restored, failed

    def restore_hpas(self):
        """Recreate every saved HPA from its complete manifest.

        Uses the same path whether called by the controller at shutdown, by the
        startup recovery path, or by the external recovery tool after a crash.
        The saved state is cleared only when every HPA is confirmed present.
        """
        if not self.cfg.restore_hpas:
            return 0
        items = self.load_saved_state()
        if not items:
            return 0
        restored, failed = self._restore_items(items)
        if failed:
            names = [f["name"] for f in failed]
            print(f"[aurora] restore incomplete ({restored}/{len(items)}); "
                  f"state kept for: {names}")
            self.saved_hpas = items
            return restored
        # Verified: clear the recovery record so a later restart does not
        # resurrect HPAs that were deliberately removed.
        self.saved_hpas = []
        self._clear_saved_state()
        return restored

    def _clear_saved_state(self):
        try:
            os.remove(self.cfg.hpa_state_file)
        except OSError:
            pass
        try:
            self.core.delete_namespaced_config_map(
                "aurora-hpa-state", self.cfg.namespace)
        except ApiException:
            pass

    # ------------------------------------------------------------------
    def pending_recovery(self):
        """True when the saved state describes HPAs that are not present.

        A restart that finds the namespace without HPAs while a saved manifest
        exists has inherited an obligation to restore them. Acting on that is
        what makes a crash before the first successful restore recoverable.
        """
        items = self.load_saved_state()
        if not items:
            return False
        for it in items:
            try:
                self.custom.get_namespaced_custom_object(
                    group="autoscaling", version="v2",
                    namespace=it["namespace"],
                    plural="horizontalpodautoscalers", name=it["name"])
            except ApiException as exc:
                if getattr(exc, "status", None) == 404:
                    return True
                raise
        return False

    def bounds_from_saved_state(self):
        """Per-service (min, max) from the saved HPA manifests.

        On a restart the live HPAs are usually absent, so the bounds used to
        size the replica model must come from the same manifests that will be
        restored -- otherwise every service silently falls back to the global
        configuration limits.
        """
        bounds = {}
        for it in self.load_saved_state():
            manifest = it.get("manifest") or {}
            target = it.get("target") or (
                (manifest.get("spec", {}).get("scaleTargetRef") or {}).get("name"))
            if not target:
                continue
            spec = manifest.get("spec", {}) or {}
            lo = spec.get("minReplicas")
            hi = spec.get("maxReplicas")
            bounds[target] = (int(lo) if lo is not None else 1,
                              int(hi) if hi is not None else 1)
        return bounds