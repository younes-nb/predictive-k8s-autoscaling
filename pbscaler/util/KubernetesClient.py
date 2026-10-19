"""Kubernetes client helpers.

Adaptations vs upstream PBScaler:
  * Loads config automatically: in-cluster ServiceAccount config when running
    as a pod, otherwise a kubeconfig file (``PBS_K8S_CONFIG`` / KUBECONFIG /
    ``~/.kube/config``). Upstream only supported ``load_kube_config`` with a
    hardcoded path.
  * Adds HPA suspend/restore helpers: TrainTicket is deployed with an HPA per
    stateless deployment (CPU 80%, min 1/max 10). HPAs rewrite
    ``.spec.replicas`` every sync and would fight PBScaler's replica patch, so
    PBScaler deletes them at start (state how it should look saved to disk)
    and re-creates them when the run ends (paper runs PBScaler as *the*
    autoscaler; data-collection runs keep them enabled).
"""

import json
import os

from kubernetes import client, config as k8s_config

from config.Config import Config


class KubernetesClient():
    def __init__(self, project_config: Config):
        self.namespace = project_config.namespace
        _load_config(project_config)
        self.core_api = client.CoreV1Api()  # namespace,pod,service,pv,pvc
        self.apps_api = client.AppsV1Api()  # deployment
        self.custom_api = client.CustomObjectsApi()  # hpa

    # Get all microservices (deployments)
    def get_svcs(self):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        svcs = [i.metadata.name for i in ret.items if i.metadata.name != 'loadgenerator']
        svcs.sort()
        return svcs

    # Get stateless microservices（exclude redis，mq，mongo，mysql, and gateway)
    def get_svcs_without_state(self):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        def judge_state_svc(svc):
            state_svcs = ['redis', 'rabbitmq', 'mongo', 'mysql']
            for state_svc in state_svcs:
                if state_svc in svc:
                    return True
            return False
        svcs = [i.metadata.name for i in ret.items
                if not judge_state_svc(i.metadata.name)
                and i.metadata.name != 'loadgenerator']
        svcs.sort()
        return svcs

    def get_svcs_counts(self):
        """Current pod count per deployment."""
        dic = {}
        pod_ret = self.core_api.list_namespaced_pod(self.namespace, watch=False)
        svcs = self.get_svcs()
        for svc in svcs:
            dic[svc] = 0
            for i in pod_ret.items:
                if i.metadata.name.find(svc) != -1:
                    dic[svc] = dic[svc] + 1
        return dic

    def get_svc_count(self, svc):
        ret_deployment = self.apps_api.read_namespaced_deployment_scale(svc, self.namespace)
        return ret_deployment.spec.replicas

    def get_all_counts(self):
        """Replica count straight from the Deployment objects (exact for the
        collector; pod-name matching breaks with long names)."""
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        return {i.metadata.name: (i.spec.replicas or 0) for i in ret.items}

    def all_avaliable(self):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        for item in ret.items:
            if item.status.ready_replicas != item.spec.replicas:
                return False
        return True

    # Determine the status of the service (available?)
    def svcs_avaliable(self, svcs):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        for item in ret.items:
            if item.metadata.name in svcs and item.status.ready_replicas != item.spec.replicas:
                return False
        return True

    def patch_scale(self, svc, count):
        """Set the replica count of a Deployment directly.

        Works only while the deployment's HPA is suspended/absent
        (see suspend_hpas/restore_hpas).
        """
        body = {'spec': {'replicas': int(count)}}
        self.apps_api.patch_namespaced_deployment_scale(
            name=svc, namespace=self.namespace, body=body)

    def update_yaml(self, k8s_yaml):
        os.system('kubectl apply -f %s > temp.log' % k8s_yaml)

    # ---------------- HPA suspend / restore -------------------------------
    def list_hpas(self):
        return self.custom_api.list_namespaced_custom_object(
            group="autoscaling", version="v2", namespace=self.namespace,
            plural="horizontalpodautoscalers")

    def hpa_manifest(self, name: str) -> dict:
        """Raw HPA object in a re-appliable form (metadata cleaned)."""
        hpa = self.custom_api.get_namespaced_custom_object(
            group="autoscaling", version="v2", namespace=self.namespace,
            plural="horizontalpodautoscalers", name=name)
        meta = json.loads(json.dumps(hpa.get("metadata", {})))
        for k in ("managedFields", "uid", "resourceVersion",
                  "creationTimestamp", "generation"):
            meta.pop(k, None)
        meta.pop("annotations", None)
        return {"apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": meta,
                "spec": json.loads(json.dumps(hpa.get("spec", {})))}

    def suspend_hpas(self, state_file: str) -> int:
        """Delete every HPA in the namespace, saving manifests first.

        Returns the number of HPAs suspended. Idempotent: with 0 HPAs present
        and a state file already on disk, this is a no-op (already suspended).
        """
        hpas = self.list_hpas().get("items", [])
        if not hpas:
            already = os.path.exists(state_file)
            print(f"no HPAs present (already suspended: {already}) — nothing to do")
            return 0
        manifests = [self.hpa_manifest(h["metadata"]["name"]) for h in hpas]
        os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(manifests, f, indent=2)
        for h in hpas:
            self.custom_api.delete_namespaced_custom_object(
                group="autoscaling", version="v2", namespace=self.namespace,
                plural="horizontalpodautoscalers",
                name=h["metadata"]["name"])
        print(f"suspended {len(hpas)} HPAs in namespace '{self.namespace}' "
              f"(manifests saved to {state_file})")
        return len(hpas)

    def restore_hpas(self, state_file: str) -> int:
        """Re-create the HPAs deleted by suspend_hpas."""
        if not os.path.exists(state_file):
            print("no HPA state file found, nothing to restore")
            return 0
        with open(state_file) as f:
            manifests = json.load(f)
        existing = {h["metadata"]["name"] for h in self.list_hpas().get("items", [])}
        n = 0
        for m in manifests:
            name = m["metadata"]["name"]
            if name in existing:
                continue
            m["metadata"].pop("finalizers", None)
            self.custom_api.create_namespaced_custom_object(
                group="autoscaling", version="v2", namespace=self.namespace,
                plural="horizontalpodautoscalers", body=m)
            n += 1
        os.remove(state_file)
        print(f"restored {n} HPAs in namespace '{self.namespace}'")
        return n

    def hpa_status(self) -> list:
        hpas = self.list_hpas().get("items", [])
        rows = []
        for h in hpas:
            cur = h.get("status", {}).get("currentReplicas")
            mn = h.get("spec", {}).get("minReplicas", 1)
            mx = h.get("spec", {}).get("maxReplicas")
            rows.append((h["metadata"]["name"], cur, mn, mx))
        return rows


def _load_config(project_config: Config):
    """Prefer in-cluster ServiceAccount; fall back to a kubeconfig file."""
    sa_token = os.path.join(
        os.environ.get("KUBERNETES_SERVICE_ROOT", "/var/run/secrets/kubernetes.io"),
        "serviceaccount/token")
    if os.path.exists(sa_token):
        k8s_config.load_incluster_config()
        return
    kubeconfig = project_config.k8s_config
    if not kubeconfig or not os.path.exists(kubeconfig):
        # last resort: default loading order of the client lib
        k8s_config.load_config()
        return
    k8s_config.load_kube_config(config_file=kubeconfig)
