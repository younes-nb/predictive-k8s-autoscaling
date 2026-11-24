#!/usr/bin/env python3
"""Save and restore the namespace HPAs around an AURORA-R run.

AURORA-R refuses to act while a native HPA targets one of its managed
Deployments (``KubernetesReader.assert_single_writer``), so the run hook must
remove them and the post-run hook must put them back. The complete autoscaling
manifest is preserved (metrics, behaviour, min/max), not just the target.

    python3 -m aurora_r.tools.hpa_state save   [--namespace NS]
    python3 -m aurora_r.tools.hpa_state restore [--namespace NS]
    python3 -m aurora_r.tools.hpa_state status  [--namespace NS]
"""

import argparse
import json
import os
import sys
import tempfile
import time

CM_NAME = "aurora-r-hpa-state"
STATE_KEY = "hpa_state.json"


def _client():
    from kubernetes import client, config as kc
    try:
        kc.load_incluster_config()
    except Exception:
        kc.load_kube_config()
    return client


def _managed_targets(cfg_path):
    """Deployment names AURORA-R is allowed to write."""
    if not cfg_path or not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    return {t["deployment"] for t in cfg.get("targets", [])}


def save(namespace, targets=None):
    c = _client()
    custom = c.CustomObjectsApi()
    core = c.CoreV1Api()
    raw = custom.list_namespaced_custom_object(
        group="autoscaling", version="v2", namespace=namespace,
        plural="horizontalpodautoscalers")
    items = []
    for h in raw.get("items", []):
        spec = h.get("spec", {}) or {}
        ref = spec.get("scaleTargetRef") or {}
        if ref.get("kind") not in (None, "Deployment"):
            continue
        if targets and ref.get("name") not in targets:
            continue
        items.append({
            "name": h["metadata"]["name"],
            "namespace": h["metadata"].get("namespace") or namespace,
            "target": ref.get("name"),
            "manifest": h,
        })
    if not items:
        print(f"[hpa] no managed HPAs found in {namespace}")
        return 0
    state = os.environ.get("AURORA_R_HPA_STATE", "/var/lib/aurora-r/hpa_state.json")
    os.makedirs(os.path.dirname(state) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(state) or ".", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(items, fh)
    os.replace(tmp, state)
    body = {"apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": CM_NAME, "namespace": namespace},
            "data": {STATE_KEY: json.dumps(items)}}
    try:
        core.create_namespaced_config_map(namespace, body)
    except Exception:
        cur = core.read_namespaced_config_map(CM_NAME, namespace)
        body["metadata"]["resourceVersion"] = cur.metadata.resource_version
        core.replace_namespaced_config_map(CM_NAME, namespace, body)
    deleted = 0
    for it in items:
        try:
            custom.delete_namespaced_custom_object(
                group="autoscaling", version="v2", namespace=it["namespace"],
                plural="horizontalpodautoscalers", name=it["name"])
            deleted += 1
        except Exception as exc:
            if getattr(exc, "status", None) != 404:
                raise
    print(f"[hpa] saved {len(items)} manifests, suspended {deleted}")
    return len(items)


def _load(namespace):
    for path in (os.environ.get("AURORA_R_HPA_STATE", "/var/lib/aurora-r/hpa_state.json"),):
        if path and os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    c = _client()
    try:
        cm = c.CoreV1Api().read_namespaced_config_map(CM_NAME, namespace)
        return json.loads((cm.data or {}).get(STATE_KEY) or "null")
    except Exception:
        return None


def restore(namespace):
    items = _load(namespace)
    if not items:
        print("[hpa] no saved state; nothing to restore")
        return 0
    c = _client()
    custom = c.CustomObjectsApi()
    core = c.CoreV1Api()
    restored = 0
    for it in items:
        body = it.get("manifest")
        if not body:
            continue
        try:
            custom.get_namespaced_custom_object(
                group="autoscaling", version="v2", namespace=it["namespace"],
                plural="horizontalpodautoscalers", name=it["name"])
            restored += 1
            continue
        except Exception as exc:
            if getattr(exc, "status", None) != 404:
                raise
        body = json.loads(json.dumps(body))
        meta = body.setdefault("metadata", {})
        for k in ("resourceVersion", "uid", "creationTimestamp",
                  "managedFields", "generation"):
            meta.pop(k, None)
        meta["name"] = it["name"]
        meta["namespace"] = it["namespace"]
        try:
            custom.create_namespaced_custom_object(
                group="autoscaling", version="v2", namespace=it["namespace"],
                plural="horizontalpodautoscalers", body=body)
            restored += 1
        except Exception as exc:
            if getattr(exc, "status", None) != 409:
                raise
    if restored >= len(items):
        try:
            core.delete_namespaced_config_map(CM_NAME, namespace)
        except Exception:
            pass
        try:
            os.remove(os.environ.get("AURORA_R_HPA_STATE",
                                     "/var/lib/aurora-r/hpa_state.json"))
        except OSError:
            pass
    print(f"[hpa] restored {restored}/{len(items)} HPAs (complete manifests)")
    return restored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("save", "restore", "status"))
    ap.add_argument("--namespace", default="train-ticket")
    ap.add_argument("--config", default="",
                    help="AURORA-R config; only HPAs for its targets are saved")
    a = ap.parse_args()
    targets = _managed_targets(a.config)
    if a.action == "save":
        return 0 if save(a.namespace, targets) >= 0 else 1
    if a.action == "restore":
        return 0
    c = _client()
    hpas = c.AutoscalingV2Api().list_namespaced_horizontal_pod_autoscaler(a.namespace)
    saved = _load(a.namespace) or []
    print(f"[hpa] live={len(hpas.items)} saved={len(saved)}")


if __name__ == "__main__":
    sys.exit(main())