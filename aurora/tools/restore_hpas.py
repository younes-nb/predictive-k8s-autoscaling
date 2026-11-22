"""Restore the namespace HPAs from the aurora-hpa-state ConfigMap.

Used by the POST_RUN_HOOK when the controller died before restoring them, so a
crashed run can never leave the namespace without HPAs.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                              # noqa: E402
from k8s.scale_client import ScaleClient               # noqa: E402


def main():
    cfg = Config()
    sc = ScaleClient(cfg)
    core = sc.core
    try:
        cm = core.read_namespaced_config_map("aurora-hpa-state", cfg.namespace)
    except Exception as exc:                             # noqa: BLE001
        print(f"[restore] no aurora-hpa-state ConfigMap ({exc}); nothing to do")
        return 0

    raw = (cm.data or {}).get("hpa_state.json")
    if not raw:
        print("[restore] ConfigMap has no hpa_state.json")
        return 0
    items = json.loads(raw)
    restored = 0
    for it in items:
        body = {
            "apiVersion": "autoscaling/v1",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": it["name"], "namespace": it["namespace"]},
            "spec": {
                "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment",
                                   "name": it["target"]},
                "minReplicas": it.get("min") or 1,
                "maxReplicas": it.get("max") or 1,
            },
        }
        try:
            sc.autoscaling.create_namespaced_horizontal_pod_autoscaler(
                it["namespace"], body)
            restored += 1
        except Exception as exc:                         # noqa: BLE001
            if getattr(exc, "status", None) != 409:
                print(f"[restore] {it['name']}: {exc}")
    print(f"[restore] recreated {restored}/{len(items)} HPAs")
    return restored


if __name__ == "__main__":
    main()