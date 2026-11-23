#!/usr/bin/env python3
"""Deep cold/clean reset of the TrainTicket namespace between experiment runs.

Protocol (matched between runs so every k6 replay starts from the same state):
  1. ensure the 42 HPAs exist (restore from the saved hpa_state file if a
     suspended run left them deleted)  — HPAs stay untouched when present
  2. scale every stateless deployment (ts-*, gateway) to replicas=1
     (the `pbscaler` controller deployment is scaled to 0)
  3. delete the pods of those deployments only (fresh process generation —
     the datastores, not touched)
  4. wait every deployment for spec==available==ready and all pods Ready
  5. summary: replica histogram, HPA count, restart/OOM deltas (should be 0)

Run modes:
    --assert-only   verify cold state without changing anything (exit 1 if hot)
    (default)       perform the full reset

    pbscaler/.venv/bin/python pbscaler/scripts/launchers/reset_cluster_cold.py \
        --kubeconfig /proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf [--assert-only] [--out /tmp/x.json]
"""
import argparse
import collections
import json
import os
import sys
import time
from datetime import datetime, timezone

import kubernetes

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

STATELESS_PREFIXES = ("ts-",)   # the ingress gateway is istio-ingressgateway in istio-system now
STORE_HINTS = ("mongo", "mysql", "zookeeper", "rabbitmq")


def classify(deploy_name: str) -> str:
    if deploy_name == "pbscaler":
        return "controller"
    if any(h in deploy_name for h in STORE_HINTS):
        return "other"
    return "stateless"


def stateless_deployments(v1, ns):
    out = {}
    for d in v1.list_namespaced_deployment(ns).items:
        kind = classify(d.metadata.name)
        if kind == "other":
            continue
        out[d.metadata.name] = {"kind": kind, "spec": d.spec.replicas or 0}
    return out


def rs_owner_map(apps, ns):
    """replica-set name -> deployment name"""
    m = {}
    for rs in apps.list_namespaced_replica_set(ns).items:
        refs = rs.metadata.owner_references or []
        for ref in refs:
            if ref.kind == "Deployment":
                m[rs.metadata.name] = ref.name
                break
    return m


def pod_owner(pod):
    for ref in (pod.metadata.owner_references or []):
        if ref.kind == "ReplicaSet":
            return ref.name
    return None


def snapshot_state(v1, apps, autos, ns, hpa_count_expected=43):
    deps = stateless_deployments(apps, ns)
    hist = collections.Counter(d["spec"] for d in deps.values())
    hpas = autos.list_namespaced_horizontal_pod_autoscaler(ns).items
    pods = v1.list_namespaced_pod(ns).items
    # Count only pods whose CURRENT (not historical lastState) container state is
    # OOMKilled, i.e. a pod that is failing right now. Historical OOMKilled pods
    # (e.g. a calico-node that died days ago, or a mongo pod from a failed restore
    # that has since been recreated) must not fail the reset gate, otherwise the
    # gate is impossible to pass on this cluster.
    oom = sum(1 for p in pods for cs in (p.status.container_statuses or [])
              if cs.state is not None and cs.state.terminated is not None
              and cs.state.terminated.reason == "OOMKilled")
    restarts = sum(cs.restart_count or 0 for p in pods
                   for cs in (p.status.container_statuses or []))
    not_running = [p.metadata.name for p in pods if p.status.phase != "Running"]
    unready = []
    for p in pods:
        conds = [c for c in (p.status.conditions or []) if c.type == "Ready"]
        if conds and conds[0].status != "True":
            unready.append(p.metadata.name)
    return {
        "ts": datetime.now(timezone.utc).isoformat() + "Z",
        "deployments_stateless": sum(1 for d in deps.values() if d["kind"] == "stateless"),
        "replica_histogram": {"%d" % k: v for k, v in sorted(hist.items())},
        "pbscaler_replicas": deps.get("pbscaler", {}).get("spec"),
        "hpa_count": len(hpas),
        "pods_total": len(pods),
        "pods_not_running": not_running,
        "pods_not_ready": unready,
        "oomkilled": oom,
        "container_restarts": restarts,
    }


def wait_ready(v1, apps, ns, stateless_names, timeout_s=900, poll_s=10):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        deps = {d.metadata.name: d for d in apps.list_namespaced_deployment(ns).items}
        bad = []
        for name in stateless_names:
            d = deps.get(name)
            if d is None:
                bad.append(name + ":missing")
                continue
            spec = d.spec.replicas or 0
            avail = (d.status.available_replicas or 0)
            ready = (d.status.ready_replicas or 0)
            if not (spec == 1 and avail == 1 and ready == 1):
                bad.append("%s:%s/%s/%s" % (name, spec, avail, ready))
        if not bad:
            return True, list(bad)
        time.sleep(poll_s)
    return False, []  # caller re-checks for details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kubeconfig", default=os.environ.get("KUBECONFIG",
                    os.path.expanduser("~/admin.conf")))
    ap.add_argument("--namespace", default="train-ticket")
    ap.add_argument("--assert-only", action="store_true")
    ap.add_argument("--hpa-expected", type=int, default=42,
                    help="expected HPA count in train-ticket (43 stateless deployments minus "
                         "the ingress gateway, which is istio-ingressgateway in istio-system since 2026-11-21)")
    ap.add_argument("--wait-timeout", type=int, default=900)
    ap.add_argument("--out", default=None, help="write summary json here")
    args = ap.parse_args()

    kubernetes.config.load_kube_config(config_file=args.kubeconfig)
    core = kubernetes.client.CoreV1Api()
    apps = kubernetes.client.AppsV1Api()
    autos = kubernetes.client.AutoscalingV1Api()
    ns = args.namespace

    pre = snapshot_state(core, apps, autos, ns, args.hpa_expected)
    pre.pop("cold_ok", None)
    print("PRE : " + json.dumps(pre, indent=None))

    before_restarts = pre["container_restarts"]

    if args.assert_only:
        hpas_ok = pre["hpa_count"] >= args.hpa_expected
        clean = (pre["replica_histogram"].get("1", 0) == pre["deployments_stateless"]
                 and set(pre["replica_histogram"]) <= {"0", "1"}
                 and pre["pbscaler_replicas"] in (0, None)
                 and hpas_ok and not pre["pods_not_running"] and not pre["pods_not_ready"]
                 and pre["oomkilled"] == 0)
        print("ASSERT cold_ok=%s (histogram=%s hpas=%s not_running=%d not_ready=%d)"
              % (clean, pre["replica_histogram"], pre["hpa_count"],
                 len(pre["pods_not_running"]), len(pre["pods_not_ready"])))
        if args.out:
            json.dump({**pre, "cold_ok": bool(clean)}, open(args.out, "w"), indent=2)
        return 0 if clean else 1

    managed = aurora_manages(apps, ns)
    if managed:
        # AURORA is the replica writer. Scaling the workloads behind its back
        # would make its next decision inconsistent with what it observed, and
        # the HPAs are supposed to be absent while it runs.
        print("AURORA is managing this namespace -- skipping replica/HPA reset")
        deps = {}
    else:
        deps = stateless_deployments(apps, ns)
    stateless_names = [n for n, d in deps.items() if d["kind"] == "stateless"]

    # 1) HPAs: only restore when some were left deleted by a crashed run
    hpas = autos.list_namespaced_horizontal_pod_autoscaler(ns).items
    if not managed and len(hpas) < args.hpa_expected:
        state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "data", "hpa_state.json")
        print("HPAs missing (%d < %d) — restoring from %s" % (len(hpas), args.hpa_expected, state_file))
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config.Config import Config                      # noqa: E402
        from util.KubernetesClient import KubernetesClient    # noqa: E402
        k8s = KubernetesClient(Config())
        k8s.restore_hpas(state_file)
        time.sleep(3)
    else:
        print("HPAs intact (%d)" % len(hpas))

    # 2) scale to targets
    for name, d in deps.items():
        target = 0 if d["kind"] == "controller" else 1
        if d["spec"] != target:
            apps.patch_namespaced_deployment_scale(
                name, ns, {"spec": {"replicas": target}})
    print("scaled: pbscaler->0, %d stateless ->1" % len(stateless_names))
    time.sleep(10)

    # 3) delete stateless pods once (fresh generation; stores untouched)
    rs2dep = rs_owner_map(apps, ns)
    pods = core.list_namespaced_pod(ns).items
    doomed = []
    for p in pods:
        rs = pod_owner(p)
        dep = rs2dep.get(rs)
        if dep in stateless_names:
            doomed.append(p.metadata.name)
    BATCH = 30
    for i in range(0, len(doomed), BATCH):
        chunk = doomed[i:i + BATCH]
        for pn in chunk:
            try:
                core.delete_namespaced_pod(pn, ns)
            except kubernetes.client.rest.ApiException as e:
                if e.status != 404:
                    print("delete %s failed: %s" % (pn, e.status))
        time.sleep(2)
    print("deleted %d stateless pods" % len(doomed))

    # 4) wait ready
    ok, _ = wait_ready(core, apps, ns, stateless_names, args.wait_timeout)
    print("wait_ready: %s" % ("OK" if ok else "TIMEOUT — see summary below"))

    post = snapshot_state(core, apps, autos, ns, args.hpa_expected)
    post.pop("cold_ok", None)
    post["restart_delta"] = post["container_restarts"] - before_restarts
    n_stateless = sum(1 for d in deps.values() if d["kind"] == "stateless")
    hist_ok = post["replica_histogram"] == {"1": n_stateless}
    # pbscaler is not a Deployment in this namespace when it is not deployed;
    # absent == 0 replicas (it must not be serving during calibration).
    pbscaler_ok = post["pbscaler_replicas"] in (0, None)
    hpas_ok = post["hpa_count"] >= args.hpa_expected
    # Pod recreation legitimately resets restart counters downward, so the
    # delta must be <= 0; a positive delta means survivng pods accumulated restarts.
    restarts_ok = (post["restart_delta"] or 0) <= 0
    reasons = []
    if not hist_ok: reasons.append("replica_histogram=%s expected 1x%d" % (post["replica_histogram"], n_stateless))
    if not pbscaler_ok: reasons.append("pbscaler_replicas=%s" % post["pbscaler_replicas"])
    if not hpas_ok: reasons.append("hpa_count=%s < %s" % (post["hpa_count"], args.hpa_expected))
    if post["pods_not_running"]: reasons.append("pods_not_running=%d" % len(post["pods_not_running"]))
    if post["pods_not_ready"]: reasons.append("pods_not_ready=%d" % len(post["pods_not_ready"]))
    if post["oomkilled"]: reasons.append("oomkilled=%d" % post["oomkilled"])
    if not restarts_ok: reasons.append("restart_delta=%s > 0" % post["restart_delta"])
    clean_ok = not reasons
    post["reset_reasons"] = reasons
    post["reset_ok"] = bool(clean_ok)
    print("POST: " + json.dumps(post, indent=None))
    if args.out:
        json.dump({**post, "pre": pre}, open(args.out, "w"), indent=2)
    return 0 if clean_ok else 1


if __name__ == "__main__":
    sys.exit(main())
