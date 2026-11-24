#!/bin/bash
# Cold start for an AURORA-R arm: every target at its configured min_replicas.
#
# AURORA-R refuses to write a replica count outside the per-service bounds it
# captured from the live HPAs (min 1/3/4). A blanket `--replicas=1` therefore
# makes every proposal "out of replica bounds", so the cold state has to be the
# configured minimum, not 1.
set -uo pipefail
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

CONFIG=${CONFIG:-aurora_r/deploy/config-trainticket.json}
NS=${NS:-train-ticket}

python3 - "$CONFIG" "$NS" <<'PY'
import json, subprocess, sys
cfg, ns = sys.argv[1], sys.argv[2]
targets = json.load(open(cfg))["targets"]
for t in targets:
    name, want = t["deployment"], int(t["min_replicas"])
    cur = subprocess.run(["kubectl", "-n", ns, "get", "deploy", name,
                          "-o", "jsonpath={.spec.replicas}"],
                         capture_output=True, text=True).stdout.strip()
    if cur == str(want):
        continue
    subprocess.run(["kubectl", "-n", ns, "scale", "deploy", name,
                    "--replicas=%d" % want], capture_output=True, text=True)
    print(f"{name}: {cur or '?'} -> {want}")
PY

echo "waiting for every target to be Ready"
for i in $(seq 1 60); do
  bad=$(kubectl -n "$NS" get pods \
        -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.status.phase}{"|"}{.status.containerStatuses[*].ready}{"\n"}{end}' 2>/dev/null \
        | awk -F'|' '$1 ~ /^ts-/ {bad=($2!="Running"); if($3=="" || $3 ~ /false/) bad=1; if(bad)n++} END{print n+0}')
  [ "${bad:-0}" = "0" ] && { echo "all targets Ready"; exit 0; }
  sleep 10
done
echo "targets not Ready after 600s" >&2
kubectl -n "$NS" get pods -o wide >&2 || true
exit 1