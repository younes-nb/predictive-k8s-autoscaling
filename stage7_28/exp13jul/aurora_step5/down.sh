#!/bin/bash
# POST_RUN_HOOK for the AURORA-HPA 1 h run: tear it down before the
# orchestrator's cold reset. Deleting the Deployment is its clean-exit path
# (SIGTERM -> restore HPAs), so the namespace is back to native HPA before the
# reset runs.
set -uo pipefail
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

echo "deleting aurora deployment"
kubectl -n train-ticket delete deployment aurora --timeout=120s --ignore-not-found

echo "waiting for the aurora pod to disappear"
for i in $(seq 1 24); do
  n=$(kubectl -n train-ticket get pods -l app=aurora --no-headers 2>/dev/null | grep -c . || true)
  [ "${n:-0}" = "0" ] && break
  sleep 5
done

# If AURORA died before restoring, put the HPAs back from its ConfigMap.
if ! kubectl -n train-ticket get hpa --no-headers 2>/dev/null | grep -q .; then
  echo "no HPAs present - restoring from the aurora-hpa-state ConfigMap"
  aurora/.venv/bin/python aurora/tools/restore_hpas.py || true
fi
echo "HPA count: $(kubectl -n train-ticket get hpa --no-headers 2>/dev/null | grep -c . || echo 0)"