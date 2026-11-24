#!/bin/bash
# POST_RUN_HOOK for step 4: tear PBScaler down before the orchestrator's cold
# reset. Deleting the Deployment is its clean-exit path (SIGTERM -> restore
# HPAs), so the namespace is back to native HPA before the reset runs.
set -uo pipefail
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

echo "deleting pbscaler deployment"
kubectl -n train-ticket delete deployment pbscaler --timeout=120s --ignore-not-found

echo "waiting for pbscaler pod to disappear"
for i in $(seq 1 24); do
  n=$(kubectl -n train-ticket get pods -l app=pbscaler --no-headers 2>/dev/null | grep -c . || true)
  [ "${n:-0}" = "0" ] && break
  sleep 5
done

# If PBScaler died before restoring, put the HPAs back explicitly.
if ! kubectl -n train-ticket get hpa --no-headers 2>/dev/null | grep -q .; then
  echo "no HPAs present - restoring from hpa_control.py"
  pbscaler/.venv/bin/python pbscaler/scripts/hpa_control.py restore || true
fi
echo "HPA count: $(kubectl -n train-ticket get hpa --no-headers 2>/dev/null | grep -c . || echo 0)"