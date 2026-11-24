#!/bin/bash
# PRE_K6_HOOK for step 4: bring up PBScaler after the orchestrator's cold gate.
#
# Order matters: the cold gate asserts a clean namespace with all 43 HPAs
# present, and PBScaler deletes those HPAs on start. So PBScaler must come up
# after the gate and before k6.
set -euo pipefail
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

echo "applying pbscaler rbac"
kubectl apply -f pbscaler/deploy/rbac.yaml

echo "applying pbscaler deployment (1h manifest)"
kubectl apply -f benchmark/pbscaler_step4/deployment_pbscaler_1h.yaml

echo "waiting for pbscaler pod"
for i in $(seq 1 36); do
  ready=$(kubectl -n train-ticket get pods -l app=pbscaler \
            -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || true)
  if [ "$ready" = "true" ]; then
    echo "pbscaler ready"
    kubectl -n train-ticket get pods -l app=pbscaler
    exit 0
  fi
  sleep 5
done
echo "pbscaler not ready after 180s" >&2
kubectl -n train-ticket get pods -l app=pbscaler -o wide >&2 || true
exit 1