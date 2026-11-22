#!/bin/bash
# PRE_K6_HOOK for the AURORA-HPA 1 h run.
#
# Order matters: the orchestrator's cold gate asserts a clean namespace with all
# 42 HPAs present, and AURORA deletes those HPAs on start. So it must come up
# after the gate and before k6.
set -euo pipefail
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

IMAGE="${AURORA_IMAGE:-$(cat aurora/data/IMAGE 2>/dev/null)}"
if [ -z "${IMAGE:-}" ]; then
  echo "AURORA_IMAGE is not set and aurora/data/IMAGE is missing" >&2
  echo "run: aurora/deploy/build_and_push.sh" >&2
  exit 1
fi
echo "using image: $IMAGE"

echo "applying aurora rbac + state pvc"
kubectl apply -f aurora/deploy/rbac.yaml
kubectl apply -f aurora/deploy/deployment.yaml >/dev/null

echo "rendering deployment with $IMAGE"
sed "s|__AURORA_IMAGE__|$IMAGE|" aurora/deploy/deployment.yaml | kubectl apply -f -

echo "waiting for the aurora pod"
for i in $(seq 1 48); do
  ready=$(kubectl -n train-ticket get pods -l app=aurora \
            -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || true)
  if [ "$ready" = "true" ]; then
    echo "aurora ready"
    kubectl -n train-ticket get pods -l app=aurora
    kubectl -n train-ticket logs deploy/aurora --tail=8 2>/dev/null || true
    exit 0
  fi
  sleep 5
done

echo "aurora not ready after 240s" >&2
kubectl -n train-ticket get pods -l app=aurora -o wide >&2 || true
kubectl -n train-ticket logs deploy/aurora --tail=40 >&2 || true
exit 1