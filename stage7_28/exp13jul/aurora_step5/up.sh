#!/bin/bash
# PRE_K6_HOOK for AURORA development runs.
#
# AURORA stays in the cluster across runs and owns the replica count: it
# suspends the namespace HPAs on start and restores them on exit. So this hook
# is idempotent -- it (re)applies the RBAC/PVC, makes sure the Deployment is
# running the image in aurora/data/IMAGE, and waits for the pod to be Ready.
#
# It deliberately does NOT delete and recreate AURORA, and it does NOT bring the
# HPAs back: AURORA already holds them suspended, which is the state a
# development run is supposed to start from.
set -uo pipefail
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

# Only re-render when the image actually changed: an unnecessary apply would
# restart the controller and suspend/restore the HPAs again mid-development.
CURRENT=$(kubectl -n train-ticket get deploy aurora \
            -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || true)
if [ "$CURRENT" != "$IMAGE" ]; then
  echo "image changed ($CURRENT -> $IMAGE); rendering deployment"
  sed "s|__AURORA_IMAGE__|$IMAGE|" aurora/deploy/deployment.yaml | kubectl apply -f -
else
  echo "aurora already running $IMAGE"
fi

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