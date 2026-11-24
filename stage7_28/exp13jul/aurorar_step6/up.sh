#!/bin/bash
# PRE_K6_HOOK for the AURORA-R arm.
#
# AURORA-R refuses to act while a native HPA targets one of its Deployments, so
# this hook saves the complete HPA manifests and suspends them, then starts the
# controller. It is idempotent: a second call with the controller already
# running and the HPAs already absent is a no-op.
set -uo pipefail
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

PY=aurora_r/.venv/bin/python
[ -x "$PY" ] || PY=.venv-aurorar/bin/python
CONFIG=aurora_r/deploy/config-trainticket.json

IMAGE="${AURORA_R_IMAGE:-$(cat aurora_r/data/IMAGE 2>/dev/null)}"
if [ -z "${IMAGE:-}" ]; then
  echo "AURORA_R_IMAGE is not set and aurora_r/data/IMAGE is missing" >&2
  exit 1
fi
echo "using image: $IMAGE"

echo "applying aurora-r rbac"
kubectl apply -f aurora_r/deploy/rbac.yaml

echo "saving and suspending native HPAs"
"$PY" -m aurora_r.tools.hpa_state save --namespace train-ticket --config "$CONFIG"

# Render the run config: mode + checkpoint + paths inside the container.
kubectl -n train-ticket create configmap aurora-r-config \
  --from-file=config.json="$CONFIG" --dry-run=client -o yaml | kubectl apply -f -

CURRENT=$(kubectl -n train-ticket get deploy aurora-r \
            -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || true)
if [ "$CURRENT" != "$IMAGE" ]; then
  echo "image changed ($CURRENT -> $IMAGE); rendering deployment"
  sed "s|__AURORA_R_IMAGE__|$IMAGE|" aurora_r/deploy/deployment.yaml | kubectl apply -f -
else
  echo "aurora-r already running $IMAGE"
fi

echo "waiting for the aurora-r pod"
for i in $(seq 1 48); do
  ready=$(kubectl -n train-ticket get pods -l app=aurora-r \
            -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || true)
  if [ "$ready" = "true" ]; then
    echo "aurora-r ready"
    kubectl -n train-ticket get pods -l app=aurora-r
    exit 0
  fi
  sleep 5
done
echo "aurora-r not ready after 240s" >&2
kubectl -n train-ticket logs deploy/aurora-r --tail=40 >&2 || true
exit 1