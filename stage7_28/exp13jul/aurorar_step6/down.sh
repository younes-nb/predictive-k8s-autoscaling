#!/bin/bash
# POST_RUN_HOOK for the AURORA-R arm: stop the controller and put the HPAs back.
#
# AURORA-R does not restore HPAs itself (it refuses to run alongside them), so
# the restoration obligation lives here. If the controller is gone and the HPAs
# are still absent, the saved complete manifests are restored.
set -uo pipefail
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

PY=aurora_r/.venv/bin/python
[ -x "$PY" ] || PY=.venv-aurorar/bin/python

kubectl -n train-ticket delete deploy aurora-r --timeout=120s --ignore-not-found

for i in $(seq 1 24); do
  n=$(kubectl -n train-ticket get pods -l app=aurora-r --no-headers 2>/dev/null | grep -c . || true)
  [ "${n:-0}" = "0" ] && break
  sleep 5
done

hpa=$(kubectl -n train-ticket get hpa --no-headers 2>/dev/null | grep -c . || true)
if [ "${hpa:-0}" = "0" ]; then
  echo "no HPAs present - restoring the saved manifests"
  "$PY" -m aurora_r.tools.hpa_state restore --namespace train-ticket || true
fi
echo "post-run: hpa=$(kubectl -n train-ticket get hpa --no-headers 2>/dev/null | grep -c . || echo 0)"