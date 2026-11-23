#!/bin/bash
# POST_RUN_HOOK for AURORA development runs.
#
# AURORA stays in the cluster: this hook does NOT delete the Deployment and
# does NOT restore the HPAs. AURORA keeps owning the replicas between runs,
# which is the point -- development iterations should not pay a
# suspend/restore cycle every time.
#
# The one thing it does guarantee is that a crashed controller cannot leave the
# namespace without HPAs forever: if AURORA is not running AND no HPAs exist,
# the saved manifests are restored. That is a recovery action, not the normal
# path.
set -uo pipefail
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

running=$(kubectl -n train-ticket get pods -l app=aurora --no-headers 2>/dev/null | grep -c . || true)
hpa_count=$(kubectl -n train-ticket get hpa --no-headers 2>/dev/null | grep -c . || true)
echo "aurora pods=${running:-0} hpa=${hpa_count:-0}"

if [ "${running:-0}" != "0" ]; then
  echo "aurora still running; leaving replicas and HPAs to it"
  exit 0
fi

if [ "${hpa_count:-0}" = "0" ]; then
  echo "aurora is gone and no HPAs exist - restoring from the aurora-hpa-state ConfigMap"
  aurora/.venv/bin/python aurora/tools/restore_hpas.py || true
fi

echo "post-run: aurora stopped, hpa=$(kubectl -n train-ticket get hpa --no-headers 2>/dev/null | grep -c . || echo 0)"