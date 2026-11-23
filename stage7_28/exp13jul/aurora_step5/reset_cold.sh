#!/usr/bin/env bash
# Cold reset for AURORA development runs.
#
# AURORA stays in the cluster and owns the replicas: it suspends the namespace
# HPAs on start and restores them on exit. So a reset must NOT bring the HPAs
# back and must NOT scale the deployments -- both would fight the controller and
# force another suspend/restore cycle.
#
# What a reset means while AURORA is deployed:
#   * every WORKLOAD pod is deleted once, so the next run starts from a fresh
#     process generation (the stores are untouched, as in the PBScaler reset);
#   * AURORA's own pod is never touched, so the controller keeps its state;
#   * every workload deployment returns to Ready;
#   * no workload OOM-kills or restarts are tolerated.
#
# Usage:
#   bash stage7_28/exp13jul/aurora_step5/reset_cold.sh
set -uo pipefail
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf
NS=train-ticket
TIMEOUT=${RESET_TIMEOUT_S:-900}

log() { echo "[reset $(date -u +%H:%M:%SZ)] $*"; }

# ---- 1. AURORA must be up and Ready, and we must not touch it -------------
aurora_ready=$(kubectl -n "$NS" get pods -l app=aurora \
  -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || true)
if [ "$aurora_ready" != "true" ]; then
  log "FATAL: aurora is not Ready; start it first (up.sh) -- refusing to reset"
  kubectl -n "$NS" get pods -l app=aurora || true
  exit 1
fi
log "aurora is running and Ready (replicas stay with it)"

# ---- 2. record the workload restart baseline ------------------------------
# AURORA's own restarts are excluded: it is not part of the workload, and a
# controller restart would restore the HPAs and hand the replicas back.
workload_restarts() {
  kubectl -n "$NS" get pods --no-headers 2>/dev/null \
    | awk -v skip="$(kubectl -n "$NS" get pods -l app=aurora \
        --no-headers -o custom-columns=N:.metadata.name 2>/dev/null)" \
      'BEGIN{n=split(skip,a,"\n"); for(i=1;i<=n;i++) s[a[i]]=1}
       !($1 in s){t+=$4} END{print t+0}'
}
before_restarts=$(workload_restarts)
log "workload container restarts before: $before_restarts"

# ---- 3. delete the workload pods once ------------------------------------
# Pods owned by a ReplicaSet belong to a Deployment; AURORA's own pod is
# selected by the app=aurora label and must never appear here.
log "deleting workload pods (aurora excluded)"
kubectl -n "$NS" get pods --no-headers -o custom-columns=N:.metadata.name,O:.metadata.ownerReferences[0].kind,L:.metadata.labels.app \
  2>/dev/null | awk '$2=="ReplicaSet" && $3!="aurora"{print $1}' \
  | xargs -r -n 20 kubectl -n "$NS" delete pod --wait=false --timeout=60s >/dev/null 2>&1 || true

# ---- 4. wait for every workload deployment to be Ready -------------------
# The readiness predicate is the *container* status, not the pod's "READY"
# column. The column counts the Istio sidecar too ("2/2"), and a sidecar can
# legitimately sit at 1/2 while the application container is Ready -- waiting
# for 2/2 would block the whole reset on a proxy that is not part of the
# workload's readiness. AURORA's own pod is excluded by name.
log "waiting for workload containers (timeout ${TIMEOUT}s)"
deadline=$(( $(date +%s) + TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  not_ready=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null \
    | awk -v skip="$(kubectl -n "$NS" get pods -l app=aurora \
        --no-headers -o custom-columns=N:.metadata.name 2>/dev/null)" \
      'BEGIN{n=split(skip,a,"\n"); for(i=1;i<=n;i++) s[a[i]]=1}
       !($1 in s) && $2 ~ /^([0-9]+)\// {
         split($2, c, "/"); if (c[1] + 0 < c[2] + 0) n++
       } END{print n+0}')
  if [ "${not_ready:-0}" = "0" ]; then
    log "all workload containers Ready"
    break
  fi
  sleep 10
done

still=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null \
  | awk -v skip="$(kubectl -n "$NS" get pods -l app=aurora \
      --no-headers -o custom-columns=N:.metadata.name 2>/dev/null)" \
    'BEGIN{n=split(skip,a,"\n"); for(i=1;i<=n;i++) s[a[i]]=1}
     !($1 in s) && $2 ~ /^([0-9]+)\// {
       split($2, c, "/"); if (c[1] + 0 < c[2] + 0) print $1, $2
     }')
if [ -n "${still:-}" ]; then
  log "FATAL: workload pod(s) still not Ready after ${TIMEOUT}s:"
  echo "$still"
  exit 1
fi

# ---- 5. verify no crash damage -------------------------------------------
after_restarts=$(workload_restarts)
oom=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null | grep -c OOMKilled || true)
log "workload container restarts after: $after_restarts (delta $((after_restarts - before_restarts)))"
if [ "${oom:-0}" != "0" ]; then
  log "WARNING: $oom pod(s) OOMKilled"
fi

hpa=$(kubectl -n "$NS" get hpa --no-headers 2>/dev/null | grep -c . || true)
reps=$(kubectl -n "$NS" get deploy --no-headers -o custom-columns=N:.metadata.name,R:.spec.replicas \
       2>/dev/null | awk '{s+=$2} END{print s+0}')
log "hpa=${hpa:-0} (must stay 0 while AURORA owns replicas), total replicas=${reps:-0}"
log "reset complete"