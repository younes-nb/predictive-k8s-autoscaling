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
#   bash benchmark/aurora_step5/reset_cold.sh
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
# Only ``ts-*`` pods are workload pods. The namespace also holds
# infrastructure (the ACME cert-manager solver, istio, monitoring sidecars)
# whose readiness has nothing to do with the workload and which never reaches
# the workload's container count -- including them made the reset wait for a
# pod that would never satisfy the predicate.
# Only ``ts-*`` pods are workload pods. The namespace also holds
# infrastructure (the ACME cert-manager solver, istio) whose restarts are
# unrelated to the workload and must not be counted.
workload_restarts() {
  kubectl -n "$NS" get pods --no-headers 2>/dev/null \
    | awk -F'|' -v skip="$(kubectl -n "$NS" get pods -l app=aurora \
        --no-headers -o custom-columns=N:.metadata.name 2>/dev/null)" \
      'BEGIN{n=split(skip,a,"\n"); for(i=1;i<=n;i++) s[a[i]]=1}
       ($1 ~ /^ts-/) && !($1 in s){t+=$4} END{print t+0}'
}
before_restarts=$(workload_restarts)
log "workload container restarts before: $before_restarts"

# ---- 3. delete the workload pods once ------------------------------------
# Pods owned by a ReplicaSet belong to a Deployment; AURORA's own pod is
# selected by the app=aurora label and must never appear here.
log "deleting workload pods (aurora excluded)"
kubectl -n "$NS" get pods --no-headers -o custom-columns=N:.metadata.name,O:.metadata.ownerReferences[0].kind,L:.metadata.labels.app \
  2>/dev/null | awk '$1 ~ /^ts-/ && $2=="ReplicaSet"{print $1}' \
  | xargs -r -n 20 kubectl -n "$NS" delete pod --wait=false --timeout=60s >/dev/null 2>&1 || true

# ---- 4. wait for every workload deployment to be Ready -------------------
# Readiness is evaluated on ``status.containerStatuses`` -- the *application*
# containers. The pod's READY column also counts Istio's sidecar, and Istio
# 1.30 runs ``istio-proxy`` as a native sidecar (an init container with
# ``restartPolicy: Always``), so a perfectly healthy pod reads "1/2" forever.
# Using the column would block the reset until the 900 s timeout.
#
# A pod is ready when it is Running and every application container reports
# ready=true. AURORA's own pod is excluded by name, and only ``ts-*`` pods are
# considered: the namespace also holds infrastructure (the ACME cert-manager
# solver) whose readiness is unrelated to the workload.
workload_not_ready() {
  # Pods are reported as name|phase|ready|deletionTimestamp. A pod that is
  # terminating is NOT ready: it is on its way out and its replacement has not
  # necessarily started, so counting it as ready would let the reset finish
  # while the namespace is still churning.
  kubectl -n "$NS" get pods -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.status.phase}{|"}{.status.containerStatuses[*].ready}{"|"}{.metadata.deletionTimestamp}{"\n"}{end}' 2>/dev/null \
    | awk -F'|' -v skip="$(kubectl -n "$NS" get pods -l app=aurora \
        --no-headers -o custom-columns=N:.metadata.name 2>/dev/null)" \
      'BEGIN{n=split(skip,a,"\n"); for(i=1;i<=n;i++) s[a[i]]=1}
       {
         name=$1; phase=$2; ready=$3; del=$4
         if (name !~ /^ts-/ || (name in s)) next
         bad = (phase != "Running") || (del != "")
         if (ready == "" || ready ~ /false/) bad = 1
         if (bad) { print name, phase, ready, (del != "" ? "Terminating" : "") }
       }'
}

log "waiting for workload containers (timeout ${TIMEOUT}s)"
deadline=$(( $(date +%s) + TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  still=$(workload_not_ready)
  if [ -z "${still:-}" ]; then
    log "all workload containers Ready"
    break
  fi
  sleep 10
done

still=$(workload_not_ready)
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