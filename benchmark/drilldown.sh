#!/bin/bash
# Quick incident drill-down helper (read-only). Usage:
#   drilldown.sh <service-or-pod-pattern> [minutes-back] [tail-lines]
set -u
PAT="${1:?usage: drilldown.sh <svc-pattern> [minutes] [tail]}"
MIN="${2:-10}"
TAILL="${3:-200}"
export KUBECONFIG=${KUBECONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf}
K=${KUBECTL:-/usr/local/bin/kubectl}
SINCE="${MIN}m"
echo "=== pods matching $PAT ==="
$K -n train-ticket get pods -o wide | grep "$PAT" || echo "none"
for POD in $($K -n train-ticket get pods -o name | grep "$PAT" | head -3); do
  echo "=== $POD container list ==="
  $K -n train-ticket get "$POD" -o jsonpath='{.spec.containers[*].name}'; echo
  for C in $($K -n train-ticket get "$POD" -o jsonpath='{.spec.containers[*].name}'); do
    echo "=== $POD / $C (last $TAILL lines, since $SINCE) ==="
    $K -n train-ticket logs "$POD" -c "$C" --since="$SINCE" --tail="$TAILL" 2>&1 | tail -"$TAILL"
  done
done
echo "=== recent events (last ${MIN}m) ==="
$K -n train-ticket get events --sort-by=.lastTimestamp 2>/dev/null | tail -20
