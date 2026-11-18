#!/bin/bash
# Self-healing runner->Loki tunnel (svc/loki:3100).
# Uses stage7_28/tunnel_lib.sh: clears a stale port holder and health-checks
# /ready before declaring the forward good.
set -u
export KUBECONFIG=${KUBECONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf}
exec >>"${LOG:-/tmp/loki_tunnel.log}" 2>&1
source "$(dirname "$0")/tunnel_lib.sh"
k8s_forward monitoring loki 3100 3100 "http://127.0.0.1:3100/ready" "${LOG:-/tmp/loki_tunnel.log}"