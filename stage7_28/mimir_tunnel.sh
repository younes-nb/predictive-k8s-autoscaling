#!/bin/bash
# Self-healing runner->Mimir tunnel (svc/mimir:18080 -> 8080).
# Uses stage7_28/tunnel_lib.sh: clears a stale port holder and health-checks
# /ready before declaring the forward good. This is the tunnel that silently
# broke on 2026-11-17 when the Mimir pod restarted.
set -u
export KUBECONFIG=${KUBECONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf}
exec >>"${LOG:-/tmp/mimir_tunnel.log}" 2>&1
source "$(dirname "$0")/tunnel_lib.sh"
k8s_forward monitoring mimir 18080 8080 "http://127.0.0.1:18080/ready" "${LOG:-/tmp/mimir_tunnel.log}"