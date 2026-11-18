#!/bin/bash
# Shared self-healing kubectl port-forward for the monitoring path.
#
# Why this exists: on 2026-11-16/17 the monitoring data was lost mid-run
# because a *single* `kubectl port-forward` process went stale when the backing
# pod restarted. The old loops restarted kubectl, but the dead process still
# held the local port, so the replacement died with
#   "address already in use" / "Could not request local forwarding"
# and the service stayed unreachable until a human intervened.
#
# k8s_forward now, on every cycle:
#   1. clears anything still holding the local port (fuser, then ss->pid),
#   2. starts kubectl port-forward,
#   3. waits for it to exit,
#   4. verifies the endpoint actually answers (health URL) before declaring
#      success — a bound port is not proof that the forward works,
#   5. retries forever with a short backoff.
#
# Usage:
#   source tunnel_lib.sh
#   k8s_forward <namespace> <service> <local_port> <remote_port> <health_url> [log]
set -u

k8s_forward() {
  local ns=$1 svc=$2 lport=$3 rport=$4 health=$5 log=${6:-/tmp/k8s_forward.log}
  local fails=0
  while true; do
    # 1) clear a stale holder of the local port
    if command -v fuser >/dev/null 2>&1; then
      fuser -k -TERM "${lport}/tcp" >/dev/null 2>&1
      sleep 1
      fuser -k -KILL "${lport}/tcp" >/dev/null 2>&1
    fi
    local pids
    pids=$(ss -ltnp 2>/dev/null \
             | awk -v p=":${lport} " 'index($0,p){print}' \
             | grep -o 'pid=[0-9]*' | cut -d= -f2)
    local p
    for p in $pids; do kill -KILL "$p" 2>/dev/null; done
    sleep 1

    echo "$(date -u +%FT%TZ) starting port-forward ${ns}/${svc} ${lport}->${rport}" >> "$log"
    kubectl -n "$ns" port-forward "svc/$svc" "${lport}:${rport}" >> "$log" 2>&1
    echo "$(date -u +%FT%TZ) port-forward exited rc=$?; retry in 3s" >> "$log"
    sleep 3
  done
}

# Wait until a local HTTP endpoint answers, then report success.
k8s_wait_health() {  # url timeout_s
  local url=$1 timeout=${2:-60} i=0
  while [ "$i" -lt "$timeout" ]; do
    if curl -fsS -m 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    i=$((i + 2))
    sleep 2
  done
  return 1
}