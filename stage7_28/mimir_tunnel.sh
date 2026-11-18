#!/bin/bash
# Self-healing loadgen -> Mimir tunnel.
#
# Mimir no longer runs in the cluster: it is a container on the monitoring host
# (inventory group `monitoring`, today mon1=10.10.1.2). This used to be a
# `kubectl port-forward svc/mimir 18080:8080`, which cannot work any more because
# the Service is gone.
#
# So: an SSH local forward to the monitoring host, with the same self-healing
# behaviour as tunnel_lib.sh (clear a stale holder, health-check /ready, retry).
# The local port stays 18080 so every script, tunnel helper and analysis tool
# keeps working unchanged.
set -u
MONITORING_HOST="${MONITORING_HOST:-mon1}"
LOCAL_PORT="${MIMIR_LOCAL_PORT:-18080}"
REMOTE_PORT="${MIMIR_REMOTE_PORT:-18080}"
HEALTH_URL="${MIMIR_HEALTH_URL:-http://127.0.0.1:${LOCAL_PORT}/ready}"
LOG="${LOG:-/tmp/mimir_tunnel.log}"
SSH_EXTRA="${SSH_EXTRA:--o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3}"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

# Clear anything already holding the local port (a stale ssh -L keeps it bound
# after the remote side dies, and the replacement then fails with
# "address already in use" -- exactly how monitoring was lost before).
clear_port() {
  local pids
  pids=$(ss -ltnp 2>/dev/null | awk -v p=":${LOCAL_PORT} " '$4 ~ p {print $0}' \
          | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
  for p in $pids; do
    log "clearing stale holder pid=$p on :${LOCAL_PORT}"
    kill -TERM "$p" 2>/dev/null || true
  done
  sleep 1
  pids=$(ss -ltnp 2>/dev/null | awk -v p=":${LOCAL_PORT} " '$4 ~ p {print $0}' \
          | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
  for p in $pids; do kill -KILL "$p" 2>/dev/null || true; done
}

while true; do
  clear_port
  # shellcheck disable=SC2086
  ssh -f -N $SSH_EXTRA -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$MONITORING_HOST" \
    >> "$LOG" 2>&1
  ok=0
  for _ in $(seq 1 12); do
    if curl -fsS -m 5 "$HEALTH_URL" >/dev/null 2>&1; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" = "1" ]; then
    log "mimir tunnel up: 127.0.0.1:${LOCAL_PORT} -> ${MONITORING_HOST}:${REMOTE_PORT}"
  else
    log "mimir tunnel health check FAILED; retrying in 10 s"
  fi
  sleep 10
done