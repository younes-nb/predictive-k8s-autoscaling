#!/bin/bash
# Self-healing loadgen -> Loki tunnel.
#
# Loki no longer runs in the cluster: it is a container on the monitoring host
# (inventory group `monitoring`, today mon1=10.10.1.2). This used to be a
# `kubectl port-forward svc/loki 3100:3100`, which cannot work any more.
#
# The forward is created ONCE and then only monitored -- see the note in
# mimir_tunnel.sh about why re-creating it every loop iteration is harmful.
# The local port stays 3100 so every script and tool keeps working unchanged.
set -u
MONITORING_HOST="${MONITORING_HOST:-mon1}"
LOCAL_PORT="${LOKI_LOCAL_PORT:-3100}"
REMOTE_PORT="${LOKI_REMOTE_PORT:-3100}"
HEALTH_URL="${LOKI_HEALTH_URL:-http://127.0.0.1:${LOCAL_PORT}/ready}"
LOG="${LOG:-/tmp/loki_tunnel.log}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes}"
FAILS_BEFORE_RESTART="${FAILS_BEFORE_RESTART:-3}"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

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

start_forward() {
  clear_port
  # shellcheck disable=SC2086
  ssh -f -N $SSH_OPTS -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$MONITORING_HOST" \
    >> "$LOG" 2>&1
}

healthy() { curl -fsS -m 5 "$HEALTH_URL" >/dev/null 2>&1; }

while true; do
  start_forward
  ok=0
  for _ in $(seq 1 12); do
    if healthy; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" = "1" ]; then
    log "loki tunnel up: 127.0.0.1:${LOCAL_PORT} -> ${MONITORING_HOST}:${REMOTE_PORT}"
  else
    log "loki tunnel health check FAILED at startup; retrying in 10 s"
    sleep 10
    continue
  fi

  fails=0
  while true; do
    sleep 10
    if healthy; then
      fails=0
    else
      fails=$((fails + 1))
      if [ "$fails" -ge "$FAILS_BEFORE_RESTART" ]; then
        log "loki tunnel unhealthy ${fails}x -- restarting"
        break
      fi
    fi
  done
done