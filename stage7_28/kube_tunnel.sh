#!/bin/bash
# Self-healing runner->kube-apiserver tunnel with MASTER FAILOVER.
#
# 127.0.0.1:16443 -> <master>:6443. The kubeconfig points at 127.0.0.1:16443,
# so every kubectl / kubernetes-client call on the runner depends on this.
#
# 2026-11-16: master1 hung (NotReady, sshd + kubelet unresponsive) and the
# single-master loop spun forever, killing kubectl + the Mimir/Loki forwards.
# 2026-11-17: the loop now also clears a stale port holder before binding.
#
# Env: MASTERS="master1:10.10.1.2 master2:10.10.1.3 master3:10.10.1.4"
#      LOCAL_PORT=16443  FAILOVER_AFTER=2
set -u
SSH_CONFIG=${SSH_CONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/ssh_config}
LOCAL_PORT=${LOCAL_PORT:-16443}
FAILOVER_AFTER=${FAILOVER_AFTER:-2}
MASTERS=${MASTERS:-"master1:10.10.1.2 master2:10.10.1.3 master3:10.10.1.4"}
LOG=${LOG:-/tmp/kube_tunnel.log}
exec >>"$LOG" 2>&1

read -r -a MASTER_ARR <<< "$MASTERS"
N=${#MASTER_ARR[@]}
idx=0
fails=0

clear_port() {
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -TERM "${LOCAL_PORT}/tcp" >/dev/null 2>&1
    sleep 1
    fuser -k -KILL "${LOCAL_PORT}/tcp" >/dev/null 2>&1
  fi
  local pids p
  pids=$(ss -ltnp 2>/dev/null | awk -v p=":${LOCAL_PORT} " 'index($0,p){print}' \
           | grep -o 'pid=[0-9]*' | cut -d= -f2)
  for p in $pids; do kill -KILL "$p" 2>/dev/null; done
}

while true; do
  entry=${MASTER_ARR[$idx]}
  host=${entry%%:*}
  ip=${entry##*:}
  clear_port
  echo "$(date -u +%FT%TZ) starting ssh tunnel 127.0.0.1:${LOCAL_PORT} -> ${host} (${ip}:6443)"
  ssh -F "$SSH_CONFIG" -N \
      -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
      -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
      -L "127.0.0.1:${LOCAL_PORT}:${ip}:6443" "$host"
  rc=$?
  echo "$(date -u +%FT%TZ) tunnel exited rc=$rc (master=${host})"
  fails=$((fails + 1))
  if [ "$fails" -ge "$FAILOVER_AFTER" ]; then
    idx=$(((idx + 1) % N))
    fails=0
    echo "$(date -u +%FT%TZ) failing over to ${MASTER_ARR[$idx]}"
  fi
  sleep 3
done