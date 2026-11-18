#!/bin/bash
# Keep the log_streamer output from filling the disk.
#
# Why: log_streamer.sh writes every container line to $RUN_DIR/logs/<pod>__<c>.log
# via gawk '>>' (append). At 30k load ts-ui-dashboard's istio-proxy access log
# alone grew ~1.9 GB in 2 h, and the orchestrator's disk watchdog only pruned
# dumps_periodic/ -- it never looked at logs/. On 2026-11-17 that filled the
# 63 GB disk at 23:50Z and killed monitor_v2, failure_recorder and k6's sample
# writer. On 2026-11-18 the step-1 run reached 11 GB of logs in 2.4 h.
#
# Safe to run alongside the streamer: gawk appends with O_APPEND, so truncating
# a file it holds open does not create a sparse hole -- the next line lands at
# the new EOF. Each cycle archives the logs (gzip, ~10x) and verifies the
# archive before truncating, so nothing is silently dropped.
#
# Exits 0 on SIGTERM so the orchestrator's phase-7 collector check does not
# record a spurious "exited rc=143".
#
# Usage: rotate_run_logs.sh <run_dir> [threshold_mb] [pidfile]
set -u
LD=$1
THRESH_MB=${2:-3072}
PIDFILE=${3:-}

term() { rm -f "$PIDFILE" 2>/dev/null; exit 0; }
trap term TERM INT

[ -n "$PIDFILE" ] && echo $$ > "$PIDFILE"

n=0
while true; do
  sz=$(du -sm "$LD/logs" 2>/dev/null | cut -f1)
  case "$sz" in ''|*[!0-9]*) sz=0 ;; esac
  if [ "$sz" -ge "$THRESH_MB" ]; then
    n=$((n + 1))
    out="$LD/logs_captured_${n}.tgz"
    tar czf "$out" -C "$LD" logs 2>/dev/null
    if gzip -t "$out" 2>/dev/null; then
      find "$LD/logs" -name '*.log' -exec truncate -s 0 {} + 2>/dev/null
      echo "$(date -u +%FT%TZ) rotated logs ${sz}MB -> $out ($(du -sh "$out" | cut -f1))" \
        >> "$LD/log_rotation.log"
    else
      echo "$(date -u +%FT%TZ) archive $out FAILED integrity -- not truncating" \
        >> "$LD/log_rotation.log"
      rm -f "$out"
    fi
  fi
  sleep 300
done