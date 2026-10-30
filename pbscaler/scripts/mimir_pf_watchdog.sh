#!/bin/bash
# Self-healing Mimir port-forward (18081 -> svc/mimir:8080).
# Spawns a PF whenever /ready fails; kills a half-dead listener BY PORT
# (ss -ltnp), never by name pattern (pattern-matching kills own shell).
LOG=/tmp/opencode/mimir_pf_watch.log
PIDF=/tmp/opencode/mimir_pf_watchdog.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) watchdog already running (pid $(cat "$PIDF"))" >>"$LOG"
  exit 0
fi
echo $$ >"$PIDF"
while true; do
  if ! curl -sf -m 5 http://127.0.0.1:18081/ready >/dev/null 2>&1; then
    LP=$(ss -ltnp 2>/dev/null | grep ':18081 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    if [ -n "$LP" ]; then
      echo "$(date -u +%FT%TZ) killing half-dead listener pid $LP" >>"$LOG"
      kill "$LP" 2>/dev/null
      sleep 2
    fi
    echo "$(date -u +%FT%TZ) respawning PF" >>"$LOG"
    kubectl -n monitoring port-forward svc/mimir 18081:8080 >>"$LOG" 2>&1 &
    sleep 25
  fi
  sleep 20
done
