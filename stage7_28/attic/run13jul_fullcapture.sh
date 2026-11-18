#!/bin/bash
# 13-July HPA run — full-capture orchestrator (1h + 10min warmup).
# Derived from run32k_fullcapture.sh; differences:
#   * 13-Jul slice (START_MIN=529, MCR jul13 slice, md5 3657e395...)
#   * MAX_REQUESTS=25000 (25k ladder point, same as runs 1&3)
#   * higher VUs: USER_POOL=6000 + PRE_ALLOCATED_VUS=2000 (avoids saturation)
#   * hardened script (md5 ae3a10d8...) with auth/login fixes
#   * 10 min warm-up + 1 h measured (same as 32k run)
# Usage: setsid bash stage7_28/exp13jul/run13jul_fullcapture.sh > /tmp/run13jul.log 2>&1 &
set -u
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH=/users/younes02/.local/bin:$PATH
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

# ---- parameters (13-Jul slice, 25k shape, 70-min duration) ----
MAX_REQUESTS=25000
USER_POOL=6000
PRE_ALLOC=2000
START_MIN=529
TEST_MIN=70
WARMUP_S=600
K6_STOP_OFFSET=4200
COLLECT_DUR=3600
TOOLS_END_OFFSET=4800
MCR_SLICE_LOCAL="$PWD/stage7_28/exp13jul/mcr_jul13_slice_local.csv"
MCR_SLICE_LOADGEN=/opt/loadgen/mcr_jul13_slice.csv

DRY=${1:-}

RUNID=hpa13jul_rr_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$PWD/stage7_28/exp13jul/$RUNID"
mkdir -p "$RUN_DIR/logs"
CONSOLE_SRC="/opt/loadgen/k6/console_${RUNID}.log"
echo "$RUNID" > "$RUN_DIR/run_id"
echo "$RUNID" > "$PWD/stage7_28/exp13jul/CURRENT_RUN_ID"
LOG="$RUN_DIR/orchestrator.log"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

log "=== run13jul_fullcapture START runid=$RUNID dry=${DRY:-no} ==="
log "params: MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL PRE_ALLOCATED_VUS=$PRE_ALLOC START_MIN=$START_MIN TEST_MIN=$TEST_MIN"

# ---------- phase 0: cold gate ----------
log "phase 0: cold reset assert gate"
if ! python3 pbscaler/scripts/launchers/reset_cluster_cold.py --assert-only \
     > "$RUN_DIR/cold_gate.log" 2>&1; then
  log "FATAL: cold gate failed (see cold_gate.log)"; exit 1
fi
log "cold gate passed"

# ---------- phase 0.5: Mimir serves CURRENT data ----------
UPNOW=$(curl -s -m 10 "http://127.0.0.1:18080/prometheus/api/v1/query" \
        --data-urlencode 'query=up' 2>/dev/null \
        | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin); print(len(d["data"]["result"]))
except Exception:
  print(0)' 2>/dev/null)
if [ "${UPNOW:-0}" -lt 1 ]; then
  log "FATAL: mimir up@now=$UPNOW (ingester not serving current data) — aborting"; exit 1
fi
log "mimir up@now=$UPNOW — healthy"

# ---------- phase 1: tunnels ----------
log "phase 1: starting mimir + loki tunnels"
pgrep -f "mimir_tunnel.sh" > /dev/null || setsid bash "$PWD/stage7_28/mimir_tunnel.sh" < /dev/null > /dev/null 2>&1 &
pgrep -f "loki_tunnel.sh"  > /dev/null || setsid bash "$PWD/stage7_28/loki_tunnel.sh"  < /dev/null > /dev/null 2>&1 &
sleep 5
curl -fsS -m 8 "http://127.0.0.1:18080/prometheus/api/v1/query" --data 'query=up' > /dev/null \
  || { log "FATAL: mimir tunnel not reachable"; exit 1; }
curl -fsS -m 8 "http://127.0.0.1:3100/ready" > /dev/null \
  || { log "FATAL: loki tunnel not reachable"; exit 1; }
log "both tunnels verified up"

if [ "$DRY" = "--dry" ]; then
  log "DRY mode: stopping here (no tools, no k6)"
  exit 0
fi

# ---------- phase 2: capture tools ----------
LAUNCH_EPOCH=$(date +%s)
END_EPOCH=$((LAUNCH_EPOCH + TOOLS_END_OFFSET))
echo "$LAUNCH_EPOCH" > "$RUN_DIR/launch_epoch"
echo "$END_EPOCH" > "$RUN_DIR/end_epoch"
log "phase 2: tools start (launch=$LAUNCH_EPOCH end=$END_EPOCH)"

setsid bash "$PWD/stage7_28/log_streamer.sh" "$RUN_DIR" "$END_EPOCH" \
  < /dev/null >> "$RUN_DIR/log_streamer.out" 2>&1 &
log "log_streamer started (pid $!)"

setsid env PBS_MIMIR_URL=http://127.0.0.1:18080 python3 \
  "$PWD/stage7_28/monitor_v2.py" --run-dir "$RUN_DIR" --t0 0 \
  --end "$END_EPOCH" --mcr "$MCR_SLICE_LOCAL" --max-requests "$MAX_REQUESTS" \
  --start-min "$START_MIN" < /dev/null >> "$RUN_DIR/monitor_v2.out" 2>&1 &
log "monitor_v2 started (pid $!)"

setsid env PBS_MIMIR_URL=http://127.0.0.1:18080 LOKI_URL=http://127.0.0.1:3100 \
  python3 "$PWD/stage7_28/failure_recorder.py" --run-dir "$RUN_DIR" \
  --end "$END_EPOCH" --runid "$RUNID" < /dev/null >> "$RUN_DIR/failure_recorder.out" 2>&1 &
log "failure_recorder started (pid $!)"

# console shipper
(
  off=1
  while [ "$(date +%s)" -lt $((END_EPOCH + 120)) ]; do
    sz=$(ssh -o ConnectTimeout=8 loadgen "wc -c < '$CONSOLE_SRC' 2>/dev/null" 2>/dev/null || echo 0)
    case "$sz" in (''|*[!0-9]*) sz=0;; esac
    if [ "$sz" -gt "$off" ]; then
      ssh -o ConnectTimeout=8 loadgen "tail -c +$off '$CONSOLE_SRC'" >> "$RUN_DIR/k6_console.log" 2>/dev/null
      off=$((sz + 1))
    fi
    sleep 15
  done
) < /dev/null > /dev/null 2>&1 &
log "console shipper started (pid $!)"

# failures shipper
ssh -o ConnectTimeout=8 loadgen "mkdir -p /opt/loadgen/failure_logs/$RUNID" 2>/dev/null
(
  foff=0
  while [ "$(date +%s)" -lt $((END_EPOCH + 60)) ]; do
    if [ -f "$RUN_DIR/failures.jsonl" ]; then
      fsz=$(stat -c%s "$RUN_DIR/failures.jsonl" 2>/dev/null || echo 0)
      if [ "$fsz" -gt "$foff" ]; then
        tail -c +$((foff + 1)) "$RUN_DIR/failures.jsonl" \
          | ssh -o ConnectTimeout=8 loadgen "cat >> /opt/loadgen/failure_logs/$RUNID/failures.jsonl" 2>/dev/null
        foff=$fsz
      fi
    fi
    sleep 10
  done
) < /dev/null > /dev/null 2>&1 &
log "failures shipper started -> loadgen:/opt/loadgen/failure_logs/$RUNID/failures.jsonl (pid $!)"

# ---------- phase 3: launch k6 on loadgen ----------
log "phase 3: launching k6 (RUNID=$RUNID, maxVUs=$USER_POOL, preAllocatedVUs=$PRE_ALLOC)"
ssh -o ConnectTimeout=10 loadgen "ulimit -n 65536; nohup setsid env RUNID='$RUNID' \
  MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL TRACE_COMPRESS=1 OPEN_LOOP=1 \
  PRE_ALLOCATED_VUS=$PRE_ALLOC START_MIN=$START_MIN TEST_MIN=$TEST_MIN \
  MCR_CSV=$MCR_SLICE_LOADGEN bash /opt/loadgen/launch_k6.sh \
  > '$CONSOLE_SRC' 2>&1 < /dev/null &" 2>&1 | sed 's/^/k6launch: /' >> "$LOG"
log "k6 launch command issued"

sleep 25
KSZ=$(ssh -o ConnectTimeout=8 loadgen "wc -c < '$CONSOLE_SRC' 2>/dev/null" 2>/dev/null || echo 0)
case "$KSZ" in (''|*[!0-9]*) KSZ=0;; esac
if [ "$KSZ" -lt 100 ]; then
  log "FATAL: k6 console log size=$KSZ after 25 s — k6 did not start; aborting"
  exit 1
fi
log "k6 console confirmed growing (size=$KSZ)"

# ---------- phase 4: traffic detection ----------
log "phase 4: waiting for traffic (ui rps > 0.2)"
t_traffic=""
for i in $(seq 1 240); do
  rps=$(curl -s -m 8 "http://127.0.0.1:18080/prometheus/api/v1/query" \
        --data 'query=sum(rate(istio_requests_total{reporter="destination",destination_workload="ts-ui-dashboard"}[30s]))' \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); r=d["data"]["result"]; print(r[0]["value"][1] if r else 0)' 2>/dev/null || echo 0)
  if python3 -c "import sys; sys.exit(0 if float('$rps' or 0) > 0.2 else 1)" 2>/dev/null; then
    t_traffic=$(date +%s); break
  fi
  sleep 5
done
if [ -z "$t_traffic" ]; then
  log "FATAL: no traffic detected in 20 min — aborting"
  exit 1
fi
echo "$t_traffic" > "$RUN_DIR/traffic_start_epoch"
log "traffic detected epoch=$t_traffic (+$((t_traffic - LAUNCH_EPOCH)) s after launch)"

# ---------- phase 5: NO live collector (post-hoc) ----------
log "phase 5: skipped (collector is post-hoc by design)"

# ---------- phase 6: SIGINT k6 at traffic+4200 ----------
STOP_AT=$((t_traffic + K6_STOP_OFFSET))
WAIT=$((STOP_AT - $(date +%s)))
log "phase 6: k6 SIGINT in ${WAIT}s (traffic+$K6_STOP_OFFSET)"
if [ "$WAIT" -gt 0 ]; then sleep "$WAIT"; fi
KP=$(ssh -o ConnectTimeout=8 loadgen "pgrep -x k6 | head -1" 2>/dev/null || true)
if [ -n "${KP:-}" ]; then
  ssh -o ConnectTimeout=8 loadgen "kill -INT $KP" && log "k6 SIGINT sent (pid $KP)"
else
  log "WARN: no k6 process found on loadgen at SIGINT time"
fi

# ---------- phase 5b: post-hoc collection ----------
log "phase 5b: post-hoc dataset collection (waiting 180 s for Alloy flush)"
sleep 180
DS="$RUN_DIR/dataset_$RUNID"
mkdir -p "$DS"
CSTART=$((t_traffic + WARMUP_S))
if python3 "$PWD/stage7_28/rebuild_train_data.py" \
     --start "$CSTART" --duration "$COLLECT_DUR" --interval 5 \
     --out "$DS" --workload "$MCR_SLICE_LOCAL" \
     > "$RUN_DIR/rebuild_collection.log" 2>&1; then
  log "post-hoc collection OK -> $DS"
else
  log "FATAL: post-hoc collection failed (see rebuild_collection.log)"
fi

# ---------- phase 7: wait for tools ----------
WAIT=$((END_EPOCH + 90 - $(date +%s)))
log "phase 7: waiting ${WAIT}s for tools to finish"
if [ "$WAIT" -gt 0 ]; then sleep "$WAIT"; fi
grep -c FAILURE_EVENT "$RUN_DIR/failures.jsonl" 2>/dev/null | xargs -I{} echo "failure events captured: {}" >> "$LOG"

# ---------- phase 8: validate dataset ----------
log "phase 8: validating collector meta.json"
python3 - "$DS" >> "$LOG" 2>&1 <<'EOF'
import json, sys, glob
ds = sys.argv[1]
metas = glob.glob(ds + "/**/meta.json", recursive=True) or glob.glob(ds + "/meta.json")
if not metas:
    print("meta.json NOT FOUND in", ds); raise SystemExit(1)
m = json.load(open(metas[0]))
txt = open(metas[0]).read()
print("meta:", {k: m.get(k) for k in ("result", "coverage_pct", "interval_s", "duration_s", "no_data_rows") if k in m})
print("RESULT COMPLETE:", "COMPLETE" in txt.upper())
cov = m.get("coverage_pct") or 0
print("coverage_ok:", float(cov) >= 95 if cov else False)
EOF

# ---------- phase 10: dumps ----------
log "phase 10: Mimir + Loki dumps (launch-60 .. END_EPOCH)"
DUMP_DIR="$RUN_DIR/dumps"
mkdir -p "$DUMP_DIR"
bash "$PWD/stage7_28/dump_mimir.sh" "$((LAUNCH_EPOCH - 60))" "$END_EPOCH" \
     "$DUMP_DIR/mimir" >> "$LOG" 2>&1 \
  && log "mimir dumps OK" || log "WARN: mimir dumps failed"
bash "$PWD/stage7_28/dump_loki.sh" "$((LAUNCH_EPOCH - 60))" "$END_EPOCH" \
     "$DUMP_DIR/loki" >> "$LOG" 2>&1 \
  && log "loki dump OK" || log "WARN: loki dump failed"

# ---------- phase 11: archive ----------
log "phase 11: archiving run dir to loadgen"
tar czf - -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")" \
  | ssh -o ConnectTimeout=8 loadgen "cat > /opt/loadgen/failure_logs/${RUNID}_rundir.tgz" 2>/dev/null \
  && log "archive OK -> loadgen:/opt/loadgen/failure_logs/${RUNID}_rundir.tgz" \
  || log "WARN: archive failed"

# ---------- phase 12: final cold reset ----------
log "phase 12: final cold reset"
if python3 pbscaler/scripts/launchers/reset_cluster_cold.py \
     > "$RUN_DIR/final_cold_reset.log" 2>&1; then
  log "final cold reset OK"
else
  log "WARN: final cold reset flagged issues"
fi
log "=== run13jul_fullcapture DONE runid=$RUNID ==="
