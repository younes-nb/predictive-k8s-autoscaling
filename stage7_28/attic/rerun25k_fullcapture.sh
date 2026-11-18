#!/bin/bash
# 25k HPA rerun with FULL failure capture (Loki-primary) -> RF retrain -> image re-bake.
#
# Mirrors reconciliation 07 step shape at the 25k ladder point (the previous
# hpa25k_20261114T073743Z run):
#   k6 launch -> ~65 s to first traffic -> 10-min warm-up -> collector at
#   traffic+600 s (5100 s @ 5 s) -> k6 SIGINT at traffic+5680 s -> tools stop
#   at launch+5900 s -> validate -> RF retrain (embargo 60) -> freeze+hash ->
#   build_and_push -> archive -> final cold reset.
#
# NEW vs the previous run: failure-driven event capture to
#   $RUN_DIR/failures.jsonl (service/code, hop flags, per-pod state, HPA,
#   app exceptions from Loki), live-shipped to loadgen
#   /opt/loadgen/failure_logs/<RUNID>/failures.jsonl during the run.
#
# Read-only on the cluster: Mimir/Loki queries + kubectl get only; no HPA or
# limit changes. Multi-hour: launch with setsid, e.g.
#   setsid bash stage7_28/exp13jul/rerun25k_fullcapture.sh > /tmp/rerun25k.log 2>&1 &
#   (--dry runs only the cold gate + tunnels + checks, no tools, no k6)
set -u
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH=/users/younes02/.local/bin:$PATH
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

# ---- parameters (mirror hpa25k_20261114T073743Z exactly) ----
MAX_REQUESTS=10500
USER_POOL=3000
START_MIN=529
TEST_MIN=250
WARMUP_S=600            # traffic -> collector attach (10-min warm-up)
K6_STOP_OFFSET=15000    # traffic -> SIGINT k6 (600 warm-up + 14400 measured = ~4h10m)
COLLECT_DUR=14400       # collector duration @ 5 s (240 min measured window)
TOOLS_END_OFFSET=15400  # launch-relative stop for all tools (traffic+~15335)
MCR_SLICE_LOCAL="$PWD/stage7_28/exp13jul/hpa25k_20261114T073743Z/mcr_slice_local.csv"
MCR_SLICE_LOADGEN=/opt/loadgen/mcr_jul13_slice.csv

DRY=${1:-}

RUNID=hpa25k_rr_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$PWD/stage7_28/exp13jul/$RUNID"
mkdir -p "$RUN_DIR/logs"
CONSOLE_SRC="/opt/loadgen/k6/console_${RUNID}.log"
echo "$RUNID" > "$RUN_DIR/run_id"
echo "$RUNID" > "$PWD/stage7_28/exp13jul/CURRENT_RUN_ID"
LOG="$RUN_DIR/orchestrator.log"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

log "=== rerun25k_fullcapture START runid=$RUNID dry=${DRY:-no} ==="
log "params: MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL START_MIN=$START_MIN TEST_MIN=$TEST_MIN"

# ---------- phase 0: cold gate ----------
log "phase 0: cold reset assert gate"
if ! pbscaler/.venv/bin/python pbscaler/scripts/launchers/reset_cluster_cold.py --assert-only \
     > "$RUN_DIR/cold_gate.log" 2>&1; then
  log "FATAL: cold gate failed (see cold_gate.log)"; exit 1
fi
log "cold gate passed"

# ---------- phase 1: tunnels (self-healing, idempotent) ----------
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

# ---------- phase 2: capture tools (all stop at END_EPOCH) ----------
LAUNCH_EPOCH=$(date +%s)
END_EPOCH=$((LAUNCH_EPOCH + TOOLS_END_OFFSET))
echo "$LAUNCH_EPOCH" > "$RUN_DIR/launch_epoch"
echo "$END_EPOCH" > "$RUN_DIR/end_epoch"
log "phase 2: tools start (launch=$LAUNCH_EPOCH end=$END_EPOCH)"

setsid bash "$PWD/stage7_28/exp13jul/log_streamer.sh" "$RUN_DIR" "$END_EPOCH" \
  < /dev/null >> "$RUN_DIR/log_streamer.out" 2>&1 &
log "log_streamer started (pid $!)"

setsid env PBS_MIMIR_URL=http://127.0.0.1:18080 python3 \
  "$PWD/stage7_28/exp13jul/monitor_v2.py" --run-dir "$RUN_DIR" --t0 0 \
  --end "$END_EPOCH" --mcr "$MCR_SLICE_LOCAL" --max-requests "$MAX_REQUESTS" \
  --start-min "$START_MIN" < /dev/null >> "$RUN_DIR/monitor_v2.out" 2>&1 &
log "monitor_v2 started (pid $!)"

setsid env PBS_MIMIR_URL=http://127.0.0.1:18080 LOKI_URL=http://127.0.0.1:3100 \
  python3 "$PWD/stage7_28/exp13jul/failure_recorder.py" --run-dir "$RUN_DIR" \
  --end "$END_EPOCH" --runid "$RUNID" < /dev/null >> "$RUN_DIR/failure_recorder.out" 2>&1 &
log "failure_recorder started (pid $!)"

# k6 console offset-pull (loadgen -> RUN_DIR/k6_console.log) for monitor VU parsing
(
  off=1
  while [ "$(date +%s)" -lt $((END_EPOCH + 120)) ]; do
    sz=$(ssh -o ConnectTimeout=8 loadgen "wc -c < '$CONSOLE_SRC' 2>/dev/null" 2>/dev/null || echo 0)
    case "$sz" in (''|*[!0-9]*) sz=0;; esac
    if [ "$sz" -gt "$off" ]; then
      ssh -o ConnectTimeout=8 loadgen "tail -c +$off '$CONSOLE_SRC'" >> "$RUN_DIR/k6_console.log" 2>/dev/null
      log_console="$(date -u +%FT%TZ) shipped console bytes $((sz-off+1)) (total $sz)"
      echo "$log_console" >> "$RUN_DIR/console_shipper.out"
      off=$((sz + 1))
    fi
    sleep 15
  done
) < /dev/null > /dev/null 2>&1 &
log "console shipper started (pid $!)"

# failures.jsonl offset-push (RUN_DIR -> loadgen, live during the run)
ssh -o ConnectTimeout=8 loadgen "mkdir -p /opt/loadgen/failure_logs/$RUNID" 2>/dev/null
(
  foff=0
  while [ "$(date +%s)" -lt $((END_EPOCH + 60)) ]; do
    if [ -f "$RUN_DIR/failures.jsonl" ]; then
      fsz=$(stat -c%s "$RUN_DIR/failures.jsonl" 2>/dev/null || echo 0)
      if [ "$fsz" -gt "$foff" ]; then
        tail -c +$((foff + 1)) "$RUN_DIR/failures.jsonl" \
          | ssh -o ConnectTimeout=8 loadgen "cat >> /opt/loadgen/failure_logs/$RUNID/failures.jsonl" 2>/dev/null
        echo "$(date -u +%FT%TZ) pushed failures bytes $((fsz - foff)) (total $fsz)" >> "$RUN_DIR/failures_shipper.out"
        foff=$fsz
      fi
    fi
    sleep 10
  done
) < /dev/null > /dev/null 2>&1 &
log "failures shipper started -> loadgen:/opt/loadgen/failure_logs/$RUNID/failures.jsonl (pid $!)"

# ---------- phase 3: launch k6 on loadgen ----------
log "phase 3: launching k6 (RUNID=$RUNID)"
ssh -o ConnectTimeout=10 loadgen "nohup setsid env RUNID='$RUNID' \
  MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL TRACE_COMPRESS=1 OPEN_LOOP=1 \
  PRE_ALLOCATED_VUS=100 START_MIN=$START_MIN TEST_MIN=$TEST_MIN \
  MCR_CSV=$MCR_SLICE_LOADGEN bash /opt/loadgen/launch_k6.sh \
  > '$CONSOLE_SRC' 2>&1 < /dev/null &" 2>&1 | sed 's/^/k6launch: /' >> "$LOG"
log "k6 launch command issued"

# ---------- phase 4: traffic detection ----------
log "phase 4: waiting for traffic (ui rps > 0.2)"
t_traffic=""
for i in $(seq 1 240); do   # up to 20 min
  rps=$(curl -s -m 8 "http://127.0.0.1:18080/prometheus/api/v1/query" \
        --data 'query=sum(rate(istio_requests_total{reporter="destination",destination_workload="ts-ui-dashboard"}[30s]))' \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); r=d["data"]["result"]; print(r[0]["value"][1] if r else 0)' 2>/dev/null || echo 0)
  if python3 -c "import sys; sys.exit(0 if float('$rps' or 0) > 0.2 else 1)" 2>/dev/null; then
    t_traffic=$(date +%s); break
  fi
  sleep 5
done
if [ -z "$t_traffic" ]; then
  log "FATAL: no traffic detected in 20 min — aborting (tools keep running until end)"
  exit 1
fi
echo "$t_traffic" > "$RUN_DIR/traffic_start_epoch"
log "traffic detected epoch=$t_traffic (+$((t_traffic - LAUNCH_EPOCH)) s after launch)"

# ---------- phase 5: collector at traffic+600 ----------
COLLECT_AT=$((t_traffic + WARMUP_S))
WAIT=$((COLLECT_AT - $(date +%s)))
log "phase 5: collector in ${WAIT}s (traffic+$WARMUP_S), duration=${COLLECT_DUR}s interval=5"
if [ "$WAIT" -gt 0 ]; then sleep "$WAIT"; fi
DS="$RUN_DIR/dataset_$RUNID"
mkdir -p "$DS"
setsid env PBS_MIMIR_URL=http://127.0.0.1:18080 \
  pbscaler/.venv/bin/python pbscaler/scripts/collect_training_data.py \
  --duration "$COLLECT_DUR" --interval 5 --out "$DS" --workload "$MCR_SLICE_LOCAL" \
  < /dev/null > "$RUN_DIR/collector.log" 2>&1 &
log "collector started -> $DS (pid $!)"

# ---------- phase 6: SIGINT k6 at traffic+5680 ----------
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

# ---------- phase 7: wait for tools to finish ----------
WAIT=$((END_EPOCH + 90 - $(date +%s)))
log "phase 7: waiting ${WAIT}s for tools to finish (END_EPOCH+90)"
if [ "$WAIT" -gt 0 ]; then sleep "$WAIT"; fi
tail -1 "$RUN_DIR/failures.jsonl" >> "$LOG" 2>/dev/null
grep -c FAILURE_EVENT "$RUN_DIR/failures.jsonl" 2>/dev/null | xargs -I{} echo "failure events captured: {}" >> "$LOG"
log "k6 outputs on loadgen:"
ssh -o ConnectTimeout=8 loadgen "ls -la /opt/loadgen/k6/*_${RUNID}.*" >> "$LOG" 2>&1

# ---------- phase 8: validate collector dataset ----------
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
print("RESULT COMPLETE:", "COMPLETE" in txt.upper() or str(m.get("result","")).upper().find("COMPLETE") >= 0)
cov = m.get("coverage_pct") or 0
print("coverage_ok:", float(cov) >= 95 if cov else False)
EOF
if grep -q "coverage_ok: True" "$LOG" && grep -q "RESULT COMPLETE: True" "$LOG"; then
  log "dataset validation PASSED"
  DATA=$(ls "$DS"/*/train_data.csv "$DS"/train_data.csv 2>/dev/null | head -1)
else
  log "WARN: dataset validation flags above — proceeding to retrain only if train_data.csv exists"
  DATA=$(ls "$DS"/*/train_data.csv "$DS"/train_data.csv 2>/dev/null | head -1)
fi
if [ -z "${DATA:-}" ]; then
  log "FATAL: no train_data.csv — skipping retrain/bake. Archive + reset still run below."
else
  # ---------- phase 9: RF retrain ----------
  log "phase 9: RF retrain on $DATA"
  pbscaler/.venv/bin/python pbscaler/simulation/RandomForestClassify.py \
    --data "$DATA" --bench train_ticket --embargo-seconds 60 \
    >> "$RUN_DIR/rf_retrain.log" 2>&1 \
    && log "rf retrain OK" || log "FATAL: rf retrain FAILED (see rf_retrain.log)"

  # ---------- phase 10: freeze + hash + bake ----------
  log "phase 10: sha256 freeze + build_and_push"
  sha256sum pbscaler/simulation/train_ticket/RandomForestClassify.model \
            pbscaler/simulation/train_ticket/RandomForestClassify.svcs \
            > "$RUN_DIR/model_sha256.txt" 2>&1
  cat "$RUN_DIR/model_sha256.txt" >> "$LOG"
  if bash pbscaler/deploy/build_and_push.sh >> "$RUN_DIR/build_push.log" 2>&1; then
    log "build_and_push OK (image younesnb/predictive-k8s-autoscaler re-baked with new RF)"
  else
    log "FATAL: build_and_push FAILED (see build_push.log)"
  fi
  ls pbscaler/simulation/train_ticket/support_envelope.json \
     pbscaler/simulation/train_ticket/model_card.json >> "$LOG" 2>&1 \
     && log "support_envelope + model_card present"
fi

# ---------- phase 11: archive RUN_DIR + k6 outputs to loadgen ----------
log "phase 11: archiving run dir to loadgen"
tar czf - -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")" \
  | ssh -o ConnectTimeout=8 loadgen "cat > /opt/loadgen/failure_logs/${RUNID}_rundir.tgz" 2>/dev/null \
  && log "archive OK -> loadgen:/opt/loadgen/failure_logs/${RUNID}_rundir.tgz" \
  || log "WARN: archive failed"

# ---------- phase 12: final cold reset ----------
log "phase 12: final cold reset"
if pbscaler/.venv/bin/python pbscaler/scripts/launchers/reset_cluster_cold.py \
     > "$RUN_DIR/final_cold_reset.log" 2>&1; then
  log "final cold reset OK"
else
  log "WARN: final cold reset flagged issues (see final_cold_reset.log)"
fi
log "=== rerun25k_fullcapture DONE runid=$RUNID ==="
