#!/bin/bash
# 32k-shape 13-July HPA run — full-capture orchestrator (10 min warm-up + 1 h measured).
#
# Derived from run32k_fullcapture.sh (14-Jul 32k) + run13jul_fullcapture.sh (13-Jul
# 25k). Differences vs run13jul_fullcapture.sh:
#   * MAX_REQUESTS=32000 instead of 25000 (the 32k ladder point; peak arrival
#     ~533 iters/s vs 417 for 25k) — same VU pool as the 14-Jul 32k run
#     (USER_POOL=6000 / PRE_ALLOCATED_VUS=2000) so the generator does not
#     saturate before the cluster does.
#   * the capture tools now really start: run13jul_fullcapture.sh referenced
#     stage7_28/{failure_recorder.py,monitor_v2.py,log_streamer.sh} which do not
#     exist (they live in stage7_28/exp13jul/) — that run captured no failure
#     events at all.
#   * tunnels are resolved from stage7_28/ with a self-healing kube-apiserver
#     forward (kube_tunnel.sh), because the runner's 127.0.0.1:16443 forward is
#     what every kubectl/kubernetes-client call depends on.
#   * colder failure-event thresholds (5XX 1 %, 4XX 5 %, 45 s cooldown) so the
#     recorder logs *every* failing service, not just the loud ones.
#   * k6 summary + samples census are pulled into RUN_DIR at the end.
#
# Run shape: 13-Jul NASA slice (md5 3657e395...), START_MIN=529, TEST_MIN=70
#            (10 min warm-up + 60 min measured), MCR jul13.
# Usage: setsid bash stage7_28/exp13jul/run32k_jul13_fullcapture.sh > /tmp/run32k_jul13.log 2>&1 &
set -u
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH=/users/younes02/.local/bin:$PATH
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

# ---- parameters (13-Jul slice, 32k shape, 70-min duration) ----
MAX_REQUESTS=32000
USER_POOL=6000
PRE_ALLOC=2000
START_MIN=529
TEST_MIN=70             # 70 min k6 scenario (10 warm-up + 60 measured)
WARMUP_S=600            # traffic -> measured window start (10-min warm-up)
K6_STOP_OFFSET=4200     # traffic -> SIGINT k6 (600 warm-up + 3600 measured)
COLLECT_DUR=3600        # post-hoc collection window @ 5 s (60 min measured)
TOOLS_END_OFFSET=4800   # launch-relative stop for all tools
MCR_SLICE_LOCAL="$PWD/stage7_28/exp13jul/mcr_jul13_slice_local.csv"
MCR_SLICE_LOADGEN=/opt/loadgen/mcr_jul13_slice.csv
SSH_CFG=/proj/k8sautoscaledl-PG0/src/cluster-access/ssh_config
SSH="ssh -F $SSH_CFG -o BatchMode=yes -o ConnectTimeout=8"
SCP="scp -F $SSH_CFG -o BatchMode=yes"
EXPECT_MCR_MD5=3657e39522f8a2e99b61eda0f58031e1
EXPECT_K6_MD5=618caae411c1416a10e9b27e0624fd2f
TOOLS="$PWD/stage7_28/exp13jul"
PY="$PWD/pbscaler/.venv/bin/python"
S5XX=${S5XX:-0.01}
S4XX=${S4XX:-0.05}
SCOOL=${SCOOL:-45}

DRY=${1:-}

RUNID=hpa32k_jul13_rr_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$PWD/stage7_28/exp13jul/$RUNID"
mkdir -p "$RUN_DIR/logs"
CONSOLE_SRC="/opt/loadgen/k6/console_${RUNID}.log"
echo "$RUNID" > "$RUN_DIR/run_id"
echo "$RUNID" > "$PWD/stage7_28/exp13jul/CURRENT_RUN_ID"
LOG="$RUN_DIR/orchestrator.log"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
die() { log "FATAL: $*"; exit 1; }

log "=== run32k_jul13_fullcapture START runid=$RUNID dry=${DRY:-no} ==="
log "params: MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL PRE_ALLOCATED_VUS=$PRE_ALLOC START_MIN=$START_MIN TEST_MIN=$TEST_MIN"
log "thresholds: 5XX_RATE_MIN=$S5XX 4XX_RATE_MIN=$S4XX COOLDOWN=$SCOOL"
cat > "$RUN_DIR/params.json" <<EOF
{"runid": "$RUNID", "bench": "train_ticket", "autoscaler": "hpa", "trace_day": "jul13",
 "max_requests": $MAX_REQUESTS, "user_pool": $USER_POOL, "pre_allocated_vus": $PRE_ALLOC,
 "start_min": $START_MIN, "test_min": $TEST_MIN, "warmup_s": $WARMUP_S,
 "k6_stop_offset": $K6_STOP_OFFSET, "collect_dur": $COLLECT_DUR, "tools_end_offset": $TOOLS_END_OFFSET,
 "mcr_csv": "$MCR_SLICE_LOADGEN", "mcr_md5": "$EXPECT_MCR_MD5", "k6_script_md5": "$EXPECT_K6_MD5",
 "five_xx_rate_min": $S5XX, "four_xx_rate_min": $S4XX, "event_cooldown_s": $SCOOL}
EOF

# ---------- phase -1: preflight (required files + generator assets) ----------
log "phase -1: preflight"
for f in "$MCR_SLICE_LOCAL" "$TOOLS/failure_recorder.py" "$TOOLS/monitor_v2.py" \
         "$TOOLS/log_streamer.sh" "$TOOLS/rebuild_train_data.py" \
         "$TOOLS/dump_mimir.sh" "$TOOLS/dump_loki.sh" "$PY"; do
  [ -f "$f" ] || die "missing required file: $f"
done
[ -x "$PY" ] || die "not executable: $PY"
SRV_CHECK=$($SSH loadgen "md5sum $MCR_SLICE_LOADGEN /opt/loadgen/trainticket.js 2>/dev/null" 2>/dev/null | sed 's/^#*//')
echo "$SRV_CHECK" | grep -q "$EXPECT_MCR_MD5" || die "loadgen slice md5 mismatch (expected $EXPECT_MCR_MD5): $SRV_CHECK"
echo "$SRV_CHECK" | grep -q "$EXPECT_K6_MD5" || die "loadgen k6 script md5 mismatch (expected $EXPECT_K6_MD5): $SRV_CHECK"
log "preflight OK: slice + k6 script md5 verified on loadgen"

# ---------- phase 0: cold gate (needs the apiserver forward) ----------
log "phase 0: tunnels + cold reset assert gate"
pgrep -f "kube_tunnel.sh"  > /dev/null || setsid bash "$PWD/stage7_28/kube_tunnel.sh"  < /dev/null > /dev/null 2>&1 &
pgrep -f "mimir_tunnel.sh" > /dev/null || setsid bash "$PWD/stage7_28/mimir_tunnel.sh" < /dev/null > /dev/null 2>&1 &
pgrep -f "loki_tunnel.sh"  > /dev/null || setsid bash "$PWD/stage7_28/loki_tunnel.sh"  < /dev/null > /dev/null 2>&1 &
for i in $(seq 1 30); do
  kubectl get nodes >/dev/null 2>&1 && break
  sleep 2
done
kubectl get nodes >/dev/null 2>&1 || die "kube apiserver forward (127.0.0.1:16443) not reachable"
curl -fsS -m 8 "http://127.0.0.1:18080/prometheus/api/v1/query" --data 'query=up' > /dev/null \
  || die "mimir tunnel not reachable"
curl -fsS -m 8 "http://127.0.0.1:3100/ready" > /dev/null || die "loki tunnel not reachable"
log "tunnels verified up (kube + mimir + loki)"

if ! "$PY" pbscaler/scripts/launchers/reset_cluster_cold.py --assert-only \
     > "$RUN_DIR/cold_gate.log" 2>&1; then
  die "cold gate failed (see cold_gate.log)"
fi
log "cold gate passed: $(grep -m1 'ASSERT cold_ok' "$RUN_DIR/cold_gate.log")"

UPNOW=$(curl -s -m 10 "http://127.0.0.1:18080/prometheus/api/v1/query" \
        --data-urlencode 'query=up' 2>/dev/null \
        | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin); print(len(d["data"]["result"]))
except Exception:
  print(0)' 2>/dev/null)
[ "${UPNOW:-0}" -lt 1 ] && die "mimir up@now=$UPNOW (ingester not serving current data)"
log "mimir up@now=$UPNOW — healthy"

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

setsid bash "$TOOLS/log_streamer.sh" "$RUN_DIR" "$END_EPOCH" \
  < /dev/null >> "$RUN_DIR/log_streamer.out" 2>&1 &
log "log_streamer started (pid $!)"

setsid env PBS_MIMIR_URL=http://127.0.0.1:18080 python3 \
  "$TOOLS/monitor_v2.py" --run-dir "$RUN_DIR" --t0 0 \
  --end "$END_EPOCH" --mcr "$MCR_SLICE_LOCAL" --max-requests "$MAX_REQUESTS" \
  --start-min "$START_MIN" < /dev/null >> "$RUN_DIR/monitor_v2.out" 2>&1 &
log "monitor_v2 started (pid $!)"

setsid env PBS_MIMIR_URL=http://127.0.0.1:18080 LOKI_URL=http://127.0.0.1:3100 \
  KUBECTL=/users/younes02/.local/bin/kubectl KUBECONFIG="$KUBECONFIG" \
  5XX_RATE_MIN="$S5XX" 4XX_RATE_MIN="$S4XX" EVENT_COOLDOWN_S="$SCOOL" \
  python3 "$TOOLS/failure_recorder.py" --run-dir "$RUN_DIR" \
  --end "$END_EPOCH" --runid "$RUNID" < /dev/null >> "$RUN_DIR/failure_recorder.out" 2>&1 &
log "failure_recorder started (pid $!)"

# k6 console offset-pull (loadgen -> RUN_DIR/k6_console.log) for monitor VU parsing
(
  off=1
  while [ "$(date +%s)" -lt $((END_EPOCH + 120)) ]; do
    sz=$($SSH loadgen "wc -c < '$CONSOLE_SRC' 2>/dev/null" 2>/dev/null || echo 0)
    case "$sz" in (''|*[!0-9]*) sz=0;; esac
    if [ "$sz" -gt "$off" ]; then
      $SSH loadgen "tail -c +$off '$CONSOLE_SRC'" >> "$RUN_DIR/k6_console.log" 2>/dev/null
      echo "$(date -u +%FT%TZ) shipped console bytes $((sz-off+1)) (total $sz)" >> "$RUN_DIR/console_shipper.out"
      off=$((sz + 1))
    fi
    sleep 15
  done
) < /dev/null > /dev/null 2>&1 &
log "console shipper started (pid $!)"

# failures.jsonl offset-push (RUN_DIR -> loadgen, live during the run)
$SSH loadgen "mkdir -p /opt/loadgen/failure_logs/$RUNID" 2>/dev/null
(
  foff=0
  while [ "$(date +%s)" -lt $((END_EPOCH + 60)) ]; do
    if [ -f "$RUN_DIR/failures.jsonl" ]; then
      fsz=$(stat -c%s "$RUN_DIR/failures.jsonl" 2>/dev/null || echo 0)
      if [ "$fsz" -gt "$foff" ]; then
        tail -c +$((foff + 1)) "$RUN_DIR/failures.jsonl" \
          | $SSH loadgen "cat >> /opt/loadgen/failure_logs/$RUNID/failures.jsonl" 2>/dev/null
        echo "$(date -u +%FT%TZ) pushed failures bytes $((fsz - foff)) (total $fsz)" >> "$RUN_DIR/failures_shipper.out"
        foff=$fsz
      fi
    fi
    sleep 10
  done
) < /dev/null > /dev/null 2>&1 &
log "failures shipper started -> loadgen:/opt/loadgen/failure_logs/$RUNID/failures.jsonl (pid $!)"

# ---------- phase 3: launch k6 on loadgen ----------
log "phase 3: launching k6 (RUNID=$RUNID, maxVUs=$USER_POOL, preAllocatedVUs=$PRE_ALLOC)"
$SSH loadgen "ulimit -n 65536; nohup setsid env RUNID='$RUNID' \
  MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL TRACE_COMPRESS=1 OPEN_LOOP=1 \
  PRE_ALLOCATED_VUS=$PRE_ALLOC START_MIN=$START_MIN TEST_MIN=$TEST_MIN \
  MCR_CSV=$MCR_SLICE_LOADGEN bash /opt/loadgen/launch_k6.sh \
  > '$CONSOLE_SRC' 2>&1 < /dev/null &" 2>&1 | sed 's/^/k6launch: /' >> "$LOG"
log "k6 launch command issued"

sleep 25
KSZ=$($SSH loadgen "wc -c < '$CONSOLE_SRC' 2>/dev/null" 2>/dev/null || echo 0)
case "$KSZ" in (''|*[!0-9]*) KSZ=0;; esac
[ "$KSZ" -lt 100 ] && die "k6 console log size=$KSZ after 25 s — k6 did not start"
log "k6 console confirmed growing (size=$KSZ)"

# ---------- phase 3.5: capture-tool liveness ----------
sleep 45
pgrep -f "$TOOLS/failure_recorder.py" >/dev/null || die "failure_recorder died at startup (see failure_recorder.out)"
[ -f "$RUN_DIR/failures.jsonl" ] || die "failure_recorder produced no failures.jsonl"
log "capture tools alive; failures.jsonl present ($(stat -c%s "$RUN_DIR/failures.jsonl") B)"
for p in log_streamer.sh monitor_v2.py; do
  pgrep -f "$p" >/dev/null && log "  $p running" || log "  WARN: $p not running"
done

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
[ -z "$t_traffic" ] && die "no traffic detected in 20 min"
echo "$t_traffic" > "$RUN_DIR/traffic_start_epoch"
log "traffic detected epoch=$t_traffic (+$((t_traffic - LAUNCH_EPOCH)) s after launch)"
log "measured window: $((t_traffic + WARMUP_S)) .. $((t_traffic + K6_STOP_OFFSET))"

# ---------- phase 5: no live collector (post-hoc by design) ----------
log "phase 5: skipped (collector is post-hoc from Mimir)"

# ---------- phase 6: SIGINT k6 at traffic+4200 ----------
STOP_AT=$((t_traffic + K6_STOP_OFFSET))
WAIT=$((STOP_AT - $(date +%s)))
log "phase 6: k6 SIGINT in ${WAIT}s (traffic+$K6_STOP_OFFSET)"
[ "$WAIT" -gt 0 ] && sleep "$WAIT"

# mid-run snapshot of the failure timeline for the report
cp -f "$RUN_DIR/failures.jsonl" "$RUN_DIR/failures_snapshot_at_stop.jsonl" 2>/dev/null || true
log "failure events so far: $(grep -c FAILURE_EVENT "$RUN_DIR/failures.jsonl" 2>/dev/null || echo 0)"

KP=$($SSH loadgen "pgrep -x k6 | head -1" 2>/dev/null || true)
if [ -n "${KP:-}" ]; then
  $SSH loadgen "kill -INT $KP" && log "k6 SIGINT sent (pid $KP)"
else
  log "WARN: no k6 process found on loadgen at SIGINT time"
fi

# wait for k6 to flush its summary
for i in $(seq 1 36); do
  $SSH loadgen "pgrep -x k6 >/dev/null" 2>/dev/null || break
  sleep 5
done
log "k6 exited; pulling summary + console sizes"
$SSH loadgen "ls -la /opt/loadgen/k6/*_${RUNID}.*" >> "$LOG" 2>&1
$SCP -q loadgen:/opt/loadgen/k6/summary_${RUNID}.json "$RUN_DIR/k6_summary.json" 2>/dev/null \
  && log "k6 summary pulled -> k6_summary.json" || log "WARN: k6 summary pull failed"

# ---------- phase 5b: post-hoc training-data collection from Mimir ----------
log "phase 5b: post-hoc dataset collection (waiting 180 s for Alloy flush)"
sleep 180
DS="$RUN_DIR/dataset_$RUNID"
mkdir -p "$DS"
CSTART=$((t_traffic + WARMUP_S))
if "$PY" "$TOOLS/rebuild_train_data.py" \
     --start "$CSTART" --duration "$COLLECT_DUR" --interval 5 \
     --out "$DS" --workload "$MCR_SLICE_LOCAL" \
     > "$RUN_DIR/rebuild_collection.log" 2>&1; then
  log "post-hoc collection OK -> $DS"
else
  log "FATAL: post-hoc collection failed (see rebuild_collection.log)"
fi
tail -14 "$RUN_DIR/rebuild_collection.log" >> "$LOG" 2>/dev/null

# ---------- phase 7: wait for tools to finish ----------
WAIT=$((END_EPOCH + 90 - $(date +%s)))
log "phase 7: waiting ${WAIT}s for tools to finish (END_EPOCH+90)"
[ "$WAIT" -gt 0 ] && sleep "$WAIT"
grep -c FAILURE_EVENT "$RUN_DIR/failures.jsonl" 2>/dev/null | xargs -I{} echo "failure events captured: {}" >> "$LOG"
log "failure event types: $(grep -o '"type": "[A-Z_]*"' "$RUN_DIR/failures.jsonl" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"

# ---------- phase 8: validate dataset ----------
log "phase 8: validating dataset meta.json"
python3 - "$DS" >> "$LOG" 2>&1 <<'EOF'
import json, sys, glob
ds = sys.argv[1]
metas = glob.glob(ds + "/**/meta.json", recursive=True) or glob.glob(ds + "/meta.json")
if not metas:
    print("meta.json NOT FOUND in", ds); raise SystemExit(1)
m = json.load(open(metas[0]))
txt = open(metas[0]).read()
print("meta:", {k: m.get(k) for k in ("result", "coverage_pct", "interval_s", "duration_s", "no_data_rows", "rows") if k in m})
print("RESULT COMPLETE:", "COMPLETE" in txt.upper())
cov = m.get("coverage_pct") or 0
print("coverage_ok:", float(cov) >= 95 if cov else False)
EOF

# ---------- phase 10: Mimir + Loki range dumps ----------
log "phase 10: Mimir + Loki dumps (launch-60 .. END_EPOCH)"
DUMP_DIR="$RUN_DIR/dumps"
mkdir -p "$DUMP_DIR"
bash "$TOOLS/dump_mimir.sh" "$((LAUNCH_EPOCH - 60))" "$END_EPOCH" \
     "$DUMP_DIR/mimir" >> "$LOG" 2>&1 \
  && log "mimir dumps OK -> $DUMP_DIR/mimir" || log "WARN: mimir dumps failed"
bash "$TOOLS/dump_loki.sh" "$((LAUNCH_EPOCH - 60))" "$END_EPOCH" \
     "$DUMP_DIR/loki" >> "$LOG" 2>&1 \
  && log "loki dump OK -> $DUMP_DIR/loki" || log "WARN: loki dump failed"

# ---------- phase 11: archive RUN_DIR + k6 outputs to loadgen ----------
log "phase 11: archiving run dir to loadgen"
tar czf - -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")" \
  | $SSH loadgen "cat > /opt/loadgen/failure_logs/${RUNID}_rundir.tgz" 2>/dev/null \
  && log "archive OK -> loadgen:/opt/loadgen/failure_logs/${RUNID}_rundir.tgz" \
  || log "WARN: archive failed"

# ---------- phase 12: final cold reset ----------
log "phase 12: final cold reset"
if "$PY" pbscaler/scripts/launchers/reset_cluster_cold.py \
     --out "$RUN_DIR/final_cold_reset.json" > "$RUN_DIR/final_cold_reset.log" 2>&1; then
  log "final cold reset OK: $(grep -m1 'POST:' "$RUN_DIR/final_cold_reset.log" | cut -c1-400)"
else
  log "WARN: final cold reset flagged issues"
  tail -5 "$RUN_DIR/final_cold_reset.log" >> "$LOG"
fi

log "=== run32k_jul13_fullcapture DONE runid=$RUNID ==="
echo "RUNID=$RUNID RUN_DIR=$RUN_DIR"
