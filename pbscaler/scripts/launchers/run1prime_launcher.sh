#!/bin/bash
# Run1' launcher: HPA-only Jul-13 training collection (paper Sec. IV-C2).
# k6 first -> TRUE 30-min warmup sleep -> collector 6 h. Detached via setsid.
set -u
MCR=/proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv
OUT_PREFIX=run1prime_hpa_jul13
OUT_SUFFIX=$(date -u +%Y%m%d_%H%M%S)
TARGET_URL=https://train-ticket.younesnb.linkpc.net
MAX_REQUESTS=7600
USER_POOL=560
START_MIN=573
TEST_MIN=390        # 30 min warmup + 6 h collection
COLLECT_MIN=360     # 6 h
LOG=/tmp/opencode/run1prime_launcher.log
cd /users/younes02/predictive-k8s-autoscaling

echo "$(date -u +%FT%TZ) k6 starting (TEST_MIN=${TEST_MIN}, START_MIN=${START_MIN}, COMPRESS=8)" >>"$LOG"
setsid env TARGET_URL="$TARGET_URL" \
  MCR_CSV="$MCR" MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL \
  SEED_FILE=load_testing/seeds/tt_users.json \
  START_MIN=$START_MIN TEST_MIN=$TEST_MIN TRACE_COMPRESS=8 \
  bash load_testing/run_test.sh >"/tmp/opencode/${OUT_PREFIX}_k6.log" 2>&1 &
K6_PID=$!
echo "$(date -u +%FT%TZ) k6 pid $K6_PID" >>"$LOG"

echo "$(date -u +%FT%TZ) warmup ${WARMUP:-30}min..." >>"$LOG"
sleep 1800

echo "$(date -u +%FT%TZ) collector starting (${COLLECT_MIN}min)" >>"$LOG"
PBS_MIMIR_URL=http://127.0.0.1:18081 \
  ./pbscaler/.venv/bin/python pbscaler/scripts/collect_training_data.py \
    --duration $((COLLECT_MIN * 60)) --interval 5 \
    --workload "$MCR" \
    --out "pbscaler/data/${OUT_PREFIX}_${OUT_SUFFIX}" >"/tmp/opencode/${OUT_PREFIX}_collector.log" 2>&1 &
COLLECT_PID=$!
echo "$(date -u +%FT%TZ) collector pid $COLLECT_PID (out: pbscaler/data/${OUT_PREFIX}_${OUT_SUFFIX})" >>"$LOG"

wait $COLLECT_PID || true
echo "$(date -u +%FT%TZ) collector done; k6 continues to ${TEST_MIN}min total" >>"$LOG"
wait $K6_PID || true
echo "$(date -u +%FT%TZ) k6 done — Run1' complete" >>"$LOG"
