#!/bin/bash
# Fast orchestrator integration test (local transport, synthetic target).
#
# Exercises the REAL orchestrator with a tiny shape against the synthetic
# server: real k6, real log_streamer, real failure_recorder, real archive.
# Never sends traffic to TrainTicket; never uses the 25k shape.
#
#   bash /tmp/opencode/test_orch_fast.sh
set -u
REPO=/proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
ORCH="$REPO/benchmark/run1h25k_jul13_fullcapture.sh"
OUT=/tmp/opencode/orch_fast_test
rm -rf "$OUT"; mkdir -p "$OUT"

PORT=${PORT:-18082}
python3 /tmp/opencode/synth_server.py "$PORT" "$OUT/synth_server.log" &
SRV=$!
sleep 1.5

START=$(date +%s)
env MAX_REQUESTS=60 TEST_MIN=2 START_MIN=0 \
    USER_POOL=5 PRE_ALLOCATED_VUS=5 MAX_VUS=5 \
    WARMUP_S=0 K6_DEADLINE_OFFSET=200 TOOLS_END_OFFSET=240 COLLECT_DUR=60 \
    TARGET_URL="http://127.0.0.1:$PORT" \
    SKIP_TRAFFIC_GATE=1 SKIP_DATASET=1 SKIP_DUMPS=1 SKIP_RESET=1 \
    bash "$ORCH" --local
RC=$?
kill "$SRV" 2>/dev/null
echo "orchestrator exit=$RC elapsed=$(( $(date +%s) - START ))s"
exit 0