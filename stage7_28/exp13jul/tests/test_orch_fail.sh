#!/bin/bash
# Criterion 8 + 12: k6's own exit status is preserved, and a dead collector
# yields an INCOMPLETE run. Both are bounded, synthetic, and never touch
# TrainTicket.
#
#   bash /tmp/opencode/test_orch_fail.sh
set -u
REPO=/proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
ORCH="$REPO/stage7_28/exp13jul/run1h25k_jul13_fullcapture.sh"
OUT=/tmp/opencode/orch_fail_test
rm -rf "$OUT"; mkdir -p "$OUT"

echo "### CASE A: k6 fails its own threshold -> k6 exit 99 must be preserved"
PORT=18083
SYNTH_FAIL=1 python3 /tmp/opencode/synth_server.py "$PORT" "$OUT/synth_a.log" &
SRV=$!
sleep 1.5
env MAX_REQUESTS=60 TEST_MIN=1 START_MIN=0 \
    USER_POOL=5 PRE_ALLOCATED_VUS=5 MAX_VUS=5 \
    WARMUP_S=0 K6_DEADLINE_OFFSET=140 TOOLS_END_OFFSET=180 COLLECT_DUR=60 \
    TARGET_URL="http://127.0.0.1:$PORT" \
    SKIP_TRAFFIC_GATE=1 SKIP_DATASET=1 SKIP_DUMPS=1 SKIP_RESET=1 \
    bash "$ORCH" --local
echo "orchestrator exit=$?"
kill "$SRV" 2>/dev/null
LD=$(ls -td "$REPO"/stage7_28/exp13jul/hpa1h25k_jul13_rr_* | head -1)
echo "run_status: $(python3 -c "import json,sys;d=json.load(open('$LD/run_status.json'));print(d['status'], d['reasons'], 'k6_rc=',d['k6_exit_code'])")"
echo "k6_exit_code file: $(cat "$LD/k6_exit_code" 2>&1)"

echo
echo "### CASE B: failure_recorder killed mid-run -> INCOMPLETE"
PORT=18084
python3 /tmp/opencode/synth_server.py "$PORT" "$OUT/synth_b.log" &
SRV=$!
sleep 1.5
(
  sleep 130
  PID=$(pgrep -f "failure_recorder.py --run-dir" | head -1)
  echo "killing failure_recorder pid=$PID at $(date -u +%FT%TZ)"
  [ -n "$PID" ] && kill -KILL "$PID"
) &
KILLER=$!
env MAX_REQUESTS=60 TEST_MIN=2 START_MIN=0 \
    USER_POOL=5 PRE_ALLOCATED_VUS=5 MAX_VUS=5 \
    WARMUP_S=0 K6_DEADLINE_OFFSET=200 TOOLS_END_OFFSET=240 COLLECT_DUR=60 \
    TARGET_URL="http://127.0.0.1:$PORT" \
    SKIP_TRAFFIC_GATE=1 SKIP_DATASET=1 SKIP_DUMPS=1 SKIP_RESET=1 \
    bash "$ORCH" --local
echo "orchestrator exit=$?"
kill "$SRV" $KILLER 2>/dev/null
LD=$(ls -td "$REPO"/stage7_28/exp13jul/hpa1h25k_jul13_rr_* | head -1)
echo "run_status: $(python3 -c "import json;d=json.load(open('$LD/run_status.json'));print(d['status'], d['reasons'], 'k6_rc=',d['k6_exit_code'])")"
exit 0