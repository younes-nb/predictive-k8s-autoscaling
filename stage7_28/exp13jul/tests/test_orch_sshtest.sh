#!/bin/bash
# Fast orchestrator integration test with ssh/scp poisoned.
#
# A fake `ssh`/`scp` that logs any invocation and exits 255 is placed first in
# PATH. If the official --local path touched the network at all, the run would
# both log it and fail. Success = STATUS=complete AND an empty ssh_calls.log.
#
#   bash /tmp/opencode/test_orch_sshtest.sh
set -u
REPO=/proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
ORCH="$REPO/stage7_28/exp13jul/run1h25k_jul13_fullcapture.sh"
OUT=/tmp/opencode/orch_sshtest
rm -rf "$OUT"; mkdir -p "$OUT"
rm -f /tmp/opencode/ssh_calls.log /tmp/opencode/scp_calls.log

PORT=${PORT:-18090}
SYNTH_FAIL=0 python3 /tmp/opencode/synth_server.py "$PORT" "$OUT/synth_server.log" &
SRV=$!
sleep 1.5

env MAX_REQUESTS=60 TEST_MIN=2 START_MIN=0 \
    USER_POOL=5 PRE_ALLOCATED_VUS=5 MAX_VUS=5 \
    WARMUP_S=0 K6_DEADLINE_OFFSET=200 TOOLS_END_OFFSET=240 COLLECT_DUR=60 \
    TARGET_URL="http://127.0.0.1:$PORT" \
    SKIP_TRAFFIC_GATE=1 SKIP_DATASET=1 SKIP_DUMPS=1 SKIP_RESET=1 \
    PATH=/tmp/opencode/fakebin:$PATH \
    bash "$ORCH" --local
RC=$?
kill "$SRV" 2>/dev/null

echo "orchestrator exit=$RC"
echo "--- ssh calls: $(wc -l < /tmp/opencode/ssh_calls.log 2>/dev/null || echo 0)"
echo "--- scp calls: $(wc -l < /tmp/opencode/scp_calls.log 2>/dev/null || echo 0)"
cat /tmp/opencode/ssh_calls.log /tmp/opencode/scp_calls.log 2>/dev/null
exit 0