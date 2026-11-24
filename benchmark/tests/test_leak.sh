#!/bin/bash
# Verify (a) stale collectors from a "previous run" are killed at phase 0, and
# (b) a complete --local run leaves no collector or kubectl stream behind.
set -u
REPO=/proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
ORCH="$REPO/benchmark/attic/run1h25k_jul13_fullcapture.sh"
OUT=/tmp/opencode/orch_leak_test
rm -rf "$OUT"; mkdir -p "$OUT"

# A stand-in for a collector orphaned by an interrupted run. The cmdline must
# contain "failure_recorder.py" and must NOT contain the new RUN_DIR.
setsid bash -c 'exec -a "python3 /opt/fake/failure_recorder.py --run-dir /opt/fake/OLD_RUN" sleep 900' \
  </dev/null >/dev/null 2>&1 &
STALE=$!
sleep 1
echo "stale stand-in pid=$STALE alive=$(kill -0 $STALE 2>/dev/null && echo yes || echo no)"
KUBE_BEFORE=$(pgrep -c -x kubectl)

PORT=18091
python3 /tmp/opencode/synth_server.py "$PORT" "$OUT/synth.log" & SRV=$!
sleep 1.5
env MAX_REQUESTS=60 TEST_MIN=2 START_MIN=0 USER_POOL=5 PRE_ALLOCATED_VUS=5 MAX_VUS=5 \
    WARMUP_S=0 K6_DEADLINE_OFFSET=200 TOOLS_END_OFFSET=240 COLLECT_DUR=60 MAX_STREAMS=8 \
    TARGET_URL="http://127.0.0.1:$PORT" \
    SKIP_TRAFFIC_GATE=1 SKIP_DATASET=1 SKIP_DUMPS=1 SKIP_RESET=1 \
    PATH=/tmp/opencode/fakebin:$PATH \
    bash "$ORCH" --local
RC=$?
kill "$SRV" 2>/dev/null

LD=$(ls -td "$REPO"/benchmark/runs/hpa1h25k_jul13_rr_* | head -1)
echo "orchestrator exit=$RC"
echo "stale stand-in alive after run: $(kill -0 $STALE 2>/dev/null && echo YES-LEAK || echo no)"
echo "kubectl: before=$KUBE_BEFORE after=$(pgrep -c -x kubectl)"
echo "collectors left: $(pgrep -cf 'log_[s]treamer.sh|failure_[r]ecorder.py|monitor_[v]2.py' 2>/dev/null || echo 0)"
echo "kubectl log streams left: $(pgrep -cf 'kubect[l].*logs -f --timestamps --all-containers' 2>/dev/null || echo 0)"
echo "ssh calls: $(wc -l < /tmp/opencode/ssh_calls.log 2>/dev/null || echo 0)"
echo "status: $(python3 -c "import json;d=json.load(open('$LD/run_status.json'));print(d['status'], d['reasons'])")"
echo "streams started: $(grep -c 'pod streams' "$LD/log_streamer.out" 2>/dev/null) -> $(head -1 "$LD/log_streamer.out")"
echo "sidecar files: $(ls "$LD/logs" | grep -c 'istio-proxy' 2>/dev/null)"
echo "stale-kill log line:"; grep -i "stale collector" "$LD/orchestrator.log" || echo "  (none)"
kill "$STALE" 2>/dev/null
exit 0