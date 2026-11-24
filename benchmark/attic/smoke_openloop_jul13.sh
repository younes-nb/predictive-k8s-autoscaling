#!/bin/bash
# Low-rate real-cluster smoke test for the corrected open-loop k6 script.
# MAX_REQUESTS=1200 (peak 20 iterations/s); the first three trace buckets at
# START_MIN=529 run at 10-14 iterations/s. Captures the k6 summary and leaves
# the cluster to the caller for a cold reset.
#
# Usage: bash benchmark/smoke_openloop_jul13.sh
set -u
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling

SSH_CFG=/proj/k8sautoscaledl-PG0/src/cluster-access/ssh_config
SSH="ssh -F $SSH_CFG -o BatchMode=yes -o ConnectTimeout=8"
SCP="scp -F $SSH_CFG -o BatchMode=yes"

MAX_REQUESTS=${MAX_REQUESTS:-1200}
USER_POOL=${USER_POOL:-300}
PRE_ALLOC=${PRE_ALLOC:-300}
MAX_VUS=${MAX_VUS:-300}
START_MIN=${START_MIN:-529}
TEST_MIN=${TEST_MIN:-3}
RUNID=${RUNID:-smoke_openloop_jul13_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-/proj/k8sautoscaledl-PG0/reports/stage7_28/$RUNID}
CONSOLE=/opt/loadgen/k6/console_${RUNID}.log

mkdir -p "$OUT"
echo "RUNID=$RUNID OUT=$OUT" | tee "$OUT/smoke.log"

$SSH loadgen "ulimit -n 65536; nohup setsid env RUNID='$RUNID' \
  MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL TRACE_COMPRESS=1 OPEN_LOOP=1 \
  PRE_ALLOCATED_VUS=$PRE_ALLOC MAX_VUS=$MAX_VUS PREFLIGHT=0 ALLOW_SEED_REUSE=0 \
  START_MIN=$START_MIN TEST_MIN=$TEST_MIN \
  MCR_CSV=/opt/loadgen/mcr_jul13_slice.csv bash /opt/loadgen/launch_k6.sh \
  > '$CONSOLE' 2>&1 < /dev/null &" 2>&1 | sed 's/^/launch: /' | tee -a "$OUT/smoke.log"

echo "waiting for k6 to finish..." | tee -a "$OUT/smoke.log"
for i in $(seq 1 60); do
  $SSH loadgen "pgrep -x k6 >/dev/null" 2>/dev/null || break
  sleep 5
done

$SSH loadgen "ls -la /opt/loadgen/k6/*_${RUNID}.*" > "$OUT/loadgen_files.txt" 2>&1
$SCP -q loadgen:/opt/loadgen/k6/summary_${RUNID}.json "$OUT/k6_summary.json" 2>/dev/null \
  && echo "summary pulled" | tee -a "$OUT/smoke.log" \
  || echo "WARN: summary pull failed" | tee -a "$OUT/smoke.log"
$SCP -q loadgen:"$CONSOLE" "$OUT/k6_console.log" 2>/dev/null || true

python3 - "$OUT/k6_summary.json" <<'EOF' | tee -a "$OUT/smoke.log"
import json, sys
d = json.load(open(sys.argv[1])); m = d["metrics"]
def g(k, f="count"):
    return (m.get(k) or {}).get(f)
print("iterations=", g("iterations"), "http_reqs=", g("http_reqs"),
      "failed=", g("http_req_failed", "value"), "dropped=", g("dropped_iterations"),
      "vus_max=", g("vus_max", "value"))
idur = m.get("iteration_duration") or {}
print("iteration_duration avg=%.1fms p95=%.1fms max=%.1fms" % (
    idur.get("avg", 0), idur.get("p(95)", 0), idur.get("max", 0)))
print("auth: attempts=%s initial_ok=%s refresh_ok=%s failures=%s retries=%s skipped_iters=%s" % (
    g("auth_login_attempts_total"), g("auth_initial_login_success_total"),
    g("auth_refresh_success_total"), g("auth_login_failures_total"),
    g("auth_login_retries_total"), g("auth_skipped_iterations_total")))
EOF
