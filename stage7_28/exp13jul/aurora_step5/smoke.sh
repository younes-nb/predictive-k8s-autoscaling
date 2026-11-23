#!/usr/bin/env bash
# One AURORA development smoke test: reset the workload pods, run a 10 min load
# test with AURORA in control, and leave AURORA running with the HPAs suspended.
#
#   bash stage7_28/exp13jul/aurora_step5/smoke.sh [name]
#
# Environment:
#   AURORA_IMAGE    image to deploy (default: aurora/data/IMAGE)
#   MAX_REQUESTS    offered load (default 30000)
#   DURATION_MIN    load-test minutes (default 10 -- keep it short)
#   RESET=0         skip the reset (the workload is already cold)
#
# The controller is NEVER deleted here, and the HPAs are NEVER re-applied: that
# is the whole point of the AURORA development loop.
set -uo pipefail
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf

NAME=${1:-smoke_aurora_$(date -u +%Y%m%dT%H%M%SZ)}
MAX_REQUESTS=${MAX_REQUESTS:-30000}
DURATION_MIN=${DURATION_MIN:-10}
RESET=${RESET:-1}

echo "=== AURORA smoke: $NAME (10 min, ${MAX_REQUESTS}/min) ==="

# 1. make sure the controller is up, running the image in data/IMAGE
if ! PRE_K6_HOOK="" bash stage7_28/exp13jul/aurora_step5/up.sh; then
  echo "FATAL: aurora did not come up" >&2
  exit 1
fi

# 2. reset the workload pods only (no HPA, no replica changes, aurora untouched)
if [ "$RESET" = "1" ]; then
  if ! bash stage7_28/exp13jul/aurora_step5/reset_cold.sh; then
    echo "FATAL: cold reset failed" >&2
    exit 1
  fi
fi

# 3. run the load test. AURORA_DEV=1 makes the orchestrator skip the HPA
#    assertions and skip the final reset, so AURORA keeps ownership.
LOG=/tmp/opencode/${NAME}.log
setsid env PATH="$HOME/.local/bin:$PATH" \
  KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf \
  AURORA_DEV=1 SKIP_DUMPS=1 \
  SEED_FILE_LOADGEN="$PWD/load_testing/seeds/tt_users_12k.json" \
  VUS_MODE=auto \
  PRE_K6_HOOK="bash stage7_28/exp13jul/aurora_step5/up.sh" \
  POST_RUN_HOOK="bash stage7_28/exp13jul/aurora_step5/down.sh" \
  bash stage7_28/exp13jul/run_experiment.sh \
    --day 1995-07-13 --duration "$DURATION_MIN" \
    --max-requests "$MAX_REQUESTS" --seed-pool 11995 \
    --local --shape aurora --name "$NAME" > "$LOG" 2>&1 < /dev/null &
echo $! > /tmp/opencode/${NAME}.pid
echo "orchestrator pid=$(cat /tmp/opencode/${NAME}.pid), log=$LOG"

RUN=$(ls -td stage7_28/exp13jul/${NAME}_* 2>/dev/null | head -1)
if [ -z "$RUN" ]; then
  for _ in $(seq 1 40); do
    sleep 5
    RUN=$(ls -td stage7_28/exp13jul/${NAME}_* 2>/dev/null | head -1)
    [ -n "$RUN" ] && break
  done
fi
echo "run dir: $RUN"
echo "follow: tail -f $LOG"