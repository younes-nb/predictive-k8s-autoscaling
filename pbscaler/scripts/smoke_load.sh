#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

COLLECT_MIN="${1:-16}"
K6_MIN="${2:-15}"
WARMUP_MIN="${WARMUP_MIN:-0}"
COLLECT_INTERVAL="${COLLECT_INTERVAL:-5}"
OUT_PREFIX="${OUT_PREFIX:-smoke}"
MCR="${MCR_CSV:-/proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv}"
MAXREQ="${MAX_REQUESTS:-400}"
POOL="${USER_POOL:-100}"
OUT_SUFFIX=$(date +%Y%m%d_%H%M%S)

SEED_FILE="load_testing/seeds/tt_users.json"
if [ ! -f "$SEED_FILE" ]; then
  echo "ERROR: $SEED_FILE missing — generate it first:"
  echo "  # needs port-forwards of ts-auth/ts-user/ts-contacts (see README)"
  echo "  python3 load_testing/seed_tt_users.py --count 120 --base http://127.0.0.1:12340 \\"
  echo "      --user-base http://127.0.0.1:12342 --contacts-base http://127.0.0.1:12347 \\"
  echo "      --out load_testing/seeds/tt_users.json"
  exit 1
fi

PY_BIN="${PY_BIN:-pbscaler/.venv/bin/python}"
if [ ! -x "$PY_BIN" ]; then PY_BIN=python3; fi

K6_TOTAL=$((WARMUP_MIN + K6_MIN))
COLLECT_PID=""
K6_PID=""
echo "== load test: collector ${COLLECT_MIN}min / k6 ${K6_TOTAL}min (warmup ${WARMUP_MIN}min excluded from data), MAX_REQUESTS=${MAXREQ}, POOL=${POOL}"

k6_start() {
  TARGET_URL="${TARGET_URL:-http://10.10.1.2:30080}" \
  MCR_CSV="$MCR" MAX_REQUESTS="$MAXREQ" USER_POOL="$POOL" SEED_FILE="$SEED_FILE" \
  START_MIN="${START_MIN:-587}" TEST_MIN="$1" \
    bash load_testing/run_test.sh >"/tmp/opencode/${OUT_PREFIX}_k6.log" 2>&1 &
  K6_PID=$!
  echo "k6 pid $K6_PID (TEST_MIN=$1, START_MIN=${START_MIN:-587}) -> /tmp/opencode/${OUT_PREFIX}_k6.log"
}

collector_start() {
  PBS_MIMIR_URL=http://127.0.0.1:18081 \
    "$PY_BIN" pbscaler/scripts/collect_training_data.py \
      --duration $((COLLECT_MIN * 60)) --interval "$COLLECT_INTERVAL" \
      --workload "$MCR" \
      --out "pbscaler/data/${OUT_PREFIX}_${OUT_SUFFIX}" >"/tmp/opencode/${OUT_PREFIX}_collector.log" 2>&1 &
  COLLECT_PID=$!
  echo "collector pid $COLLECT_PID -> /tmp/opencode/${OUT_PREFIX}_collector.log (out: pbscaler/data/${OUT_PREFIX}_${OUT_SUFFIX})"
}

kubectl -n monitoring port-forward svc/mimir 18081:8080 >/tmp/opencode/smoke_mimir_pf.log 2>&1 &
PF_PID=$!
trap 'kill $PF_PID $COLLECT_PID $K6_PID 2>/dev/null || true' EXIT
for i in $(seq 1 30); do curl -sf -m 3 http://127.0.0.1:18081/prometheus/api/v1/query --data-urlencode 'query=up' >/dev/null && break; sleep 2; done

if [ "$WARMUP_MIN" -gt 0 ]; then
  k6_start "$K6_TOTAL"
  echo "warming up ${WARMUP_MIN}min (k6-only; collector starts afterwards) — waiting..."
  sleep $((WARMUP_MIN * 60))
  collector_start
else
  collector_start
  sleep 20
  k6_start "$K6_MIN"
fi

COLLECT_RC=0
wait $COLLECT_PID || COLLECT_RC=$?
echo "collector finished; waiting for k6..."
wait $K6_PID || echo "WARN: k6 exited non-zero (thresholds are slack on purpose)"
tail -12 "/tmp/opencode/${OUT_PREFIX}_collector.log"

exit $COLLECT_RC
