#!/bin/bash
# k6 NASA replay + PBScaler training-data collector (scripts/
# collect_training_data.py) running over the same window, namespace HPAs left
# enabled (the paper's data generation: labels come from native-HPA metrics).
#
# Cold-start handling (paper-faithful features): WARMUP_MIN (default 0; set 60
# for training runs) starts the k6 replay FIRST and joins the collector only
# after that many minutes — startup storms (image pulls, cold JIT/connection
# pools, replica settling) are absorbed into the warmup phase and are NOT part
# of the collected dataset. The k6 session itself is continuous across
# warmup + test; only the collector's window is trimmed.
#
# Usage (from repo root):
#   pbscaler/scripts/smoke_load.sh [COLLECT_MIN] [K6_MIN]
#     COLLECT_MIN default 16 (collector minutes = the collected dataset span)
#     K6_MIN      default 15 (k6 test minutes; k6 total = WARMUP_MIN + K6_MIN)
#   env overrides: MCR_CSV, MAX_REQUESTS, USER_POOL, START_MIN,
#     COLLECT_INTERVAL, WARMUP_MIN, OUT_PREFIX, PY_BIN, TARGET_URL
#
# Smoke defaults are deliberately light: MAX_REQUESTS=400 (peak ~= 5 rps) at
# the busiest 15-min window of the Jul 13–14 NASA slice (START_MIN=587).
# Full training-run example (12 h dataset + 1 h warmup, Jul-13 busiest window;
# trace rows: warmup 400..460, collected dataset 460..1180):
#   MAX_REQUESTS=7600 USER_POOL=560 START_MIN=400 WARMUP_MIN=60 \
#     pbscaler/scripts/smoke_load.sh 720 720
set -euo pipefail
cd "$(dirname "$0")/../.."

COLLECT_MIN="${1:-16}"
K6_MIN="${2:-15}"
WARMUP_MIN="${WARMUP_MIN:-0}"
COLLECT_INTERVAL="${COLLECT_INTERVAL:-10}"
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

# pick the collector interpreter
PY_BIN="${PY_BIN:-pbscaler/.venv/bin/python}"
if [ ! -x "$PY_BIN" ]; then PY_BIN=python3; fi

K6_TOTAL=$((WARMUP_MIN + K6_MIN))
COLLECT_PID=""
K6_PID=""
echo "== load test: collector ${COLLECT_MIN}min / k6 ${K6_TOTAL}min (warmup ${WARMUP_MIN}min excluded from data), MAX_REQUESTS=${MAXREQ}, POOL=${POOL}"

k6_start() {
  TARGET_URL="${TARGET_URL:-https://train-ticket.younesnb.linkpc.net}" \
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

# ---- collector's own Mimir port-forward (decoupled from run_test.sh) --------
kubectl -n monitoring port-forward svc/mimir 18081:8080 >/tmp/opencode/smoke_mimir_pf.log 2>&1 &
PF_PID=$!
trap 'kill $PF_PID $COLLECT_PID $K6_PID 2>/dev/null || true' EXIT
for i in $(seq 1 30); do curl -sf -m 3 http://127.0.0.1:18081/prometheus/api/v1/query --data-urlencode 'query=up' >/dev/null && break; sleep 2; done

if [ "$WARMUP_MIN" -gt 0 ]; then
  # paper-cold-start protocol: k6 first, collector joins after the warmup hour
  k6_start "$K6_TOTAL"
  echo "warming up ${WARMUP_MIN}min (k6-only; collector starts afterwards) — waiting..."
  sleep $((WARMUP_MIN * 60))
  collector_start
else
  collector_start   # pre-load baseline rows (older smoke behavior)
  sleep 20
  k6_start "$K6_MIN"
fi

# ---- wait out the collector, then the k6 tail --------------------------------
COLLECT_RC=0
wait $COLLECT_PID || COLLECT_RC=$?
echo "collector finished; waiting for k6..."
wait $K6_PID || echo "WARN: k6 exited non-zero (thresholds are slack on purpose)"
tail -12 "/tmp/opencode/${OUT_PREFIX}_collector.log"

exit $COLLECT_RC
