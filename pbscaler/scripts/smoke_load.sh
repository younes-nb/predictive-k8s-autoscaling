#!/bin/bash
# 15-minute smoke of PBScaler's data path:
#   limited k6 NASA replay load_testing/run_test.sh  +  PBScaler training-data
#   collector (pbscaler/scripts/collect_training_data.py) running over the
#   same window, HPAs left enabled (as in the paper's data generation).
#
# Usage (from repo root):
#   pbscaler/scripts/smoke_load.sh [COLLECT_MIN] [K6_MIN]
#     COLLECT_MIN default 16 (a bit longer than the load to cover ramp)
#     K6_MIN      default 15 (TEST_MIN for run_test.sh)
#
# Smoke load is deliberately light: MAX_REQUESTS=400 (peak ≈ 5 rps) at the
# busiest 15-min window of the Jul 13–14 NASA slice (START_MIN=587).
# For the full dataset run: Cells COLLECT_MIN/K6_MIN/MAX_REQUESTS higher;
# see pbscaler/README.md ("full run" notes).
set -euo pipefail
cd "$(dirname "$0")/../.."

COLLECT_MIN="${1:-16}"
K6_MIN="${2:-15}"
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

echo "== smoke: collector ${COLLECT_MIN}min / k6 ${K6_MIN}min, MAX_REQUESTS=${MAXREQ}, POOL=${POOL}"

# ---- 1. collector (its own Mimir port-forward, decoupled from run_test.sh) --
kubectl -n monitoring port-forward svc/mimir 18081:8080 >/tmp/opencode/smoke_mimir_pf.log 2>&1 &
PF_PID=$!
trap 'kill $PF_PID $COLLECT_PID $K6_PID 2>/dev/null || true' EXIT
for i in $(seq 1 30); do curl -sf -m 3 http://127.0.0.1:18081/prometheus/api/v1/query --data-urlencode 'query=up' >/dev/null && break; sleep 2; done

PBS_MIMIR_URL=http://127.0.0.1:18081 \
  "$PY_BIN" pbscaler/scripts/collect_training_data.py \
    --duration $((COLLECT_MIN * 60)) --interval 10 \
    --workload "$MCR" \
    --out "pbscaler/data/smoke_${OUT_SUFFIX}" >"/tmp/opencode/smoke_collector.log" 2>&1 &
COLLECT_PID=$!
echo "collector pid $COLLECT_PID -> /tmp/opencode/smoke_collector.log (out: pbscaler/data/smoke_${OUT_SUFFIX})"

# ---- 2. k6 NASA replay -----------------------------------------------------
sleep 20   # let the collector record a pre-load baseline
TARGET_URL="${TARGET_URL:-https://train-ticket.younesnb.linkpc.net}" \
MCR_CSV="$MCR" MAX_REQUESTS="$MAXREQ" USER_POOL="$POOL" SEED_FILE="$SEED_FILE" \
START_MIN="${START_MIN:-587}" TEST_MIN="$K6_MIN" \
  bash load_testing/run_test.sh >"/tmp/opencode/smoke_k6.log" 2>&1 &
K6_PID=$!
echo "k6 pid $K6_PID -> /tmp/opencode/smoke_k6.log"

# ---- 3. wait out the load, then the collector --------------------------------
wait $K6_PID || echo "WARN: k6 exited non-zero (thresholds are slack on purpose)"
echo "k6 finished; waiting for collector..."
COLLECT_RC=0; wait $COLLECT_PID || COLLECT_RC=$?
tail -12 "/tmp/opencode/smoke_collector.log"

exit $COLLECT_RC
