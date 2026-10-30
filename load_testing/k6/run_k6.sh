#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

TARGET_URL="${TARGET_URL:-https://train-ticket.younesnb.linkpc.net}"
MCR_CSV="${MCR_CSV:-/proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv}"
MAX_REQUESTS="${MAX_REQUESTS:-5000}"
USER_POOL="${USER_POOL:-500}"
SEED_FILE="${SEED_FILE:-load_testing/seeds/tt_users.json}"
START_MIN="${START_MIN:-0}"
TEST_MIN="${TEST_MIN:-0}"
K6_BIN="${K6_BIN:-$HOME/.local/bin/k6}"
# TRACE_COMPRESS: replay the NASA trace N times faster than its recorded
# 1-min/row cadence (default 1 = 1 trace row per wall minute). >1 scales
# offered load linearly — the trace is already normalized to [0,1] by
# MAX_REQUESTS, so this is the ONLY knob that raises pressure without
# touching MAX_REQUESTS. e.g. TRACE_COMPRESS=5 replays 5 trace rows per
# wall minute (a 2880-row trace in ~576 wall minutes).
TRACE_COMPRESS="${TRACE_COMPRESS:-1}"

case "$MCR_CSV" in /*) ;; *) MCR_CSV="$PWD/$MCR_CSV";; esac
case "$SEED_FILE" in /*) ;; *) SEED_FILE="$PWD/$SEED_FILE";; esac

echo "Starting k6 train-ticket (frontend-only) targeting $TARGET_URL"
echo "  MCR_CSV=$MCR_CSV START_MIN=$START_MIN TEST_MIN=$TEST_MIN"
echo "  MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL SEED_FILE=$SEED_FILE"

kubectl -n monitoring port-forward svc/mimir 18080:8080 >/tmp/opencode/k6_mimir_pf.log 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
for i in $(seq 1 30); do
  curl -sf -m 3 http://127.0.0.1:18080/ready >/dev/null 2>&1 && break
  sleep 2
done

# k6 remote-write to Prometheus: optional (each push writes ~1000s of series at
# 1s into Mimir and was the disk-pressure driver on the 6-node lab cluster).
# Set K6_PROMETHEUS_RW=1 to enable for a local Grafana install; default off —
# verdicts parse the k6 text summary, so RW push is not needed for the pipeline.
if [ "${K6_PROMETHEUS_RW:-0}" = "1" ]; then
  export K6_PROMETHEUS_RW_SERVER_URL="http://127.0.0.1:18080/api/v1/push"
fi
export K6_PROMETHEUS_RW_PUSH_INTERVAL="10s"
export K6_PROMETHEUS_RW_TREND_STATS="avg,p(95),p(99)"
export K6_PROMETHEUS_RW_STALE_MARKERS="true"

HOST="$TARGET_URL" MCR_CSV="$MCR_CSV" MAX_REQUESTS="$MAX_REQUESTS" \
USER_POOL="$USER_POOL" SEED_FILE="$SEED_FILE" START_MIN="$START_MIN" \
TEST_MIN="$TEST_MIN" TRACE_COMPRESS="$TRACE_COMPRESS" "$K6_BIN" run \
  -o experimental-prometheus-rw \
  load_testing/k6/trainticket.js

