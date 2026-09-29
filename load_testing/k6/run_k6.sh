#!/bin/bash
# k6 frontend-only train-ticket load.
# Same inputs: MCR curve slice paced across the VU pool through the frontend.
# Metrics stream to Mimir via experimental-prometheus-rw (port-forward below).
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

export K6_PROMETHEUS_RW_SERVER_URL="http://127.0.0.1:18080/api/v1/push"
export K6_PROMETHEUS_RW_PUSH_INTERVAL="10s"
export K6_PROMETHEUS_RW_TREND_STATS="avg,p(95),p(99)"
export K6_PROMETHEUS_RW_STALE_MARKERS="true"

HOST="$TARGET_URL" MCR_CSV="$MCR_CSV" MAX_REQUESTS="$MAX_REQUESTS" \
USER_POOL="$USER_POOL" SEED_FILE="$SEED_FILE" START_MIN="$START_MIN" \
TEST_MIN="$TEST_MIN" "$K6_BIN" run \
  -o experimental-prometheus-rw \
  load_testing/k6/trainticket.js
