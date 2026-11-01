#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

MCR_CSV="${MCR_CSV:-/proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv}"
MAX_REQUESTS="${MAX_REQUESTS:-5000}"
USER_POOL="${USER_POOL:-500}"
SEED_FILE="${SEED_FILE:-load_testing/seeds/tt_users.json}"
START_MIN="${START_MIN:-0}"
TEST_MIN="${TEST_MIN:-0}"
TARGET_URL="${TARGET_URL:-http://10.10.1.2:30080}"
TRACE_COMPRESS="${TRACE_COMPRESS:-1}"
OPEN_LOOP="${OPEN_LOOP:-1}"
FLAT_RPS="${FLAT_RPS:-0}"
PRE_ALLOCATED_VUS="${PRE_ALLOCATED_VUS:-100}"

case "$TARGET_URL" in
  http://10.10.1.2:30080) ;;
  *)
    echo "ERROR: refusing TARGET_URL='$TARGET_URL'." >&2
    echo "       Load must target the private gateway (http://10.10.1.2:30080)." >&2
    echo "       External/public targets cross the CloudLab controlled network" >&2
    echo "       and get the experiment quarantined." >&2
    exit 1
    ;;
esac

if [[ -x /opt/loadgen/launch_k6.sh ]]; then
  if [ ! -f /opt/loadgen/mcr.csv ] && [ -f "$MCR_CSV" ]; then
    cp "$MCR_CSV" /opt/loadgen/mcr.csv
  fi
  exec env TARGET_URL="$TARGET_URL" \
    MCR_CSV=/opt/loadgen/mcr.csv \
    MAX_REQUESTS="$MAX_REQUESTS" USER_POOL="$USER_POOL" \
    SEED_FILE=/opt/loadgen/seeds/tt_users.json \
    START_MIN="$START_MIN" TEST_MIN="$TEST_MIN" \
    TRACE_COMPRESS="$TRACE_COMPRESS" OPEN_LOOP="$OPEN_LOOP" \
    FLAT_RPS="$FLAT_RPS" PRE_ALLOCATED_VUS="$PRE_ALLOCATED_VUS" \
    /opt/loadgen/launch_k6.sh
fi

ssh -o ConnectTimeout=10 loadgen 'test -x /opt/loadgen/launch_k6.sh' \
  || {
    echo "ERROR: loadgen is not initialized (no /opt/loadgen/launch_k6.sh)." >&2
    echo "       Run> ./run-cluster-setup.sh loadgen   (from the cluster repo)" >&2
    exit 1
  }

if ! ssh -o ConnectTimeout=10 loadgen 'test -f /opt/loadgen/mcr.csv'; then
  scp -q "$MCR_CSV" loadgen:/opt/loadgen/mcr.csv
fi
if ! ssh -o ConnectTimeout=10 loadgen 'test -f /opt/loadgen/seeds/tt_users.json'; then
  scp -q "$SEED_FILE" loadgen:/opt/loadgen/seeds/tt_users.json 2>/dev/null || true
fi

exec ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10 loadgen \
  "env TARGET_URL='$TARGET_URL' \
   MCR_CSV=/opt/loadgen/mcr.csv \
   MAX_REQUESTS='$MAX_REQUESTS' USER_POOL='$USER_POOL' \
   SEED_FILE=/opt/loadgen/seeds/tt_users.json \
   START_MIN='$START_MIN' TEST_MIN='$TEST_MIN' \
   TRACE_COMPRESS='$TRACE_COMPRESS' OPEN_LOOP='$OPEN_LOOP' \
   FLAT_RPS='$FLAT_RPS' PRE_ALLOCATED_VUS='$PRE_ALLOCATED_VUS' \
   /opt/loadgen/launch_k6.sh"
