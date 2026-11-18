#!/bin/bash
# Launch a k6 TrainTicket run.
#
# Path-agnostic: nothing here assumes /opt/loadgen. The generator, the seed file,
# the k6 binary and the output directory all come from the environment, so the
# repo tree can live anywhere and this same script works on the load generator
# and on a laptop.
#
# Previously this lived only at /opt/loadgen/launch_k6.sh with /opt/loadgen baked
# into `cd`, OUT_DIR and the generator path, and the orchestrator copied it to
# that fixed location before every run. It now lives in the repo and the
# orchestrator points at it.
#
# Environment (all optional):
#   RUNID              run identifier, used in output filenames
#   MCR_CSV            trace slice (absolute path, or relative to $PWD)
#   SEED_FILE          seed users JSON
#   GENERATOR_JS       k6 script (default: trainticket.js next to this file)
#   K6_BIN             k6 binary (default: /usr/local/bin/k6)
#   K6_OUT_DIR         output directory (default: $PWD/k6)
#   TARGET_URL         gateway (default: http://10.10.1.2:30080)
#   ... plus the usual MAX_REQUESTS / USER_POOL / START_MIN / TEST_MIN knobs
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

GENERATOR_JS="${GENERATOR_JS:-$HERE/trainticket.js}"
K6_BIN="${K6_BIN:-/usr/local/bin/k6}"
OUT_DIR="${K6_OUT_DIR:-$PWD/k6}"

MCR_CSV="${MCR_CSV:-mcr.csv}"
MAX_REQUESTS="${MAX_REQUESTS:-5000}"
USER_POOL="${USER_POOL:-500}"
SEED_FILE="${SEED_FILE:-$HERE/../seeds/tt_users.json}"
START_MIN="${START_MIN:-0}"
TEST_MIN="${TEST_MIN:-0}"
TRACE_COMPRESS="${TRACE_COMPRESS:-1}"
OPEN_LOOP="${OPEN_LOOP:-1}"
PRE_ALLOCATED_VUS="${PRE_ALLOCATED_VUS:-100}"
# Open-loop execution capacity. maxVUs defaults to preAllocatedVUs: no hidden
# dependency on runtime VU allocation. USER_POOL is the account/VU budget.
MAX_VUS="${MAX_VUS:-$PRE_ALLOCATED_VUS}"
# Preflight traffic is opt-in and outside the measured arrival schedule.
PREFLIGHT="${PREFLIGHT:-0}"
# Account sharing is forbidden unless explicitly requested.
ALLOW_SEED_REUSE="${ALLOW_SEED_REUSE:-0}"
LOGIN_RETRIES="${LOGIN_RETRIES:-2}"
LOGIN_COOLDOWN_MS="${LOGIN_COOLDOWN_MS:-30000}"
EXEC_MODE="${EXEC_MODE:-}"
FLAT_RPS="${FLAT_RPS:-}"

TARGET_URL="${TARGET_URL:-http://10.10.1.2:30080}"
RUNID="${RUNID:-loadgen_$(date -u +%Y%m%dT%H%M%SZ)}"

# Relative paths are resolved against the caller's working directory, never
# against a baked-in prefix.
case "$MCR_CSV"   in /*) ;; *) MCR_CSV="$PWD/$MCR_CSV" ;; esac
case "$SEED_FILE" in /*) ;; *) SEED_FILE="$PWD/$SEED_FILE" ;; esac

mkdir -p "$OUT_DIR"
SUMMARY_OUT="$OUT_DIR/summary_${RUNID}.json"
SAMPLES_OUT="$OUT_DIR/samples_${RUNID}.json.gz"

echo "Starting k6 train-ticket targeting $TARGET_URL"
echo "  RUNID=$RUNID"
echo "  GENERATOR_JS=$GENERATOR_JS"
echo "  MCR_CSV=$MCR_CSV START_MIN=$START_MIN TEST_MIN=$TEST_MIN"
echo "  MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL SEED_FILE=$SEED_FILE"
echo "  TRACE_COMPRESS=$TRACE_COMPRESS OPEN_LOOP=$OPEN_LOOP EXEC_MODE=${EXEC_MODE:-<unset>}"
echo "  PRE_ALLOCATED_VUS=$PRE_ALLOCATED_VUS MAX_VUS=$MAX_VUS PREFLIGHT=$PREFLIGHT ALLOW_SEED_REUSE=$ALLOW_SEED_REUSE"
echo "  SUMMARY_OUT=$SUMMARY_OUT"
echo "  SAMPLES_OUT=$SAMPLES_OUT"

for f in "$GENERATOR_JS" "$MCR_CSV" "$SEED_FILE"; do
  [ -f "$f" ] || { echo "FATAL: required file missing: $f" >&2; exit 1; }
done

# Build the k6 argument array explicitly: no eval, no silent splitting of
# untrusted input. K6_EXTRA_ARGS is validated upstream and appended as a single
# shell word.
K6_ARGS=(
  run
  --include-system-env-vars=true
  --summary-export "$SUMMARY_OUT"
  -o json="$SAMPLES_OUT"
)

if [ -n "${K6_EXTRA_ARGS:-}" ]; then
  K6_ARGS+=($K6_EXTRA_ARGS)
fi

K6_ARGS+=("$GENERATOR_JS")

HOST="$TARGET_URL" MCR_CSV="$MCR_CSV" MAX_REQUESTS="$MAX_REQUESTS" \
USER_POOL="$USER_POOL" SEED_FILE="$SEED_FILE" START_MIN="$START_MIN" \
TEST_MIN="$TEST_MIN" TRACE_COMPRESS="$TRACE_COMPRESS" OPEN_LOOP="$OPEN_LOOP" \
PRE_ALLOCATED_VUS="$PRE_ALLOCATED_VUS" MAX_VUS="$MAX_VUS" \
PREFLIGHT="$PREFLIGHT" ALLOW_SEED_REUSE="$ALLOW_SEED_REUSE" \
LOGIN_RETRIES="$LOGIN_RETRIES" LOGIN_COOLDOWN_MS="$LOGIN_COOLDOWN_MS" \
EXEC_MODE="$EXEC_MODE" FLAT_RPS="$FLAT_RPS" \
"$K6_BIN" "${K6_ARGS[@]}"