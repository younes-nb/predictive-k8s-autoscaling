#!/bin/bash
# Train-ticket frontend-only load test (k6; replaces the locust driver).
#
# Every request enters through ts-ui-dashboard (the ingress `/` catch-all);
# the dashboard fans out to backends. Thin wrapper over load_testing/k6/.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_URL="${TARGET_URL:-https://train-ticket.younesnb.linkpc.net}"
MCR_CSV="${MCR_CSV:-/proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv}"
MAX_REQUESTS="${MAX_REQUESTS:-5000}"
USER_POOL="${USER_POOL:-500}"
SEED_FILE="${SEED_FILE:-load_testing/seeds/tt_users.json}"
START_MIN="${START_MIN:-0}"
TEST_MIN="${TEST_MIN:-0}"

export TARGET_URL MCR_CSV MAX_REQUESTS USER_POOL SEED_FILE START_MIN TEST_MIN
exec bash load_testing/k6/run_k6.sh
