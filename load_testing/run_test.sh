#!/bin/bash
set -euo pipefail

TARGET_URL="${TARGET_URL:-https://online-boutique.younesnb.linkpc.net}"
MCR_CSV="${MCR_CSV:?MCR_CSV must point to the http_mcr CSV (http_mcr_<service>_<ts>.csv) from analytics/analyze_http_mcr_oscillation.py}"
MAX_REQUESTS="${MAX_REQUESTS:-1000}"
USER_POOL="${USER_POOL:-50}"
START_HOURS="${START_HOURS:-0}"
TEST_HOURS="${TEST_HOURS:-$(python3 -c 'import csv,sys; n=(sum(1 for _ in csv.reader(open(sys.argv[1])))-1)/60; print(n-float(sys.argv[2]))' "$MCR_CSV" "$START_HOURS")}"

echo "Starting Locust targeting $TARGET_URL"
echo "  MCR_CSV=$MCR_CSV"
echo "  MAX_REQUESTS=$MAX_REQUESTS req/min at peak (http_mcr=1.0)"
echo "  USER_POOL=$USER_POOL"
echo "  START_HOURS=$START_HOURS h (skip this much of the curve before starting)"
echo "  TEST_HOURS=$TEST_HOURS h (loops CSV curve until this)"

exec locust -f locustfile.py --host "$TARGET_URL" \
    --mcr-csv "$MCR_CSV" \
    --max-requests "$MAX_REQUESTS" \
    --user-pool "$USER_POOL" \
    --start-hours "$START_HOURS" \
    --test-hours "$TEST_HOURS"
