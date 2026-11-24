#!/bin/bash
# P5 — renewal behaviour at a representative session population, against the
# mock auth service with an accelerated (test-only) token lifetime.
#
#   bash /tmp/opencode/test_population.sh [SESSIONS] [TTL_S] [TEST_MIN]
set -u
REPO=/proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
JS="$REPO/load_testing/k6/trainticket.js"
N="${1:-1150}"
TTL="${2:-180}"
TMIN="${3:-5}"
MAXR="${4:-1200}"
PORT=${PORT:-18101}
OUT="/tmp/opencode/pop_${N}"
rm -rf "$OUT"; mkdir -p "$OUT"

echo "### population=$N sessions ttl=${TTL}s test_min=$TMIN max_req=$MAXR"
start=$(date +%s)
env HOST="http://127.0.0.1:$PORT" \
    MCR_CSV=/opt/loadgen/mcr_jul13_slice.csv \
    SEED_FILE=/opt/loadgen/seeds/tt_users.json \
    MAX_REQUESTS=$MAXR USER_POOL=$N TRACE_COMPRESS=1 OPEN_LOOP=1 \
    PRE_ALLOCATED_VUS=$N MAX_VUS=$N \
    PREFLIGHT=0 ALLOW_SEED_REUSE=0 LOGIN_RETRIES=2 LOGIN_COOLDOWN_MS=30000 \
    START_MIN=529 TEST_MIN=$TMIN \
    /usr/local/bin/k6 run --include-system-env-vars=true \
      --summary-export "$OUT/summary.json" \
      "$JS" > "$OUT/k6_console.log" 2>&1
RC=$?
echo "k6 exit=$RC elapsed=$(( $(date +%s) - start ))s"

python3 - "$OUT/summary.json" "$OUT" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["metrics"]
def g(n, k="count"):
    return s.get(n, {}).get(k)
print("iterations              :", g("iterations"))
print("auth_initial_login_attempts_total:", g("auth_initial_login_attempts_total"))
print("auth_initial_login_success_total :", g("auth_initial_login_success_total"))
print("auth_initial_login_failures_total:", g("auth_initial_login_failures_total"))
print("auth_renew_scheduled_total       :", g("auth_renew_scheduled_total"))
print("auth_renew_attempts_total        :", g("auth_renew_attempts_total"))
print("auth_renew_success_total         :", g("auth_renew_success_total"))
print("auth_renew_failures_total        :", g("auth_renew_failures_total"))
print("auth_renew_retries_total         :", g("auth_renew_retries_total"))
print("auth_expired_token_total         :", g("auth_expired_token_total"))
print("auth_unavailable_actions_total   :", g("auth_unavailable_actions_total"))
print("auth_unauth_iterations_total     :", g("auth_unauth_iterations_total"))
print("http_req_failed                  :", g("http_req_failed", "rate"))
PY
exit 0