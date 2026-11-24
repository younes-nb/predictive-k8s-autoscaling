#!/bin/bash
# P3 — generator capacity at the official 2,999-VU configuration against the
# synthetic server. No TrainTicket traffic.
#
#   bash /tmp/opencode/test_vu_sizing.sh [TEST_MIN] [TAG]
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
JS="$REPO/load_testing/k6/trainticket.js"
OUT="${OUT_DIR:-/tmp/opencode}/vu_${2:-base}"
TMIN="${1:-3}"
PORT=${PORT:-18099}
rm -rf "$OUT"; mkdir -p "$OUT"

# The trace slice is generated per run from the NASA archive; the old fixed
# /opt/loadgen/mcr_jul13_slice.csv no longer exists.
MCR_CSV=${MCR_CSV:-"$OUT/trace_slice.csv"}
if [ ! -s "$MCR_CSV" ]; then
  python3 "$REPO/benchmark/nasa_trace_slice.py" \
    --day 1995-07-13 --duration "$TMIN" --out "$MCR_CSV" >/dev/null 2>&1 \
    || { echo "FATAL: could not generate a trace slice"; exit 1; }
fi

echo "### VU sizing: test_min=$TMIN port=$PORT out=$OUT slice=$MCR_CSV"
start=$(date +%s)
env HOST="http://127.0.0.1:$PORT" \
    MCR_CSV="$MCR_CSV" \
    SEED_FILE="$REPO/load_testing/seeds/tt_users.json" \
    MAX_REQUESTS=25000 USER_POOL=2999 TRACE_COMPRESS=1 OPEN_LOOP=1 \
    PRE_ALLOCATED_VUS=2999 MAX_VUS=2999 \
    PREFLIGHT=0 ALLOW_SEED_REUSE=0 LOGIN_RETRIES=2 LOGIN_COOLDOWN_MS=30000 \
    START_MIN=529 TEST_MIN=$TMIN \
    /usr/local/bin/k6 run --include-system-env-vars=true \
      --summary-export "$OUT/summary.json" \
      "$JS" > "$OUT/k6_console.log" 2>&1 &
K6=$!
echo "k6 pid=$K6"

: > "$OUT/samples.csv"
echo "t,rss_mb,fds,threads" >> "$OUT/samples.csv"
peak_rss=0; peak_fds=0
while kill -0 "$K6" 2>/dev/null; do
  rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$K6/status 2>/dev/null)
  fds=$(ls /proc/$K6/fd 2>/dev/null | wc -l)
  thr=$(ls /proc/$K6/task 2>/dev/null | wc -l)
  rss=${rss:-0}; fds=${fds:-0}
  echo "$(( $(date +%s) - start )),${rss},${fds},${thr:-0}" >> "$OUT/samples.csv"
  [ "$rss" -gt "$peak_rss" ] && peak_rss=$rss
  [ "$fds" -gt "$peak_fds" ] && peak_fds=$fds
  sleep 5
done
wait "$K6"; RC=$?
elapsed=$(( $(date +%s) - start ))
echo "k6 exit=$RC elapsed=${elapsed}s"
echo "peak rss: ${peak_rss} MB   peak fds: $peak_fds"
echo "$peak_rss $peak_fds $elapsed" > "$OUT/peaks.txt"

python3 - "$OUT/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["metrics"]
def g(n, k="count"):
    v = s.get(n, {})
    return v.get(k)
print("iterations          :", g("iterations"), " expected:", g("scheduled_iterations_expected", "value"))
print("dropped_iterations  :", g("dropped_iterations"))
print("http_reqs           :", g("http_reqs"))
print("http_req_failed     :", g("http_req_failed", "rate"))
print("vus_max             :", g("vus_max", "value"))
print("vus live            : max", g("vus", "max"), "avg", round(g("vus", "avg") or 0, 1))
print("iteration_duration  : avg", round(g("iteration_duration","avg") or 0,1),
      "p95", g("iteration_duration","p(95)"), "max", g("iteration_duration","max"))
print("http_req_duration   : avg", round(g("http_req_duration","avg") or 0,1),
      "p95", g("http_req_duration","p(95)"), "max", g("http_req_duration","max"))
print("auth_initial_login_success_total:", g("auth_initial_login_success_total"))
print("auth_renew_scheduled_total     :", g("auth_renew_scheduled_total"))
print("auth_renew_success_total       :", g("auth_renew_success_total"))
print("correlation present/missing    :", g("correlation_mesh_id_present_total"), "/", g("correlation_mesh_id_missing_total"))
PY
exit 0