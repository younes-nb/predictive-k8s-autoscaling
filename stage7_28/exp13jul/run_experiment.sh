#!/bin/bash
# TrainTicket full-capture orchestrator — THE single orchestrator.
#
# One script, every shape. You choose the trace and the duration; the script
# picks the highest-standard-deviation slice of that day of exactly that length,
# then runs the full capture (cold gate -> tools -> k6 -> dumps -> archive ->
# cold reset) and writes an explicit run_status.json.
#
# Trace source: the NASA-HTTP archive
#   https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
#   Jul 1-31 1995 (NASA_access_log_Jul95.gz) and Aug 1-7 1995.
# The selection rule is the documented one: for the chosen day, slide a window
# of `--duration` minutes across the per-minute request counts and keep the
# window with the highest standard deviation (the spikiest = hardest to autoscale).
#
# Usage:
#   run_experiment.sh --day 1995-07-13 --duration 240 --max-requests 32000 --local
#   run_experiment.sh --day 1995-07-13 --duration 60  --max-requests 25000 --ssh
#   run_experiment.sh --day 1995-07-13 --duration 240 --max-requests 32000 --trace-only
#
# Required:
#   --day YYYY-MM-DD      day of the NASA-HTTP trace (1995-07-01..1995-07-31, 1995-08-01..07)
#   --duration MINUTES    measured window length; also the length of the trace slice
#   --max-requests N      peak iterations/minute (trace normalisation peak)
#
# Options:
#   --local | --ssh        transport (default: --local; loadgen IS this node)
#   --shape hpa|pbscaler   label only (default hpa)
#   --name NAME            run-id prefix (default <shape><maxreq>_<day>)
#   --source jul|aug       NASA-HTTP archive file (default jul)
#   --seed-pool N          account/VU budget (default 2999)
#   --trace-only           print the selected slice and exit
#   --dry                  stop after the cold gate
#   -h, --help
set -u
cd /proj/k8sautoscaledl-PG0/src/predictive-k8s-autoscaling
export KUBECONFIG=${KUBECONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf}
KUBECTL_BIN=$(command -v kubectl || echo /usr/local/bin/kubectl)
TOOLS="$PWD/stage7_28/exp13jul"

usage() {
  awk 'NR>1 && /^set -u/{exit} NR>1{sub(/^# ?/,""); print}' "$0"
  exit "${1:-0}"
}

# ---- CLI -------------------------------------------------------------------
MODE=local
DRY=""
TRACE_DAY=""
DURATION=""
MAX_REQUESTS=""
SHAPE=hpa
NAME=""
TRACE_SOURCE=jul
SEED_POOL=2999
TRACE_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --day)           TRACE_DAY=${2:-}; shift 2 ;;
    --duration)      DURATION=${2:-}; shift 2 ;;
    --max-requests)  MAX_REQUESTS=${2:-}; shift 2 ;;
    --local)         MODE=local; shift ;;
    --ssh)           MODE=ssh; shift ;;
    --shape)         SHAPE=${2:-}; shift 2 ;;
    --name)          NAME=${2:-}; shift 2 ;;
    --source)        TRACE_SOURCE=${2:-}; shift 2 ;;
    --seed-pool)     SEED_POOL=${2:-}; shift 2 ;;
    --trace-only)    TRACE_ONLY=1; shift ;;
    --dry)           DRY=--dry; shift ;;
    -h|--help)       usage 0 ;;
    --*)             echo "unknown option: $1" >&2; usage 2 ;;
    *)               echo "unexpected argument: $1" >&2; usage 2 ;;
  esac
done

fail() { echo "FATAL: $*" >&2; exit 2; }

# ---- validate (fail closed, before any cluster interaction) ----------------
[ -n "$TRACE_DAY" ]   || fail "--day is required (YYYY-MM-DD)"
[ -n "$DURATION" ]    || fail "--duration is required (minutes)"
[ -n "$MAX_REQUESTS" ] || fail "--max-requests is required"
case "$TRACE_DAY" in
  [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
  *) fail "--day must be YYYY-MM-DD, got '$TRACE_DAY'" ;;
esac
case "$DURATION" in
  ''|*[!0-9]*) fail "--duration must be a positive integer, got '$DURATION'" ;;
esac
[ "$DURATION" -ge 1 ] && [ "$DURATION" -le 1440 ] || fail "--duration must be 1..1440 minutes"
case "$MAX_REQUESTS" in
  ''|*[!0-9]*) fail "--max-requests must be a positive integer, got '$MAX_REQUESTS'" ;;
esac
[ "$MAX_REQUESTS" -ge 1 ] || fail "--max-requests must be >= 1"
case "$SEED_POOL" in
  ''|*[!0-9]*) fail "--seed-pool must be a positive integer" ;;
esac
case "$SHAPE" in hpa|pbscaler) ;; *) fail "--shape must be hpa or pbscaler" ;; esac
case "$TRACE_SOURCE" in jul|aug) ;; *) fail "--source must be jul or aug" ;; esac

# ---- parameters derived from the CLI --------------------------------------
TEST_MIN=$DURATION          # measured window = slice length
START_MIN=0                 # the slice starts at the selected window
WARMUP_S=${WARMUP_S:-0}
USER_POOL=${USER_POOL:-$SEED_POOL}
PRE_ALLOC=${PRE_ALLOCATED_VUS:-$SEED_POOL}
MAX_VUS=${MAX_VUS:-$SEED_POOL}
K6_RUN_S=$((TEST_MIN * 60))
K6_DEADLINE_OFFSET=${K6_DEADLINE_OFFSET:-$((K6_RUN_S + 120))}
COLLECT_DUR=${COLLECT_DUR:-$K6_RUN_S}
TOOLS_END_OFFSET=${TOOLS_END_OFFSET:-$((K6_RUN_S + 600))}
TARGET_URL=${TARGET_URL:-}
SKIP_TRAFFIC_GATE=${SKIP_TRAFFIC_GATE:-0}
# How long to keep retrying the traffic gate when the istio metric itself is
# unreachable (Prometheus/Mimir down) before anchoring the window at launch.
TRAFFIC_GATE_FALLBACK_S=${TRAFFIC_GATE_FALLBACK_S:-180}
SKIP_DATASET=${SKIP_DATASET:-0}
SKIP_REPORT=${SKIP_REPORT:-0}      # build the one-file REPORT_*.md after the run
SKIP_DUMPS=${SKIP_DUMPS:-0}
SKIP_RESET=${SKIP_RESET:-0}
MAX_STREAMS=${MAX_STREAMS:-0}   # 0 = derive from the live pod count at launch
PERIODIC_DUMP_EVERY=${PERIODIC_DUMP_EVERY:-0}
# Rotate (gzip + truncate) $RUN_DIR/logs when it exceeds this many MB. 0 = off.
LOG_ROTATE_MB=${LOG_ROTATE_MB:-3072}
# ---- loadgen layout --------------------------------------------------------
# Everything the load generator reads/writes lives under $LOADGEN_HOME. It was
# hardcoded to /opt/loadgen in 21 places, which made the whole orchestrator
# unusable on a host with a different disk layout. Override per host in the
# environment; the defaults reproduce the current node exactly.
LOADGEN_HOME=${LOADGEN_HOME:-/opt/loadgen}
K6_BIN=${K6_BIN:-/usr/local/bin/k6}
KUBECTL_BIN=${KUBECTL_BIN:-/usr/local/bin/kubectl}
FAILURE_LOG_DIR=${FAILURE_LOG_DIR:-$LOADGEN_HOME/failure_logs}
# The k6 generator, its seed file and the launch wrapper all come from the REPO,
# not from a copy under $LOADGEN_HOME. They used to be copied to a fixed path
# before every run, which meant the repo and the deployed copy could drift (the
# generator is a pinned artifact with a recorded sha256). Defaults derive
# REPO_ROOT from this script's location.
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
K6_DIR=${K6_DIR:-$REPO_ROOT/load_testing/k6}
SEED_FILE_LOADGEN=${SEED_FILE_LOADGEN:-$REPO_ROOT/load_testing/seeds/tt_users.json}
LAUNCH_K6_SH=${LAUNCH_K6_SH:-$K6_DIR/launch_k6.sh}
GENERATOR_JS=${GENERATOR_JS:-$K6_DIR/trainticket.js}
K6_CONSOLE_DIR=${K6_CONSOLE_DIR:-$LOADGEN_HOME/k6}
# Disk watchdog thresholds are computed after RUN_DIR exists (see below).
DISK_PRUNE_BELOW_GB=${DISK_PRUNE_BELOW_GB:-}
DISK_CRITICAL_GB=${DISK_CRITICAL_GB:-}
TRACE_CACHE=${NASA_TRACE_CACHE:-/proj/k8sautoscaledl-PG0/reports/stage7_28/nasa_cache}

[ -n "$NAME" ] || NAME="${SHAPE}${MAX_REQUESTS}_$(echo "$TRACE_DAY" | tr -d '-')"
RUNID="${NAME}_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$PWD/stage7_28/exp13jul/$RUNID"
mkdir -p "$RUN_DIR/logs"
MCR_SLICE_LOCAL="$RUN_DIR/trace_slice.csv"

# Disk watchdog thresholds. Defaults scale with the filesystem that holds
# RUN_DIR (1/8 of it, floor 8 G, critical 1/20, floor 3 G) instead of being
# absolute numbers tuned to the old 63 G node. Either can be overridden.
if [ -z "${DISK_PRUNE_BELOW_GB:-}" ] || [ -z "${DISK_CRITICAL_GB:-}" ]; then
  _fs_total_gb=$(df -BG --output=size "$RUN_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
  case "${_fs_total_gb:-0}" in ''|*[!0-9]*) _fs_total_gb=0 ;; esac
  _scaled=$(( _fs_total_gb / 8 ))
  [ "$_scaled" -lt 8 ] && _scaled=8
  DISK_PRUNE_BELOW_GB=${DISK_PRUNE_BELOW_GB:-$_scaled}
  _crit=$(( _fs_total_gb / 20 ))
  [ "$_crit" -lt 3 ] && _crit=3
  if [ "$_crit" -ge "$DISK_PRUNE_BELOW_GB" ]; then _crit=$(( DISK_PRUNE_BELOW_GB - 1 )); fi
  DISK_CRITICAL_GB=${DISK_CRITICAL_GB:-$_crit}
  unset _fs_total_gb _scaled _crit
fi
: "${DISK_PRUNE_BELOW_GB:=8}"; : "${DISK_CRITICAL_GB:=3}"

# ---- trace selection -------------------------------------------------------
# The slice is derived from the real NASA-HTTP archive, not from a checked-in
# CSV, so the day/duration you ask for is exactly what runs. The selector is
# deterministic and its output hash is recorded in params.json.
echo "selecting trace: day=$TRACE_DAY duration=${DURATION}m source=$TRACE_SOURCE"
TRACE_JSON=$(python3 "$TOOLS/nasa_trace_slice.py" \
  --day "$TRACE_DAY" --duration "$DURATION" --source "$TRACE_SOURCE" \
  --cache "$TRACE_CACHE" --out "$MCR_SLICE_LOCAL") || {
  echo "FATAL: trace selection failed" >&2; exit 2; }
echo "$TRACE_JSON"
TRACE_STD=$(echo "$TRACE_JSON"  | python3 -c "import json,sys;print(json.load(sys.stdin)['window_std'])" 2>/dev/null)
TRACE_START=$(echo "$TRACE_JSON" | python3 -c "import json,sys;print(json.load(sys.stdin)['window_start_utc'])" 2>/dev/null)
TRACE_END=$(echo "$TRACE_JSON" | python3 -c "import json,sys;print(json.load(sys.stdin)['window_end_utc'])" 2>/dev/null)
TRACE_PEAK=$(echo "$TRACE_JSON"  | python3 -c "import json,sys;print(json.load(sys.stdin)['window_peak_per_min'])" 2>/dev/null)
TRACE_SHA=$(sha256sum "$MCR_SLICE_LOCAL" | cut -d' ' -f1)
[ -s "$MCR_SLICE_LOCAL" ] || { echo "FATAL: trace slice not produced" >&2; exit 2; }

if [ "$TRACE_ONLY" = "1" ]; then
  echo "trace-only: $MCR_SLICE_LOCAL"
  exit 0
fi

# Where k6 reads the trace slice from. In local mode the load generator IS this
# node, so hand k6 the run-scoped slice in RUN_DIR directly -- no copy, and no
# second copy of a 240-row CSV to keep in sync. ssh mode still needs the slice
# published on the remote loadgen.
if [ "$MODE" = local ]; then
  MCR_SLICE_LOADGEN="$MCR_SLICE_LOCAL"
else
  MCR_SLICE_LOADGEN="$K6_CONSOLE_DIR/trace_${RUNID}.csv"
fi

SSH_CFG=/proj/k8sautoscaledl-PG0/src/cluster-access/ssh_config
SSH="ssh -F $SSH_CFG -o BatchMode=yes -o ConnectTimeout=8"
SCP="scp -F $SSH_CFG -o BatchMode=yes"

# on_loadgen "<shell>"  — run a command on the load generator (ssh or local).
on_loadgen() {
  if [ "$MODE" = local ]; then bash -c "$1"; else $SSH loadgen "$1"; fi
}
# from_loadgen <remote-path> <local-path> — fetch a file from the load generator.
from_loadgen() {
  if [ "$MODE" = local ]; then cp -f "$1" "$2"; else $SCP -q "loadgen:$1" "$2"; fi
}
TOOLS="$PWD/stage7_28/exp13jul"
PY="$PWD/pbscaler/.venv/bin/python"
S5XX=${S5XX:-0.01}
S4XX=${S4XX:-0.05}
SCOOL=${SCOOL:-45}

# RUNID / RUN_DIR / MCR_SLICE_LOCAL were set from the CLI above; do not redefine.
CONSOLE_SRC="$K6_CONSOLE_DIR/console_${RUNID}.log"
echo "$RUNID" > "$RUN_DIR/run_id"
echo "$RUNID" > "$PWD/stage7_28/exp13jul/CURRENT_RUN_ID"
LOG="$RUN_DIR/orchestrator.log"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

# ---- run status -----------------------------------------------------------
# Every run ends with an explicit run_status.json: "complete" only if every
# phase and every collector finished. Anything else is "incomplete" with the
# reasons recorded, so a partial run can never be mistaken for a clean one.
RUN_STATUS="running"
STATUS_REASONS=()
COLLECTOR_PIDS=()
COLLECTOR_NAMES=()
K6_PID=""
K6_RC=""
K6_KIND=""

status_reason() {
  RUN_STATUS="incomplete"
  STATUS_REASONS+=("$1")
  log "STATUS: incomplete -- $1"
}

write_status() {
  local f="$RUN_DIR/run_status.json" tmp="$RUN_DIR/.run_status.$$.tmp"
  local args=("$tmp" "$RUNID" "$RUN_STATUS" "${LAUNCH_EPOCH:-}" "${END_EPOCH:-}"
              "${K6_RC:-}" "${K6_KIND:-}")
  if [ "${#STATUS_REASONS[@]}" -gt 0 ]; then args+=("${STATUS_REASONS[@]}"); fi
  python3 - "${args[@]}" <<'PY'
import json, sys, time
tmp, runid, status, launch, end, k6rc, k6kind = sys.argv[1:8]
reasons = sys.argv[8:]

def num(v):
    return int(v) if str(v).isdigit() else None

json.dump({
    "runid": runid,
    "status": status,
    "reasons": reasons,
    "launch_epoch": num(launch),
    "end_epoch": num(end),
    "k6_exit_code": num(k6rc),
    "k6_transport": k6kind or None,
    "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, open(tmp, "w"), indent=2)
PY
  mv -f "$tmp" "$f"
}

k6_signal_int() {
  if [ "$MODE" = local ]; then
    local kreal
    kreal=$(pgrep -P "$K6_PID" -x k6 2>/dev/null | head -1)
    if [ -n "$kreal" ]; then kill -INT "$kreal" 2>/dev/null; return 0; fi
    kill -INT "$K6_PID" 2>/dev/null
  else
    local KP
    KP=$(on_loadgen "pgrep -x k6 | head -1" 2>/dev/null | grep -E '^[0-9]+$' | head -1)
    [ -n "${KP:-}" ] && on_loadgen "kill -INT $KP"
  fi
}

cleanup() {
  local sig=$1 code=$2
  trap - INT TERM
  log "signal SIG$sig received -- stopping run-owned processes only"
  status_reason "terminated by SIG$sig"
  if [ -n "${K6_PID:-}" ] && k6_alive_local "$K6_PID"; then
    log "stopping k6 pid=$K6_PID"
    k6_signal_int || true
  fi
  local i p
  for i in "${!COLLECTOR_PIDS[@]}"; do
    p=${COLLECTOR_PIDS[$i]}
    kill -0 "$p" 2>/dev/null || continue
    log "stopping collector ${COLLECTOR_NAMES[$i]:-pid $p} (pid $p)"
    kill -TERM "$p" 2>/dev/null || true
  done
  sleep 3
  for p in ${COLLECTOR_PIDS[@]+"${COLLECTOR_PIDS[@]}"}; do
    kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null || true
  done
  # log_streamer's per-pod kubectl streams live in their own sessions, so they
  # survive the streamer's death. Reap them by the pidfiles it wrote.
  local pf sid p
  for pf in "$RUN_DIR"/.pids/*.pid; do
    [ -s "$pf" ] || continue
    sid=$(cat "$pf" 2>/dev/null)
    [ -n "$sid" ] || continue
    for p in $(pgrep -s "$sid" 2>/dev/null); do
      kill -KILL "$p" 2>/dev/null || true
    done
  done
  write_status
  log "=== run_experiment ABORTED runid=$RUNID ==="
  exit "$code"
}
trap 'cleanup INT 130' INT
trap 'cleanup TERM 143' TERM

track_collector() { COLLECTOR_NAMES+=("$1"); COLLECTOR_PIDS+=("$2"); }

# k6 liveness must ignore zombies: a k6 killed outside this script can survive
# as <defunct> under an unrelated reaper, and `pgrep -x k6` still lists it.
k6_pids_live() {
  local p st
  for p in $(pgrep -x k6 2>/dev/null); do
    st=$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ')
    case "$st" in ""|Z*) continue ;; esac
    echo "$p"
  done
}
loadgen_k6_pids() {
  if [ "$MODE" = local ]; then k6_pids_live; else
    on_loadgen "pgrep -x k6" 2>/dev/null | grep -E '^[0-9]+$' || true
  fi
}
k6_alive_local() {   # $1 = pid recorded by launch_k6_local
  kill -0 "$1" 2>/dev/null || return 1
  local st
  st=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ')
  case "$st" in ""|Z*) return 1 ;; esac
  return 0
}

die() { log "FATAL: $*"; status_reason "fatal: $*"; write_status; exit 1; }

log "=== run_experiment START runid=$RUNID mode=$MODE dry=${DRY:-no} ==="
log "transport: $( [ "$MODE" = local ] && echo "loadgen=localhost (this node; ssh disabled)" || echo "loadgen over ssh ($SSH_CFG)" )"
log "log_streamer: MAX_STREAMS=$MAX_STREAMS (one kubectl per pod, ~50 MB RSS each)"
log "periodic dumps: every ${PERIODIC_DUMP_EVERY}s (0 = end of run only)"
log "params: MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL PRE_ALLOCATED_VUS=$PRE_ALLOC MAX_VUS=$MAX_VUS START_MIN=$START_MIN TEST_MIN=$TEST_MIN COMPRESS=1 WARMUP_S=$WARMUP_S TARGET_URL=${TARGET_URL:-<default>}"
log "thresholds: 5XX_RATE_MIN=$S5XX 4XX_RATE_MIN=$S4XX COOLDOWN=$SCOOL"

# ---------- phase -1a: publish the trace slice to the load generator ----------
# In local mode the load generator IS this node and MCR_SLICE_LOADGEN is the
# slice in RUN_DIR, so there is nothing to copy -- copying a file onto itself
# used to fail the run at 07:47Z on the smoke test (2026-11-18).
if [ "$MODE" = local ]; then
  if [ "$MCR_SLICE_LOADGEN" != "$MCR_SLICE_LOCAL" ]; then
    cp -f "$MCR_SLICE_LOCAL" "$MCR_SLICE_LOADGEN" \
      || die "could not copy the trace slice to $MCR_SLICE_LOADGEN"
  fi
  log "trace slice used in place (local mode) -> $MCR_SLICE_LOADGEN"
else
  $SCP -q "$MCR_SLICE_LOCAL" "loadgen:$MCR_SLICE_LOADGEN" \
    || die "could not copy the trace slice to loadgen:$MCR_SLICE_LOADGEN"
  log "trace slice published -> $MCR_SLICE_LOADGEN (sha256=$TRACE_SHA)"
fi

# ---------- phase -1: preflight (files, hashes, k6 config) ----------
log "phase -1: preflight"
for f in "$MCR_SLICE_LOCAL" "$TOOLS/failure_recorder.py" "$TOOLS/monitor_v2.py" \
         "$TOOLS/log_streamer.sh" "$TOOLS/rebuild_train_data.py" \
         "$TOOLS/dump_mimir.sh" "$TOOLS/dump_loki.sh" "$PY"; do
  [ -f "$f" ] || die "missing required file: $f"
done
[ -x "$PY" ] || die "not executable: $PY"

LOCAL_K6_SHA=$(sha256sum "$PWD/load_testing/k6/trainticket.js" | awk '{print $1}')
SRV_CHECK=$(on_loadgen "sha256sum $MCR_SLICE_LOADGEN; sha256sum $GENERATOR_JS; sha256sum $LAUNCH_K6_SH" 2>/dev/null | sed 's/^#*//')
echo "$SRV_CHECK" | grep -q "$TRACE_SHA" || die "loadgen trace slice sha256 mismatch (expected $TRACE_SHA): $SRV_CHECK"
echo "$SRV_CHECK" | grep -q "$LOCAL_K6_SHA" || die "loadgen k6 script sha256 mismatch (expected $LOCAL_K6_SHA): $SRV_CHECK"
K6_SHA=$(echo "$SRV_CHECK" | awk '/trainticket.js/{print $1}')
LAUNCH_SHA=$(echo "$SRV_CHECK" | awk '/launch_k6.sh/{print $1}')
log "preflight OK: slice md5, k6 sha256=$K6_SHA, launcher sha256=$LAUNCH_SHA"

SEED_N=$(on_loadgen "python3 -c \"import json;print(len(json.load(open('$SEED_FILE_LOADGEN'))))\"" 2>/dev/null | grep -E '^[0-9]+$' | tail -1)
case "${SEED_N:-}" in (''|*[!0-9]*) die "cannot read seed pool size from loadgen" ;; esac
[ "$SEED_N" -ge "$USER_POOL" ] || die "seed pool too small: loadgen has $SEED_N accounts, USER_POOL=$USER_POOL"
[ "$SEED_N" -ge "$PRE_ALLOC" ] || die "seed pool too small: loadgen has $SEED_N accounts, PRE_ALLOCATED_VUS=$PRE_ALLOC"
log "seed pool: $SEED_N unique accounts (>= USER_POOL=$USER_POOL, >= PRE_ALLOCATED_VUS=$PRE_ALLOC)"

on_loadgen "env MCR_CSV=$MCR_SLICE_LOADGEN SEED_FILE=$SEED_FILE_LOADGEN \
  MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL PRE_ALLOCATED_VUS=$PRE_ALLOC MAX_VUS=$MAX_VUS \
  START_MIN=$START_MIN TEST_MIN=$TEST_MIN TRACE_COMPRESS=1 PREFLIGHT=0 \
  $K6_BIN inspect --include-system-env-vars=true $GENERATOR_JS" \
  > "$RUN_DIR/inspect.json" 2> "$RUN_DIR/inspect.err" \
  || die "k6 inspect failed (see inspect.err)"
python3 - "$RUN_DIR/inspect.json" "$TEST_MIN" >> "$LOG" 2>&1 <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); rows = int(sys.argv[2])
s = d["scenarios"]["load"]
st = s["stages"]
assert s["executor"] == "ramping-arrival-rate", s["executor"]
assert "1m" in s["timeUnit"], s["timeUnit"]
assert len(st) == 2 * rows, (len(st), rows)
assert st[0]["target"] == s["startRate"], (st[0], s["startRate"])
assert abs(float(s["gracefulStop"].rstrip("s")) - 30) < 0.001, s["gracefulStop"]
print(f"inspect OK: {len(st)} stages, startRate={s['startRate']}, timeUnit={s['timeUnit']}, pre={s['preAllocatedVUs']}, max={s['maxVUs']}")
EOF
[ $? -eq 0 ] || die "inspect config validation failed (see orchestrator.log)"
log "k6 inspect config validated"

cat > "$RUN_DIR/params.json" <<EOF
{"runid": "$RUNID", "bench": "train_ticket", "autoscaler": "$SHAPE",
 "max_requests": $MAX_REQUESTS, "user_pool": $USER_POOL, "pre_allocated_vus": $PRE_ALLOC, "max_vus": $MAX_VUS,
 "start_min": $START_MIN, "test_min": $TEST_MIN, "warmup_s": $WARMUP_S, "k6_run_s": $K6_RUN_S,
 "target_url": "${TARGET_URL:-<launch_k6.sh default>}", "transport": "$MODE",
 "trace_day": "$TRACE_DAY", "trace_source": "$TRACE_SOURCE", "trace_duration_min": $DURATION,
 "trace_window_start_utc": "$TRACE_START", "trace_window_end_utc": "$TRACE_END", "trace_window_std": ${TRACE_STD:-0},
 "trace_window_peak_per_min": ${TRACE_PEAK:-0}, "trace_slice_sha256": "$TRACE_SHA",
 "trace_slice_rows": $DURATION, "shape": "$SHAPE", "seed_pool": $SEED_POOL,
 "collect_dur": $COLLECT_DUR, "tools_end_offset": $TOOLS_END_OFFSET, "exec_mode": "open",
 "time_unit": "1m", "preflight": false, "seed_count": $SEED_N,
 "mcr_csv": "$MCR_SLICE_LOADGEN", "mcr_sha256": "$TRACE_SHA",
 "k6_script_sha256": "$K6_SHA", "launch_k6_sha256": "$LAUNCH_SHA",
 "five_xx_rate_min": $S5XX, "four_xx_rate_min": $S4XX, "event_cooldown_s": $SCOOL}
EOF

# ---------- phase 0: tunnels + cold gate ----------
log "phase 0: tunnels + cold reset assert gate"
pgrep -f "kube_tunnel.sh"  > /dev/null || setsid bash "$PWD/stage7_28/kube_tunnel.sh"  < /dev/null > /dev/null 2>&1 &
# Monitoring runs OUTSIDE the cluster now (cluster-repo playbook/monitoring.yaml):
# Mimir on 127.0.0.1:18080 and Loki on 127.0.0.1:3100, both on this host. Only
# start a port-forward when nothing is already serving the endpoint -- otherwise
# the tunnel and the local container fight for the port, the tunnel loses with
# "address already in use", and the checks below silently read the tunnel's dead
# socket instead of Mimir.
if ! curl -fsS -m 5 "http://127.0.0.1:18080/ready" > /dev/null 2>&1; then
  pgrep -f "mimir_tunnel.sh" > /dev/null || setsid bash "$PWD/stage7_28/mimir_tunnel.sh" < /dev/null > /dev/null 2>&1 &
fi
if ! curl -fsS -m 5 "http://127.0.0.1:3100/ready" > /dev/null 2>&1; then
  pgrep -f "loki_tunnel.sh"  > /dev/null || setsid bash "$PWD/stage7_28/loki_tunnel.sh"  < /dev/null > /dev/null 2>&1 &
fi
for i in $(seq 1 30); do
  kubectl get nodes >/dev/null 2>&1 && break
  sleep 2
done
kubectl get nodes >/dev/null 2>&1 || die "kube apiserver forward (127.0.0.1:16443) not reachable"
curl -fsS -m 8 "http://127.0.0.1:18080/prometheus/api/v1/query" --data 'query=up' > /dev/null \
  || die "mimir tunnel not reachable"
curl -fsS -m 8 "http://127.0.0.1:3100/ready" > /dev/null || die "loki tunnel not reachable"
log "tunnels verified up (kube + mimir + loki)"

if ! "$PY" pbscaler/scripts/launchers/reset_cluster_cold.py --assert-only \
     > "$RUN_DIR/cold_gate.log" 2>&1; then
  die "cold gate failed (see cold_gate.log)"
fi
log "cold gate passed: $(grep -m1 'ASSERT cold_ok' "$RUN_DIR/cold_gate.log")"

# ---------- resolve the gateway address ----------
# Used to be hardcoded (http://10.10.1.2:30080). That was master1's IP on the
# OLD cluster; on the new one 10.10.1.2 is the monitoring node and the gateway
# NodePort is whatever the Service has. Derive both from the live Service so a
# new cluster needs no edit here.
if [ -z "$TARGET_URL" ]; then
  _gw=$(kubectl -n train-ticket get svc train-ticket-gateway-istio -o json 2>/dev/null)
  if [ -n "$_gw" ]; then
    _ip=$(printf '%s' "$_gw" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); s=d["spec"]
    print((s.get("externalIPs") or [""])[0])
except Exception:
    print("")' 2>/dev/null)
    _np=$(printf '%s' "$_gw" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin)
    for p in d["spec"]["ports"]:
        if p.get("port") == 80: print(p.get("nodePort", "")); break
    else: print("")
except Exception:
    print("")' 2>/dev/null)
    if [ -z "$_ip" ]; then
      _ip=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.addresses[?(@.type=="InternalIP")].address}{"\n"}{end}' 2>/dev/null | grep -E '^10\.' | head -1)
    fi
    if [ -n "$_ip" ] && [ -n "$_np" ]; then
      TARGET_URL="http://${_ip}:${_np}"
      log "resolved TARGET_URL=$TARGET_URL (gateway externalIP/nodeIP + NodePort from the live Service)"
    fi
    unset _gw _ip _np
  fi
  [ -z "$TARGET_URL" ] && log "WARN: could not resolve the gateway address; launch_k6.sh default will be used"
fi

UPNOW=$(curl -s -m 10 "http://127.0.0.1:18080/prometheus/api/v1/query" \
        --data-urlencode 'query=up' 2>/dev/null \
        | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin); print(len(d["data"]["result"]))
except Exception:
  print(0)' 2>/dev/null)
[ "${UPNOW:-0}" -lt 1 ] && die "mimir up@now=$UPNOW (ingester not serving current data)"
log "mimir up@now=$UPNOW — healthy"

# The istio dataplane request rate is what the SLO analysis, the 5xx attribution
# and the traffic gate all read.
#
# There is NO Prometheus in this stack: Alloy (in-cluster) scrapes the istio
# ServiceMonitors and remote-writes to Mimir, which runs on the monitoring host.
# This block used to check for a Prometheus CR and log a WARN when it was absent
# -- a leftover from a since-deleted Prometheus server. It fired on every single
# run and said nothing useful, so it is gone. Absence of istio series right after
# the cold reset is EXPECTED (every pod is new, so its sidecar has no series yet)
# and is handled by the traffic gate in phase 4, which distinguishes "metric
# answered 0" from "metric unreachable".

# Kill collectors orphaned by a previous interrupted run. Without this, each
# interrupted run leaves a full set of kubectl log streams behind; two or three
# of those exhaust this 11 GB node (each kubectl is ~50 MB RSS).
kill_stale_collectors() {
  local pat p cmdline
  local -a victims=()
  for pat in 'log_streamer.sh' 'failure_recorder.py' 'monitor_v2.py'; do
    for p in $(pgrep -f "$pat" 2>/dev/null); do
      [ "$p" = "$$" ] && continue
      cmdline=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
      case "$cmdline" in *"$RUN_DIR"*) continue ;; esac
      victims+=("$p")
    done
  done
  [ "${#victims[@]}" -eq 0 ] && return 0
  log "killing ${#victims[@]} stale collector(s) from a previous run: ${victims[*]}"
  for p in "${victims[@]}"; do
    pkill -TERM -P "$p" 2>/dev/null
    kill -TERM "$p" 2>/dev/null
  done
  sleep 3
  for p in "${victims[@]}"; do
    pkill -KILL -P "$p" 2>/dev/null
    kill -KILL "$p" 2>/dev/null
  done
  # and any kubectl log stream that outlived its streamer
  local left
  left=$(pgrep -f 'kubect[l].*logs -f --timestamps --all-containers' 2>/dev/null | wc -l)
  log "kubectl log streams still running after stale cleanup: $left"
}
kill_stale_collectors

# loadgen must be idle
K6_PRE=$(loadgen_k6_pids | head -1)
[ -z "${K6_PRE:-}" ] || die "k6 already running on loadgen (pid $K6_PRE)"
log "loadgen has no running k6"

if [ "$DRY" = "--dry" ]; then
  log "DRY mode: stopping here (no tools, no k6)"
  exit 0
fi

# ---------- phase 2: capture tools (all stop at END_EPOCH) ----------
LAUNCH_EPOCH=$(date +%s)
END_EPOCH=$((LAUNCH_EPOCH + TOOLS_END_OFFSET))
echo "$LAUNCH_EPOCH" > "$RUN_DIR/launch_epoch"
echo "$END_EPOCH" > "$RUN_DIR/end_epoch"
log "phase 2: tools start (launch=$LAUNCH_EPOCH end=$END_EPOCH)"

# MAX_STREAMS caps how many pods get a log stream. A fixed 96 was fine for 65
# pods but silently skips pods on a bigger cluster (they are recorded in
# logs_skipped.txt, so a run looks complete while missing logs). Derive it from
# the live pod count: one stream per pod, plus headroom. Explicit override wins.
if [ "$MAX_STREAMS" -eq 0 ]; then
  _npods=$(kubectl -n train-ticket get pods --no-headers 2>/dev/null | grep -c . || echo 0)
  case "$_npods" in ''|*[!0-9]*) _npods=0 ;; esac
  MAX_STREAMS=$(( _npods + 32 ))
  [ "$MAX_STREAMS" -lt 96 ] && MAX_STREAMS=96
  [ "$MAX_STREAMS" -gt 400 ] && MAX_STREAMS=400
  unset _npods
  log "MAX_STREAMS derived from live pod count -> $MAX_STREAMS"
fi

MAX_STREAMS="$MAX_STREAMS" bash "$TOOLS/log_streamer.sh" "$RUN_DIR" "$END_EPOCH" \
  < /dev/null >> "$RUN_DIR/log_streamer.out" 2>&1 &
LS_PID=$!
track_collector log_streamer "$LS_PID"
log "log_streamer started (pid $LS_PID)"

env PBS_MIMIR_URL=http://127.0.0.1:18080 python3 \
  "$TOOLS/monitor_v2.py" --run-dir "$RUN_DIR" --t0 0 \
  --end "$END_EPOCH" --mcr "$MCR_SLICE_LOCAL" --max-requests "$MAX_REQUESTS" \
  --start-min "$START_MIN" < /dev/null >> "$RUN_DIR/monitor_v2.out" 2>&1 &
MON_PID=$!
track_collector monitor_v2 "$MON_PID"
log "monitor_v2 started (pid $MON_PID)"

env PBS_MIMIR_URL=http://127.0.0.1:18080 LOKI_URL=http://127.0.0.1:3100 \
  KUBECTL="$KUBECTL_BIN" KUBECONFIG="$KUBECONFIG" \
  5XX_RATE_MIN="$S5XX" 4XX_RATE_MIN="$S4XX" EVENT_COOLDOWN_S="$SCOOL" \
  python3 "$TOOLS/failure_recorder.py" --run-dir "$RUN_DIR" \
  --end "$END_EPOCH" --runid "$RUNID" < /dev/null >> "$RUN_DIR/failure_recorder.out" 2>&1 &
FR_PID=$!
track_collector failure_recorder "$FR_PID"
log "failure_recorder started (pid $FR_PID)"

# k6 console offset-pull (loadgen -> RUN_DIR/k6_console.log)
(
  # `off` = bytes already copied (0-based): data exists iff sz > off, and the
  # file was truncated iff sz < off.
  off=0
  while [ "$(date +%s)" -lt $((END_EPOCH + 120)) ]; do
    sz=$(on_loadgen "wc -c < '$CONSOLE_SRC' 2>/dev/null" 2>/dev/null || echo 0)
    case "$sz" in (''|*[!0-9]*) sz=0;; esac
    # Truncation / rotation: the offset would otherwise be past EOF forever and
    # every later line would be lost silently.
    if [ "$sz" -lt "$off" ]; then
      echo "$(date -u +%FT%TZ) console truncated ($sz < $off copied) -- restarting" >> "$RUN_DIR/console_shipper.out"
      off=0
    fi
    if [ "$sz" -gt "$off" ]; then
      # Copy to a temp file first, then advance by the bytes ACTUALLY copied.
      # Advancing to the pre-copy size re-reads anything k6 wrote during the
      # copy and duplicates output (measured 17x on the integration run).
      tmp="${TMPDIR:-/tmp}/console_chunk_${RUNID}.$$"
      on_loadgen "tail -c +$((off + 1)) '$CONSOLE_SRC'" > "$tmp" 2>/dev/null
      n=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
      case "$n" in (''|*[!0-9]*) n=0;; esac
      if [ "$n" -gt 0 ]; then
        cat "$tmp" >> "$RUN_DIR/k6_console.log"
        off=$((off + n))
      fi
      rm -f "$tmp"
      echo "$(date -u +%FT%TZ) shipped console bytes $n (offset now $off, size $sz)" >> "$RUN_DIR/console_shipper.out"
    fi
    sleep 15
  done
) < /dev/null > /dev/null 2>&1 &
log "console shipper started (pid $!)"

# failures.jsonl offset-push (RUN_DIR -> loadgen, live during the run)
on_loadgen "mkdir -p $FAILURE_LOG_DIR/$RUNID" 2>/dev/null
(
  foff=0
  while [ "$(date +%s)" -lt $((END_EPOCH + 60)) ]; do
    if [ -f "$RUN_DIR/failures.jsonl" ]; then
      fsz=$(stat -c%s "$RUN_DIR/failures.jsonl" 2>/dev/null || echo 0)
      if [ "$fsz" -gt "$foff" ]; then
        if [ "$MODE" = local ]; then
          # loadgen is this node: the ledger is already local, so snapshot it.
          # `tail | cat >>` would append the file onto itself and duplicate rows.
          cp -f "$RUN_DIR/failures.jsonl" \
                "$FAILURE_LOG_DIR/$RUNID/failures.jsonl" 2>/dev/null
        else
          tail -c +$((foff + 1)) "$RUN_DIR/failures.jsonl" \
            | $SSH loadgen "cat >> $FAILURE_LOG_DIR/$RUNID/failures.jsonl" 2>/dev/null
        fi
        echo "$(date -u +%FT%TZ) pushed failures bytes $((fsz - foff)) (total $fsz)" >> "$RUN_DIR/failures_shipper.out"
        foff=$fsz
      fi
    fi
    sleep 10
  done
) < /dev/null > /dev/null 2>&1 &
log "failures shipper started -> $FAILURE_LOG_DIR/$RUNID/failures.jsonl (pid $!)"

# ---------- phase 2a: disk watchdog ----------
if [ "${DISK_WATCHDOG:-1}" = "1" ]; then
  (
    while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
      free_gb=$(df -BG --output=avail "$RUN_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
      case "$free_gb" in ''|*[!0-9]*) free_gb=99 ;; esac
      if [ "$free_gb" -lt "$DISK_CRITICAL_GB" ]; then
        echo "$(date -u +%FT%TZ) CRITICAL free=${free_gb}G -- pruning periodic snapshots" >> "$RUN_DIR/disk_watchdog.log"
        rm -rf "$RUN_DIR"/dumps_periodic/* 2>/dev/null
        status_reason "disk free ${free_gb}G below critical ${DISK_CRITICAL_GB}G"
      elif [ "$free_gb" -lt "$DISK_PRUNE_BELOW_GB" ]; then
        echo "$(date -u +%FT%TZ) LOW free=${free_gb}G -- removing oldest periodic snapshot" >> "$RUN_DIR/disk_watchdog.log"
        oldest=$(ls -d "$RUN_DIR"/dumps_periodic/*/ 2>/dev/null | sort | head -1)
        [ -n "$oldest" ] && rm -rf "$oldest"
      fi
      sleep 60
    done
  ) < /dev/null > /dev/null 2>&1 &
  track_collector disk_watchdog "$!"
  log "phase 2a: disk watchdog started (prune below ${DISK_PRUNE_BELOW_GB}G, critical ${DISK_CRITICAL_GB}G)"
fi

# ---------- phase 2b: periodic Mimir/Loki snapshots ----------
if [ "$PERIODIC_DUMP_EVERY" -gt 0 ]; then
  (
    last=$LAUNCH_EPOCH
    while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
      # Sleep in short slices: a single `sleep 1800` overshoots END_EPOCH, so the
      # collector was still napping when phase 7 killed it (rc=143, "did not stop
      # at END_EPOCH"). Waking every 60 s makes the exit land on the boundary.
      s=0
      while [ "$s" -lt "$PERIODIC_DUMP_EVERY" ] && [ "$(date +%s)" -lt "$END_EPOCH" ]; do
        sleep 60; s=$((s + 60))
      done
      now=$(date +%s)
      [ "$now" -ge "$END_EPOCH" ] && break
      # Skip a snapshot that cannot finish before END_EPOCH: phase 7 would kill
      # it mid-dump and leave a partial snapshot behind.
      if [ $((END_EPOCH - now)) -lt 420 ]; then
        echo "$(date -u +%FT%TZ) skipping periodic dump: only $((END_EPOCH - now))s left" \
          >> "$RUN_DIR/periodic_dump.out"
        break
      fi
      d="$RUN_DIR/dumps_periodic/$(date -u +%Y%m%dT%H%M%SZ)"
      mkdir -p "$d"
      bash "$TOOLS/dump_mimir.sh" "$last" "$now" "$d/mimir" \
        >> "$RUN_DIR/periodic_dump.out" 2>&1
      bash "$TOOLS/dump_loki.sh" "$last" "$now" "$d/loki" \
        >> "$RUN_DIR/periodic_dump.out" 2>&1
      echo "$(date -u +%FT%TZ) periodic dump $d (window ${last}..${now})" \
        >> "$RUN_DIR/periodic_dump.out"
      last=$now
    done
  ) < /dev/null > /dev/null 2>&1 &
  track_collector periodic_dump "$!"
  log "phase 2b: periodic Mimir+Loki dumps every ${PERIODIC_DUMP_EVERY}s -> dumps_periodic/"
fi

# ---------- phase 2c: log rotation ----------
# log_streamer's per-pod files are the fastest-growing thing in RUN_DIR and the
# disk watchdog does not cover them. Rotate them into .tgz archives instead.
if [ "${LOG_ROTATE_MB:-0}" -gt 0 ]; then
  bash "$TOOLS/rotate_run_logs.sh" "$RUN_DIR" "$LOG_ROTATE_MB" \
    < /dev/null > /dev/null 2>&1 &
  track_collector log_rotation "$!"
  log "phase 2c: log rotation started (gzip+truncate logs/ above ${LOG_ROTATE_MB} MB)"
fi

# ---------- phase 3: launch k6 on the load generator ----------
# Local mode runs k6 as a direct child of this script so its real exit status
# is observable; ssh mode keeps the detached remote launch (exit status is not
# recoverable across ssh, so the console marker is used instead).
launch_k6_local() {
  (
    ulimit -n 65536 2>/dev/null || ulimit -n 8192
    exec env RUNID="$RUNID" \
      MAX_REQUESTS="$MAX_REQUESTS" USER_POOL="$USER_POOL" TRACE_COMPRESS=1 OPEN_LOOP=1 \
      PRE_ALLOCATED_VUS="$PRE_ALLOC" MAX_VUS="$MAX_VUS" \
      PREFLIGHT=0 ALLOW_SEED_REUSE=0 LOGIN_RETRIES=2 LOGIN_COOLDOWN_MS=30000 \
      START_MIN="$START_MIN" TEST_MIN="$TEST_MIN" \
      TARGET_URL="$TARGET_URL" \
      MCR_CSV="$MCR_SLICE_LOADGEN" \
      bash "$LAUNCH_K6_SH"
  ) > "$CONSOLE_SRC" 2>&1 < /dev/null &
  K6_PID=$!
  echo "$K6_PID" > "$RUN_DIR/k6.pid"
}

launch_k6_ssh() {
  on_loadgen "ulimit -n 65536; nohup setsid env RUNID='$RUNID' \
    MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL TRACE_COMPRESS=1 OPEN_LOOP=1 \
    PRE_ALLOCATED_VUS=$PRE_ALLOC MAX_VUS=$MAX_VUS \
    PREFLIGHT=0 ALLOW_SEED_REUSE=0 LOGIN_RETRIES=2 LOGIN_COOLDOWN_MS=30000 \
    START_MIN=$START_MIN TEST_MIN=$TEST_MIN \
    TARGET_URL='$TARGET_URL' \
    MCR_CSV=$MCR_SLICE_LOADGEN bash $LAUNCH_K6_SH \
    > '$CONSOLE_SRC' 2>&1 < /dev/null &" 2>&1 | sed 's/^/k6launch: /' >> "$LOG"
}

log "phase 3: launching k6 (RUNID=$RUNID, mode=$MODE, preAllocatedVUs=$PRE_ALLOC, maxVUs=$MAX_VUS)"
if [ "$MODE" = local ]; then
  launch_k6_local
  K6_KIND=local
  log "k6 launched locally pid=$K6_PID (console=$CONSOLE_SRC)"
else
  launch_k6_ssh
  K6_KIND=ssh
  log "k6 launch command issued over ssh"
fi

sleep 25
KSZ=$(on_loadgen "wc -c < '$CONSOLE_SRC' 2>/dev/null" 2>/dev/null || echo 0)
case "$KSZ" in (''|*[!0-9]*) KSZ=0;; esac
[ "$KSZ" -lt 100 ] && die "k6 console log size=$KSZ after 25 s — k6 did not start"
log "k6 console confirmed growing (size=$KSZ)"

# ---------- phase 3.5: capture-tool liveness ----------
sleep 45
kill -0 "$FR_PID" 2>/dev/null || die "failure_recorder died at startup (see failure_recorder.out)"
[ -f "$RUN_DIR/failures.jsonl" ] || die "failure_recorder produced no failures.jsonl"
log "capture tools alive; failures.jsonl present ($(stat -c%s "$RUN_DIR/failures.jsonl") B)"
for i in "${!COLLECTOR_PIDS[@]}"; do
  p=${COLLECTOR_PIDS[$i]}; n=${COLLECTOR_NAMES[$i]}
  if kill -0 "$p" 2>/dev/null; then
    log "  $n running (pid $p)"
  else
    log "  WARN: $n (pid $p) not running"
    [ "$n" = log_streamer ] || status_reason "$n died during startup"
  fi
done

# ---------- phase 4: traffic detection ----------
if [ "$SKIP_TRAFFIC_GATE" = "1" ]; then
  t_traffic=$LAUNCH_EPOCH
  echo "$t_traffic" > "$RUN_DIR/traffic_start_epoch"
  log "phase 4: SKIPPED (SKIP_TRAFFIC_GATE=1) -- measured window anchored at launch"
else
log "phase 4: waiting for traffic (ui rps > 0.2)"
# The istio dataplane request rate is the primary signal. If that metric is
# unreachable (Prometheus/Mimir down) the gate must not fail the run: an
# infrastructure gap is not a no-traffic condition. So the query distinguishes
# "no answer" (metric unavailable) from "answer 0" (no traffic), and only the
# latter counts toward the 20 min limit. A metric outage longer than
# TRAFFIC_GATE_FALLBACK_S anchors the window at launch with a recorded reason.
t_traffic=""
metric_missing_since=""
for i in $(seq 1 240); do   # up to 20 min of "metric answering zero"
  resp=$(curl -s -m 8 "http://127.0.0.1:18080/prometheus/api/v1/query" \
          --data 'query=sum(rate(istio_requests_total{reporter="destination",destination_workload="ts-ui-dashboard"}[30s]))' 2>/dev/null)
  if [ -z "$resp" ]; then
    # no answer at all -> metric unavailable
    [ -z "$metric_missing_since" ] && metric_missing_since=$(date +%s)
    if [ $(( $(date +%s) - metric_missing_since )) -ge "$TRAFFIC_GATE_FALLBACK_S" ]; then
      log "WARN: istio metric unavailable for ${TRAFFIC_GATE_FALLBACK_S}s -- anchoring the measured window at launch"
      status_reason "traffic gate fell back to launch anchor: Mimir istio_requests_total unreachable"
      t_traffic=$LAUNCH_EPOCH
      break
    fi
    sleep 5
    continue
  fi
  rps=$(printf '%s' "$resp" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); r=d["data"]["result"]
    print(json.loads(r[0]["value"][1]) if r else 0)
except Exception:
    print(0)' 2>/dev/null || echo 0)
  metric_missing_since=""
  if python3 -c "import sys; sys.exit(0 if float('$rps' or 0) > 0.2 else 1)" 2>/dev/null; then
    t_traffic=$(date +%s); break
  fi
  sleep 5
done
[ -z "$t_traffic" ] && die "no traffic detected in 20 min (istio metric answering zero)"
echo "$t_traffic" > "$RUN_DIR/traffic_start_epoch"
log "traffic detected epoch=$t_traffic (+$((t_traffic - LAUNCH_EPOCH)) s after launch)"
log "measured window starts at the first selected bucket (WARMUP_S=$WARMUP_S)"
fi

# ---------- phase 5: wait for natural k6 completion ----------
K6_DEADLINE=$((LAUNCH_EPOCH + K6_DEADLINE_OFFSET))
log "phase 5: waiting for k6 to finish naturally (deadline launch+$K6_DEADLINE_OFFSET s)"
if [ "$MODE" = local ]; then
  while [ "$(date +%s)" -lt "$K6_DEADLINE" ]; do
    k6_alive_local "$K6_PID" || break
    sleep 10
  done
  if k6_alive_local "$K6_PID"; then
    log "k6 still running at the deadline -- SIGINT (gracefulStop 30 s)"
    k6_signal_int || true
    for i in $(seq 1 36); do k6_alive_local "$K6_PID" || break; sleep 5; done
  else
    log "k6 exited naturally"
  fi
  if k6_alive_local "$K6_PID"; then
    log "k6 did not stop after SIGINT -- SIGKILL"
    kill -KILL "$K6_PID" 2>/dev/null || true
    status_reason "k6 had to be SIGKILLed at the deadline"
  fi
  wait "$K6_PID"; K6_RC=$?
  echo "$K6_RC" > "$RUN_DIR/k6_exit_code"
  log "k6 exited rc=$K6_RC (pid $K6_PID)"
  [ "$K6_RC" -eq 0 ] || status_reason "k6 exited with rc=$K6_RC"
else
  while [ "$(date +%s)" -lt "$K6_DEADLINE" ]; do
    on_loadgen "pgrep -x k6 >/dev/null" || break
    sleep 10
  done
  if on_loadgen "pgrep -x k6 >/dev/null"; then
    KP=$(on_loadgen "pgrep -x k6 | head -1" 2>/dev/null | grep -E '^[0-9]+$' | head -1)
    [ -n "${KP:-}" ] && on_loadgen "kill -INT $KP" && log "k6 SIGINT sent at deadline (pid $KP)"
  else
    log "k6 exited naturally"
  fi
fi

# mid-run snapshot of the failure timeline for the report
timeout 300 timeout 300 timeout 600 cp -f "$CONSOLE_SRC" "$RUN_DIR/k6_console_full.log" 2>/dev/null \
log "failure events so far: $(grep -c FAILURE_EVENT "$RUN_DIR/failures.jsonl" 2>/dev/null || echo 0)"

# wait for k6 to flush its summary (remote only: local k6 was already reaped)
if [ "$MODE" != local ]; then
  for i in $(seq 1 36); do
    on_loadgen "pgrep -x k6 >/dev/null" || break
    sleep 5
  done
fi
log "k6 exited; pulling summary + console sizes"
# Authoritative full console copy, independent of the incremental shipper.
if [ "$MODE" = local ]; then
  cp -f "$CONSOLE_SRC" "$RUN_DIR/k6_console_full.log" 2>/dev/null \
    && log "console captured in full -> k6_console_full.log ($(stat -c%s "$RUN_DIR/k6_console_full.log" 2>/dev/null) B)" \
    || status_reason "could not capture the full k6 console"
else
  $SCP -q "loadgen:$CONSOLE_SRC" "$RUN_DIR/k6_console_full.log" 2>/dev/null \
    && log "console captured in full -> k6_console_full.log" \
    || status_reason "could not capture the full k6 console"
fi
on_loadgen "ls -la $K6_CONSOLE_DIR/*_${RUNID}.*" >> "$LOG" 2>&1
from_loadgen "$K6_CONSOLE_DIR/summary_${RUNID}.json" "$RUN_DIR/k6_summary.json" 2>/dev/null \
  && log "k6 summary pulled -> k6_summary.json" || log "WARN: k6 summary pull failed"

# ---------- phase 5b: post-hoc training-data collection from Mimir ----------
DS="$RUN_DIR/dataset_$RUNID"
mkdir -p "$DS"
if [ "$SKIP_DATASET" = "1" ]; then
  log "phase 5b: SKIPPED (SKIP_DATASET=1)"
else
  log "phase 5b: post-hoc dataset collection (waiting 180 s for Alloy flush)"
  sleep 180
  CSTART=$((t_traffic + WARMUP_S))
  if "$PY" "$TOOLS/rebuild_train_data.py" \
       --start "$CSTART" --duration "$COLLECT_DUR" --interval 5 \
       --out "$DS" --workload "$MCR_SLICE_LOCAL" \
       > "$RUN_DIR/rebuild_collection.log" 2>&1; then
    log "post-hoc collection OK -> $DS"
  else
    log "FATAL: post-hoc collection failed (see rebuild_collection.log)"
    status_reason "post-hoc dataset collection failed"
  fi
  tail -14 "$RUN_DIR/rebuild_collection.log" >> "$LOG" 2>/dev/null
fi

# ---------- phase 7: wait for tools to finish ----------
WAIT=$((END_EPOCH + 90 - $(date +%s)))
log "phase 7: waiting ${WAIT}s for tools to finish (END_EPOCH+90)"
[ "$WAIT" -gt 0 ] && sleep "$WAIT"

# Reap only the collectors this run started. A collector still alive past
# END_EPOCH+90 is a failure, not something to leave behind.
: > "$RUN_DIR/collector_exit_codes.txt"
for i in "${!COLLECTOR_PIDS[@]}"; do
  p=${COLLECTOR_PIDS[$i]}; n=${COLLECTOR_NAMES[$i]}
  if kill -0 "$p" 2>/dev/null; then
    # log_rotation is designed to still be running at END_EPOCH -- it is not a
    # "did not stop" failure. Everything else is.
    #
    # disk_watchdog and periodic_dump are backgrounded *subshells* containing
    # loops, so $! is the subshell and its children (sleep, dump_mimir.sh,
    # dump_loki.sh, xargs, curl) are NOT killed with it -- they survive as
    # orphans and keep hammering Mimir/Loki after the run. Reap the children
    # first, then the collector.
    pkill -TERM -P "$p" 2>/dev/null || true
    if [ "$n" = log_rotation ]; then
      log "  $n (pid $p) still running as designed -- stopping"
      kill -TERM "$p" 2>/dev/null || true
    else
      log "  WARN: $n (pid $p) still running past END_EPOCH+90 -- terminating"
      kill -TERM "$p" 2>/dev/null || true
      status_reason "$n did not stop at END_EPOCH"
    fi
  fi
  wait "$p" 2>/dev/null; rc=$?
  echo "$n $rc" >> "$RUN_DIR/collector_exit_codes.txt"
  log "  collector $n exit=$rc"
  [ "$rc" -eq 0 ] || status_reason "$n exited rc=$rc"
done
# Collector completion markers (independent of exit codes).
grep -q "log_streamer: stopped" "$RUN_DIR/log_streamer.out" 2>/dev/null \
  || status_reason "log_streamer has no stop marker (see log_streamer.out)"
grep -q '"type": "RECORDER_STOP"' "$RUN_DIR/failures.jsonl" 2>/dev/null \
  || status_reason "failure_recorder has no RECORDER_STOP record"
[ -s "$RUN_DIR/live.csv" ] || status_reason "monitor_v2 produced no live.csv rows"
log "collector completion markers checked"

grep -c FAILURE_EVENT "$RUN_DIR/failures.jsonl" 2>/dev/null | xargs -I{} echo "failure events captured: {}" >> "$LOG"
log "failure event types: $(grep -o '"type": "[A-Z_]*"' "$RUN_DIR/failures.jsonl" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"

# ---------- phase 8: validate dataset ----------
if [ "$SKIP_DATASET" = "1" ]; then
  log "phase 8: SKIPPED (SKIP_DATASET=1)"
else
log "phase 8: validating dataset meta.json"
python3 - "$DS" >> "$LOG" 2>&1 <<'EOF'
import json, sys, glob
ds = sys.argv[1]
metas = glob.glob(ds + "/**/meta.json", recursive=True) or glob.glob(ds + "/meta.json")
if not metas:
    print("meta.json NOT FOUND in", ds); raise SystemExit(1)
m = json.load(open(metas[0]))
txt = open(metas[0]).read()
print("meta:", {k: m.get(k) for k in ("result", "coverage_pct", "interval_s", "duration_s", "no_data_rows", "rows") if k in m})
print("RESULT COMPLETE:", "COMPLETE" in txt.upper())
cov = m.get("coverage_pct") or 0
print("coverage_ok:", float(cov) >= 95 if cov else False)
EOF
fi

# ---------- phase 9: build the one-file report ----------
# Reads this run's artifacts (k6 summary, live.csv, failures.jsonl, the dataset,
# hpa/dep history) plus Mimir, and writes a single timestamped REPORT_*.md into
# RUN_DIR. With 60-day retention the per-run dumps are no longer the archival
# copy -- this report is.
if [ "${SKIP_REPORT:-0}" = "1" ]; then
  log "phase 9: SKIPPED (SKIP_REPORT=1)"
elif [ "$SKIP_DATASET" = "1" ]; then
  log "phase 9: SKIPPED (no dataset -- SKIP_DATASET=1)"
else
  log "phase 9: building the run report"
  if python3 "$TOOLS/build_run_report.py" "$RUN_DIR" \
       --mimir "${PBS_MIMIR_URL:-http://127.0.0.1:18080}" >> "$LOG" 2>&1; then
    _rep=$(ls -t "$RUN_DIR"/REPORT_*.md 2>/dev/null | head -1)
    log "phase 9: report -> ${_rep:-<none>}"
  else
    log "WARN: report generation failed"
    status_reason "report generation failed"
  fi
fi

# ---------- phase 10: Mimir + Loki range dumps ----------
if [ "$SKIP_DUMPS" = "1" ]; then
  log "phase 10: SKIPPED (SKIP_DUMPS=1)"
else
  log "phase 10: Mimir + Loki dumps (launch-60 .. END_EPOCH)"
  DUMP_DIR="$RUN_DIR/dumps"
  mkdir -p "$DUMP_DIR"
  bash "$TOOLS/dump_mimir.sh" "$((LAUNCH_EPOCH - 60))" "$END_EPOCH" \
       "$DUMP_DIR/mimir" >> "$LOG" 2>&1 \
    && log "mimir dumps OK -> $DUMP_DIR/mimir" \
    || { log "WARN: mimir dumps failed"; status_reason "mimir dump failed"; }
  bash "$TOOLS/dump_loki.sh" "$((LAUNCH_EPOCH - 60))" "$END_EPOCH" \
       "$DUMP_DIR/loki" >> "$LOG" 2>&1 \
    && log "loki dump OK -> $DUMP_DIR/loki" \
    || { log "WARN: loki dump failed"; status_reason "loki dump failed"; }
fi

# ---------- phase 11: archive RUN_DIR + k6 outputs ----------
log "phase 11: archiving run dir (mode=$MODE)"
ARCHIVE="$FAILURE_LOG_DIR/${RUNID}_rundir.tgz"
# Manifest of what the run produced, written before the tarball so the archive
# and the manifest can be compared after the fact.
MANIFEST="$RUN_DIR/archive_manifest.txt"
( cd "$RUN_DIR" && find . -type f | sort | while read -r f; do
    printf "%s  %s\n" "$(sha256sum "$f" | cut -d' ' -f1)" "$f"
  done ) > "$MANIFEST" 2>/dev/null
log "archive manifest: $(wc -l < "$MANIFEST") files"

archive_ok=0
tar_rc=0
# tar exits 1 when a file changes while it is being read. RUN_DIR still holds
# the log_streamer files, so that is expected and benign; integrity is proven
# below by `gzip -t` plus a member count. rc >= 2 is a real error.
if [ "$MODE" = local ]; then
  tar czf "$ARCHIVE" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")" \
      2> "$RUN_DIR/archive_tar.err"
  tar_rc=$?
else
  tar czf - -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")" \
      2> "$RUN_DIR/archive_tar.err" \
    | $SSH loadgen "cat > $ARCHIVE" 2>/dev/null
  tar_rc=$?
fi
case "$tar_rc" in
  0) archive_ok=1 ;;
  1) log "archive: tar rc=1 (file changed while reading; benign for live logs)"
     grep -v "file changed as we read it" "$RUN_DIR/archive_tar.err" 2>/dev/null | head -5
     archive_ok=1 ;;
  *) log "archive: tar rc=$tar_rc"
     head -5 "$RUN_DIR/archive_tar.err" 2>/dev/null ;;
esac

if [ "$archive_ok" -eq 1 ]; then
  # Verify integrity and membership, not just that the command exited 0.
  if [ "$MODE" = local ]; then
    gzip -t "$ARCHIVE" 2>/dev/null; gz_ok=$?
    n_members=$(tar tzf "$ARCHIVE" 2>/dev/null | grep -vc '/$')
  else
    on_loadgen "gzip -t '$ARCHIVE'" >/dev/null 2>&1; gz_ok=$?
    n_members=$(on_loadgen "tar tzf '$ARCHIVE' 2>/dev/null | grep -vc '/\$'" 2>/dev/null | tail -1)
  fi
  if [ "$gz_ok" -eq 0 ] && [ "${n_members:-0}" -gt 0 ]; then
    log "archive OK -> $ARCHIVE (gzip integrity ok, $n_members files)"
    echo "$ARCHIVE" > "$RUN_DIR/archive_path.txt"
  else
    log "WARN: archive integrity check failed (gzip rc=$gz_ok, members=$n_members)"
    status_reason "archive failed its integrity check"
  fi
else
  log "WARN: archive creation failed"
  status_reason "archive creation failed"
fi

# ---------- phase 12: final cold reset ----------
if [ "$SKIP_RESET" = "1" ]; then
  log "phase 12: SKIPPED (SKIP_RESET=1)"
else
  log "phase 12: final cold reset"
  if "$PY" pbscaler/scripts/launchers/reset_cluster_cold.py \
       --out "$RUN_DIR/final_cold_reset.json" > "$RUN_DIR/final_cold_reset.log" 2>&1; then
    log "final cold reset OK: $(grep -m1 'POST:' "$RUN_DIR/final_cold_reset.log" | cut -c1-400)"
  else
    log "WARN: final cold reset flagged issues"
    tail -5 "$RUN_DIR/final_cold_reset.log" >> "$LOG"
    status_reason "final cold reset failed"
  fi
fi

# Reaching the end with no recorded reason is the only path to "complete".
[ "$RUN_STATUS" = running ] && RUN_STATUS=complete
write_status
if [ "$RUN_STATUS" = complete ]; then
  log "=== run_experiment COMPLETE runid=$RUNID ==="
else
  log "=== run_experiment INCOMPLETE runid=$RUNID ==="
  log "reasons: ${STATUS_REASONS[*]-none}"
fi
echo "RUNID=$RUNID RUN_DIR=$RUN_DIR STATUS=$RUN_STATUS"
