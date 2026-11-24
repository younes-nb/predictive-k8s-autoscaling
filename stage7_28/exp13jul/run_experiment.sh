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
#   --select transitions  window objective: transitions|std|tv (default transitions)
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
# Trace window selection objective (nasa_trace_slice.py --select).
TRACE_SELECT=${TRACE_SELECT:-transitions}
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
case "$SHAPE" in hpa|pbscaler|aurora|aurora-r) ;; *) fail "--shape must be hpa, pbscaler, aurora or aurora-r" ;; esac
case "$TRACE_SOURCE" in jul|aug) ;; *) fail "--source must be jul or aug" ;; esac

# ---- parameters derived from the CLI --------------------------------------
# TRACE_MIN is the length of the trace slice selected from the archive; TEST_MIN
# is how many of its rows k6 actually replays. They differ when you want the
# shape/scale of a long window but only its head, e.g. the first 15 min of the
# 60 min 13-Jul window: TRACE_MIN=60 (so the slice keeps its own 1.0 peak and
# MAX_REQUESTS means the same thing as in the 60 min run) + TEST_MIN=15.
TRACE_MIN=${TRACE_MIN:-$DURATION}
TEST_MIN=${TEST_MIN:-$DURATION}   # measured window = replayed rows
START_MIN=${START_MIN:-0}         # the slice starts at the selected window
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
# AURORA development mode: the controller stays in the cluster and owns the
# replicas. The cold gate must then not demand HPAs that AURORA deliberately
# suspended, and phase 12 must not hand the replicas back mid-development.
AURORA_DEV=${AURORA_DEV:-0}
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
echo "selecting trace: day=$TRACE_DAY duration=${TRACE_MIN}m source=$TRACE_SOURCE"
TRACE_JSON=$(python3 "$TOOLS/nasa_trace_slice.py" \
  --day "$TRACE_DAY" --duration "$TRACE_MIN" --source "$TRACE_SOURCE" \
  --select "$TRACE_SELECT" \
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
  # No collectors to stop any more; only the k6 process group is run-owned.
  write_status
  log "=== run_experiment ABORTED runid=$RUNID ==="
  exit "$code"
}
trap 'cleanup INT 130' INT
trap 'cleanup TERM 143' TERM

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
log "collectors: none (all capture is post-hoc from Mimir/Loki)"
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
for f in "$MCR_SLICE_LOCAL" "$TOOLS/rebuild_train_data.py" \
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

# ---- VU sizing: auto = derived from the offered load (2026-11-19) ----------
# A fixed pool equal to the seed count (2999) is simultaneously far more than the
# run needs at low load (120k used 7-94 active VUs) and the binding constraint at
# high load (480k pinned 2999 and still dropped 63 % of iterations). Size the cap
# from the offered peak instead:
#
#   VUs = ceil(peak_iterations_per_s * VUS_D_MAX * VUS_SAFETY)
#
# VUS_D_MAX is a conservative per-iteration wall-clock estimate (0.5 s covers the
# 30 s-timeout regime only partially, which is why SAFETY exists). Set
# VUS_MODE=static to pin PRE_ALLOCATED_VUS/MAX_VUS as before.
VUS_MODE=${VUS_MODE:-auto}
VUS_D_MAX=${VUS_D_MAX:-0.5}
VUS_SAFETY=${VUS_SAFETY:-2}
if [ "$VUS_MODE" = auto ] && [ -z "${MAX_VUS_OVERRIDE:-}" ] && [ -z "${PRE_ALLOCATED_VUS_OVERRIDE:-}" ]; then
  _vu_need=$(python3 -c "
import math
peak = max(1.0, $MAX_REQUESTS / 60.0)
print(int(math.ceil(peak * $VUS_D_MAX * $VUS_SAFETY)))")
  MAX_VUS=$_vu_need
  PRE_ALLOC=$_vu_need
  unset _vu_need
fi
PRE_ALLOC=${PRE_ALLOCATED_VUS_OVERRIDE:-$PRE_ALLOC}
MAX_VUS=${MAX_VUS_OVERRIDE:-$MAX_VUS}
# The VU cap is derived from the offered load, NOT from the seed count: the
# generator must be able to deliver the requested rate and the CLUSTER must be
# the bottleneck. One VU owns one account, so the seed pool has to be large
# enough -- fail closed with the exact number needed rather than silently
# clamping the load down.
if [ "$MAX_VUS" -gt "$SEED_N" ] && [ "${ALLOW_SEED_REUSE:-0}" != "1" ]; then
  die "VU sizing needs MAX_VUS=$MAX_VUS accounts but the seed pool has $SEED_N. Generate more seeds (load_testing/seed_tt_users.py --count $MAX_VUS --out <file>), or set ALLOW_SEED_REUSE=1 to share accounts."
fi
[ "$SEED_N" -ge "$USER_POOL" ] || die "seed pool too small: loadgen has $SEED_N accounts, USER_POOL=$USER_POOL"
[ "$SEED_N" -ge "$PRE_ALLOC" ] || die "seed pool too small: loadgen has $SEED_N accounts, PRE_ALLOCATED_VUS=$PRE_ALLOC"
log "seed pool: $SEED_N unique accounts (>= USER_POOL=$USER_POOL, >= PRE_ALLOCATED_VUS=$PRE_ALLOC)"
log "VU sizing: mode=$VUS_MODE d_max=${VUS_D_MAX}s safety=$VUS_SAFETY -> PRE_ALLOCATED_VUS=$PRE_ALLOC MAX_VUS=$MAX_VUS for MAX_REQUESTS=$MAX_REQUESTS"

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
 "trace_day": "$TRACE_DAY", "trace_source": "$TRACE_SOURCE", "trace_select": "$TRACE_SELECT", "trace_duration_min": $TRACE_MIN,
 "replay_min": $TEST_MIN, "replay_start_min": $START_MIN, "measured_min": $DURATION,
 "trace_window_start_utc": "$TRACE_START", "trace_window_end_utc": "$TRACE_END", "trace_window_std": ${TRACE_STD:-0},
 "trace_window_peak_per_min": ${TRACE_PEAK:-0}, "trace_slice_sha256": "$TRACE_SHA",
 "trace_slice_rows": $TRACE_MIN, "shape": "$SHAPE", "seed_pool": $SEED_POOL,
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
# Mimir answers slowly while it drains a query backlog (measured 9-16 s on
# 2026-11-20 after a heavy post-hoc collection), so the health check has to be
# patient: a short timeout here used to abort perfectly good runs.
MIMIR_TIMEOUT=${MIMIR_TIMEOUT:-40}
if ! curl -fsS -m "$MIMIR_TIMEOUT" "http://127.0.0.1:18080/ready" > /dev/null 2>&1; then
  pgrep -f "mimir_tunnel.sh" > /dev/null || setsid bash "$PWD/stage7_28/mimir_tunnel.sh" < /dev/null > /dev/null 2>&1 &
fi
if ! curl -fsS -m 10 "http://127.0.0.1:3100/ready" > /dev/null 2>&1; then
  pgrep -f "loki_tunnel.sh"  > /dev/null || setsid bash "$PWD/stage7_28/loki_tunnel.sh"  < /dev/null > /dev/null 2>&1 &
fi
for i in $(seq 1 30); do
  kubectl get nodes >/dev/null 2>&1 && break
  sleep 2
done
kubectl get nodes >/dev/null 2>&1 || die "kube apiserver forward (127.0.0.1:16443) not reachable"
MIMIR_UP=0
for i in $(seq 1 6); do
  if curl -fsS -m "$MIMIR_TIMEOUT" "http://127.0.0.1:18080/prometheus/api/v1/query" \
       --data 'query=up' > /dev/null 2>&1; then MIMIR_UP=1; break; fi
  log "mimir not answering yet (attempt $i/6), retrying"
  sleep 10
done
[ "$MIMIR_UP" = "1" ] || die "mimir tunnel not reachable after 6 attempts"
curl -fsS -m 15 "http://127.0.0.1:3100/ready" > /dev/null || die "loki tunnel not reachable"
log "tunnels verified up (kube + mimir + loki)"

if [ "${SKIP_COLD_GATE:-0}" = "1" ]; then
  # The operator has already verified the cold state (every pod Ready, no load).
  # Re-asserting "all replicas == 1" is wrong for a workload whose HPAs declare
  # minReplicas > 1, because the HPAs legitimately scale the deployments back up
  # before the gate can run.
  NOT_READY=$(kubectl -n train-ticket get pods \
      -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.status.phase}{"|"}{.status.containerStatuses[*].ready}{"|"}{.metadata.deletionTimestamp}{"\n"}{end}' 2>/dev/null \
      | awk -F'|' '$1 !~ /^ts-/ {next} {bad=($2!="Running") || ($4!=""); if($3=="" || $3 ~ /false/) bad=1; if(bad) n++} END{print n+0}')
  echo "ASSERT cold_ok=$([ "${NOT_READY:-0}" = "0" ] && echo True || echo False) SKIP_COLD_GATE not_ready=${NOT_READY:-0}" \
      > "$RUN_DIR/cold_gate.log"
  [ "${NOT_READY:-0}" = "0" ] || die "cold gate failed (see cold_gate.log)"
  log "cold gate skipped (SKIP_COLD_GATE=1): operator-verified cold state"
elif [ "$AURORA_DEV" = "1" ]; then
  # AURORA is already deployed and owns the replicas; the HPAs are supposed to
  # be absent. The only thing worth asserting is that the controller itself is
  # up and every pod is Ready.
  AURORA_READY=$(kubectl -n train-ticket get pods -l app=aurora \
      -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || true)
  # Readiness is read from ``containerStatuses``, not the READY column: Istio
  # 1.30 runs istio-proxy as a native sidecar, so a healthy pod reads "1/2".
  NOT_READY=$(kubectl -n train-ticket get pods \
      -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.status.phase}{"|"}{.status.containerStatuses[*].ready}{"|"}{.metadata.deletionTimestamp}{"\\n"}{end}' 2>/dev/null \
      | awk -F'|' '$1 !~ /^ts-/ {next} {bad=($2!="Running") || ($4!=""); if($3=="" || $3 ~ /false/) bad=1; if(bad) n++} END{print n+0}')
  if [ "$AURORA_READY" != "true" ] || [ "${NOT_READY:-0}" != "0" ]; then
    echo "ASSERT cold_ok=False AURORA_DEV aurora_ready=$AURORA_READY not_ready=$NOT_READY" \
        > "$RUN_DIR/cold_gate.log"
    die "cold gate failed (see cold_gate.log)"
  fi
  echo "ASSERT cold_ok=True AURORA_DEV aurora_ready=true not_ready=0 (AURORA owns replicas)" \
      > "$RUN_DIR/cold_gate.log"
  log "cold gate passed (AURORA_DEV): aurora Ready, all pods Ready, HPAs owned by AURORA"
else
  if ! "$PY" pbscaler/scripts/launchers/reset_cluster_cold.py --assert-only \
       > "$RUN_DIR/cold_gate.log" 2>&1; then
    die "cold gate failed (see cold_gate.log)"
  fi
  log "cold gate passed: $(grep -m1 'ASSERT cold_ok' "$RUN_DIR/cold_gate.log")"
fi

# ---------- phase 0c: optional pre-k6 hook ----------
# Used to bring up an active autoscaler (PBScaler) AFTER the cold gate has
# asserted a clean cluster, because PBScaler suspends the HPAs on start and the
# cold gate requires them to be present. Nothing else runs in this window.
if [ -n "${PRE_K6_HOOK:-}" ]; then
  log "phase 0c: running PRE_K6_HOOK"
  if bash -c "$PRE_K6_HOOK" >> "$RUN_DIR/pre_k6_hook.log" 2>&1; then
    log "phase 0c: PRE_K6_HOOK OK"
  else
    die "PRE_K6_HOOK failed (see pre_k6_hook.log)"
  fi
fi

# ---------- resolve the gateway address ----------
# Used to be hardcoded (http://10.10.1.2:30080). That was master1's IP on the
# OLD cluster; on the new one 10.10.1.2 is the monitoring node and the gateway
# NodePort is whatever the Service has. Derive both from the live Service so a
# new cluster needs no edit here.
if [ -z "$TARGET_URL" ]; then
  _gw=$(kubectl -n istio-system get svc istio-ingressgateway -o json 2>/dev/null)
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

# ---------- phase 0b: mongo-path smoke probe ----------
# A cold reset recreates every stateless pod, so every service builds a fresh
# connection pool. If a pooled connection is already dead (2026-11-20: the first
# request after an idle period returned 500 "Prematurely reached end of stream"),
# the run would start with a login storm that has nothing to do with the load.
# Probe a mongo-backed read path before any k6 traffic is generated.
MONGO_PROBE_OK=0
for _i in $(seq 1 12); do
  # login with bogus credentials exercises ts-auth-service -> ts-auth-mongo and
  # returns HTTP 200 with status=0 in the body, so it is safe and repeatable.
  _code=$(curl -s -m 15 -o /dev/null -w '%{http_code}' -X POST \
    "$TARGET_URL/api/v1/users/login" -H 'Content-Type: application/json' \
    -d '{"username":"coldgate_probe_nope","password":"wrong"}' 2>/dev/null || echo 000)
  if [ "$_code" = "200" ]; then MONGO_PROBE_OK=1; break; fi
  log "mongo probe attempt $_i: HTTP $_code"
  sleep 10
done
if [ "$MONGO_PROBE_OK" = "1" ]; then
  log "mongo-path probe OK (ts-auth-service -> ts-auth-mongo)"
else
  log "FATAL: mongo-path probe failed after 12 attempts (see above)"
  die "mongo-path probe failed - refusing to start a run on a stale connection pool"
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
# collectors removed (2026-11-20): all capture is post-hoc from Mimir/Loki

# loadgen must be idle
K6_PRE=$(loadgen_k6_pids | head -1)
[ -z "${K6_PRE:-}" ] || die "k6 already running on loadgen (pid $K6_PRE)"
log "loadgen has no running k6"

if [ "$DRY" = "--dry" ]; then
  log "DRY mode: stopping here (no tools, no k6)"
  exit 0
fi

# ---------- phase 2: window anchors ----------
# No live collectors. Everything is queried post-hoc from Mimir/Loki (60-day
# retention): the dataset comes from rebuild_train_data.py in phase 5b, the
# report from build_run_report.py in phase 9.
LAUNCH_EPOCH=$(date +%s)
END_EPOCH=$((LAUNCH_EPOCH + TOOLS_END_OFFSET))
echo "$LAUNCH_EPOCH" > "$RUN_DIR/launch_epoch"
echo "$END_EPOCH" > "$RUN_DIR/end_epoch"
log "phase 2: window anchors set (launch=$LAUNCH_EPOCH end=$END_EPOCH); no live collectors"

# ---------- phase 3: launch k6 on the load generator ----------
# Local mode runs k6 as a direct child of this script so its real exit status
# is observable; ssh mode keeps the detached remote launch (exit status is not
# recoverable across ssh, so the console marker is used instead).
launch_k6_local() {
  (
    ulimit -n 65536 2>/dev/null || ulimit -n 8192
    exec env RUNID="$RUNID" \
      MAX_REQUESTS="$MAX_REQUESTS" USER_POOL="$USER_POOL" TRACE_COMPRESS=1 OPEN_LOOP=1 \
      K6_OUT_DIR="$K6_CONSOLE_DIR" \
      PRE_ALLOCATED_VUS="$PRE_ALLOC" MAX_VUS="$MAX_VUS" \
      PREFLIGHT=0 ALLOW_SEED_REUSE="${ALLOW_SEED_REUSE:-0}" \
      LOGIN_RETRIES=2 LOGIN_COOLDOWN_MS=30000 \
      LOGIN_START_JITTER_S="${LOGIN_START_JITTER_S:-300}" \
      LOGIN_BACKOFF_JITTER="${LOGIN_BACKOFF_JITTER:-0.5}" \
      START_MIN="$START_MIN" TEST_MIN="$TEST_MIN" \
      TARGET_URL="$TARGET_URL" \
      MCR_CSV="$MCR_SLICE_LOADGEN" \
      SEED_FILE="$SEED_FILE_LOADGEN" \
      bash "$LAUNCH_K6_SH"
  ) > "$CONSOLE_SRC" 2>&1 < /dev/null &
  K6_PID=$!
  echo "$K6_PID" > "$RUN_DIR/k6.pid"
}

launch_k6_ssh() {
  on_loadgen "ulimit -n 65536; nohup setsid env RUNID='$RUNID' K6_OUT_DIR='$K6_CONSOLE_DIR' \
    MAX_REQUESTS=$MAX_REQUESTS USER_POOL=$USER_POOL TRACE_COMPRESS=1 OPEN_LOOP=1 \
    PRE_ALLOCATED_VUS=$PRE_ALLOC MAX_VUS=$MAX_VUS \
    PREFLIGHT=0 ALLOW_SEED_REUSE=${ALLOW_SEED_REUSE:-0} \
    LOGIN_RETRIES=2 LOGIN_COOLDOWN_MS=30000 \
    LOGIN_START_JITTER_S=${LOGIN_START_JITTER_S:-300} \
    LOGIN_BACKOFF_JITTER=${LOGIN_BACKOFF_JITTER:-0.5} \
    START_MIN=$START_MIN TEST_MIN=$TEST_MIN \
    TARGET_URL='$TARGET_URL' \
    MCR_CSV=$MCR_SLICE_LOADGEN SEED_FILE='$SEED_FILE_LOADGEN' bash $LAUNCH_K6_SH \
    > '$CONSOLE_SRC' 2>&1 < /dev/null &" 2>&1 | sed 's/^/k6launch: /' >> "$LOG"
}

# k6 writes its console/summary/samples into $K6_CONSOLE_DIR. The loadgen role
# creates /opt/loadgen but not necessarily the k6 subdirectory, and when it is
# missing the redirect fails, the subshell dies, and the orchestrator reports
# "k6 console log size=0" (2026-11-18, new loadgen).
mkdir -p "$K6_CONSOLE_DIR" 2>/dev/null || true
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
  # The collection runs minutes after k6 ends, while Alloy is still flushing
  # and Mimir is compacting. A single attempt can come back with the tail of the
  # window missing (measured 2026-11-22: 64% coverage, last 21 min NaN, while
  # Mimir had the data and an immediate re-run gave 100%). Retry on low coverage
# instead of accepting a partial dataset as if it were complete.
  COVER_MIN=${AURORA_COVER_MIN:-95}
  COVER_TRIES=${COLLECT_RETRIES:-3}
  attempt=1
  while [ "$attempt" -le "$COVER_TRIES" ]; do
    if "$PY" "$TOOLS/rebuild_train_data.py" \
         --start "$CSTART" --duration "$COLLECT_DUR" --interval 5 \
         --out "$DS" --workload "$MCR_SLICE_LOCAL" \
         > "$RUN_DIR/rebuild_collection.log" 2>&1; then
      COV=$("$PY" -c "import json;print(json.load(open('$DS/meta.json')).get('coverage_pct') or 0)" 2>/dev/null || echo 0)
      log "post-hoc collection attempt $attempt: coverage=${COV}%"
      if awk "BEGIN{exit !($COV >= $COVER_MIN)}"; then
        log "post-hoc collection OK -> $DS"
        break
      fi
      log "coverage ${COV}% < ${COVER_MIN}% -- retrying in 300 s (Mimir may still be ingesting)"
    else
      log "post-hoc collection attempt $attempt failed (see rebuild_collection.log)"
    fi
    attempt=$((attempt + 1))
    [ "$attempt" -le "$COVER_TRIES" ] && sleep 300
  done
  if [ "$attempt" -gt "$COVER_TRIES" ]; then
    COV=$("$PY" -c "import json;print(json.load(open('$DS/meta.json')).get('coverage_pct') or 0)" 2>/dev/null || echo 0)
    log "WARNING: coverage ${COV}% after ${COVER_TRIES} attempts"
    [ "$COV" -lt 50 ] 2>/dev/null && { log "FATAL: post-hoc collection failed"; status_reason "post-hoc dataset collection failed"; }
  fi
  tail -14 "$RUN_DIR/rebuild_collection.log" >> "$LOG" 2>/dev/null
fi

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
# the log files, so that is expected and benign; integrity is proven
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

# ---------- phase 11b: optional post-run hook ----------
# Tears down whatever PRE_K6_HOOK started, before the cold reset so the reset
# sees the namespace the next experiment expects.
if [ -n "${POST_RUN_HOOK:-}" ]; then
  log "phase 11b: running POST_RUN_HOOK"
  if bash -c "$POST_RUN_HOOK" >> "$RUN_DIR/post_run_hook.log" 2>&1; then
    log "phase 11b: POST_RUN_HOOK OK"
  else
    log "WARN: POST_RUN_HOOK failed (see post_run_hook.log)"
    status_reason "post-run hook failed"
  fi
fi

# ---------- phase 12: final cold reset ----------
if [ "$SKIP_RESET" = "1" ]; then
  log "phase 12: SKIPPED (SKIP_RESET=1)"
elif [ "$AURORA_DEV" = "1" ]; then
  # The controller is still running and still owns the replicas. Resetting
  # would scale the workloads behind its back and restore HPAs it suspended.
  log "phase 12: SKIPPED (AURORA_DEV=1 -- aurora owns replicas)"
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
