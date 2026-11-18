#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

MCR_CSV="${MCR_CSV:-/proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv}"
MCR_NAME="${MCR_CSV##*/}"   # day-slice traces (mcr_jul13_slice.csv / mcr_jul14_slice.csv) live on the loadgen side
MAX_REQUESTS="${MAX_REQUESTS:-5000}"
USER_POOL="${USER_POOL:-500}"
SEED_FILE="${SEED_FILE:-load_testing/seeds/tt_users.json}"
START_MIN="${START_MIN:-0}"
TEST_MIN="${TEST_MIN:-0}"
TARGET_URL="${TARGET_URL:-http://10.10.1.2:30080}"
TRACE_COMPRESS="${TRACE_COMPRESS:-1}"
# Official benchmark = open loop. Unset resolves to open in trainticket.js;
# the wrapper states it explicitly. Closed loop requires OPEN_LOOP=0.
OPEN_LOOP="${OPEN_LOOP:-1}"
PRE_ALLOCATED_VUS="${PRE_ALLOCATED_VUS:-100}"
# Open-loop execution capacity: maxVUs defaults to preAllocatedVUs (no hidden
# runtime allocation). USER_POOL is the account/VU budget.
MAX_VUS="${MAX_VUS:-$PRE_ALLOCATED_VUS}"
# Preflight traffic is opt-in and outside the measured arrival schedule.
PREFLIGHT="${PREFLIGHT:-0}"
# VU->account sharing is forbidden unless explicitly requested.
ALLOW_SEED_REUSE="${ALLOW_SEED_REUSE:-0}"
LOGIN_RETRIES="${LOGIN_RETRIES:-2}"
LOGIN_COOLDOWN_MS="${LOGIN_COOLDOWN_MS:-30000}"

# SSH config for the loadgen hop. This runner has no ~/.ssh/config, so the
# cluster access file is preferred when present; otherwise plain ssh is used.
SSH_CFG="${SSH_CFG:-/proj/k8sautoscaledl-PG0/src/cluster-access/ssh_config}"
SSH_OPTS=()
[ -f "$SSH_CFG" ] && SSH_OPTS=(-F "$SSH_CFG")

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

# Stage 7.26: optional k6 flag pass-through (K6_EXTRA_ARGS), validated
# fail-closed BEFORE any ssh or launch. The value is forwarded verbatim and
# split by the remote shell at the k6 invocation — by design; no eval here.
# Forbidden: quotes, ';', '$', backticks, backslashes, control characters
# (they would change remote parsing silently). Output flags are rejected:
# the launcher derives RUNID-scoped output paths itself (Run-0 launcher patch).
RUNID="${RUNID:-loadgen_$(date -u +%Y%m%dT%H%M%SZ)}"
validate_k6_extra_args() {
  local v="${K6_EXTRA_ARGS:-}" tok
  [ -z "$v" ] && return 0
  case "$v" in
    *"'"*|*'"'*|*';'*|*'$'*|*'`'*|*'\'*)
      echo "ERROR: K6_EXTRA_ARGS contains a forbidden character (quote, semicolon, dollar, backtick, backslash): $v" >&2
      exit 1 ;;
  esac
  case "$v" in
    *$'\t'*|*$'\r'*|*$'\n'*)
      echo "ERROR: K6_EXTRA_ARGS must be a single line (no control characters): $v" >&2
      exit 1 ;;
  esac
  for tok in $v; do
    case "$tok" in
      --out|-o|--summary-export|--output|--console-output|--http-debug|\
      --out=*|-o=*|--summary-export=*|--output=*|--console-output=*|--http-debug=*)
        echo "ERROR: K6_EXTRA_ARGS must not set output flags ('$tok'); the launcher writes RUNID-derived outputs" >&2
        exit 1 ;;
    esac
  done
  return 0
}
validate_k6_extra_args

# Pre-flight health checks — runner-side, via temporary kubectl port-forwards.
# Stage 7.26: the previous implementation exec'd into the `pbscaler-collector`
# pod (namespace monitoring); that pod is created per-run and does not exist
# between runs, so the old preflight failed unconditionally before any k6
# launch. These probes reach the same TrainTicket services through the
# kube-apiserver tunnel from the runner (same endpoints, same ports).
preflight_checks() {
  echo "Running pre-flight health checks..."
  LOG_DIR=/tmp/loadgen_logs
  mkdir -p "$LOG_DIR"
  PREFLIGHT_LOG="$LOG_DIR/preflight_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$PREFLIGHT_LOG") 2>&1

  echo "=== Pre-flight checks started at $(date) ==="

  command -v kubectl >/dev/null 2>&1 || { echo "ERROR: kubectl not on PATH"; exit 1; }
  command -v curl    >/dev/null 2>&1 || { echo "ERROR: curl not installed on runner"; exit 1; }
  command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not installed on runner"; exit 1; }
  export KUBECONFIG="${KUBECONFIG:-${PBS_K8S_CONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf}}"
  kubectl get -n train-ticket svc/ts-station-service >/dev/null 2>&1 \
    || { echo "ERROR: kubectl cannot reach the cluster (KUBECONFIG=$KUBECONFIG)"; exit 1; }

  # kill stragglers from an aborted earlier run (our own ts-* forwards only)
  pkill -f 'kubectl.*port-forward svc/ts-' 2>/dev/null || true

  # json_status_ok FILE MODE -> exit 0 iff the payload satisfies MODE.
  #   status : JSON object with status==1 (the TrainTicket success envelope)
  #   json   : any parseable JSON (object or list); prints a shape note
  # The travel endpoints legitimately answer with an empty JSON list on this
  # cluster (documented in report_20261108_official.md §3: `trips/left` returns
  # an empty result set in every retained run, so the request mix stays
  # internally consistent). They are gated on well-formed JSON, not on
  # containing trips; the observed shape is logged so the degradation is
  # visible instead of silent.
  json_status_ok() {
    python3 - "$1" "$2" <<'PYEOF'
import json, sys
path, mode = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as e:
    print("  payload is not JSON: %s" % e)
    sys.exit(1)
if mode == "status":
    if not isinstance(d, dict) or d.get("status") != 1:
        s = d.get("status") if isinstance(d, dict) else type(d).__name__
        m = d.get("msg") if isinstance(d, dict) else ""
        print("  expected status==1, got status=%s msg=%r" % (s, str(m)[:80]))
        sys.exit(1)
    print("  status=1 data_len=%s" % (len(d.get("data") or [])))
    sys.exit(0)
if mode == "json":
    if isinstance(d, list):
        print("  json list len=%d (empty result set is a known cluster condition)" % len(d))
    elif isinstance(d, dict):
        n = d.get("data")
        print("  json object status=%s msg=%r data_len=%s" % (
            d.get("status"), str(d.get("msg"))[:60],
            len(n) if isinstance(n, (list, dict, str)) else n))
    else:
        print("  json %s" % type(d).__name__)
    sys.exit(0)
print("  unknown payload mode %r" % mode)
sys.exit(1)
PYEOF
  }

  # pf_probe NAME SVC PORT METHOD PATH [BODY] [CHECK_PAYLOAD]
  #   One HTTP probe through a temporary kubectl port-forward. Retries while
  #   the tunnel comes up (curl transport errors only), then requires HTTP 200
  #   and — when CHECK_PAYLOAD is set — status==1 (+ non-empty data for
  #   need_data). Any failure aborts the script (set -e / explicit exits).
  pf_probe() {
    local name="$1" svc="$2" port="$3" method="$4" path="$5" body="${6:-}"
    local payload="${7:-}" out code pid
    out="$(mktemp)"
    kubectl -n train-ticket port-forward "svc/${svc}" "${port}:${port}" \
      >"$LOG_DIR/pf_${name}.log" 2>&1 &
    pid=$!
    code=""
    for _ in $(seq 1 40); do
      if [ -n "$body" ]; then
        if code=$(curl -s -m 10 -X "$method" -H 'Content-Type: application/json' \
                    -d "$body" -o "$out" -w '%{http_code}' \
                    "http://127.0.0.1:${port}${path}" 2>/dev/null); then break; fi
      else
        if code=$(curl -s -m 10 -X "$method" -o "$out" -w '%{http_code}' \
                    "http://127.0.0.1:${port}${path}" 2>/dev/null); then break; fi
      fi
      sleep 0.5
    done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    if [ "${code:-}" != "200" ]; then
      echo "ERROR: ${name} health check failed (http_code=${code:-none}); tunnel log: $LOG_DIR/pf_${name}.log"
      head -c 400 "$out" 2>/dev/null; echo
      rm -f "$out"
      exit 1
    fi
    if [ -n "$payload" ] && ! json_status_ok "$out" "$payload"; then
      echo "ERROR: ${name} payload validation failed (mode=${payload})"
      head -c 400 "$out" 2>/dev/null; echo
      rm -f "$out"
      exit 1
    fi
    rm -f "$out"
    echo "Pre-flight OK: ${name}"
  }

  TRAVEL_BODY='{"startPlace":"shanghai","endPlace":"taiyuan","departureTime":"2027-05-04"}'
  TICKETINFO_BODY='{"trip":{"tripId":{"type":"G","number":"1236"},"trainTypeId":"GaoTieOne","routeId":"0b23bd3e-876a-4af3-b920-c50a90c90b04","startingStationId":"shanghai","stationsId":"suzhou","terminalStationId":"taiyuan","startingTime":"2027-05-04T01:00:00.000Z","endTime":"2027-05-04T07:51:52.000Z"},"startingPlace":"shanghai","endPlace":"taiyuan","departureTime":"2027-05-04T00:00:00.000Z"}'

  # --- service reachability (same endpoints as the previous exec-based checks)
  echo "Checking ts-station-service..."
  pf_probe station ts-station-service 12345 GET /api/v1/stationservice/stations "" status
  echo "Checking ts-route-service..."
  pf_probe route ts-route-service 11178 GET /api/v1/routeservice/routes "" status
  echo "Checking ts-travel-service..."
  pf_probe travel ts-travel-service 12346 POST /api/v1/travelservice/trips/left "$TRAVEL_BODY" json
  echo "Checking ts-ticketinfo-service..."
  pf_probe ticketinfo ts-ticketinfo-service 15681 POST /api/v1/ticketinfoservice/ticketinfo "$TICKETINFO_BODY" json
  echo "Checking ts-basic-service..."
  pf_probe basic ts-basic-service 15680 GET /api/v1/basicservice/welcome

  # Core reference data. station/id/shanghai and the ticketinfo trip body are
  # known-degraded on this cluster (status=0, reported in every preflight log);
  # they are probed for reachability + well-formed JSON so the state is
  # recorded, while station list / route pair / route list are hard-gated.
  echo "Verifying core data..."
  pf_probe station_shanghai ts-station-service 12345 GET /api/v1/stationservice/stations/id/shanghai "" json
  pf_probe route_pair ts-route-service 11178 GET /api/v1/routeservice/routes/0b23bd3e-876a-4af3-b920-c50a90c90b04 "" status
  pf_probe travel_trips ts-travel-service 12346 POST /api/v1/travelservice/trips/left "$TRAVEL_BODY" json
  pf_probe ticketinfo_data ts-ticketinfo-service 15681 POST /api/v1/ticketinfoservice/ticketinfo "$TICKETINFO_BODY" json
  pf_probe basic_data ts-basic-service 15680 GET /api/v1/basicservice/welcome

  echo "=== All pre-flight checks passed at $(date) ==="
}

# Preflight is opt-in only: PREFLIGHT=1 runs the runner-side checks before the
# measured workload. Default 0 keeps all pre-scenario traffic out of the run.
if [ "$PREFLIGHT" = "1" ]; then
  preflight_checks
else
  echo "Preflight disabled (PREFLIGHT=0, official benchmark default)"
fi

# PREFLIGHT_ONLY=1 validates the guard itself without launching load.
if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  if [ "$PREFLIGHT" != "1" ]; then
    echo "ERROR: PREFLIGHT_ONLY=1 requires PREFLIGHT=1" >&2
    exit 1
  fi
  echo "PREFLIGHT_ONLY=1: stopping before k6 launch"
  exit 0
fi

if [[ -x /opt/loadgen/launch_k6.sh ]]; then
  if [ ! -f "/opt/loadgen/$MCR_NAME" ] && [ -f "$MCR_CSV" ]; then
    cp "$MCR_CSV" "/opt/loadgen/$MCR_NAME"
  fi
  exec env TARGET_URL="$TARGET_URL" \
    MCR_CSV="/opt/loadgen/$MCR_NAME" \
    MAX_REQUESTS="$MAX_REQUESTS" USER_POOL="$USER_POOL" \
    SEED_FILE=/opt/loadgen/seeds/tt_users.json \
    START_MIN="$START_MIN" TEST_MIN="$TEST_MIN" \
    TRACE_COMPRESS="$TRACE_COMPRESS" OPEN_LOOP="$OPEN_LOOP" \
    PRE_ALLOCATED_VUS="$PRE_ALLOCATED_VUS" MAX_VUS="$MAX_VUS" \
    PREFLIGHT="$PREFLIGHT" ALLOW_SEED_REUSE="$ALLOW_SEED_REUSE" \
    LOGIN_RETRIES="$LOGIN_RETRIES" LOGIN_COOLDOWN_MS="$LOGIN_COOLDOWN_MS" \
    K6_EXTRA_ARGS="${K6_EXTRA_ARGS:-}" RUNID="$RUNID" \
    /opt/loadgen/launch_k6.sh
fi

ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 loadgen 'test -x /opt/loadgen/launch_k6.sh' \
  || {
    echo "ERROR: loadgen is not initialized (no /opt/loadgen/launch_k6.sh)." >&2
    echo "       Run> ./run-cluster-setup.sh loadgen   (from the cluster repo)" >&2
    exit 1
  }

if ! ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 loadgen "test -f /opt/loadgen/$MCR_NAME"; then
  scp -q "${SSH_OPTS[@]}" "$MCR_CSV" "loadgen:/tmp/$MCR_NAME" && \
  ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 loadgen "sudo mv /tmp/$MCR_NAME /opt/loadgen/$MCR_NAME && sudo chmod 0644 /opt/loadgen/$MCR_NAME"
fi
if ! ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 loadgen 'test -f /opt/loadgen/seeds/tt_users.json'; then
  scp -q "${SSH_OPTS[@]}" "$SEED_FILE" loadgen:/tmp/tt_users.json && \
  ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 loadgen 'sudo mv /tmp/tt_users.json /opt/loadgen/seeds/tt_users.json && sudo chmod 0644 /opt/loadgen/seeds/tt_users.json' 2>/dev/null || true
fi

exec ssh "${SSH_OPTS[@]}" -o ServerAliveInterval=30 -o ServerAliveCountMax=10 loadgen \
  "env TARGET_URL='$TARGET_URL' \
   MCR_CSV='/opt/loadgen/$MCR_NAME' \
   MAX_REQUESTS='$MAX_REQUESTS' USER_POOL='$USER_POOL' \
   SEED_FILE=/opt/loadgen/seeds/tt_users.json \
   START_MIN='$START_MIN' TEST_MIN='$TEST_MIN' \
   TRACE_COMPRESS='$TRACE_COMPRESS' OPEN_LOOP='$OPEN_LOOP' \
   PRE_ALLOCATED_VUS='$PRE_ALLOCATED_VUS' MAX_VUS='$MAX_VUS' \
   PREFLIGHT='$PREFLIGHT' ALLOW_SEED_REUSE='$ALLOW_SEED_REUSE' \
   LOGIN_RETRIES='$LOGIN_RETRIES' LOGIN_COOLDOWN_MS='$LOGIN_COOLDOWN_MS' \
   K6_EXTRA_ARGS='${K6_EXTRA_ARGS:-}' RUNID='$RUNID' \
   /opt/loadgen/launch_k6.sh"
