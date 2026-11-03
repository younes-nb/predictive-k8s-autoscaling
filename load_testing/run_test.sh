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

# Pre-flight health checks (run on loadgen, save results there)
preflight_checks() {
  echo "Running pre-flight health checks on loadgen..."
  ssh -o ConnectTimeout=10 loadgen '
    set -euo pipefail
    LOG_DIR=/opt/loadgen/logs
    mkdir -p "$LOG_DIR"
    PREFLIGHT_LOG="$LOG_DIR/preflight_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$PREFLIGHT_LOG") 2>&1
    
    echo "=== Pre-flight checks started at $(date) ==="
    
    checks=(
      "ts-station-service:12345:/api/v1/stationservice/stations"
      "ts-route-service:11178:/api/v1/routeservice/routes"
      "ts-travel-service:12346:/api/v1/travelservice/trips/left"
      "ts-ticketinfo-service:15681:/api/v1/ticketinfoservice/ticketinfo"
      "ts-basic-service:15680:/api/v1/basicservice/welcome"
    )
    
    for check in "${checks[@]}"; do
      IFS=":" read -r svc port path <<< "$check"
      echo "Checking $svc..."
      if ! curl -sf -o /dev/null --max-time 10 "http://$svc.train-ticket.svc.cluster.local:$port$path"; then
        echo "ERROR: $svc health check failed"
        exit 1
      fi
    done
    
    echo "Verifying core data..."
    
    # Station service - shanghai must exist
    if ! curl -sf "http://ts-station-service.train-ticket.svc.cluster.local:12345/api/v1/stationservice/stations/id/shanghai" | jq -e ".status==1" >/dev/null; then
      echo "ERROR: Station service missing shanghai"
      exit 1
    fi
    
    # Route service - shanghai->taiyuan must exist
    if ! curl -sf "http://ts-route-service.train-ticket.svc.cluster.local:11178/api/v1/routeservice/routes/0b23bd3e-876a-4af3-b920-c50a90c90b04" | jq -e ".status==1" >/dev/null; then
      echo "ERROR: Route service missing shanghai->taiyuan route"
      exit 1
    fi
    
    # Travel service - must have valid trips
    if ! curl -sf "http://ts-travel-service.train-ticket.svc.cluster.local:12346/api/v1/travelservice/trips/left" \
         -X POST -H "Content-Type: application/json" \
         -d "{\"startPlace\":\"shanghai\",\"endPlace\":\"taiyuan\",\"departureTime\":\"2027-05-04\"}" \
         | jq -e ".status==1 and .data|length>0" >/dev/null; then
      echo "ERROR: Travel service has no valid trips"
      exit 1
    fi
    
    # Ticketinfo service
    if ! curl -sf "http://ts-ticketinfo-service.train-ticket.svc.cluster.local:15681/api/v1/ticketinfoservice/ticketinfo" \
         -X POST -H "Content-Type: application/json" \
         -d "{\"trip\":{\"tripId\":{\"type\":\"G\",\"number\":\"1236\"},\"trainTypeId\":\"GaoTieOne\",\"routeId\":\"0b23bd3e-876a-4af3-b920-c50a90c90b04\",\"startingStationId\":\"shanghai\",\"stationsId\":\"suzhou\",\"terminalStationId\":\"taiyuan\",\"startingTime\":\"2027-05-04T01:00:00.000Z\",\"endTime\":\"2027-05-04T07:51:52.000Z\"},\"startingPlace\":\"shanghai\",\"endPlace\":\"taiyuan\",\"departureTime\":\"2027-05-04T00:00:00.000Z\"}" \
         | jq -e ".status==1" >/dev/null; then
      echo "ERROR: Ticketinfo service failed"
      exit 1
    fi
    
    # Basic service
    if ! curl -sf "http://ts-basic-service.train-ticket.svc.cluster.local:15680/api/v1/basicservice/welcome" | jq -e ".status==1" >/dev/null; then
      echo "ERROR: Basic service failed"
      exit 1
    fi
    
    echo "=== All pre-flight checks passed at $(date) ==="
  ' || { echo "Pre-flight checks failed, aborting"; exit 1; }
}

preflight_checks

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
  scp -q "$MCR_CSV" loadgen:/tmp/mcr.csv && \
  ssh -o ConnectTimeout=10 loadgen 'sudo mv /tmp/mcr.csv /opt/loadgen/mcr.csv && sudo chmod 0644 /opt/loadgen/mcr.csv'
fi
if ! ssh -o ConnectTimeout=10 loadgen 'test -f /opt/loadgen/seeds/tt_users.json'; then
  scp -q "$SEED_FILE" loadgen:/tmp/tt_users.json && \
  ssh -o ConnectTimeout=10 loadgen 'sudo mv /tmp/tt_users.json /opt/loadgen/seeds/tt_users.json && sudo chmod 0644 /opt/loadgen/seeds/tt_users.json' 2>/dev/null || true
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
