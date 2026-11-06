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

# Pre-flight health checks (run locally from runner with kubectl access, save logs locally)
preflight_checks() {
  echo "Running pre-flight health checks..."
  LOG_DIR=/tmp/loadgen_logs
  mkdir -p "$LOG_DIR"
  PREFLIGHT_LOG="$LOG_DIR/preflight_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$PREFLIGHT_LOG") 2>&1
  
  echo "=== Pre-flight checks started at $(date) ==="
  
  # Use kubectl port-forward or exec to check cluster services
  echo "Checking cluster services via kubectl exec..."
  
  # Check station service
  echo "Checking ts-station-service..."
  if ! kubectl -n monitoring exec pbscaler-collector -- python3 -c "
import requests, sys
try:
    r = requests.get('http://ts-station-service.train-ticket.svc.cluster.local:12345/api/v1/stationservice/stations', timeout=10)
    sys.exit(0 if r.status_code == 200 else 1)
except Exception as e:
    sys.exit(1)
  "; then
    echo "ERROR: ts-station-service health check failed"
    exit 1
  fi
  
  # Check route service
  echo "Checking ts-route-service..."
  if ! kubectl -n monitoring exec pbscaler-collector -- python3 -c "
import requests, sys
try:
    r = requests.get('http://ts-route-service.train-ticket.svc.cluster.local:11178/api/v1/routeservice/routes', timeout=10)
    sys.exit(0 if r.status_code == 200 else 1)
except Exception as e:
    sys.exit(1)
  "; then
    echo "ERROR: ts-route-service health check failed"
    exit 1
  fi
  
  # Check travel service
  echo "Checking ts-travel-service..."
  if ! kubectl -n monitoring exec pbscaler-collector -- python3 -c "
import requests, sys
try:
    r = requests.post('http://ts-travel-service.train-ticket.svc.cluster.local:12346/api/v1/travelservice/trips/left', 
                      json={'startPlace':'shanghai','endPlace':'taiyuan','departureTime':'2027-05-04'},
                      headers={'Content-Type': 'application/json'}, timeout=10)
    sys.exit(0 if r.status_code == 200 and r.json().get('status') == 1 else 1)
except Exception as e:
    sys.exit(1)
  "; then
    echo "ERROR: ts-travel-service health check failed"
    exit 1
  fi
  
  # Check ticketinfo service
  echo "Checking ts-ticketinfo-service..."
  if ! kubectl -n monitoring exec pbscaler-collector -- python3 -c "
import requests, sys
try:
    r = requests.post('http://ts-ticketinfo-service.train-ticket.svc.cluster.local:15681/api/v1/ticketinfoservice/ticketinfo',
                      json={'trip':{'tripId':{'type':'G','number':'1236'},'trainTypeId':'GaoTieOne','routeId':'0b23bd3e-876a-4af3-b920-c50a90c90b04','startingStationId':'shanghai','stationsId':'suzhou','terminalStationId':'taiyuan','startingTime':'2027-05-04T01:00:00.000Z','endTime':'2027-05-04T07:51:52.000Z'},'startingPlace':'shanghai','endPlace':'taiyuan','departureTime':'2027-05-04T00:00:00.000Z'},
                      headers={'Content-Type': 'application/json'}, timeout=10)
    sys.exit(0 if r.status_code == 200 and r.json().get('status') == 1 else 1)
except Exception as e:
    sys.exit(1)
  "; then
    echo "ERROR: ts-ticketinfo-service health check failed"
    exit 1
  fi
  
  # Check basic service
  echo "Checking ts-basic-service..."
  if ! kubectl -n monitoring exec pbscaler-collector -- python3 -c "
import requests, sys
try:
    r = requests.get('http://ts-basic-service.train-ticket.svc.cluster.local:15680/api/v1/basicservice/welcome', timeout=10)
    sys.exit(0 if r.status_code == 200 else 1)
except Exception as e:
    sys.exit(1)
  "; then
    echo "ERROR: ts-basic-service health check failed"
    exit 1
  fi
  
  echo "Verifying core data..."
  
  # Station service - shanghai must exist
  if ! kubectl -n monitoring exec pbscaler-collector -- python3 -c "
import requests, sys
try:
    r = requests.get('http://ts-station-service.train-ticket.svc.cluster.local:12345/api/v1/stationservice/stations/id/shanghai', timeout=10)
    data = r.json()
    sys.exit(0 if r.status_code == 200 and data.get('status') == 1 else 1)
except Exception as e:
    sys.exit(1)
  "; then
    echo "ERROR: Station service missing shanghai"
    exit 1
  fi
  
  # Route service - shanghai->taiyuan must exist
  if ! kubectl -n monitoring exec pbscaler-collector -- python3 -c "
import requests, sys
try:
    r = requests.get('http://ts-route-service.train-ticket.svc.cluster.local:11178/api/v1/routeservice/routes/0b23bd3e-876a-4af3-b920-c50a90c90b04', timeout=10)
    data = r.json()
    sys.exit(0 if r.status_code == 200 and data.get('status') == 1 else 1)
except Exception as e:
    sys.exit(1)
  "; then
    echo "ERROR: Route service missing shanghai->taiyuan route"
    exit 1
  fi
  
  # Travel service - must have valid trips
  if ! kubectl -n monitoring exec pbscaler-collector -- python3 -c "
import requests, sys
try:
    r = requests.post('http://ts-travel-service.train-ticket.svc.cluster.local:12346/api/v1/travelservice/trips/left',
         json={'startPlace':'shanghai','endPlace':'taiyuan','departureTime':'2027-05-04'},
         headers={'Content-Type': 'application/json'}, timeout=10)
    data = r.json()
    sys.exit(0 if r.status_code == 200 and data.get('status') == 1 and len(data.get('data', [])) > 0 else 1)
except Exception as e:
    sys.exit(1)
  "; then
    echo "ERROR: Travel service has no valid trips"
    exit 1
  fi
  
  # Ticketinfo service
  if ! kubectl -n monitoring exec pbscaler-collector -- python3 -c "
import requests, sys
try:
    r = requests.post('http://ts-ticketinfo-service.train-ticket.svc.cluster.local:15681/api/v1/ticketinfoservice/ticketinfo',
         json={'trip':{'tripId':{'type':'G','number':'1236'},'trainTypeId':'GaoTieOne','routeId':'0b23bd3e-876a-4af3-b920-c50a90c90b04','startingStationId':'shanghai','stationsId':'suzhou','terminalStationId':'taiyuan','startingTime':'2027-05-04T01:00:00.000Z','endTime':'2027-05-04T07:51:52.000Z'},'startingPlace':'shanghai','endPlace':'taiyuan','departureTime':'2027-05-04T00:00:00.000Z'},
         headers={'Content-Type': 'application/json'}, timeout=10)
    data = r.json()
    sys.exit(0 if r.status_code == 200 and data.get('status') == 1 else 1)
except Exception as e:
    sys.exit(1)
  "; then
    echo "ERROR: Ticketinfo service failed"
    exit 1
  fi
  
  # Basic service
  if ! kubectl -n monitoring exec pbscaler-collector -- python3 -c "
import requests, sys
try:
    r = requests.get('http://ts-basic-service.train-ticket.svc.cluster.local:15680/api/v1/basicservice/welcome', timeout=10)
    data = r.json()
    sys.exit(0 if r.status_code == 200 and data.get('status') == 1 else 1)
except Exception as e:
    sys.exit(1)
  "; then
    echo "ERROR: Basic service failed"
    exit 1
  fi
  
  echo "=== All pre-flight checks passed at $(date) ==="
} || { echo "Pre-flight checks failed, aborting"; exit 1; }

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
