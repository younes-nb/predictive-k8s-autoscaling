#!/bin/bash
# Collector wrapper: waits until real k6 traffic is observed in Mimir, then
# waits until traffic_start + WARMUP_S, then runs the collector for the
# 240-minute scored window at 5 s. Detached; writes its own log.
set -u
RUN_DIR="$1"; WARMUP_S="$2"; DURATION_S="$3"; DATASET_DIR="$4"
MIMIR="${PBS_MIMIR_URL:-http://127.0.0.1:18080}/prometheus"
KUBECTL=${KUBECTL:-/usr/local/bin/kubectl}
export KUBECONFIG=${KUBECONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf}
export PBS_MIMIR_URL=http://127.0.0.1:18080
QURL="$MIMIR/api/v1/query"

echo "$(date -u +%FT%TZ) wrapper started, waiting for k6 traffic..."
for i in $(seq 1 240); do
  R=$(curl -s --max-time 8 "$QURL" --data 'query=sum(rate(istio_requests_total{reporter="destination",destination_workload="ts-ui-dashboard"}[30s]))' | python3 -c 'import json,sys; d=json.load(sys.stdin); r=d["data"]["result"]; print(r[0]["value"][1] if r else 0)' 2>/dev/null || echo 0)
  if python3 -c "exit(0 if float('''$R''') > 0.2 else 1)" 2>/dev/null; then
    TRAFFIC_START=$(date +%s); TRAFFIC_TS=$(date -u +%FT%TZ)
    echo "$(date -u +%FT%TZ) traffic detected (ui rps30=$R) at $TRAFFIC_TS epoch=$TRAFFIC_START"
    break
  fi
  sleep 5
done
if [ -z "${TRAFFIC_START:-}" ]; then echo "ERROR: no traffic detected in 20 min; aborting" ; exit 1; fi
echo "$TRAFFIC_START" > "$RUN_DIR/traffic_start_epoch"
echo "$TRAFFIC_TS" > "$RUN_DIR/traffic_start_utc"

TARGET=$((TRAFFIC_START + WARMUP_S))
WAIT=$((TARGET - $(date +%s)))
echo "$(date -u +%FT%TZ) sleeping ${WAIT}s until scored-window start ($(date -u -d @$TARGET +%FT%TZ))"
if [ "$WAIT" -gt 0 ]; then sleep "$WAIT"; fi
MEASURE_START_UTC=$(date -u +%FT%TZ)
echo "$MEASURE_START_UTC" > "$RUN_DIR/measure_start_utc"
echo "$(date -u +%FT%TZ) MEASUREMENT START $MEASURE_START_UTC — attaching collector (duration=${DURATION_S}s interval=5s)"
exec pbscaler/.venv/bin/python pbscaler/scripts/collect_training_data.py \
  --duration "$DURATION_S" --interval 5 --out "$DATASET_DIR" \
  --workload /opt/loadgen/mcr_jul13_slice.csv
