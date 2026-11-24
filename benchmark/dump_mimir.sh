#!/bin/bash
# Dump Mimir range data over a time window to JSON files (one per query).
# Usage: dump_mimir.sh <start_epoch> <end_epoch> <out_prefix>
set -u
START=${1:?start_epoch}
END=${2:?end_epoch}
OUT=${3:?out_prefix}
MIMIR_URL="${PBS_MIMIR_URL:-http://127.0.0.1:18080/prometheus}"

mkdir -p "$(dirname "$OUT")"

QUERIES=(
  'up'
  'sum(rate(istio_requests_total{namespace="train-ticket",reporter="destination"}[1m])) by (destination_workload,response_code)'
  'sum(rate(istio_requests_total{namespace="train-ticket",reporter="source"}[1m])) by (source_workload,destination_workload,response_code)'
  'sum(container_memory_usage_bytes{namespace="train-ticket",container!="",container!="POD",container!="istio-proxy"}) by (pod)'
  'sum(rate(container_cpu_usage_seconds_total{namespace="train-ticket",container!="",container!="POD",container!="istio-proxy"}[1m])) by (pod)'
  'sum(rate(container_network_receive_bytes_total{namespace="train-ticket",container!="",container!="POD",container!="istio-proxy"}[1m])) by (pod)'
  'sum(rate(container_network_transmit_bytes_total{namespace="train-ticket",container!="",container!="POD",container!="istio-proxy"}[1m])) by (pod)'
  'sum(rate(container_cpu_cfs_throttled_periods_total{namespace="train-ticket",container!="",container!="POD",container!="istio-proxy"}[1m])) by (pod)'
  'kube_pod_info{namespace="train-ticket"}'
  'kube_horizontalpodautoscaler_status_current_replicas'
  'kube_horizontalpodautoscaler_status_desired_replicas'
  'kube_horizontalpodautoscaler_spec_min_replicas'
  'kube_horizontalpodautoscaler_spec_max_replicas'
)

ok=0; fail=0
for q in "${QUERIES[@]}"; do
  qsafe=$(echo "$q" | tr '/,={}\[\] ' '_')
  if curl -s -m 120 --get \
      --data-urlencode "query=$q" \
      --data-urlencode "start=$START" \
      --data-urlencode "end=$END" \
      --data-urlencode 'step=5' \
      "$MIMIR_URL/api/v1/query_range" > "${OUT}_${qsafe}.json" 2>/dev/null \
     && grep -q '"status":"success"' "${OUT}_${qsafe}.json"; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "FAILED: $q"
  fi
done
echo "mimir dump done: $ok ok, $fail failed -> ${OUT}_*.json"
