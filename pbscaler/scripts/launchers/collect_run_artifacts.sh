#!/bin/bash
# Post-run artifact collection for the official 2h10m runs.
# Usage: bash collect_run_artifacts.sh <run_name> <collector_pod_outdir>
# Example: bash collect_run_artifacts.sh jul14_hpa /tmp/jul14_hpa_trace1100_official_2h10m
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"

RUN_NAME="${1:?usage: $0 <run_name> <collector_outdir>}"
COL_DIR="${2:?usage: $0 <run_name> <collector_outdir>}"
LOCAL_DIR="pbscaler/data/${RUN_NAME}_trace1100_official_2h10m"
K="sudo kubectl --kubeconfig /etc/kubernetes/admin.conf"

echo "=== Collecting artifacts for $RUN_NAME ==="
mkdir -p "$LOCAL_DIR/k6"

# 1. Copy collector output from pbscaler-collector pod
echo "[1/5] Copying collector data from pod..."
ssh -o ConnectTimeout=8 master1 "$K -n monitoring cp pbscaler-collector:$COL_DIR/train_data.csv /tmp/td_copy.csv && \
  $K -n monitoring cp pbscaler-collector:$COL_DIR/metrics.csv /tmp/met_copy.csv && \
  $K -n monitoring cp pbscaler-collector:$COL_DIR/meta.json /tmp/meta_copy.json" 2>/dev/null
scp -q master1:/tmp/td_copy.csv "$LOCAL_DIR/train_data.csv"
scp -q master1:/tmp/met_copy.csv "$LOCAL_DIR/metrics.csv"
scp -q master1:/tmp/meta_copy.json "$LOCAL_DIR/meta.json"
ssh master1 "rm -f /tmp/td_copy.csv /tmp/met_copy.csv /tmp/meta_copy.json" 2>/dev/null

# 2. k6 summary JSON from loadgen
echo "[2/5] Fetching k6 summary..."
scp -q loadgen:/tmp/k6_${RUN_NAME}_summary.json "$LOCAL_DIR/k6/summary.json" 2>/dev/null || echo "  (no summary file)"

# 3. k6 log tail (last 200 lines with final stats)
echo "[3/5] Fetching k6 log tail..."
ssh -o ConnectTimeout=8 loadgen "tail -200 /tmp/k6_${RUN_NAME}_trace1100.log" > "$LOCAL_DIR/k6/log_tail.txt" 2>/dev/null

# 4. Mimir dump (run promptly — bucket TTL ~15-20h)
echo "[4/5] Running Mimir dump..."
# Compute start/end from meta.json
START=$(python3 -c "import json; m=json.load(open('$LOCAL_DIR/meta.json')); print(m['started_at'])")
END=$(python3 -c "import json; m=json.load(open('$LOCAL_DIR/meta.json')); print(m.get('ended_at',''))")
if [ -n "$END" ]; then
  ssh -o ConnectTimeout=8 master1 "$K -n monitoring exec pbscaler-collector -- python3 /tmp/mimir_dump.py \
    --start '$START' --end '$END' --out /tmp/mimir_dump_${RUN_NAME} --namespace train-ticket" 2>/dev/null
  # Copy dump back
  ssh master1 "cd /tmp && tar czf mimir_dump_${RUN_NAME}.tar.gz mimir_dump_${RUN_NAME}/" 2>/dev/null
  scp -q master1:/tmp/mimir_dump_${RUN_NAME}.tar.gz "$LOCAL_DIR/mimir_dump.tar.gz"
  ssh master1 "rm -rf /tmp/mimir_dump_${RUN_NAME} /tmp/mimir_dump_${RUN_NAME}.tar.gz" 2>/dev/null
  echo "  Mimir dump saved to $LOCAL_DIR/mimir_dump.tar.gz"
else
  echo "  SKIP: no ended_at in meta.json"
fi

# 5. Cluster state snapshot
echo "[5/5] Cluster state snapshot..."
ssh -o ConnectTimeout=8 master1 "$K -n train-ticket get deploy -o wide" > "$LOCAL_DIR/deployments_final.txt" 2>/dev/null
ssh -o ConnectTimeout=8 master1 "$K -n train-ticket get hpa" > "$LOCAL_DIR/hpas_final.txt" 2>/dev/null

echo "=== Done: $LOCAL_DIR ==="
ls -la "$LOCAL_DIR/"
