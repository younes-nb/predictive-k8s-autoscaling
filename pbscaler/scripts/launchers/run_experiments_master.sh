#!/bin/bash
# Autonomous Master Orchestrator for Run B′ (HPA Jul-14) and Run D (PBScaler Jul-14)
# Runs via nohup/setsid to stay alive even if the session drops.
#
# Protocol:
#   Stage 1: Monitor Run B′ to completion -> dump Mimir -> collect artifacts -> cold reset
#   Stage 2: Launch Run D (PBScaler Jul-14) -> monitor -> collect artifacts -> scale 0 -> cold reset
#   Stage 3: Generate 3-way comparison report (HPA-j13, HPA-j14, PB-j14)
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"

LOG=/tmp/opencode/master_orchestrator.log
exec >> "$LOG" 2>&1

echo "================================================================="
echo "MASTER ORCHESTRATOR STARTED AT $(date -u +%FT%TZ)"
echo "================================================================="

K="sudo kubectl --kubeconfig /etc/kubernetes/admin.conf"
PY="pbscaler/.venv/bin/python"

# ----------------------------------------------------------------------
# STAGE 1: Wait for Run B′ (HPA Jul-14) to finish
# ----------------------------------------------------------------------
echo "[Stage 1] Waiting for Run B′ collector to complete..."
t0=$(date +%s)
while true; do
  STATUS=$(ssh -o ConnectTimeout=8 master1 "$K -n monitoring exec pbscaler-collector -- tail -1 /tmp/col_jul14_hpa_official.log" 2>/dev/null | tail -1 || true)
  if echo "$STATUS" | grep -q "collection summary"; then
    echo "$(date -u +%FT%TZ) Run B′ collector finished!"
    break
  fi
  now=$(date +%s)
  elapsed=$((now - t0))
  if [ $elapsed -gt 8700 ]; then  # 145 min safety timeout
    echo "$(date -u +%FT%TZ) WARN: Run B′ timed out at $elapsed seconds; proceeding to collection"
    break
  fi
  sleep 30
done

# Wait for k6 on loadgen to finish completely
echo "[Stage 1] Waiting for k6 to exit on loadgen..."
while ssh -o ConnectTimeout=8 loadgen "pgrep -c k6" >/dev/null 2>&1; do
  sleep 15
done
echo "$(date -u +%FT%TZ) k6 process exited on loadgen"

# Collect Run B′ artifacts
echo "[Stage 1] Collecting Run B′ artifacts..."
bash pbscaler/scripts/launchers/collect_run_artifacts.sh jul14_hpa /tmp/jul14_hpa_trace1100_official_2h10m

# Deep cold reset after Run B′
echo "[Stage 1] Resetting cluster to cold state..."
$PY pbscaler/scripts/launchers/reset_cluster_cold.py --kubeconfig "${PBS_K8S_CONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf}" \
  --out pbscaler/data/jul14_hpa_trace1100_official_2h10m/reset_after.json

echo "$(date -u +%FT%TZ) Stage 1 complete! Cluster is cold and clean."

# ----------------------------------------------------------------------
# STAGE 2: Run D (PBScaler Jul-14)
# ----------------------------------------------------------------------
echo "================================================================="
echo "[Stage 2] Starting Run D (PBScaler Jul-14) at $(date -u +%FT%TZ)"
echo "================================================================="

RUN_D_DIR="pbscaler/data/jul14_pbscaler_trace1100_official_2h10m"
mkdir -p "$RUN_D_DIR/k6"

# Verify cold state
$PY pbscaler/scripts/launchers/reset_cluster_cold.py --kubeconfig "${PBS_K8S_CONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf}" \
  --assert-only --out "$RUN_D_DIR/reset_before.json"

cat > "$RUN_D_DIR/RUN_NOTES.md" <<EOF
# Run D: PBScaler Jul-14 (official 2h10m protocol)

- Image: docker.io/younesnb/pbscaler:v2.0.0-official2h (RF trained on official Jul-13 data)
- PBScaler Deployment: PBS_DURATION=7920 (132m), PBS_MAX_POD=8, PBS_SLO=200, in-cluster Mimir
- k6 (loadgen): MCR_CSV=/opt/loadgen/mcr_jul14_slice.csv MAX_REQUESTS=66000 USER_POOL=3000
  START_MIN=145 TEST_MIN=130 COMPRESS=1 OPEN_LOOP=1
- collector: in-cluster pbscaler-collector pod, 7800s, 5s interval
- Pre-state: verified cold by reset_cluster_cold.py

UTC Timeline:
- started_at: $(date -u +%FT%TZ)
EOF

# 1. Apply PBScaler controller deployment
echo "[Stage 2] Applying PBScaler deployment..."
ssh -o ConnectTimeout=8 master1 "$K apply -f $REPO_ROOT/pbscaler/deploy/rbac.yaml"
ssh -o ConnectTimeout=8 master1 "$K apply -f $REPO_ROOT/pbscaler/deploy/deployment_runD.yaml"

# Wait for PBScaler controller pod to be Running and check its startup log (HPA suspend)
echo "[Stage 2] Waiting for PBScaler controller pod..."
sleep 10
for i in $(seq 1 30); do
  POD_PHASE=$(ssh -o ConnectTimeout=8 master1 "$K -n train-ticket get pod -l app=pbscaler -o jsonpath='{.items[0].status.phase}' 2>/dev/null" || true)
  if [ "$POD_PHASE" = "Running" ]; then
    echo "  PBScaler pod is Running!"
    break
  fi
  sleep 3
done

ssh -o ConnectTimeout=8 master1 "$K -n train-ticket logs -l app=pbscaler --tail=20" || true

# 2. Start collector in-pod
echo "[Stage 2] Starting collector in pod..."
ssh -o ConnectTimeout=8 master1 "$K -n monitoring exec pbscaler-collector -- sh -c 'rm -rf /tmp/jul14_pbscaler_trace1100_official_2h10m; nohup python3 /tmp/collect_training_data.py --duration 7800 --interval 5 --workload /opt/loadgen/mcr_jul14_slice.csv --out /tmp/jul14_pbscaler_trace1100_official_2h10m > /tmp/col_jul14_pbscaler_official.log 2>&1 &'"

# 3. Start k6 load on loadgen
echo "[Stage 2] Launching k6 on loadgen..."
ssh -o ConnectTimeout=8 loadgen "nohup setsid env MCR_CSV=/opt/loadgen/mcr_jul14_slice.csv MAX_REQUESTS=66000 USER_POOL=3000 START_MIN=145 TEST_MIN=130 K6_EXTRA_ARGS='--summary-export /tmp/k6_jul14_pbscaler_summary.json' /opt/loadgen/launch_k6.sh > /tmp/k6_jul14_pbscaler_trace1100.log 2>&1 < /dev/null & echo launched-runD"

sleep 15
echo "[Stage 2] Verifying k6 process..."
ssh -o ConnectTimeout=8 loadgen "pgrep -c k6" && echo "  k6 running on loadgen!"

# 4. Monitor Run D to completion
echo "[Stage 2] Monitoring Run D..."
t0_d=$(date +%s)
while true; do
  STATUS=$(ssh -o ConnectTimeout=8 master1 "$K -n monitoring exec pbscaler-collector -- tail -1 /tmp/col_jul14_pbscaler_official.log" 2>/dev/null | tail -1 || true)
  if echo "$STATUS" | grep -q "collection summary"; then
    echo "$(date -u +%FT%TZ) Run D collector finished!"
    break
  fi
  now=$(date +%s)
  elapsed=$((now - t0_d))
  if [ $elapsed -gt 8700 ]; then  # 145 min safety timeout
    echo "$(date -u +%FT%TZ) WARN: Run D timed out at $elapsed seconds; proceeding"
    break
  fi
  sleep 30
done

# Wait for k6 process
while ssh -o ConnectTimeout=8 loadgen "pgrep -c k6" >/dev/null 2>&1; do
  sleep 10
done

# Scale PBScaler deployment to 0 so restartPolicy doesn't restart it
echo "[Stage 2] Scaling PBScaler deployment to 0..."
ssh -o ConnectTimeout=8 master1 "$K -n train-ticket scale deployment pbscaler --replicas=0"

# Fetch controller logs
ssh -o ConnectTimeout=8 master1 "$K -n train-ticket logs -l app=pbscaler --tail=1000" > "$RUN_D_DIR/pbscaler_controller.log" 2>/dev/null || true

# Collect Run D artifacts
echo "[Stage 2] Collecting Run D artifacts..."
bash pbscaler/scripts/launchers/collect_run_artifacts.sh jul14_pbscaler /tmp/jul14_pbscaler_trace1100_official_2h10m

# Deep cold reset after Run D
echo "[Stage 2] Resetting cluster to cold state..."
$PY pbscaler/scripts/launchers/reset_cluster_cold.py --kubeconfig "${PBS_K8S_CONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf}" \
  --out "$RUN_D_DIR/reset_after.json"

echo "$(date -u +%FT%TZ) Stage 2 complete!"

# ----------------------------------------------------------------------
# STAGE 3: Generate comparison report
# ----------------------------------------------------------------------
echo "================================================================="
echo "[Stage 3] Generating 3-way comparison report at $(date -u +%FT%TZ)"
echo "================================================================="

REPORT_FILE="report_$(date +%Y%m%d)_official2h.md"
python3 generate_report.py \
  --hpa-jul13 pbscaler/data/jul13_hpa_trace1100_official_2h10m/train_data.csv \
  --hpa-jul14 pbscaler/data/jul14_hpa_trace1100_official_2h10m/train_data.csv \
  --pbscaler-jul14 pbscaler/data/jul14_pbscaler_trace1100_official_2h10m/train_data.csv \
  --output "$REPORT_FILE" || echo "generate_report.py failed"

echo "================================================================="
echo "ALL EXPERIMENTS AND REPORTS COMPLETED AT $(date -u +%FT%TZ)"
echo "Report: $REPORT_FILE"
echo "================================================================="
