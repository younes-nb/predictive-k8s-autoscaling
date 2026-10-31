#!/bin/bash
# Full experiment pipeline for HPA Jul-13, HPA Jul-14, PBScaler Jul-14
# Run on head node after cluster is deployed with private network (10.10.1.0/24)
# Load generator is on lb/loadgen node (10.10.1.1) targeting master1 (10.10.1.2:30080)

set -euo pipefail
cd /users/younes02/predictive-k8s-autoscaling

export KUBECONFIG=/users/younes02/cluster-repo/admin.conf
export PATH=$HOME/.local/bin:$PATH

# Verify cluster is up
kubectl get nodes -o wide

# Copy NASA trace to loadgen
scp /proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv loadgen:/opt/loadgen/mcr.csv

# Parameters
MAX_REQUESTS=2600   # Calibrated for SLO violations
USER_POOL=150
WARMUP_MIN=60       # 1h warmup per paper
K6_MIN=120          # 2h test per requirement
TEST_MIN=$((WARMUP_MIN + K6_MIN))
COLLECT_MIN=120     # Collector runs for test duration only

# ===== Run A': Jul-13 HPA (START_MIN=497, busiest Jul-13 window) =====
echo "=== Run A': Jul-13 HPA (START_MIN=497) ==="
ssh loadgen "nohup bash -c '
  MAX_REQUESTS=2600 USER_POOL=150 START_MIN=497 TEST_MIN=180 \
  /opt/loadgen/launch_k6.sh > /tmp/k6_jul13.log 2>&1
' &" 2>&1

# Wait for k6 to start
sleep 30

# Start collector on head (with Mimir port-forward)
PBS_MIMIR_URL=http://127.0.0.1:18080 \
  pbscaler/.venv/bin/python pbscaler/scripts/collect_training_data.py \
    --duration $((COLLECT_MIN * 60)) --interval 5 \
    --workload /proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv \
    --out pbscaler/data/exp_hpa_jul13_$(date +%Y%m%d_%H%M%S) \
    > /tmp/collector_jul13.log 2>&1 &

echo "Waiting for Run A' to complete (~3.5h)..."
wait

# ===== Retrain RF on Jul-13 data =====
echo "=== Retraining RF on Jul-13 data ==="
cd pbscaler/simulation/train_ticket
pbscaler/.venv/bin/python RandomForestClassify.py --data ../../pbscaler/data/exp_hpa_jul13_*/train_data.csv --bench train_ticket --seed 42
# Copy new model to deploy
cp RandomForestClassify.model ../../pbscaler/deploy/RandomForestClassify.model
cp RandomForestClassify.svcs ../../pbscaler/deploy/RandomForestClassify.svcs

# ===== Rebuild PBScaler image with new RF =====
cd ../../pbscaler/deploy
./build_and_push.sh

# ===== Run B': Jul-14 HPA (START_MIN=2324, busiest Jul-14 window) =====
echo "=== Run B': Jul-14 HPA (START_MIN=2324) ==="
ssh loadgen "nohup bash -c '
  MAX_REQUESTS=2600 USER_POOL=150 START_MIN=2324 TEST_MIN=180 \
  /opt/loadgen/launch_k6.sh > /tmp/k6_jul14.log 2>&1
' &" 2>&1

sleep 30

PBS_MIMIR_URL=http://127.0.0.1:18080 \
  pbscaler/.venv/bin/python pbscaler/scripts/collect_training_data.py \
    --duration $((COLLECT_MIN * 60)) --interval 5 \
    --workload /proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv \
    --out pbscaler/data/exp_hpa_jul14_$(date +%Y%m%d_%H%M%S) \
    > /tmp/collector_jul14.log 2>&1 &

echo "Waiting for Run B' to complete (~3.5h)..."
wait

# ===== Run D: Jul-14 PBScaler (paper mode) =====
echo "=== Run D: Jul-14 PBScaler (paper mode) ==="
cd /users/younes02/predictive-k8s-autoscaling/pbscaler

# Suspend HPAs
PBS_K8S_CONFIG=/users/younes02/cluster-repo/admin.conf python scripts/hpa_control.py suspend

# Run PBScaler controller (7200s = 2h)
PBS_K8S_CONFIG=/users/younes02/cluster-repo/admin.conf \
PBS_MAX_POD=8 \
PBS_MIMIR_URL=http://127.0.0.1:18080 \
python main.py --duration 7200 > /tmp/pbscaler_jul14.log 2>&1 &

# Start k6 load on loadgen
ssh loadgen "nohup bash -c '
  MAX_REQUESTS=2600 USER_POOL=150 START_MIN=2324 TEST_MIN=180 \
  /opt/loadgen/launch_k6.sh > /tmp/k6_pbscaler.log 2>&1
' &" 2>&1

# Start collector for PBScaler run
PBS_MIMIR_URL=http://127.0.0.1:18080 \
  pbscaler/.venv/bin/python pbscaler/scripts/collect_training_data.py \
    --duration $((COLLECT_MIN * 60)) --interval 5 \
    --workload /proj/k8sautoscaledl-PG0/nasa/jul13-14/http_mcr_NASA_jul95.csv \
    --out pbscaler/data/exp_pbscaler_jul14_$(date +%Y%m%d_%H%M%S) \
    > /tmp/collector_pbscaler.log 2>&1 &

echo "Waiting for PBScaler run to complete (~3.5h)..."
wait

# Restore HPAs
PBS_K8S_CONFIG=/users/younes02/cluster-repo/admin.conf python scripts/hpa_control.py restore

echo "=== All runs complete! ==="
echo "Generating comparison report..."

# Generate report
cd /users/younes02/predictive-k8s-autoscaling
python generate_report.py \
  --hpa-jul13 pbscaler/data/exp_hpa_jul13_*/train_data.csv \
  --hpa-jul14 pbscaler/data/exp_hpa_jul14_*/train_data.csv \
  --pbscaler-jul14 pbscaler/data/exp_pbscaler_jul14_*/train_data.csv \
  --output report_$(date +%Y%m%d).md
