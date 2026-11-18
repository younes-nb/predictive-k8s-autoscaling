#!/bin/bash
# Stage 12K endgame — run AFTER the k6 window closes (~15:46Z) and monitor end.
# Mirrors the 13 Jul endgame, plus streamer-based log extracts.
# Usage: bash endgame_12k.sh RUNDIR RUNID
set -euo pipefail
cd /users/younes02/predictive-k8s-autoscaling
RUNDIR=$1; RUNID=$2
export PATH=/users/younes02/.local/bin:$PATH
export KUBECONFIG=/users/younes02/cluster-repo/admin.conf
K=/users/younes02/.local/bin/kubectl
PB=pbscaler/.venv/bin/python

echo "== [1/6] verify processes exited =="
pgrep -af 'monitor_v2|log_streamer|collector_wrapper' | grep -v grep || echo "all instrumentation exited"
tail -2 "$RUNDIR/collector_wrapper.log"
tail -1 "$RUNDIR/monitor.log" 2>/dev/null || true

echo "== [2/6] pull loadgen artifacts =="
ssh -o ConnectTimeout=10 loadgen "ls -la /opt/loadgen/k6/ | grep $RUNID" | tail -5
scp -q "loadgen:/opt/loadgen/k6/summary_${RUNID}.json" "$RUNDIR/summary_${RUNID}.json"
scp -q "loadgen:/opt/loadgen/k6/samples_${RUNID}.json.gz" "$RUNDIR/samples_${RUNID}.json.gz"
ls -la "$RUNDIR"/summary_${RUNID}.json "$RUNDIR"/samples_${RUNID}.json.gz

echo "== [3/6] Mimir exports (warm-up + scored, 5 s) =="
$PB "$RUNDIR/exports/mimir_export.py" "$RUNDIR/exports"

echo "== [4/6] end-state snapshot =="
{
  echo "=== end state $(date -u +%FT%TZ) ==="
  echo "--- deployments (replicas) ---"
  $K -n train-ticket get deploy -o wide | awk '{print $1, $NF}' | head -50
  echo "--- hpa current/max ---"
  $K -n train-ticket get hpa --no-headers | awk '{print $1, "cur="$4, "tgt="$5, "min="$6, "max="$7}'
  echo "--- nodes ---"
  $K get nodes --no-headers | awk '{print $1, $2}'
} > "$RUNDIR/exports/end_state_snapshot.txt" 2>&1
wc -l "$RUNDIR/exports/end_state_snapshot.txt"

echo "== [5/6] log extracts from streamer files =="
LOGD="$RUNDIR/logs"
if [ -d "$LOGD" ]; then
  grep -hiE 'error|exception|failed' "$LOGD"/ts-auth-service-*.log 2>/dev/null | tail -2000 > "$RUNDIR/exports/logs_auth_errors.log" || true
  grep -hE '" 5[0-9][0-9] ' "$LOGD"/train-ticket-gateway*.log 2>/dev/null | tail -2000 > "$RUNDIR/exports/logs_gateway_5xx.log" || true
  grep -hiE '\[(alert|emerg|error)\]' "$LOGD"/ts-ui-dashboard-*.log 2>/dev/null | tail -2000 > "$RUNDIR/exports/logs_ui_nginx_errors.log" || true
  grep -hiE 'error|exception|Traceback' "$LOGD"/ts-voucher-service-*.log 2>/dev/null | tail -2000 > "$RUNDIR/exports/logs_voucher_errors.log" || true
  wc -l "$RUNDIR"/exports/logs_*.log || true
fi

echo "== [6/6] dataset validation =="
$PB - <<'EOF'
import csv, json
rd = "pbscaler/data/hpa12k_20261113T112649Z"
rows = list(csv.DictReader(open(f"{rd}/train_data.csv")))
n = len(rows)
import collections
lab = collections.Counter(r.get("slo_reward") for r in rows)
nodata = sum(1 for r in rows if r.get("front&p90") in ("", "nan", "NaN", None))
first, last = rows[0].get("timestamp"), rows[-1].get("timestamp")
print(f"train_data rows: {n} (expected 2880 = 14400s/5s)")
print(f"first ts: {first}  last ts: {last}")
print(f"slo labels: {dict(lab)}")
print(f"no-data front_p90 rows: {nodata}")
cols = list(rows[0].keys())
print(f"columns: {len(cols)} (e.g. {cols[:4]} ...)")
EOF
echo "ENDGAME COMPLETE"
