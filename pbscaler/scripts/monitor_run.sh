#!/bin/bash
# Periodic health monitor for a load-test run (collector + k6, both optional).
# Usage: monitor_run.sh RUN_PREFIX [INTERVAL_S] [SLO_MS]
#   RUN_PREFIX  e.g. fullrun3 -> expects /tmp/opencode/${RUN_PREFIX}_{collector,k6}.log
#               and pbscaler/data/${RUN_PREFIX}_*/train_data.csv
#   INTERVAL_S  sampling interval (default 90)
#   SLO_MS      violation threshold for front_p90 warnings (default 200)
#
# Exits when neither the collector nor k6 is running anymore.
# Designed to run detached:  setsid bash scripts/monitor_run.sh fullrun3 90 &

set -u
cd "$(dirname "$0")/../.."

PREFIX="${1:?usage: monitor_run.sh RUN_PREFIX [INTERVAL_S] [SLO_MS]}"
IV="${2:-90}"
SLO="${3:-200}"

CLOG="/tmp/opencode/${PREFIX}_collector.log"
KLOG="/tmp/opencode/${PREFIX}_k6.log"

last_restart_snapshot=""
warn() { echo "  *** $(date +%H:%M:%S) WARN: $*"; }

echo "== monitor start $(date '+%F %H:%M:%S') prefix=${PREFIX} interval=${IV}s slo=${SLO}ms"

while :; do
  COL_ALIVE=$(pgrep -cf "collect_training_data" 2>/dev/null || echo 0)
  K6_ALIVE=$(pgrep -cf "[k]6 run" 2>/dev/null || echo 0)
  CSV=$(ls -1 pbscaler/data/${PREFIX}_*/train_data.csv 2>/dev/null | head -1)
  ROWS=0; VIOL=0; P90MAX=0
  if [ -f "$CSV" ]; then
    read -r ROWS VIOL P90MAX < <(python3 - "$CSV" <<'EOF'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[0]
i_p90, i_label = hdr.index("front_p90"), hdr.index("slo_reward")
v, n, mx = 0, 0, 0.0
for r in rows[1:]:
    n += 1
    if r[i_label] == "0": v += 1
    try:
        f = float(r[i_p90])
        if f == f and f > mx: mx = f
    except (ValueError, IndexError): pass
print(n, v, f"{mx:.0f}")
EOF
)
  fi
  COL_LAST=$(tail -n1 "$CLOG" 2>/dev/null | head -c 120)
  K6_L=$(tail -n1 "$KLOG" 2>/dev/null | sed -E 's/\x1b\[[0-9;]*m//g' | head -c 100)
  NODES=$(kubectl top node --no-headers 2>/dev/null | awk '{cpu=cpu+$3} END {gsub("%","",cpu); printf "%.0f%%", cpu/NR}')
  RESTARTS=""
  snap=$(kubectl get pods -n train-ticket -o json 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin)
items=[]
for it in d["items"]:
    for c in it.get("status",{}).get("containerStatuses",[]) or []:
        if c.get("restartCount",0)>0 or (c.get("lastState",{}).get("terminated",{}) or {}).get("reason")=="OOMKilled":
            items.append(f"{it[\"metadata\"][\"name\"]}({c.get(\"restartCount\",0)})")
print(" ".join(sorted(items)[:6]))' 2>/dev/null || true)
  if [ -n "$snap" ] && [ "$snap" != "$last_restart_snapshot" ]; then
    RESTARTS="  RESTART-DIFF: ${snap}"
    last_restart_snapshot="$snap"
  fi
  HPA=$(kubectl get hpa -n train-ticket -o custom-columns=S:.status.currentReplicas 2>/dev/null | awk 'NR>1 && $1>1 {n++} END {print n+0}')
  LINE="$(date '+%H:%M:%S') collector=${COL_ALIVE} k6=${K6_ALIVE} rows=${ROWS} viol=${VIOL} p90max=${P90MAX}ms clusterCPU=${NODES} hpa>1=${HPA}${RESTARTS}"
  echo "$LINE"
  echo "      col: ${COL_LAST}"
  echo "      k6 : ${K6_L}"
  [ "${MAXROWS:-0}" -gt 0 ] 2>/dev/null && [ "$ROWS" -ge "$MAXROWS" ] && warn "row target ${MAXROWS} reached"
  [ "$COL_ALIVE" = "0" ] && [ "$K6_ALIVE" = "0" ] && { echo "== run complete $(date '+%F %H:%M:%S')"; exit 0; }
  sleep "$IV"
done
