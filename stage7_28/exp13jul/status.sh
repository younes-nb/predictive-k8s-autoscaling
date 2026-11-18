#!/bin/bash
# Compact experiment status: sleep then print. Usage: status.sh [sleep_s] [run_dir]
#
# Both the repo path and the run directory used to be hardcoded
# (the pre-move /users/younes02 path and a run from 2026-11-13), so
# this script always reported on a run that no longer existed. Now it derives the
# repo root from its own location and takes the run directory as an argument,
# defaulting to the newest run.
set -u
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
S=${1:-540}
sleep "$S"

if [ "${2:-}" = "" ]; then
  RD=$(ls -dt "$REPO_ROOT"/stage7_28/exp13jul/*/ 2>/dev/null | grep -v attic | head -1)
  RD=${RD%/}
else
  RD=$2
fi
[ -d "$RD" ] || { echo "run dir not found: $RD" >&2; exit 1; }
echo "=== $(date -u +%FT%TZ)  run=$RD ==="
python3 - "$RD" <<'EOF'
import json, sys
rd = sys.argv[1]
rows = []
for l in open(f"{rd}/live.csv"):
    l = l.strip()
    if l.startswith("{"):
        try: rows.append(json.loads(l))
        except Exception: pass
print(f'{"ts":>20} {"p90":>8} {"5xx/s":>7} {"authP90":>8} {"auth":>5} {"gw":>4} {"ui":>3} {"idx":>4} {"offrd":>6} {"k6iters":>9} {"vus":>5}')
for r in rows[-9:]:
    fp, ap, orp, idx = r.get('front_p90'), r.get('auth_p90'), r.get('offered_rps'), r.get('trace_idx')
    print(f"{r['ts']:>20} {(fp or 0):8.1f} {(r.get('five_xx_rate') or 0):7.1f} {((ap or 0)/1000):8.1f} {int(r.get('auth_rep') or 0)//2}/{int(r.get('auth_avail') or 0)//2} {int(r.get('gw_rep') or 0)//2:>4} {int(r.get('ui_rep') or 0)//2:>3} {idx if idx is not None else -1:>4} {(orp or 0):6.1f} {str(r.get('iters_complete') or '-'):>9} {str(r.get('vus_active') or '-'):>5}")
EOF
echo "--- incidents (last 5) ---"
tail -5 "$RD/incidents.jsonl" 2>/dev/null
echo "--- k6 console tail ---"
grep -E 'level=(error|warning)' "$RD/k6_console.log" 2>/dev/null | tail -2
tail -1 "$RD/k6_console.log" 2>/dev/null | cut -c1-160
echo "--- collector last row ---"
tail -1 "$RD/collector_wrapper.log" 2>/dev/null | cut -c1-140
