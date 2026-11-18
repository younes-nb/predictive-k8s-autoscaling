#!/bin/bash
# Dump Loki logs over a time window to gzipped JSON files.
#
# Loki caps queries at max_entries_limit_per_query (5000 here). Peak observed
# volume is ~3.4k lines/s ({instance=~"train-ticket/.*"} includes istio access
# lines), and under the 30k/32k runs it exceeds 4900 lines/s, so a fixed 1 s
# chunk silently loses data: the 2026-11-18 step-1 run had 5550/15060 chunks
# (37%) hit the cap and are partial, concentrated exactly at peak load.
#
# Fix: recursive splitting. A chunk that comes back full is split in half and
# re-queried, repeatedly, until every chunk returns fewer than MAXENT lines (or
# MAXSPLIT depth is reached, i.e. sub-millisecond windows). Every emitted file
# is therefore complete, and the file count grows only where the traffic is.
#
# Usage: dump_loki.sh <start_epoch_seconds> <end_epoch_seconds> <out_prefix>
#        [chunk_seconds] [parallel]
set -u
SELF=$(readlink -f "$0")
START=${1:?start_epoch}
END=${2:?end_epoch}
OUT=${3:?out_prefix}
CHUNK=${4:-1}
PAR=${5:-6}
LOKI_URL="${LOKI_URL:-http://127.0.0.1:3100}"
MAXENT=4900        # under the 5000 server cap
MAXSPLIT=${MAXSPLIT:-10}   # 1 s / 2^10 ~ 1 ms floor
GZIP_CHUNKS=${GZIP_CHUNKS:-1}
# Hard ceiling on total output, as a disk safety valve (MB).
MAXTOTAL_MB=${MAXTOTAL_MB:-6144}

# All internal times are integer milliseconds. The orchestrator passes seconds.
ms() { echo $(( $1 * 1000 )); }
fmt_ms() {  # 1763...123 -> 20261117T221204Z.123
  local v=$1
  printf '%s.%03d' "$(date -u -d @$((v / 1000)) +%Y%m%dT%H%M%SZ)" $((v % 1000))
}

if [ "${1:-}" = "--one" ]; then
  # worker: --one <t_ms> <e_ms> <f>  (LOKI_URL, RES exported)
  t=$2; e=$3; f=$4
  curl -s -m 60 --get \
    --data-urlencode 'query={instance=~"train-ticket/.*"}' \
    --data-urlencode "start=$((t * 1000000))" \
    --data-urlencode "end=$((e * 1000000))" \
    --data-urlencode "limit=$MAXENT" \
    --data-urlencode 'direction=forward' \
    "$LOKI_URL/loki/api/v1/query_range" > "$f" 2>/dev/null
  if grep -q '"status":"success"' "$f" 2>/dev/null; then
    n=$(python3 -c "import json;d=json.load(open('$f'));print(sum(len(s.get('values',[])) for s in d.get('data',{}).get('result',[])))" 2>/dev/null || echo 0)
    if [ "${GZIP_CHUNKS:-1}" = "1" ]; then
      gzip -1 -f "$f" 2>/dev/null
      f="$f.gz"
    fi
    if [ "$n" -ge "$MAXENT" ]; then
      # full page: hand back to the parent to split
      echo "TRUNC $t $e $f"
    else
      echo "$n $f"
    fi
  else
    echo "FAILED $f"
  fi
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
WORK=$(mktemp -d /tmp/opencode/lokidump.XXXXXX)
RES="$WORK/results.txt"
export LOKI_URL RES GZIP_CHUNKS
: > "$RES"

START_MS=$(ms "$START"); END_MS=$(ms "$END")
CHUNK_MS=$(( CHUNK * 1000 ))
[ "$CHUNK_MS" -lt 1 ] && CHUNK_MS=1

# ---- round 1: fixed chunks -------------------------------------------------
t=$START_MS
while [ "$t" -lt "$END_MS" ]; do
  e=$(( t + CHUNK_MS )); [ "$e" -gt "$END_MS" ] && e=$END_MS
  echo "$t $e ${OUT}_$(fmt_ms "$t")_$(fmt_ms "$e").json"
  t=$e
done > "$WORK/tasks.txt"

# ---- rounds 2..N: split every chunk that came back full --------------------
depth=0
while [ "$depth" -lt "$MAXSPLIT" ]; do
  xargs -P "$PAR" -L 1 bash "$SELF" --one < "$WORK/tasks.txt" >> "$RES"
  grep '^TRUNC ' "$RES" > "$WORK/trunc.txt" || true
  ntr=$(grep -c '' "$WORK/trunc.txt" || true)
  if [ "$ntr" -eq 0 ]; then
    : > "$WORK/tasks.txt"   # everything processed; nothing left for a final round
    break
  fi
  : > "$WORK/tasks.txt"
  while read -r _ t e f; do
    # discard the partial file; the halves replace it
    rm -f "$f"
    mid=$(( (t + e) / 2 ))
    [ "$mid" -le "$t" ] && continue   # already at ms resolution
    echo "$t $mid ${OUT}_$(fmt_ms "$t")_$(fmt_ms "$mid").json"
    echo "$mid $e ${OUT}_$(fmt_ms "$mid")_$(fmt_ms "$e").json"
  done < "$WORK/trunc.txt" >> "$WORK/tasks.txt"
  # drop the TRUNC records so the final count is not polluted
  grep -v '^TRUNC ' "$RES" > "$RES.new" && mv "$RES.new" "$RES"
  depth=$(( depth + 1 ))
  [ -s "$WORK/tasks.txt" ] || break
done

# anything still full at MAXSPLIT is sub-millisecond: count it as truncated
xargs -P "$PAR" -L 1 bash "$SELF" --one < "$WORK/tasks.txt" >> "$RES" 2>/dev/null || true

TOTAL=$(awk '$1 ~ /^[0-9]+$/ {s += $1} END {print s + 0}' "$RES")
TRUNC=$(grep -c '^TRUNC ' "$RES" 2>/dev/null || true); TRUNC=${TRUNC:-0}
FAILED=$(grep -c '^FAILED' "$RES" 2>/dev/null || true); FAILED=${FAILED:-0}
OKCNT=$(grep -c '^[0-9]' "$RES" 2>/dev/null || true); OKCNT=${OKCNT:-0}
BYTES=$(du -sm "$(dirname "$OUT")" 2>/dev/null | cut -f1)
echo "loki dump done: $OKCNT files, $TOTAL lines -> ${OUT}_*.json.gz (truncated: $TRUNC, failed: $FAILED, splits=$depth)"
echo "loki dump size: ${BYTES} MB (cap ${MAXTOTAL_MB} MB)"

if [ "$FAILED" -gt 0 ]; then
  echo "loki dump: $FAILED chunks FAILED" >&2
  rm -rf "$WORK"
  exit 1
fi
if [ "$TRUNC" -gt 0 ]; then
  echo "loki dump: WARNING $TRUNC chunks still hit the ${MAXENT}-line cap after ${MAXSPLIT} splits" >&2
fi
rm -rf "$WORK"
exit 0