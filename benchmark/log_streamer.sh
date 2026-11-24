#!/bin/bash
# Stage 12K — continuous per-pod log capture for the whole run.
#
# Problem it fixes: this cluster's containerd keeps only ~2 min of pod logs
# after the stream closes, so post-hoc `kubectl logs` is useless for
# root-cause work (the 13 Jul run lost every warm-up/saturation log except
# the tail). This streamer attaches to EVERY ts-*/gateway pod at run start and
# keeps the logs until END_EPOCH.
#
# WHY ONE PROCESS PER POD (2026-11-16):
#   The first version spawned one `kubectl logs -f` per (pod, container). With
#   Istio 1.30 running istio-proxy as a native sidecar that is 2 streams per
#   pod, i.e. ~130 kubectl processes. Each kubectl is ~50 MB RSS, so 130 of
#   them is ~6.5 GB on an 11 GB load generator — it exhausted the node.
#   `kubectl logs --all-containers --prefix` returns every container of a pod
#   from ONE process, prefixing each line with `[pod/<pod>/<container>]`; the
#   lines are split into per-container files locally with gawk. That halves
#   the process count and makes the footprint predictable.
#
# Behaviour:
#   - enumerate Running ts-*/train-ticket-gateway* pods, one stream per pod
#   - `--all-containers` includes the app container AND the istio-proxy /
#     istio-init native sidecars; `--prefix` keeps them separable
#   - split locally into RUN_DIR/logs/<pod>__<container>.log
#   - a re-enumerator every 60 s attaches to pods that appear later (HPA
#     scale-out); MAX_STREAMS caps the total (default 96) and any skipped pod
#     is recorded in logs_skipped.txt rather than silently dropped
#   - on a dead stream it reconnects with --since-time=<last line timestamp>;
#     a parse failure falls back to --since=2m (small overlap, documented)
#   - each stream runs in its own session (setsid) so the whole group,
#     including kubectl, can be killed by PID
#
# Usage: log_streamer.sh RUN_DIR END_EPOCH [NS]
#        MAX_STREAMS=96 overrides the concurrency cap.
set -u
RUN_DIR=$1; END_EPOCH=$2; NS=${3:-train-ticket}
MAX_STREAMS=${MAX_STREAMS:-96}
# A stream whose session has no live process is respawned, at most once per
# this many seconds per pod. Without it a control-plane outage that outlives the
# 3-failure budget stopped capture for the rest of the run.
REVIVE_COOLDOWN_S=${REVIVE_COOLDOWN_S:-300}
KUBECTL=${KUBECTL:-/usr/local/bin/kubectl}
export KUBECONFIG=${KUBECONFIG:-/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf}
LOGD="$RUN_DIR/logs"
mkdir -p "$LOGD"
MANIFEST="$RUN_DIR/logs_manifest.jsonl"
PIDDIR="$RUN_DIR/.pids"; mkdir -p "$PIDDIR"
SKIPPED="$RUN_DIR/logs_skipped.txt"

declare -A STREAMED=()   # pod -> 1
PIDS=()                  # session-leader pid per stream (== pgid)

ts_epoch() { date -u -d "$1" +%s 2>/dev/null || echo ""; }

cleanup() {
  trap - INT TERM
  local p sid pf
  # Kill by SESSION id: `timeout` calls setpgid(), so the kubectl tree can end
  # up in a different process group than the stream's and `kill -PGID` misses
  # it (measured: 4 kubectl survived every cleanup before this).
  for pf in "$PIDDIR"/*.pid; do
    [ -s "$pf" ] || continue
    sid=$(cat "$pf" 2>/dev/null)
    [ -n "$sid" ] || continue
    for p in $(pgrep -s "$sid" 2>/dev/null); do kill -TERM "$p" 2>/dev/null || true; done
  done
  sleep 2
  for pf in "$PIDDIR"/*.pid; do
    [ -s "$pf" ] || continue
    sid=$(cat "$pf" 2>/dev/null)
    [ -n "$sid" ] || continue
    for p in $(pgrep -s "$sid" 2>/dev/null); do kill -KILL "$p" 2>/dev/null || true; done
  done
  # Fallback: anything still in a group we recorded.
  for p in "${PIDS[@]:-}"; do
    [ -n "$p" ] || continue
    kill -KILL "-$p" 2>/dev/null || true
  done
  echo "log_streamer: stopped $(date -u +%FT%TZ), files: $(ls "$LOGD" | wc -l)" >&2
}
trap cleanup INT TERM

enumerate() {
  "$KUBECTL" -n "$NS" get pods -o json 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for p in d.get("items", []):
    n = p["metadata"]["name"]
    if not (n.startswith("ts-") or n.startswith("train-ticket-gateway")):
        continue
    if p.get("status", {}).get("phase") != "Running":
        continue
    st = p.get("status", {})
    started = st.get("startTime") or ""
    names = [c["name"] for c in (p.get("spec", {}).get("containers") or [])]
    # Native sidecars (k8s >= 1.28) live in spec.initContainers with
    # restartPolicy=Always; Istio 1.30 runs istio-proxy that way. They are
    # reported in initContainerStatuses and must be enumerated too.
    names += [c["name"] for c in (p.get("spec", {}).get("initContainers") or [])
              if c.get("restartPolicy") == "Always"]
    if not names:
        continue
    print("%s\t%s\t%s" % (n, started, ",".join(names)))
'
}

# One stream per pod: every container of that pod arrives on one pipe and is
# split locally. Runs in its own session so the whole group dies with the pid.
stream_one() { # pod since_epoch
  local pod=$1 since=$2 fails=0 lastts="" rc=0
  local base="$LOGD/${pod}"
  echo "=== stream start $(date -u +%FT%TZ) pod=$pod ===" >> "${base}__stream.log"
  while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
    local args=(logs -f --timestamps --all-containers --prefix)
    if [ -n "$lastts" ]; then
      args+=(--since-time="$(date -u -d "@$lastts" +%FT%TZ 2>/dev/null || echo "")")
    elif [ -n "$since" ]; then
      args+=(--since-time="$(date -u -d "@$since" +%FT%TZ 2>/dev/null || echo "")")
    fi
    set -o pipefail
    timeout --foreground 1800 "$KUBECTL" -n "$NS" "${args[@]}" "$pod" 2>>"${base}__stream.err" \
      | gawk -v dir="$LOGD" -v pod="$pod" '
        {
          line = $0
          if (match(line, /^\[pod\/[^\/]+\/([^\]]+)\] /, m)) {
            cont = m[1]
            gsub(/[^A-Za-z0-9._-]/, "_", cont)
            f = dir "/" pod "__" cont ".log"
            print substr(line, RLENGTH + 1) >> f
            fflush(f)
            if (match(substr(line, RLENGTH + 1),
                       /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+([+-][0-9]{2}:[0-9]{2})?/)) {
              print substr(substr(line, RLENGTH + 1), RSTART, RLENGTH) > (dir "/" pod ".lastts")
            }
          } else {
            print line >> (dir "/" pod "__raw.log")
            fflush()
          }
        }'
    rc=${PIPESTATUS[0]}
    set +o pipefail
    [ "$(date +%s)" -ge "$END_EPOCH" ] && break
    if [ "$rc" -eq 124 ]; then
      fails=0   # our own 30-min timeout: healthy cycle
    else
      fails=$((fails + 1))
    fi
    if [ -s "$LOGD/${pod}.lastts" ]; then
      lastts=$(tail -n 1 "$LOGD/${pod}.lastts")
      lastts=$(ts_epoch "$lastts")
    fi
    [ -n "$lastts" ] || lastts=$since
    if [ "$fails" -ge 3 ]; then
      echo "=== stream dead after 3 failed connects $(date -u +%FT%TZ) pod=$pod ===" \
        >> "${base}__stream.log"
      break
    fi
    [ "$fails" -gt 0 ] && sleep 5
  done
  echo "=== stream end $(date -u +%FT%TZ) pod=$pod ===" >> "${base}__stream.log"
}

# stream_one runs inside a setsid'd child bash, so export it and its inputs.
export LOGD END_EPOCH NS KUBECTL
export -f ts_epoch stream_one

spawn() { # pod since_epoch csv_containers
  local pod=$1 since=$2 conts=$3
  [ -n "${STREAMED[$pod]:-}" ] && return 0
  if [ "${#PIDS[@]}" -ge "$MAX_STREAMS" ]; then
    echo "$(date -u +%FT%TZ) pod=$pod SKIPPED (MAX_STREAMS=$MAX_STREAMS reached)" >> "$SKIPPED"
    return 1
  fi
  STREAMED[$pod]=1
  local IFS=,
  for c in $conts; do
    echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"pod\":\"$pod\",\"container\":\"$c\",\"log\":\"logs/${pod}__${c}.log\"}" \
      >> "$MANIFEST"
  done
  # setsid(1) forks when the caller is already a process-group leader, so $!
  # is NOT reliably the new session leader and `kill -PGID` would miss it.
  # The child writes its own PID (== its PGID as session leader) to a pidfile
  # and cleanup kills the whole group from that.
  local pf="$PIDDIR/$pod.pid"
  rm -f "$pf"
  setsid bash -c 'echo "$$" > "$1"; shift; stream_one "$@"' _ "$pf" "$pod" "$since" \
    < /dev/null >> "$RUN_DIR/log_streamer.out" 2>&1 &
  local i
  for i in $(seq 1 20); do [ -s "$pf" ] && break; sleep 0.1; done
  if [ -s "$pf" ]; then PIDS+=("$(cat "$pf")"); else PIDS+=("$!"); fi
}

# initial sweep (wait for the API to be reachable)
for _ in $(seq 1 12); do
  while IFS=$'\t' read -r pod started conts; do
    [ -z "$pod" ] && continue
    since=$(ts_epoch "$started")
    spawn "$pod" "$since" "$conts"
  done < <(enumerate)
  [ "${#PIDS[@]}" -gt 0 ] && break
  sleep 5
done
echo "log_streamer: ${#PIDS[@]} pod streams at $(date -u +%FT%TZ) (cap $MAX_STREAMS)" >&2

# re-enumerate every 60 s for scale-out pods; exit when the run window ends
while [ "$(date +%s)" -lt "$((END_EPOCH - 60))" ]; do
  now_s=$(date +%s)
  while IFS=$'\t' read -r pod started conts; do
    [ -z "$pod" ] && continue
    if [ -n "${STREAMED[$pod]:-}" ]; then
      # Revive a dead stream: its session leader has no live process left.
      pf="$PIDDIR/$pod.pid"
      sid=$(cat "$pf" 2>/dev/null)
      if [ -n "$sid" ] && ! pgrep -s "$sid" >/dev/null 2>&1; then
        stamp="$PIDDIR/$pod.revived"
        last=0
        [ -f "$stamp" ] && last=$(cat "$stamp" 2>/dev/null || echo 0)
        if [ $((now_s - last)) -ge "$REVIVE_COOLDOWN_S" ]; then
          echo "$now_s" > "$stamp"
          unset 'STREAMED[$pod]'
          echo "$(date -u +%FT%TZ) reviving dead stream pod=$pod" >> "$RUN_DIR/log_streamer.out"
        fi
      fi
      continue
    fi
    since=$(ts_epoch "$started")
    spawn "$pod" "$since" "$conts"
  done < <(enumerate)
  sleep 60
done

# graceful stop
sleep 2
cleanup
exit 0