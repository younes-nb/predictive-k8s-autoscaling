#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash fetchData.sh start_date=0d0 end_date=1d0 [--aria2]
#
# Notes:
# - Downloads ONLY MSResource (MSMetricsUpdate) and MSRTMCR (MCRRTUpdate).
# - Output dirs:
#     /dataset/raw/alibaba_v2022/MSResource
#     /dataset/raw/alibaba_v2022/MSRTMCR
#
# Mapping:
#   Local dir/file prefix                                Remote path prefix
#   /dataset/raw/alibaba_v2022/MSResource/MSResource     MSMetricsUpdate/MSMetricsUpdate
#   /dataset/raw/alibaba_v2022/MSRTMCR/MSRTMCR           MCRRTUpdate/MCRRTUpdate

# >>> NEW: where to store things on this node
DATA_ROOT="/dataset/raw/alibaba_v2022"

USE_ARIA2=0
for ARGUMENT in "$@"; do
  if [[ "$ARGUMENT" == "--aria2" ]]; then
    USE_ARIA2=1
  else
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VAL="${ARGUMENT#*=}"
    export "$KEY"="$VAL"
  fi
done

if [[ -z "${start_date:-}" || -z "${end_date:-}" ]]; then
  echo "ERROR: Provide start_date and end_date like: start_date=0d0 end_date=1d0 [--aria2]" >&2
  exit 2
fi

parse_dh() {
  local s="$1"
  local d h

  IFS='d' read -r d h <<< "$s"

  if [[ -z "$d" || -z "$h" ]]; then
    echo "ERROR: Bad time spec '$s'. Expected something like '0d0' or '1d12'." >&2
    exit 2
  fi

  echo "$d" "$h"
}

read SD SH < <(parse_dh "$start_date")
read ED EH < <(parse_dh "$end_date")
SD=${SD:-0}; SH=${SH:-0}; ED=${ED:-0}; EH=${EH:-0}

START_MIN=$(( SD*24*60 + SH*60 ))
END_MIN=$(( ED*24*60 + EH*60 ))

mkdir -p "${DATA_ROOT}/MSResource" "${DATA_ROOT}/MSRTMCR"

declare -a local_prefix=(
  "${DATA_ROOT}/MSResource/MSResource"
  "${DATA_ROOT}/MSRTMCR/MSRTMCR"
)
declare -a remote_prefix=("MSMetricsUpdate/MSMetricsUpdate" "MCRRTUpdate/MCRRTUpdate")
declare -a ratio=(30 3)

BASE_URL="https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces"

if [[ "$USE_ARIA2" -eq 1 ]]; then
  TMP_LIST=$(mktemp)
  trap 'rm -f "$TMP_LIST"' EXIT

  for i in 0 1; do
    r=${ratio[$i]}
    start_idx=$(( START_MIN / r ))
    end_idx=$(( END_MIN / r - 1 ))

    for idx in $(seq "$start_idx" "$end_idx"); do
      dst="${local_prefix[$i]}_${idx}.tar.gz"
      url="${BASE_URL}/${remote_prefix[$i]}_${idx}.tar.gz"
      {
        echo "$url"
        echo "  out=$dst"
      } >> "$TMP_LIST"
    done
  done

  echo ">>> Using aria2c to download…"
  aria2c -i "$TMP_LIST" -c -m 0 -x16 -s16 -k1M --retry-wait=3 --timeout=60

else
  for i in 0 1; do
    r=${ratio[$i]}
    start_idx=$(( START_MIN / r ))
    end_idx=$(( END_MIN / r - 1 ))

    for idx in $(seq "$start_idx" "$end_idx"); do
      dst="${local_prefix[$i]}_${idx}.tar.gz"
      url="${BASE_URL}/${remote_prefix[$i]}_${idx}.tar.gz"
      echo "Fetching: $url -> $dst"
      wget -c --retry-connrefused --tries=0 --timeout=60 -O "$dst" "$url"
    done
  done
fi

echo "All requested files downloaded (MSResource + MSRTMCR)."
echo
echo "Next (extract all downloaded archives in-place):"
echo "  for f in ${DATA_ROOT}/MSResource/*.tar.gz; do tar -xzf \"\$f\" -C ${DATA_ROOT}/MSResource; done"
echo "  for f in ${DATA_ROOT}/MSRTMCR/*.tar.gz; do tar -xzf \"\$f\" -C ${DATA_ROOT}/MSRTMCR; done"
