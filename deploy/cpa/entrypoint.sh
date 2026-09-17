#!/bin/sh
set -eu

HORIZON_MIN="${HORIZON:-5}"
case "$HORIZON_MIN" in
  ''|*[!0-9]*) HORIZON_MIN=5 ;;
esac
INTERVAL_MS=$((HORIZON_MIN * 60 * 1000))

cat > /config.yaml <<EOF
evaluate:
  type: "shell"
  timeout: 15000
  shell:
    entrypoint: "python"
    command: ["/app/evaluate.py"]
metric:
  type: "shell"
  timeout: 15000
  shell:
    entrypoint: "python"
    command: ["/app/metric.py"]
runMode: "per-resource"
interval: ${INTERVAL_MS}
EOF

exec /custom-pod-autoscaler "$@"
