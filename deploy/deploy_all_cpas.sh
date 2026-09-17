#!/bin/bash

NAMESPACE="online-boutique"
IMAGE="docker.io/younesnb/predictive-k8s-autoscaler:v1.0.0"
PROMETHEUS_URL="http://prometheus-stack-kube-prom-prometheus.monitoring.svc.cluster.local:9090"
FEATURE_SET="cpu_mem_both"
MODEL_TYPE="dpam"
PREPROCESS_APPROACH="swt"
WINDOW_SIZE="32"
SWT_LEVEL="5"
MEM_SWT_LEVEL="5"
HORIZON="5"
# CPA eval interval is aligned to the prediction horizon: the CPA runs once
# per HORIZON minutes so each cycle consumes exactly one horizon-ahead
# prediction (e.g. HORIZON=5 -> CPA every 5 minutes = 300000 ms).
INTERVAL_MS=$((HORIZON * 60 * 1000))
EVAL_INTERVAL_SECONDS=$((HORIZON * 60))

# Adaptive threshold: base 80%, live range = base +/- range
# (e.g. range 10 -> threshold floats in [70, 90]).
BASE_THRESHOLD="80"
ADAPTIVE_THRESHOLD_RANGE="10"

for DEPLOYMENT in $(kubectl get deployments -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}'); do
    
    if [ "$DEPLOYMENT" == "loadgenerator" ] || [ "$DEPLOYMENT" == "redis-cart" ]; then
        continue
    fi

    echo "Deploying CPA for $DEPLOYMENT..."

    cat <<EOF | kubectl apply -f -
apiVersion: custompodautoscaler.com/v1
kind: CustomPodAutoscaler
metadata:
  name: ${DEPLOYMENT}-cpa
  namespace: ${NAMESPACE}
spec:
  template:
    spec:
      volumes:
      - name: metrics-vol
        emptyDir: {}
      containers:
      - name: autoscaler
        image: ${IMAGE}
        imagePullPolicy: Always
        volumeMounts:
        - name: metrics-vol
          mountPath: /app/metrics
        env:
          - name: PROMETHEUS_URL
            value: "${PROMETHEUS_URL}"
          - name: FEATURE_SET
            value: "${FEATURE_SET}"
          - name: MODEL_TYPE
            value: "${MODEL_TYPE}"
          - name: PREPROCESS_APPROACH
            value: "${PREPROCESS_APPROACH}"
          - name: WINDOW_SIZE
            value: "${WINDOW_SIZE}"
          - name: SWT_LEVEL
            value: "${SWT_LEVEL}"
          - name: MEM_SWT_LEVEL
            value: "${MEM_SWT_LEVEL}"
          - name: HORIZON
            value: "${HORIZON}"
          - name: EVAL_INTERVAL_SECONDS
            value: "${EVAL_INTERVAL_SECONDS}"
          - name: BASE_THRESHOLD
            value: "${BASE_THRESHOLD}"
          - name: ADAPTIVE_THRESHOLD_RANGE
            value: "${ADAPTIVE_THRESHOLD_RANGE}"
          - name: TARGET_DEPLOYMENT
            value: "${DEPLOYMENT}"
          - name: TARGET_NAMESPACE
            valueFrom:
              fieldRef:
                fieldPath: metadata.namespace
          - name: EXPERIMENT_METRICS_FILE
            value: "/app/metrics/experiment_metrics.csv"
      - name: metrics-exporter
        image: ${IMAGE}
        imagePullPolicy: Always
        command: ["python", "/app/metrics_exporter.py"]
        ports:
        - containerPort: 8000
          name: metrics
        volumeMounts:
        - name: metrics-vol
          mountPath: /app/metrics
        env:
          - name: EXPERIMENT_METRICS_FILE
            value: "/app/metrics/experiment_metrics.csv"
          - name: METRICS_PORT
            value: "8000"
          - name: TARGET_DEPLOYMENT
            value: "${DEPLOYMENT}"
          - name: POD_NAME
            valueFrom:
              fieldRef:
                fieldPath: metadata.name
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${DEPLOYMENT}
  config:
    - name: interval
      value: "${INTERVAL_MS}"
    - name: logVerbosity
      value: "3"
EOF
done
