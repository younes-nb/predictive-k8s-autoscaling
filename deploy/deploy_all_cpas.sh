#!/bin/bash

NAMESPACE="online-boutique"
IMAGE="docker.io/younesnb/predictive-k8s-autoscaler:v1.0.0"
PROMETHEUS_URL="http://prometheus-stack-kube-prom-prometheus.monitoring.svc.cluster.local:9090"
FEATURE_SET="cpu_mem_both"
MODEL_TYPE="dpam"
PREPROCESS_APPROACH="swt"
WINDOW_SIZE="128"
SWT_LEVEL="5"
MEM_SWT_LEVEL="5"
HORIZON="5"

# Conformal Prediction Configuration
CONFORMAL_WINDOW="500"
CONFORMAL_TARGET_ALPHA="0.05"
CONFORMAL_ETA="0.01"
CONFORMAL_ALPHA_MIN="0.01"
CONFORMAL_ALPHA_MAX="0.20"
SPIKE_THRESHOLD="0.6099"

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
          - name: CONFORMAL_WINDOW
            value: "${CONFORMAL_WINDOW}"
          - name: CONFORMAL_TARGET_ALPHA
            value: "${CONFORMAL_TARGET_ALPHA}"
          - name: CONFORMAL_ETA
            value: "${CONFORMAL_ETA}"
          - name: CONFORMAL_ALPHA_MIN
            value: "${CONFORMAL_ALPHA_MIN}"
          - name: CONFORMAL_ALPHA_MAX
            value: "${CONFORMAL_ALPHA_MAX}"
          - name: SPIKE_THRESHOLD
            value: "${SPIKE_THRESHOLD}"
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
      value: "60000"
    - name: logVerbosity
      value: "3"
EOF
done
