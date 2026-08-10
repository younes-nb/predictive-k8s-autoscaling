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
RESIDUAL="true"
RESIDUAL_CORRECTION="true"
AR_ORDER="2"
FORGETTING_FACTOR="0.95"
QUANTILE_ALPHA="0.9"
RESIDUAL_WINDOW="${WINDOW_SIZE}"

for DEPLOYMENT in $(kubectl get deployments -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}'); do
    
    if [ "$DEPLOYMENT" == "loadgenerator" ]; then
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
          - name: RESIDUAL
            value: "${RESIDUAL}"
          - name: RESIDUAL_CORRECTION
            value: "${RESIDUAL_CORRECTION}"
          - name: AR_ORDER
            value: "${AR_ORDER}"
          - name: FORGETTING_FACTOR
            value: "${FORGETTING_FACTOR}"
          - name: QUANTILE_ALPHA
            value: "${QUANTILE_ALPHA}"
          - name: RESIDUAL_WINDOW
            value: "${RESIDUAL_WINDOW}"
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