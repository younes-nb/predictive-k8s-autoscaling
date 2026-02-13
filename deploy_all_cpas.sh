#!/bin/bash

NAMESPACE="online-boutique"
IMAGE="docker.io/younesnb/predictive-k8s-autoscaler:v1.0.0"
PROMETHEUS_URL="http://prometheus-stack-kube-prom-prometheus.monitoring.svc.cluster.local:9090"

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
      containers:
      - name: autoscaler
        image: ${IMAGE}
        imagePullPolicy: Always
        env:
          - name: PROMETHEUS_URL
            value: "${PROMETHEUS_URL}"
          - name: FEATURE_SET
            value: "cpu_mem"
          - name: TARGET_NAMESPACE
            valueFrom:
              fieldRef:
                fieldPath: metadata.namespace
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${DEPLOYMENT}
  config:
    - name: interval
      value: "60000"
    - name: evaluate
      value: |
        {
          "type": "shell",
          "timeout": 15000,
          "shell": {
            "entrypoint": "python",
            "command": ["/app/evaluate.py"]
          }
        }
    - name: metric
      value: |
        {
          "type": "shell",
          "timeout": 15000,
          "shell": {
            "entrypoint": "python",
            "command": ["/app/metric.py"]
          }
        }
EOF
done