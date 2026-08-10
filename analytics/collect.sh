#!/bin/bash

DEPLOYMENTS=("frontend" "recommendationservice" "productcatalogservice" "cartservice" "currencyservice" "shippingservice" "emailservice" "paymentservice" "checkoutservice" "adservice")
NAMESPACE="online-boutique"
CONTAINER="autoscaler"
OUTPUT_DIR="./data"

mkdir -p $OUTPUT_DIR

echo "Starting collection from Namespace: $NAMESPACE..."

{
    for DEPLOY in "${DEPLOYMENTS[@]}"; do
        echo "Searching for CPA pod for: $DEPLOY..." >&2

        POD_NAME=$(kubectl get pods -n $NAMESPACE --no-headers -o custom-columns=":metadata.name" | grep "${DEPLOY}-cpa" | head -n 1)

        if [ -z "$POD_NAME" ]; then
            echo "No CPA pod found for $DEPLOY. Skipping." >&2
            echo "."
            continue
        fi

        echo "Downloading from $POD_NAME..." >&2

        kubectl cp $NAMESPACE/$POD_NAME:/tmp/experiment_metrics.csv $OUTPUT_DIR/$DEPLOY.csv -c $CONTAINER

        if [ $? -eq 0 ]; then
            echo "Saved $OUTPUT_DIR/$DEPLOY.csv" >&2
        else
            echo "Failed to copy. Checking if file exists in $POD_NAME..." >&2
            kubectl exec -n $NAMESPACE $POD_NAME -c $CONTAINER -- ls -l /tmp/experiment_metrics.csv 2>/dev/null
            if [ $? -ne 0 ]; then
                echo "   (Error: /tmp/experiment_metrics.csv not found inside the container)" >&2
            fi
        fi
        echo "."
    done
} | tqdm --total "${#DEPLOYMENTS[@]}" --desc "Collecting metrics" >/dev/null

echo "Collection complete!"
