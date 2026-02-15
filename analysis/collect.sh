#!/bin/bash

DEPLOYMENTS=("frontend" "recommendationservice" "productcatalogservice" "cartservice" "currencyservice" "shippingservice" "emailservice" "paymentservice" "checkoutservice" "adservice")
NAMESPACE="online-boutique"
CONTAINER="autoscaler"
OUTPUT_DIR="./data"

mkdir -p $OUTPUT_DIR

echo "Starting collection from Namespace: $NAMESPACE..."

for DEPLOY in "${DEPLOYMENTS[@]}"; do
    POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=$DEPLOY -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)

    if [ -z "$POD_NAME" ]; then
        echo "⚠️  No pod found for $DEPLOY. Skipping."
        continue
    fi

    echo "⬇️  Downloading from $DEPLOY ($POD_NAME)..."
    
    kubectl cp $NAMESPACE/$POD_NAME:/tmp/experiment_metrics.csv $OUTPUT_DIR/$DEPLOY.csv -c $CONTAINER
    
    if [ $? -eq 0 ]; then
        echo "✅ Saved $OUTPUT_DIR/$DEPLOY.csv"
    else
        echo "❌ Failed to copy from $DEPLOY"
    fi
done

echo "Collection complete!"