#!/bin/bash
# Build + push the PBScaler controller image to docker.io (registry-based
# deploy; the old ctr side-load onto worker1 is retired).
#
# Requires:
#   * docker login done on this host (docker.io credentials);
#   * the trained paper-mode predictor baked in:
#       pbscaler/simulation/train_ticket/RandomForestClassify.model
#       pbscaler/simulation/train_ticket/RandomForestClassify.svcs
#     (the .model is a gitignored artifact; retrain after each new
#      collection run — the baked model must match the reverted paper-mode
#      trainer, i.e. the [i, qps, count] feature row).
#
# Usage: pbscaler/deploy/build_and_push.sh [TAG]   (default v1.1.0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REGISTRY="${REGISTRY:-docker.io/younesnb}"
IMAGE_NAME="pbscaler"
TAG="${1:-v1.1.0}"
FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

for f in "$PBS_DIR/simulation/train_ticket/RandomForestClassify.model" \
         "$PBS_DIR/simulation/train_ticket/RandomForestClassify.svcs" \
         "$PBS_DIR/main.py" "$PBS_DIR/PBScaler.py"; do
  if [ ! -f "$f" ]; then
    echo -e "${RED}Error: required file missing: $f${NC}"
    echo "Train the RF first: pbscaler/.venv/bin/python pbscaler/simulation/RandomForestClassify.py"
    exit 1
  fi
done

echo -e "${GREEN}Building $FULL_IMAGE ...${NC}"
docker build -t "$FULL_IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$PBS_DIR"

echo -e "${GREEN}Pushing $FULL_IMAGE ...${NC}"
if docker push "$FULL_IMAGE"; then
  echo -e "${GREEN}Pushed $FULL_IMAGE — deployment.yaml pulls it via imagePullPolicy: IfNotPresent${NC}"
else
  echo -e "${RED}Push failed. Run 'docker login' on this host first.${NC}"
  exit 1
fi
