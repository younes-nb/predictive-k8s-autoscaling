#!/bin/bash
# Build + push the PBScaler controller image (mirrors deploy/build_and_push.sh
# style of this repo). Then apply RBAC + Deployment:
#   kubectl apply -f pbscaler/deploy/rbac.yaml
#   kubectl apply -f pbscaler/deploy/deployment.yaml
set -euo pipefail

REGISTRY="docker.io/younesnb"
IMAGE_NAME="pbscaler"
TAG="${TAG:-v1.0.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

# The trained predictor is baked into the image if present (optional but
# expected for real runs; without it run with PBS_DRY_RUN=true).
MODEL_PATH="${PBS_DIR}/simulation/train_ticket/RandomForestClassify.model"
if [ ! -f "$MODEL_PATH" ]; then
  echo "WARNING: no trained model at ${MODEL_PATH} (${PBS_SLO_PREDICTOR})"
  echo "         build anyway for dry-run testing."
fi

echo "Building ${IMAGE} (context=${PBS_DIR})"
docker build -t "$IMAGE" -f "${SCRIPT_DIR}/Dockerfile" "${PBS_DIR}"

echo "Pushing to ${REGISTRY} (docker login required)"
docker push "$IMAGE"
