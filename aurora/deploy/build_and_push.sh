#!/bin/bash
# Build + push the AURORA-HPA controller image.
#
# The frozen model must be published first (train/calibrate.py writes it into
# aurora/data/registry); it is baked into the image, so an image built before
# calibration would silently contain no model and run hold-only.
#
# Usage: deploy/build_and_push.sh [TAG]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AURORA_DIR="$(cd "${HERE}/.." && pwd)"
REGISTRY="${REGISTRY:-docker.io/younesnb}"
IMAGE_NAME="aurora"
TAG="${1:-v0.1.0-$(date -u +%Y%m%dT%H%M%SZ)}"
FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

MODEL_DIR="${AURORA_REGISTRY_DIR:-${AURORA_DIR}/data/registry}"
if [ ! -f "$MODEL_DIR/bundle.pt" ]; then
  echo -e "${RED}Error: no model bundle at $MODEL_DIR/bundle.pt${NC}" >&2
  echo "Run: .venv/bin/python train/train_models.py" >&2
  exit 1
fi
echo -e "${GREEN}bundle: $(ls -la $MODEL_DIR/bundle.pt | awk '{print $5" bytes"}')${NC}"

# stage the model into the build context so the Dockerfile can COPY it
rm -rf "${AURORA_DIR}/build_model"
mkdir -p "${AURORA_DIR}/build_model"
cp -r "$MODEL_DIR" "${AURORA_DIR}/build_model/registry"

echo -e "${GREEN}Building $FULL_IMAGE ...${NC}"
docker build -t "$FULL_IMAGE" -f "${AURORA_DIR}/Dockerfile" "$AURORA_DIR"

echo -e "${GREEN}Pushing $FULL_IMAGE ...${NC}"
docker push "$FULL_IMAGE"

echo "$FULL_IMAGE" > "${AURORA_DIR}/data/IMAGE"
echo -e "${GREEN}Pushed $FULL_IMAGE (recorded in aurora/data/IMAGE)${NC}"