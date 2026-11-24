#!/bin/bash
# Build + push the AURORA-R controller image.
#
# The bundle must exist first (train/ or a bootstrap artifact), otherwise the
# image would deploy a controller that refuses to start.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "${HERE}/../.." && pwd)"
REGISTRY="${REGISTRY:-docker.io/younesnb}"
TAG="${1:-v0.1.0-$(date -u +%Y%m%dT%H%M%SZ)}"
IMAGE="$REGISTRY/aurora-r:$TAG"

BUNDLE="${AURORA_R_BUNDLE:-${PKG}/data/bundle.pt}"
if [ ! -f "$BUNDLE" ]; then
  echo "Error: no bundle at $BUNDLE" >&2
  echo "Train one: python -m aurora_r train --snapshots ... --output $BUNDLE" >&2
  exit 1
fi

rm -rf "${PKG}/build_ctx"
mkdir -p "${PKG}/build_ctx/bundle"
cp -r "${PKG}/aurora_r" "${PKG}/build_ctx/aurora_r"
cp "${PKG}/aurora_r/requirements.txt" "${PKG}/build_ctx/"
cp "${HERE}/Dockerfile" "${PKG}/build_ctx/Dockerfile"
cp "$BUNDLE" "${PKG}/build_ctx/bundle/bundle.pt"

docker build -t "$IMAGE" "${PKG}/build_ctx"
docker push "$IMAGE"
echo "$IMAGE" > "${PKG}/aurora_r/data/IMAGE"
echo "Pushed $IMAGE (recorded in aurora_r/data/IMAGE)"
