#!/bin/bash

REGISTRY="docker.io/younesnb"
IMAGE_NAME="predictive-k8s-autoscaler"
TAG="v1.0.0"
DOCKERFILE_PATH="deploy/cpa/Dockerfile"
FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$TAG"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting build process for $FULL_IMAGE_NAME...${NC}"

if [ ! -f "model.pt" ]; then
    echo -e "${RED}Error: model.pt not found in root directory!${NC}"
    echo "Please ensure your trained model is named model.pt and located in the repo root."
    exit 1
fi

if [ ! -f "common/models.py" ]; then
    echo -e "${RED}Error: common/models.py not found!${NC}"
    exit 1
fi

echo -e "${GREEN}Building Docker image...${NC}"
docker build -t "$FULL_IMAGE_NAME" -f "$DOCKERFILE_PATH" .

if [ $? -ne 0 ]; then
    echo -e "${RED}Docker build failed.${NC}"
    exit 1
fi

echo -e "${GREEN}Pushing image to $REGISTRY...${NC}"
docker push "$FULL_IMAGE_NAME"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully built and pushed $FULL_IMAGE_NAME${NC}"
else
    echo -e "${RED}Failed to push image. Are you logged in to the registry?${NC}"
    exit 1
fi