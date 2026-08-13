#!/bin/bash

# Accept image name as first argument, default to "packages/drone-edge:00" if omitted
IMAGE="${1:-packages/drone-edge:00}"

# Ensure container is removed even if the script is interrupted
trap 'docker rm -f drone-edge >/dev/null 2>&1' EXIT

# Determine if platform flag is needed
PLATFORM_FLAG=""

# 1. Try inspecting local image architecture directly
INSPECT_ARCH=$(docker image inspect --format '{{.Architecture}}' "$IMAGE" 2>/dev/null)

if [ "$INSPECT_ARCH" = "arm64" ] || [[ "$IMAGE" =~ arm ]]; then
    PLATFORM_FLAG="--platform linux/arm64"
    echo "Detected ARM architecture. Running with $PLATFORM_FLAG"
fi

# Start container using platform flag if set
docker run -d $PLATFORM_FLAG --name drone-edge -p 8787:22111 "$IMAGE" >/dev/null

START=$(date +%s.%N)

until curl -sf http://localhost:8787/health | grep -q '"model_loaded":true'; do
    sleep 0.1
done

END=$(date +%s.%N)

# Cleanup container immediately
docker rm -f drone-edge >/dev/null

DURATION=$(awk "BEGIN {print $END - $START}")

echo "Cold start for $IMAGE: ${DURATION}s"

