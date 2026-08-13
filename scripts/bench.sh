#!/bin/bash
#
# bench.sh
# ========
# Runs the full edge-deployment benchmark suite against one image and
# prints a markdown table (also appended to $OUT). This stitches
# together the manual steps you were already running by hand:
#   1. image size          (docker save | wc -c)
#   2. cold start           (tests/measure_cold_start.sh)
#   3. RSS memory, idle + peak under load (docker stats + simulate_sensor.py load)
#   4. inference latency, p50/p95          (simulate_sensor.py --bench, server-side)
#                                           + optionally hey (client-side, run separately)
#
# Nothing here replaces the step-by-step runs you did to find things
# like the .venv size issue -- this is for repeat runs once the numbers
# are stable, so every re-run (new model version, new Dockerfile) is
# one command instead of four manual procedures.
#
# Usage:
#   ./scripts/bench.sh <image> [options]
#
# Options (all have defaults you'll likely want to override once, then leave):
#   --host-port PORT         host port to map            (default: 8787)
#   --container-port PORT    container's PREDICTOR_API    (default: 22111)
#   --sim-module MODULE      python -m module for simulate_sensor.py
#                             (default: packages.drone-dev.src.drone_dev.simulate_sensor)
#   --cold-start-script PATH (default: tests/measure_cold_start.sh)
#   --scenario SPEC          load-generation scenario     (default: background:20,bebop:10,ar:10,phantom:10)
#   --speed N                simulate_sensor.py --speed for the load step (default: 50)
#   --bench-n N               requests for latency p50/p95 step (default: 200)
#   --out PATH                markdown output file (default: bench_results.md)
#
# Example:
#   ./scripts/bench.sh packages/drone-edge:00 --host-port 8787

set -uo pipefail

IMAGE="${1:?Usage: $0 <image> [options]}"
shift

HOST_PORT=8787
CONTAINER_PORT=22111
SIM_MODULE="packages.drone-dev.src.drone_dev.simulate_sensor"
COLD_START_SCRIPT="scripts/measure_cold_start.sh"
SCENARIO="background:20,bebop:10,ar:10,phantom:10"
SPEED=50
BENCH_N=200
OUT="bench_results.md"
CONTAINER_NAME="drone-edge-bench"
MEM_LOG="log/mem_log.txt"

while [ $# -gt 0 ]; do
    case "$1" in
        --host-port) HOST_PORT="$2"; shift 2 ;;
        --container-port) CONTAINER_PORT="$2"; shift 2 ;;
        --sim-module) SIM_MODULE="$2"; shift 2 ;;
        --cold-start-script) COLD_START_SCRIPT="$2"; shift 2 ;;
        --scenario) SCENARIO="$2"; shift 2 ;;
        --speed) SPEED="$2"; shift 2 ;;
        --bench-n) BENCH_N="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p log
API_URL="http://localhost:${HOST_PORT}"

# Always clean up any container we started, even on failure/interrupt.
trap 'docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1; kill "$POLLER_PID" >/dev/null 2>&1' EXIT

echo "=== Benchmarking $IMAGE ==="
echo

# ---------------------------------------------------------------------
# 1. Image size -- no container needed
# ---------------------------------------------------------------------
echo "--- 1. Image size ---"
IMAGE_SIZE_SI=$(docker save "$IMAGE" | wc -c | numfmt --to=si)
IMAGE_SIZE_IEC=$(docker save "$IMAGE" | wc -c | numfmt --to=iec)
IMAGE_LS_SIZE=$(docker images "$IMAGE" --format "{{.Size}}")
echo "  save size: ${IMAGE_SIZE_SI} (SI) / ${IMAGE_SIZE_IEC} (IEC)"
echo "  docker images reported size (incl. layer overhead): ${IMAGE_LS_SIZE}"
echo

# ---------------------------------------------------------------------
# 2. Cold start -- delegates to the existing script, which starts and
#    tears down its own container, so this must run before we start
#    the long-lived container for steps 3-4.
# ---------------------------------------------------------------------
echo "--- 2. Cold start ---"
if [ -x "$COLD_START_SCRIPT" ] || [ -f "$COLD_START_SCRIPT" ]; then
    COLD_START_OUTPUT=$(sh "$COLD_START_SCRIPT" "$IMAGE" 2>&1)
    echo "$COLD_START_OUTPUT"
    COLD_START_SECS=$(echo "$COLD_START_OUTPUT" | grep -oE '[0-9]+\.[0-9]+s?$' | head -1 | tr -d 's')
else
    echo "  [!] $COLD_START_SCRIPT not found -- skipping cold start measurement"
    COLD_START_SECS="N/A"
fi
echo

# ---------------------------------------------------------------------
# 3. RSS memory: idle, then peak under load
# ---------------------------------------------------------------------
echo "--- 3. Memory (idle + under load) ---"

PLATFORM_FLAG=""
INSPECT_ARCH=$(docker image inspect --format '{{.Architecture}}' "$IMAGE" 2>/dev/null)
if [ "$INSPECT_ARCH" = "arm64" ] || [[ "$IMAGE" =~ arm ]]; then
    PLATFORM_FLAG="--platform linux/arm64"
    echo "Detected ARM architecture. Running with $PLATFORM_FLAG"
fi

docker run -d $PLATFORM_FLAG --name "$CONTAINER_NAME" -p "${HOST_PORT}:${CONTAINER_PORT}" "$IMAGE" >/dev/null

echo "  waiting for /health..."
until curl -sf "${API_URL}/health" | grep -q '"model_loaded":true'; do
    sleep 0.1
done

IDLE_MEM=$(docker stats --no-stream "$CONTAINER_NAME" --format "{{.MemUsage}}")
echo "  idle: $IDLE_MEM"

# Background poller: sample every 0.5s while the load step runs below.
: > "$MEM_LOG"
( while true; do
    docker stats --no-stream "$CONTAINER_NAME" --format "{{.MemUsage}}" >> "$MEM_LOG" 2>/dev/null
    sleep 0.5
done ) &
POLLER_PID=$!

echo "  generating load: scenario='$SCENARIO' speed=${SPEED}x against $API_URL ..."
python -m "$SIM_MODULE" \
    --scenario "$SCENARIO" \
    --speed "$SPEED" \
    --api-url "$API_URL" \
    > log/load_run.log 2>&1

kill "$POLLER_PID" >/dev/null 2>&1
wait "$POLLER_PID" 2>/dev/null

MEM_SUMMARY=$(python3 scripts/parse_mem_log.py "$MEM_LOG")
echo "  under load: $MEM_SUMMARY"
PEAK_MEM=$(echo "$MEM_SUMMARY" | grep -oE 'peak=[0-9.]+MB' | grep -oE '[0-9.]+')
echo

# ---------------------------------------------------------------------
# 4. Inference latency p50/p95 -- server-reported inference_time_ms,
#    via simulate_sensor.py --bench (single-stream, back-to-back, real
#    test-set windows). Run `hey` separately against the still-running
#    container for the client-side round-trip number.
# ---------------------------------------------------------------------
echo "--- 4. Inference latency (server-side, single-stream) ---"
python -m "$SIM_MODULE" --api-url "$API_URL" --bench "$BENCH_N" | tee log/bench_latency.log
P50=$(grep -oE 'p50 = [0-9.]+' log/bench_latency.log | grep -oE '[0-9.]+')
P95=$(grep -oE 'p95 = [0-9.]+' log/bench_latency.log | grep -oE '[0-9.]+')
echo
echo "  API container is still running at $API_URL -- run \`hey\` now if you want the"
echo "  end-to-end client-side p50/p95 too, e.g.:"
echo "    hey -n 200 -c 1 -m POST -T \"application/json\" -D payload.json ${API_URL}/v1/predict"
echo

# ---------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M UTC")

SUMMARY=$(cat <<EOF

## Edge Deployment Benchmarks

Image: \`$IMAGE\` -- measured $TIMESTAMP

| Metric                          | Value                    |
|----------------------------------|--------------------------|
| Image size (docker save)         | ${IMAGE_SIZE_SI} |
| Image size (docker images)       | ${IMAGE_LS_SIZE} |
| Cold start (to /health ready)    | ${COLD_START_SECS}s |
| RSS, idle                        | ${IDLE_MEM} |
| RSS, peak under load             | ${PEAK_MEM} MB |
| Inference latency, p50 (server)  | ${P50} ms |
| Inference latency, p95 (server)  | ${P95} ms |

Reproduce with \`scripts/bench.sh $IMAGE\`. Client-side (HTTP round-trip)
latency measured separately via \`hey\` -- see \`scripts/dump_sample_payload.py\`.
EOF
)

echo "$SUMMARY"
echo "$SUMMARY" >> "$OUT"
echo
echo "Appended to $OUT"
