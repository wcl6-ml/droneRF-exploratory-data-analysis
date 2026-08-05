# Build from the REPO ROOT (same reasoning as the edge image, even
# though this one doesn't currently need anything outside its own
# package -- keeps both build commands consistent):
#
#   docker build -f packages/dronerf-sensor-sim/Dockerfile -t dronerf-sensor-sim .
#
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY packages/dronerf-sensor-sim/pyproject.toml packages/dronerf-sensor-sim/uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY packages/dronerf-sensor-sim/src ./src
RUN uv sync --locked --no-editable --no-dev


FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

# NOTE: dronerf.h5 and dronerf_splits.json are NOT copied into this
# image. They're real (large, dataset-side) files, not application
# artifacts -- baking them in would bloat every rebuild and tie the
# image to one specific dataset snapshot. Instead they're mounted at
# container run time (docker-compose `volumes:`, or `docker run -v`).
# This container only ever expects them to already exist at the paths
# below inside its own filesystem, wherever they came from.
ENV H5_PATH=/data/dronerf.h5 \
    SPLITS_PATH=/data/dronerf_splits.json

USER appuser

# No EXPOSE -- this is a client, not a server; it makes outbound
# requests to an edge node's /v1/predict, it doesn't accept connections.

# ENTRYPOINT fixes the module being run; CMD supplies default args that
# docker-compose (or `docker run ... dronerf-sensor-sim --api-url ...`)
# can override per-instance without needing a different image per
# sensor -- e.g. `command: ["--api-url", "http://edge-2:8000", ...]`
# in compose for a second simulated sensor.
ENTRYPOINT ["python", "-m", "dronerf_sensor_sim.simulate_sensor", "--h5", "/data/dronerf.h5", "--splits", "/data/dronerf_splits.json"]
CMD ["--scenario", "background:20,bebop:10,ar:10,phantom:10", "--speed", "50", "--api-url", "http://localhost:8000"]
