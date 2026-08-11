# Build from the REPO ROOT, not this file's directory, so we can reach
# models/ which lives outside packages/drone-edge/:
#
#   docker build -f packages/drone-edge/Dockerfile -t drone-edge .
#
# ---------------------------------------------------------------------------
# Stage 1: builder -- resolves + installs dependencies and the package
# itself. Nothing in this stage ships in the final image; it's discarded
# after the COPY --from=builder lines below.
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Bytecode-compile at build time (slightly faster cold start), and copy
# packages into the image instead of hardlinking from uv's cache (the
# cache doesn't exist / isn't reused across build stages anyway).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# --- Dependency layer -------------------------------------------------
# Copy ONLY the dependency manifests first. This layer is cached and
# skipped entirely on every rebuild unless pyproject.toml or uv.lock
# actually change -- i.e. editing service.py does not re-trigger an
# onnxruntime re-resolve.
COPY packages/drone-edge/pyproject.toml packages/drone-edge/uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# --- App layer ----------------------------------------------------------
# Now copy source and install the project itself into the already-built
# environment. --no-editable: we want a real install, not a symlink back
# to /app/src (which won't exist in the runtime stage).
COPY packages/drone-edge/src ./src
COPY packages/drone-edge/README.md ./README.md
RUN uv sync --locked --no-editable --no-dev


# ---------------------------------------------------------------------------
# Stage 2: runtime -- what actually ships. No uv, no compilers, no
# pyproject.toml/uv.lock -- just the resolved venv and the app.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# If a build/run ever fails with something like
#   ImportError: libgomp.so.1: cannot open shared object file
# onnxruntime needs OpenMP and it's not present in slim by default.
# Uncomment if/when that happens -- don't add speculatively.
# RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# Run as non-root -- standard hardening, costs nothing here.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Model artifacts: named explicitly, not the whole models/ directory --
# model.pt (the PyTorch training checkpoint) has no business in the
# edge image. Paths match main.py's MODEL_PATH/SCALER_PATH/LABELS_PATH
# ("models/model.onnx" etc.), relative to this WORKDIR.
COPY --chown=appuser:appuser models/model.onnx models/model.onnx.data models/scaler.json models/labels.json ./models/

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Exec form (JSON array), not a bare string -- so SIGTERM from
# `docker stop` / Kubernetes reaches uvicorn directly as PID 1 instead
# of being swallowed by an intermediate shell. 0.0.0.0, not 127.0.0.1 --
# uvicorn's default loopback bind is unreachable from outside the
# container's own network namespace.
CMD ["uvicorn", "dronerf_edge.main:app", "--host", "0.0.0.0", "--port", "8000"]
