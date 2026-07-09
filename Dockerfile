# Inference service image for Cloud Run.
#
# Multi-stage: the builder resolves dependencies with uv, the runtime carries
# only the virtualenv and the source. Keeps the image small enough that Cloud
# Run cold starts stay inside the latency budget.

FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer caches across
# source changes. --no-install-project: the project itself is copied in below.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY README.md ./
RUN uv sync --frozen --no-dev


FROM python:3.11-slim AS runtime

# libgomp is the OpenMP runtime that both XGBoost and LightGBM link against.
# Without it, `import lightgbm` fails at startup with an opaque loader error.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Never run the service as root.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src

# Model + explainer + metrics.json. Produced by the training job and baked in at
# build time, so a revision's artefacts are immutable and match its image tag.
COPY --chown=appuser:appuser artifacts/ /app/artifacts/

# The sample backs the local customer-state store. Production overrides this
# with the Vertex AI Feature Store reader.
COPY --chown=appuser:appuser data/sample/ /app/data/sample/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_ARTIFACTS_DIR=/app/artifacts \
    SERVING_VARIANT=xgboost

USER appuser

# Cloud Run injects $PORT and ignores EXPOSE; 8080 is the documented default.
ENV PORT=8080
EXPOSE 8080

# Cloud Run terminates TLS and load-balances, so one uvicorn worker per
# container is correct -- concurrency is a Cloud Run setting, not a uvicorn one.
CMD exec uvicorn src.inference.app:app --host 0.0.0.0 --port ${PORT} --workers 1
