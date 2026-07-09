"""FastAPI inference service for real-time fraud detection.

One Cloud Run revision per model variant; Cloud Run's revision traffic splitting
performs the A/B allocation. The service loads its model and SHAP explainer once
at startup and holds them for the process lifetime.

Endpoints:
    GET  /health   -- liveness/readiness. Fails if artefacts did not load.
    POST /predict  -- score one transaction, with its SHAP explanation.

Run locally:
    uv run python -m src.training.train --backend local   # produce artefacts
    uv run uvicorn src.inference.app:app --port 8080
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status

from src.evaluation.explainer import POSITIVE_CLASS
from src.inference.features import build_serving_features
from src.inference.registry import ArtifactError, ServingBundle, load_bundle
from src.inference.schemas import (
    AttributionResponse,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.inference.state import CustomerState, InMemoryStateStore

logger = logging.getLogger(__name__)

#: How many SHAP attributions to return per prediction.
TOP_K_FEATURES = 5


def _load_state_store() -> Any:
    """Build the customer state store.

    Local and test runs use the committed sample. Production swaps in the Vertex
    AI Feature Store reader, which satisfies the same `lookup(customer_id)`
    signature. Both return a `CustomerState`.
    """
    from src.features.sample_data import load_sample

    return InMemoryStateStore.from_transactions(load_sample())


def to_naive_utc(timestamp: Any) -> pd.Timestamp:
    """Normalise an incoming timestamp to a timezone-naive UTC `Timestamp`.

    The feature store and the training data hold naive UTC timestamps. A request
    carrying `+02:00` must be *converted* to UTC, not have its offset discarded
    -- dropping it would shift the transaction two hours and change
    `is_night`, `hour_of_day`, and every velocity window.
    """
    parsed = pd.Timestamp(timestamp)
    if parsed.tzinfo is None:
        return parsed
    return parsed.tz_convert("UTC").tz_localize(None)


def _require_bundle(request: Request) -> ServingBundle:
    """Fetch the loaded bundle, or fail the request with a 503."""
    bundle: ServingBundle | None = getattr(request.app.state, "bundle", None)
    if bundle is None:
        detail = getattr(request.app.state, "startup_error", None) or "model artefacts not loaded"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artefacts once, at startup.

    A failure here is recorded rather than raised: the process stays up so that
    `/health` can report *why* it is unhealthy. A container that exits on
    startup gives Cloud Run nothing to show an operator but a crash loop.
    """
    app.state.bundle = None
    app.state.store = None
    app.state.startup_error = None
    try:
        app.state.bundle = load_bundle()
        app.state.store = _load_state_store()
        logger.info("service ready, serving variant %s", app.state.bundle.variant)
    except (ArtifactError, FileNotFoundError) as exc:
        app.state.startup_error = str(exc)
        logger.exception("startup failed; /health will report unhealthy")
    yield
    app.state.bundle = None
    app.state.store = None


def create_app() -> FastAPI:
    """Build the application.

    State lives on `app.state`, not in module globals: two app instances (a
    test client per variant, say) must not clobber each other's loaded model.
    """
    app = FastAPI(
        title="Fraud Detection Inference API",
        description="Real-time fraud scoring with per-prediction SHAP explanations.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Readiness probe. Returns 503 when the artefacts failed to load."""
        bundle = _require_bundle(request)
        return HealthResponse(
            status="ok",
            variant=bundle.variant,
            model_loaded=bundle.model is not None,
            explainer_loaded=bundle.explainer is not None,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: Request, transaction: TransactionRequest) -> PredictionResponse:
        """Score one transaction and explain the decision.

        The explanation is computed in the request path deliberately: an
        unexplained decision is not auditable, and `TreeExplainer` over a
        pre-built explainer is a tree traversal, not a retraining.
        """
        started = time.perf_counter()
        bundle = _require_bundle(request)

        event_time = to_naive_utc(transaction.timestamp)
        # `as_of` keeps the lookup causal: a customer's state must never include
        # the transaction being scored, nor anything after it.
        customer_state: CustomerState = request.app.state.store.lookup(
            transaction.customer_id, as_of=event_time
        )
        features = build_serving_features(
            timestamp=event_time,
            amount=transaction.amount,
            country=transaction.country,
            customer_home_country=transaction.customer_home_country,
            card_present=transaction.card_present,
            state=customer_state,
        )

        probability = float(bundle.model.predict_proba(features)[0, POSITIVE_CLASS])
        explanation = bundle.explainer.explain_one(features)

        latency_ms = (time.perf_counter() - started) * 1000.0

        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            variant=bundle.variant,
            fraud_probability=probability,
            threshold=bundle.threshold,
            is_flagged=probability >= bundle.threshold,
            base_value=explanation.base_value,
            top_features=[
                AttributionResponse(
                    feature=attribution.feature,
                    value=attribution.value,
                    shap_value=attribution.shap_value,
                    direction=attribution.direction,
                )
                for attribution in explanation.top_contributions(TOP_K_FEATURES)
            ],
            latency_ms=latency_ms,
            new_customer=customer_state.is_new_customer,
        )

    @app.middleware("http")
    async def add_latency_header(request: Request, call_next):
        """Expose server-side latency on every response, for Cloud Run metrics."""
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    return app


#: The ASGI application uvicorn serves. See the Dockerfile CMD.
app = create_app()
