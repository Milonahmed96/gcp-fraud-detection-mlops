"""Cloud Run service that Cloud Scheduler invokes for the daily drift check.

Deliberately a **separate service from the inference API**, not another route on
it. Two reasons, both concrete:

* The inference image has no BigQuery SDK -- that is what keeps it 500 MB
  smaller (see `pyproject.toml`). The drift check reads a day of features from
  BigQuery, so it needs the `gcp` extra. Bolting `/drift-check` onto the
  inference app would drag those SDKs into every serving container.
* The drift check is a minutes-long batch job. It must not share a request
  pool, a CPU budget, or a scale-to-zero policy with a latency-critical
  endpoint that answers in 10 ms.

Cloud Scheduler authenticates with an OIDC token, which is why this is a Cloud
Run *service* rather than a Cloud Run *job*: services accept OIDC directly.

    POST /drift-check   -- run the check; retrain if drift is significant
    GET  /health        -- readiness
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.monitoring.drift import DriftError
from src.monitoring.monitor import DEFAULT_ARTIFACTS_DIR, MonitorError, check_and_maybe_retrain

logger = logging.getLogger(__name__)


class DriftCheckRequest(BaseModel):
    """Cloud Scheduler posts an empty body; every field has a safe default."""

    dry_run: bool = Field(
        default=False, description="Report drift but never submit a retraining job."
    )
    start_date: str | None = Field(default=None, description="Inclusive lower bound.")
    end_date: str | None = Field(default=None, description="Exclusive upper bound.")


class DriftCheckResponse(BaseModel):
    drifted: bool
    retraining_triggered: bool
    dry_run: bool
    n_current_rows: int
    worst_feature: str | None
    worst_psi: float | None
    importance_shift: float | None
    significant_features: list[str]


def _artifacts_dir() -> Path:
    return Path(os.environ.get("MODEL_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))


def _source() -> str:
    """`bigquery` in production; `sample` for a local smoke test."""
    return os.environ.get("DRIFT_SOURCE", "sample")


def _load_current_features(source: str, request: DriftCheckRequest):
    """Fetch the features observed since the last check."""
    from src.training.dataset import load_features

    if source == "sample":
        return load_features("sample"), None

    from google.cloud import bigquery as bq_sdk

    from src.features.config import load_config

    config = load_config()
    frame = load_features(
        "bigquery",
        config=config,
        client=bq_sdk.Client(project=config.project_id),
        start_date=request.start_date,
        end_date=request.end_date,
    )
    return frame, config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the explainer once. Its absence degrades the check, it does not kill it."""
    app.state.explainer = None
    app.state.startup_error = None
    try:
        from src.inference.registry import load_bundle

        app.state.explainer = load_bundle(artifacts_dir=_artifacts_dir()).explainer
    except Exception as exc:  # noqa: BLE001 -- explanation drift is the secondary signal
        app.state.startup_error = str(exc)
        logger.warning("no explainer loaded; explanation drift will be skipped: %s", exc)
    yield
    app.state.explainer = None


def create_app() -> FastAPI:
    """Build the monitoring application."""
    app = FastAPI(
        title="Fraud Detection Drift Monitor",
        description="Scheduled feature-drift check with automatic retraining.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Readiness. The reference profile is what this service cannot work without."""
        profile = _artifacts_dir() / "reference_profile.json"
        if not profile.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{profile} not found; run the training job first",
            )
        return {"status": "ok", "source": _source(), "artifacts_dir": str(_artifacts_dir())}

    @app.post("/drift-check", response_model=DriftCheckResponse)
    def drift_check(request: Request, body: DriftCheckRequest | None = None) -> DriftCheckResponse:
        """Run the drift check. Invoked by Cloud Scheduler every 24h."""
        body = body or DriftCheckRequest()

        try:
            current, config = _load_current_features(_source(), body)
            result = check_and_maybe_retrain(
                current,
                artifacts_dir=_artifacts_dir(),
                explainer=request.app.state.explainer,
                config=config,
                dry_run=body.dry_run,
            )
        except (DriftError, MonitorError) as exc:
            # A 5xx makes Cloud Scheduler retry. That is right for a transient
            # failure and wrong for a missing reference profile, but Scheduler
            # cannot tell the difference -- so cap retries in the job definition.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

        report = result.report
        return DriftCheckResponse(
            drifted=report.drifted,
            retraining_triggered=result.retraining_triggered,
            dry_run=result.dry_run,
            n_current_rows=report.n_current_rows,
            worst_feature=report.worst.feature if report.worst else None,
            worst_psi=report.worst.psi if report.worst else None,
            importance_shift=report.importance_shift,
            significant_features=[f.feature for f in report.significant],
        )

    return app


#: The ASGI application uvicorn serves.
app = create_app()
