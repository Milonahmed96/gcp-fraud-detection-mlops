"""Log training runs to Vertex AI Experiments.

Each variant becomes one run inside a shared experiment, carrying its
hyperparameters and its evaluation metrics. That gives the A/B comparison an
auditable home: a reviewer can see exactly which parameters produced which cost
per 1,000 transactions, months later.

Phase 4 attaches SHAP artefacts to these same runs.

Failures here are logged, not raised. A completed model that could not be logged
is still a completed model; losing it because the experiment tracker was
unreachable would be the worse outcome.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.features.config import GCPConfig

if TYPE_CHECKING:  # pragma: no cover
    from src.training.train import ComparisonResult, TrainingResult

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "fraud-detection-ab-test"


def _aiplatform():
    """Import the Vertex AI SDK lazily."""
    from google.cloud import aiplatform

    return aiplatform


def _stringify_params(params: dict[str, Any]) -> dict[str, str | float | int | bool]:
    """Vertex AI accepts only scalar parameter values."""
    scalar: dict[str, str | float | int | bool] = {}
    for key, value in params.items():
        scalar[key] = value if isinstance(value, (str, float, int, bool)) else str(value)
    return scalar


def log_training_run(
    config: GCPConfig,
    result: TrainingResult,
    *,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
) -> str | None:
    """Log one variant's params and metrics as a Vertex AI Experiments run.

    Returns the run name, or None if logging failed.
    """
    aiplatform = _aiplatform()
    run_name = run_name or f"{result.variant}-run"

    try:
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            experiment=experiment_name,
        )
        with aiplatform.start_run(run=run_name) as run:
            run.log_params(_stringify_params(result.hyperparameters))
            run.log_metrics(result.evaluation.as_dict())
        logger.info("logged run %r to experiment %r", run_name, experiment_name)
        return run_name
    except Exception:  # noqa: BLE001 -- tracking must never fail the training job
        logger.exception("failed to log run %r to Vertex AI Experiments", run_name)
        return None


def log_comparison(
    config: GCPConfig,
    comparison: ComparisonResult,
    *,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> list[str]:
    """Log every variant in an A/B comparison. Returns the runs that succeeded."""
    logged = [
        run
        for result in comparison.results.values()
        if (run := log_training_run(config, result, experiment_name=experiment_name)) is not None
    ]
    logger.info(
        "A/B winner %s (delta %.2f per 1k, significant=%s)",
        comparison.winner,
        comparison.cost_difference_per_1000,
        comparison.significant,
    )
    return logged
