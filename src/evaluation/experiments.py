"""Log SHAP explanations to Vertex AI Experiments and BigQuery.

Two destinations, two purposes:

* **Vertex AI Experiments** -- global feature importance per training run, so a
  reviewer can see which features drove which model version, months later.
* **BigQuery `prediction_log`** -- per-prediction attributions, queryable:
  *why was transaction X blocked on date Y?* That is the question a regulator
  actually asks, and it needs SQL, not a metadata store.

As in `src/training/experiments.py`, logging failures are caught and reported.
An explanation that could not be written must not fail the prediction that
produced it -- the customer's transaction has already been decided.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.features.config import GCPConfig
from src.features.schema import PREDICTIONS_TABLE

if TYPE_CHECKING:  # pragma: no cover
    from src.evaluation.explainer import Explanation

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "fraud-detection-ab-test"

#: Prefix for global-importance metrics so they do not collide with the
#: evaluation metrics logged by `src/training/experiments.py` on the same run.
IMPORTANCE_METRIC_PREFIX = "shap_importance__"

#: How many attributions to persist per prediction. The full vector is 13 floats
#: today, but the top-k is what an auditor reads, and it bounds the row size if
#: the feature set grows.
DEFAULT_TOP_K = 5


def _aiplatform():
    """Import the Vertex AI SDK lazily."""
    from google.cloud import aiplatform

    return aiplatform


def log_global_importance(
    config: GCPConfig,
    variant: str,
    importance: pd.Series,
    *,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
) -> str | None:
    """Attach mean-absolute-SHAP per feature to a variant's experiment run.

    Reuses the run name minted by `src/training/experiments.py` so the SHAP
    artefacts land on the same run as the metrics they explain.

    Returns the run name, or None if logging failed.
    """
    aiplatform = _aiplatform()
    run_name = run_name or f"{variant}-run"

    try:
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            experiment=experiment_name,
        )
        with aiplatform.start_run(run=run_name, resume=True) as run:
            run.log_metrics(
                {
                    f"{IMPORTANCE_METRIC_PREFIX}{name}": float(value)
                    for name, value in importance.items()
                }
            )
        logger.info("logged SHAP importance for %s to run %r", variant, run_name)
        return run_name
    except Exception:  # noqa: BLE001 -- tracking must never fail the training job
        logger.exception("failed to log SHAP importance for %s", variant)
        return None


def explanation_rows(
    transaction_ids: list[str],
    variant: str,
    explanations: list[Explanation],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Shape explanations into rows for the BigQuery audit log.

    The attributions are stored as a JSON string rather than a repeated record:
    it keeps the table schema stable as the feature set evolves, and BigQuery's
    `JSON_VALUE` makes it queryable regardless.
    """
    if len(transaction_ids) != len(explanations):
        raise ValueError(
            f"got {len(transaction_ids)} transaction ids for {len(explanations)} explanations"
        )

    return pd.DataFrame(
        [
            {
                "transaction_id": transaction_id,
                "variant": variant,
                "fraud_probability": explanation.probability,
                "base_value": explanation.base_value,
                "top_features": json.dumps(explanation.as_dict(top_k)["top_features"]),
            }
            for transaction_id, explanation in zip(transaction_ids, explanations, strict=True)
        ]
    )


def log_predictions_to_bigquery(
    client: Any,
    config: GCPConfig,
    transaction_ids: list[str],
    variant: str,
    explanations: list[Explanation],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> int:
    """Append per-prediction explanations to the BigQuery audit log.

    Returns the number of rows written, or 0 if the write failed. The prediction
    has already been served by the time this runs; losing the audit row is bad,
    but failing the request would be worse.
    """
    rows = explanation_rows(transaction_ids, variant, explanations, top_k=top_k)
    try:
        job = client.load_table_from_dataframe(rows, config.table_ref(PREDICTIONS_TABLE))
        job.result()
        logger.info("wrote %d explanation rows to %s", len(rows), PREDICTIONS_TABLE)
        return len(rows)
    except Exception:  # noqa: BLE001 -- the prediction is already served
        logger.exception("failed to write explanations to %s", PREDICTIONS_TABLE)
        return 0
