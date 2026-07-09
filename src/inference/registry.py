"""Load the serving artefacts for one A/B variant.

Each Cloud Run revision serves exactly one variant, selected by the
`SERVING_VARIANT` environment variable. Cloud Run's native revision traffic
splitting then does the A/B allocation -- no routing logic in application code,
and no way for the two variants to share mutable state.

The decision threshold is **not** 0.5. It is the cost-minimising threshold that
`src/training/train.py` fitted on the validation split, and it is read from the
`metrics.json` the training run wrote. Hardcoding 0.5 would silently discard the
entire business-cost calibration and block a different set of customers than the
A/B test measured.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.evaluation.explainer import FraudExplainer
from src.training.models import VARIANTS

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
METRICS_FILENAME = "metrics.json"


class ArtifactError(RuntimeError):
    """Raised when the serving artefacts are missing, mismatched, or unreadable."""


@dataclass(frozen=True)
class ServingBundle:
    """Everything one revision needs to score and explain a transaction."""

    variant: str
    model: Any
    explainer: FraudExplainer
    threshold: float

    @property
    def feature_names(self) -> list[str]:
        return self.explainer.feature_names


def resolve_variant(env: dict[str, str] | None = None) -> str:
    """Read `SERVING_VARIANT` from the environment.

    Defaults to the incumbent (XGBoost) rather than guessing, and rejects an
    unknown name loudly at startup -- a revision serving a nonexistent variant
    should fail its health check, not 500 on every request.
    """
    env = os.environ if env is None else env
    variant = env.get("SERVING_VARIANT", VARIANTS[0]).strip().lower()
    if variant not in VARIANTS:
        raise ArtifactError(f"SERVING_VARIANT={variant!r} is not one of {VARIANTS}")
    return variant


def resolve_artifacts_dir(env: dict[str, str] | None = None) -> Path:
    """Read `MODEL_ARTIFACTS_DIR`, defaulting to the local `artifacts/`."""
    env = os.environ if env is None else env
    return Path(env.get("MODEL_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))


def load_threshold(artifacts_dir: Path, variant: str) -> float:
    """Read the variant's cost-minimising threshold from the training metrics.

    Raises:
        ArtifactError: If `metrics.json` is absent, malformed, or has no entry
            for this variant. Falling back to 0.5 would be worse than failing:
            the service would come up healthy and quietly mis-calibrated.
    """
    metrics_path = artifacts_dir / METRICS_FILENAME
    if not metrics_path.exists():
        raise ArtifactError(
            f"{metrics_path} not found. Run: uv run python -m src.training.train --backend local"
        )

    try:
        payload = json.loads(metrics_path.read_text())
        threshold = payload["variants"][variant]["evaluation"]["threshold"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ArtifactError(f"cannot read threshold for {variant!r} from {metrics_path}") from exc

    if not 0.0 <= float(threshold) <= 1.0:
        raise ArtifactError(f"threshold {threshold!r} for {variant!r} is not a probability")
    return float(threshold)


def load_bundle(
    variant: str | None = None,
    artifacts_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> ServingBundle:
    """Load the model, explainer, and threshold for the serving variant.

    Called once at application startup. Loading the explainer here rather than
    per request is what keeps SHAP off the hot path.
    """
    variant = variant or resolve_variant(env)
    artifacts_dir = artifacts_dir or resolve_artifacts_dir(env)

    model_path = artifacts_dir / f"model_{variant}.joblib"
    explainer_path = artifacts_dir / f"explainer_{variant}.joblib"

    missing = [str(p) for p in (model_path, explainer_path) if not p.exists()]
    if missing:
        raise ArtifactError(
            f"missing serving artefacts: {', '.join(missing)}. "
            "Run: uv run python -m src.training.train --backend local"
        )

    model = joblib.load(model_path)
    explainer = FraudExplainer.load(explainer_path)
    threshold = load_threshold(artifacts_dir, variant)

    # An explainer paired with the wrong model would produce explanations that
    # do not justify the prediction actually served -- the exact failure the
    # audit log exists to prevent.
    if type(explainer.model) is not type(model):
        raise ArtifactError(
            f"explainer wraps {type(explainer.model).__name__} "
            f"but the model is {type(model).__name__}"
        )

    logger.info(
        "loaded %s bundle from %s (threshold=%.6f, %d features)",
        variant,
        artifacts_dir,
        threshold,
        len(explainer.feature_names),
    )
    return ServingBundle(variant=variant, model=model, explainer=explainer, threshold=threshold)
