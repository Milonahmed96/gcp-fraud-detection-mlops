"""The drift check that Cloud Scheduler triggers every 24h.

Two independent signals, answering different questions:

1. **Feature drift** (`src/monitoring/drift.py`) -- has the input distribution
   moved away from what the model was trained on? Unsupervised, so it works
   even though fraud labels arrive weeks late via chargebacks.
2. **Explanation drift** (`importance_shift` on SHAP profiles) -- has *what
   drives the model* changed? A model can see a stable input distribution and
   still reweight its features. This catches drift that (1) misses.

Retraining is triggered only on signal (1). Signal (2) is reported, because a
shift in attribution without a shift in inputs usually means the model is fine
and the world is subtly different -- worth a human look, not an automatic
retrain that would burn a Vertex AI job on nothing.

Run locally:
    uv run python -m src.monitoring.monitor --source sample --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.explainer import FraudExplainer, importance_shift
from src.features.schema import feature_names
from src.monitoring.drift import (
    REFERENCE_FILENAME,
    DriftError,
    DriftReport,
    ReferenceProfile,
    build_reference,
    detect_drift,
)

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
IMPORTANCE_FILENAME = "reference_importance.json"

#: Cap on how many rows the SHAP importance profile is computed over. Exact
#: TreeExplainer is linear in rows; the daily batch can be large, and a few
#: thousand rows estimate a mean-absolute-SHAP profile perfectly well.
IMPORTANCE_SAMPLE_ROWS = 2000


class MonitorError(RuntimeError):
    """Raised when the drift check cannot run."""


@dataclass(frozen=True)
class DriftCheckResult:
    """The outcome of one scheduled drift check."""

    report: DriftReport
    retraining_triggered: bool
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.report.as_dict(),
            "retraining_triggered": self.retraining_triggered,
            "dry_run": self.dry_run,
        }


def save_reference(
    features: pd.DataFrame,
    importance: pd.Series,
    artifacts_dir: Path,
) -> tuple[Path, Path]:
    """Persist the training distribution and SHAP importance profile.

    Called at the end of training. Without these, the drift monitor has nothing
    to compare against.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    profile_path = build_reference(features).save(artifacts_dir / REFERENCE_FILENAME)

    importance_path = artifacts_dir / IMPORTANCE_FILENAME
    importance_path.write_text(
        json.dumps({name: float(value) for name, value in importance.items()}, indent=2)
    )
    return profile_path, importance_path


def load_reference_importance(artifacts_dir: Path) -> pd.Series:
    """Load the SHAP importance profile captured at training time."""
    path = artifacts_dir / IMPORTANCE_FILENAME
    if not path.exists():
        raise MonitorError(
            f"{path} not found. Run: uv run python -m src.training.train --backend local"
        )
    return pd.Series(json.loads(path.read_text()))


def current_importance(explainer: FraudExplainer, features: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP over the current batch, capped for cost."""
    sample = features.head(IMPORTANCE_SAMPLE_ROWS)
    return explainer.global_importance(sample)


def run_drift_check(
    current_features: pd.DataFrame,
    *,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    explainer: FraudExplainer | None = None,
) -> DriftReport:
    """Compare a batch of production features against the training reference.

    `explainer` is optional: without it the explanation-drift signal is skipped
    and only the feature-distribution comparison runs.
    """
    profile = ReferenceProfile.load(artifacts_dir / REFERENCE_FILENAME)

    model_features = [name for name in feature_names() if name in current_features.columns]
    if not model_features:
        raise MonitorError("current sample contains none of the model's features")

    shift: float | None = None
    if explainer is not None:
        try:
            reference_importance = load_reference_importance(artifacts_dir)
            shift = importance_shift(
                reference_importance,
                current_importance(explainer, current_features[model_features]),
            )
        except (MonitorError, ValueError):
            logger.exception("could not compute explanation drift; continuing with feature drift")

    return detect_drift(profile, current_features[model_features], importance_shift=shift)


def trigger_retraining(config: Any, *, dry_run: bool) -> bool:
    """Submit a Vertex AI retraining job. Returns whether it was submitted.

    A drift check that cannot start a retraining job has still done its job --
    it reported drift. So a submission failure is logged, not raised: the
    Cloud Scheduler invocation should not be marked failed and retried into a
    thundering herd of training jobs.
    """
    if dry_run:
        logger.warning("drift detected; retraining SKIPPED (--dry-run)")
        return False

    try:
        from src.training.vertex import submit_training_job

        submit_training_job(config, source="bigquery", args=[])
        logger.warning("drift detected; retraining job submitted")
        return True
    except Exception:  # noqa: BLE001 -- a failed submission must not fail the check
        logger.exception("drift detected but retraining submission failed")
        return False


def check_and_maybe_retrain(
    current_features: pd.DataFrame,
    *,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    explainer: FraudExplainer | None = None,
    config: Any = None,
    dry_run: bool = True,
) -> DriftCheckResult:
    """The full scheduled job: detect drift, and retrain if it is significant."""
    report = run_drift_check(current_features, artifacts_dir=artifacts_dir, explainer=explainer)
    logger.info(report.summary())
    if report.importance_shift is not None:
        logger.info("explanation drift (SHAP importance shift): %.4f", report.importance_shift)

    triggered = False
    if report.drifted:
        for feature in report.significant:
            logger.warning("  %s psi=%.4f (%s)", feature.feature, feature.psi, feature.severity)
        triggered = trigger_retraining(config, dry_run=dry_run)
    else:
        logger.info("no significant drift; no retraining needed")

    return DriftCheckResult(report=report, retraining_triggered=triggered, dry_run=dry_run)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drift check for the fraud detection model.")
    parser.add_argument("--source", choices=("sample", "bigquery"), default="sample")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument(
        "--start-date", default=None, help="BigQuery source: inclusive lower bound."
    )
    parser.add_argument("--end-date", default=None, help="BigQuery source: exclusive upper bound.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report drift but never submit a retraining job.",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Skip the SHAP explanation-drift signal (faster).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    from src.training.dataset import load_features

    load_kwargs: dict[str, Any] = {}
    config = None
    if args.source == "bigquery":
        from google.cloud import bigquery as bq_sdk

        from src.features.config import load_config

        config = load_config()
        load_kwargs = {
            "config": config,
            "client": bq_sdk.Client(project=config.project_id),
            "start_date": args.start_date,
            "end_date": args.end_date,
        }

    current = load_features(args.source, **load_kwargs)

    explainer = None
    if not args.no_explain:
        from src.inference.registry import load_bundle

        try:
            explainer = load_bundle(artifacts_dir=args.artifacts_dir).explainer
        except Exception:  # noqa: BLE001 -- explanation drift is optional
            logger.exception("could not load an explainer; skipping explanation drift")

    try:
        result = check_and_maybe_retrain(
            current,
            artifacts_dir=args.artifacts_dir,
            explainer=explainer,
            config=config,
            dry_run=args.dry_run,
        )
    except (DriftError, MonitorError):
        logger.exception("drift check failed")
        return 1

    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
