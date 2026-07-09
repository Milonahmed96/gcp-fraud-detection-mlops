"""Train the two fraud model variants and compare them.

Runs one of two ways, selected by `--backend`:

* `local`  -- fits in this process against the committed sample. Free, fast,
  no GCP. This is what CI and `pytest` exercise.
* `vertex` -- packages this same module and submits it as a Vertex AI Custom
  Training job. This is what produces the portfolio artefacts.

The code path that actually fits the model is identical in both cases; only the
machine it runs on differs. That is deliberate -- a training script that behaves
differently in the cloud is a script you cannot debug.

    uv run python -m src.training.train --backend local --variant both
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib

from src.training.dataset import DataSource, Dataset, load_features, temporal_split
from src.training.metrics import (
    CostModel,
    Evaluation,
    bootstrap_cost_difference,
    evaluate,
    optimal_threshold,
)
from src.training.models import (
    VARIANTS,
    FittedModel,
    Variant,
    build_model,
    hyperparameters_for,
    predict_fraud_probability,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("artifacts")

#: Rows used to estimate the training SHAP importance baseline. Exact
#: TreeExplainer is linear in rows; a few thousand estimates a mean fine.
IMPORTANCE_PROFILE_ROWS = 2000


@dataclass(frozen=True)
class TrainingResult:
    """Everything one variant's training run produced."""

    variant: str
    evaluation: Evaluation
    hyperparameters: dict[str, Any]
    n_train: int
    n_validation: int
    n_test: int
    train_fraud_rate: float
    model_path: str | None = None
    explainer_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evaluation"] = self.evaluation.as_dict()
        return payload


@dataclass(frozen=True)
class ComparisonResult:
    """The A/B verdict, with an honest interval rather than a bare point estimate."""

    winner: str
    cost_difference_per_1000: float
    confidence_interval: tuple[float, float]
    significant: bool
    results: dict[str, TrainingResult]

    def summary(self) -> str:
        low, high = self.confidence_interval
        verdict = "significant" if self.significant else "NOT significant (interval straddles zero)"
        return (
            f"winner: {self.winner} | "
            f"cost delta per 1k: {self.cost_difference_per_1000:+.2f} "
            f"[{low:+.2f}, {high:+.2f}] | {verdict}"
        )


def train_variant(
    dataset: Dataset,
    variant: Variant,
    *,
    cost_model: CostModel | None = None,
) -> tuple[FittedModel, TrainingResult]:
    """Fit one variant, tune its threshold on validation, score it on test.

    The threshold is chosen to minimise business cost on the *validation* slice
    and then applied unchanged to test. Choosing it on test would leak, and
    every variant would look better than it is.
    """
    cost_model = cost_model or CostModel()
    scale_pos_weight = dataset.scale_pos_weight
    params = hyperparameters_for(variant, scale_pos_weight=scale_pos_weight)

    logger.info(
        "fitting %s on %d rows (scale_pos_weight=%.1f)",
        variant,
        len(dataset.train),
        scale_pos_weight,
    )
    model = build_model(variant, scale_pos_weight=scale_pos_weight)
    model.fit(dataset.train.X, dataset.train.y)

    threshold_source = dataset.validation if len(dataset.validation) else dataset.train
    validation_scores = predict_fraud_probability(model, threshold_source.X)
    threshold = optimal_threshold(
        threshold_source.y,
        validation_scores,
        amounts=threshold_source.amounts,
        cost_model=cost_model,
    )

    test_scores = predict_fraud_probability(model, dataset.test.X)
    evaluation = evaluate(
        dataset.test.y,
        test_scores,
        threshold=threshold,
        amounts=dataset.test.amounts,
        cost_model=cost_model,
    )

    result = TrainingResult(
        variant=variant,
        evaluation=evaluation,
        hyperparameters=params,
        n_train=len(dataset.train),
        n_validation=len(dataset.validation),
        n_test=len(dataset.test),
        train_fraud_rate=dataset.train.fraud_rate,
    )
    return model, result


def save_model(model: FittedModel, variant: str, output_dir: Path) -> Path:
    """Persist a fitted model as a joblib artefact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"model_{variant}.joblib"
    joblib.dump(model, path)
    return path


def save_explainer(model: FittedModel, variant: str, dataset: Dataset, output_dir: Path) -> Path:
    """Build the SHAP explainer once at training time and ship it as an artefact.

    Constructing a `TreeExplainer` per request would put tree traversal on the
    hot path. Building it here and persisting it means the inference service
    loads it alongside the model.
    """
    from src.evaluation.explainer import FraudExplainer

    explainer = FraudExplainer.from_model(model, list(dataset.train.X.columns))
    return explainer.save(output_dir / f"explainer_{variant}.joblib")


def save_drift_reference(model: FittedModel, dataset: Dataset, output_dir: Path) -> None:
    """Capture the training feature distribution and SHAP importance profile.

    These are the baselines the Phase 6 drift monitor compares production
    traffic against. Written from the *training* split -- the distribution the
    model actually learned -- not from test, which the model never saw.

    Only the incumbent variant's explainer is profiled: `importance_shift` is a
    comparison against one baseline, and mixing variants would make it
    meaningless.
    """
    from src.evaluation.explainer import FraudExplainer
    from src.monitoring.monitor import save_reference

    explainer = FraudExplainer.from_model(model, list(dataset.train.X.columns))
    importance = explainer.global_importance(dataset.train.X.head(IMPORTANCE_PROFILE_ROWS))
    save_reference(dataset.train.X, importance, output_dir)


def compare_variants(
    dataset: Dataset,
    *,
    cost_model: CostModel | None = None,
    output_dir: Path | None = None,
    n_resamples: int = 1000,
) -> ComparisonResult:
    """Train both variants on identical data and decide which wins on cost.

    The winner is the variant with the lower expected business cost per 1,000
    transactions -- not the higher AUC. If the bootstrap interval for the
    difference contains zero, the result is reported as not significant and the
    incumbent (XGBoost, variant A) should be kept.
    """
    cost_model = cost_model or CostModel()
    models: dict[str, FittedModel] = {}
    results: dict[str, TrainingResult] = {}
    scores: dict[str, Any] = {}

    for variant in VARIANTS:
        model, result = train_variant(dataset, variant, cost_model=cost_model)
        models[variant] = model
        results[variant] = result
        scores[variant] = predict_fraud_probability(model, dataset.test.X)
        if output_dir is not None:
            model_path = save_model(model, variant, output_dir)
            explainer_path = save_explainer(model, variant, dataset, output_dir)
            results[variant] = replace(
                result, model_path=str(model_path), explainer_path=str(explainer_path)
            )

    a, b = VARIANTS  # xgboost, lightgbm

    if output_dir is not None:
        # Baselines for the drift monitor, from the incumbent variant only.
        save_drift_reference(models[a], dataset, output_dir)
    point, low, high = bootstrap_cost_difference(
        dataset.test.y,
        scores[a],
        scores[b],
        threshold_a=results[a].evaluation.threshold,
        threshold_b=results[b].evaluation.threshold,
        amounts=dataset.test.amounts,
        cost_model=cost_model,
        n_resamples=n_resamples,
    )

    significant = not (low <= 0.0 <= high)
    # point = cost(A) - cost(B). Positive means A costs more, so B wins.
    winner = b if point > 0 else a
    if not significant:
        winner = a  # keep the incumbent when the evidence is inconclusive

    return ComparisonResult(
        winner=winner,
        cost_difference_per_1000=point,
        confidence_interval=(low, high),
        significant=significant,
        results=results,
    )


def run_local_training(
    *,
    source: DataSource = "sample",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cost_model: CostModel | None = None,
    n_resamples: int = 1000,
    **load_kwargs: Any,
) -> ComparisonResult:
    """Load, split, train both variants, and write results to `output_dir`."""
    frame = load_features(source, **load_kwargs)
    dataset = temporal_split(frame)
    comparison = compare_variants(
        dataset, cost_model=cost_model, output_dir=output_dir, n_resamples=n_resamples
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "winner": comparison.winner,
                "significant": comparison.significant,
                "cost_difference_per_1000": comparison.cost_difference_per_1000,
                "confidence_interval": list(comparison.confidence_interval),
                "variants": {k: v.as_dict() for k, v in comparison.results.items()},
            },
            indent=2,
        )
    )
    logger.info("wrote metrics to %s", metrics_path)
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        choices=("local", "vertex"),
        default="local",
        help="Where training runs. 'local' fits in-process; 'vertex' submits a Custom Training job.",
    )
    parser.add_argument(
        "--source",
        choices=("sample", "bigquery"),
        default="sample",
        help="Where features come from.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--start-date", default=None, help="BigQuery source: inclusive lower bound."
    )
    parser.add_argument("--end-date", default=None, help="BigQuery source: exclusive upper bound.")
    parser.add_argument(
        "--false-positive-cost",
        type=float,
        default=CostModel().false_positive_cost,
        help="Cost of wrongly blocking one genuine transaction.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--log-experiment", action="store_true", help="Log to Vertex AI Experiments."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Resolve argv explicitly: the Vertex backend forwards these flags to the
    # remote job, and `argv or []` would silently forward nothing when invoked
    # from a real command line (where argv is None and argparse reads sys.argv).
    argv = sys.argv[1:] if argv is None else argv
    args = _build_parser().parse_args(argv)
    cost_model = CostModel(false_positive_cost=args.false_positive_cost)

    if args.backend == "vertex":
        from src.features.config import load_config
        from src.training.vertex import submit_training_job

        config = load_config()
        job = submit_training_job(config, source=args.source, args=argv)
        logger.info("submitted Vertex AI training job: %s", job)
        return 0

    load_kwargs: dict[str, Any] = {}
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

    comparison = run_local_training(
        source=args.source,
        output_dir=args.output_dir,
        cost_model=cost_model,
        n_resamples=args.bootstrap_resamples,
        **load_kwargs,
    )

    for variant, result in comparison.results.items():
        e = result.evaluation
        logger.info(
            "%-9s roc_auc=%.4f pr_auc=%.4f f1=%.4f cost/1k=%.2f (fp=%d fn=%d)",
            variant,
            e.roc_auc,
            e.pr_auc,
            e.f1,
            e.cost_per_1000,
            e.false_positives,
            e.false_negatives,
        )
    logger.info(comparison.summary())

    if args.log_experiment:
        from src.features.config import load_config
        from src.training.experiments import log_comparison

        log_comparison(load_config(), comparison)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
