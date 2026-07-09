"""Load the A/B test result for reporting.

Reads what the training run wrote (`artifacts/metrics.json` and the per-variant
SHAP importance profiles) and turns it into a shape the dashboard can render
without knowing anything about JSON layouts.

**Where the numbers come from, and where they do not.** The inference service
does not yet write to the BigQuery `prediction_log`, so there is no production
prediction data to aggregate. Everything here is the *offline* evaluation on the
held-out temporal test split. Serving latency in particular is unavailable, and
this module says so rather than inventing it -- see README > Known limitations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

METRICS_FILENAME = "metrics.json"
IMPORTANCE_TEMPLATE = "importance_{variant}.json"

#: Metrics shown on the grouped comparison chart, in display order. All live on
#: the same 0-1 scale, which is why they may share one axis.
RANKING_METRICS: tuple[tuple[str, str], ...] = (
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("f1", "F1"),
    ("precision", "Precision"),
    ("recall", "Recall"),
)


class ReportError(ValueError):
    """Raised when the A/B result cannot be loaded."""


@dataclass(frozen=True)
class VariantResult:
    """One model variant's offline evaluation."""

    variant: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    cost_per_1000: float
    total_cost: float
    false_positives: int
    false_negatives: int
    true_positives: int
    n_train: int
    n_test: int
    importance: dict[str, float]

    @property
    def label(self) -> str:
        return {"xgboost": "XGBoost", "lightgbm": "LightGBM"}.get(self.variant, self.variant)

    def metric(self, key: str) -> float:
        return float(getattr(self, key))


@dataclass(frozen=True)
class ABReport:
    """The full A/B verdict, ready to render."""

    variants: tuple[VariantResult, ...]
    winner: str
    significant: bool
    cost_difference_per_1000: float
    confidence_interval: tuple[float, float]

    @property
    def winner_result(self) -> VariantResult:
        return next(v for v in self.variants if v.variant == self.winner)

    @property
    def verdict(self) -> str:
        """The headline. An inconclusive result must say so."""
        if self.significant:
            return f"{self.winner_result.label} wins on business cost"
        return "No significant difference"

    @property
    def verdict_detail(self) -> str:
        if self.significant:
            return (
                f"The bootstrap interval excludes zero, so the difference is unlikely "
                f"to be chance. Ship {self.winner_result.label}."
            )
        return (
            "The bootstrap interval for the cost difference straddles zero: on this "
            "test set the two variants are statistically indistinguishable. The "
            f"incumbent ({self.winner_result.label}) is kept rather than shipping a coin flip."
        )

    def cheaper(self) -> VariantResult:
        """The variant with the lower expected cost, significant or not."""
        return min(self.variants, key=lambda v: v.cost_per_1000)

    def best_on(self, key: str) -> VariantResult:
        """The variant with the highest value of a ranking metric."""
        return max(self.variants, key=lambda v: v.metric(key))

    @property
    def metrics_disagree(self) -> bool:
        """True when the F1 winner and the cost winner are different variants.

        This is the whole reason the business cost metric exists, so the
        dashboard calls it out explicitly when it happens.
        """
        return self.best_on("f1").variant != self.cheaper().variant


def _load_importance(artifacts_dir: Path, variant: str) -> dict[str, float]:
    """Per-variant SHAP importance. Absent for an older training run."""
    path = artifacts_dir / IMPORTANCE_TEMPLATE.format(variant=variant)
    if not path.exists():
        return {}
    return {name: float(value) for name, value in json.loads(path.read_text()).items()}


def load_report(artifacts_dir: Path) -> ABReport:
    """Load the A/B result written by `src.training.train`.

    Raises:
        ReportError: If `metrics.json` is absent or malformed.
    """
    metrics_path = artifacts_dir / METRICS_FILENAME
    if not metrics_path.exists():
        raise ReportError(
            f"{metrics_path} not found. Run: uv run python -m src.training.train --backend local"
        )

    try:
        payload = json.loads(metrics_path.read_text())
        variants = tuple(
            VariantResult(
                variant=name,
                importance=_load_importance(artifacts_dir, name),
                n_train=entry["n_train"],
                n_test=entry["n_test"],
                **{
                    key: entry["evaluation"][key]
                    for key in (
                        "roc_auc",
                        "pr_auc",
                        "precision",
                        "recall",
                        "f1",
                        "threshold",
                        "cost_per_1000",
                        "total_cost",
                    )
                },
                **{
                    key: int(entry["evaluation"][key])
                    for key in ("false_positives", "false_negatives", "true_positives")
                },
            )
            for name, entry in payload["variants"].items()
        )
        low, high = payload["confidence_interval"]
        report = ABReport(
            variants=variants,
            winner=payload["winner"],
            significant=bool(payload["significant"]),
            cost_difference_per_1000=float(payload["cost_difference_per_1000"]),
            confidence_interval=(float(low), float(high)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ReportError(f"cannot parse {metrics_path}: {exc}") from exc

    if not report.variants:
        raise ReportError(f"{metrics_path} contains no variants")
    if report.winner not in {v.variant for v in report.variants}:
        raise ReportError(f"winner {report.winner!r} is not among the variants")
    return report
