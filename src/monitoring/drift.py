"""Feature distribution drift detection.

Fraud models decay because the world moves: attack patterns emerge, get
detected, and die. Labels arrive weeks late (a chargeback is not instant), so we
cannot watch AUC in production. Drift detection is therefore **unsupervised** --
it compares the distribution of the features the model is currently seeing
against the distribution it was trained on.

Two statistics, because they fail differently:

* **PSI (Population Stability Index)** -- a symmetric, binned divergence. Bounded
  in practice, interpretable, and the metric a risk function will ask for by
  name. Insensitive to sample size, which is what makes it usable on a daily
  batch of a few thousand transactions.
* **KS (two-sample Kolmogorov-Smirnov)** -- sensitive to shifts PSI's coarse bins
  can hide, but its p-value shrinks with sample size, so a large batch flags
  drift that is real yet negligible. We report it; we do not gate on it.

The binning is where PSI implementations quietly go wrong. `txn_count_1h` is
almost always 1, `is_night` is a bool, and quantile bins over either produce
duplicate edges, empty bins, and a division by zero that surfaces as `inf` or
`nan` rather than an error. `ReferenceProfile` decides per feature whether to
bin as numeric or treat as categorical, and every proportion is floored at
`EPSILON` before any logarithm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

Severity = Literal["none", "moderate", "significant"]

#: Industry-conventional PSI thresholds.
PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25

#: Floor applied to every bin proportion before taking a logarithm. Without it,
#: a bin that is empty in either sample yields log(0) -> -inf and the PSI for the
#: whole feature becomes inf or nan.
EPSILON = 1e-6

DEFAULT_N_BINS = 10

#: A feature with at most this many distinct values in the reference is treated
#: as categorical rather than quantile-binned. Bools and near-constant counts
#: land here, which is exactly the point.
MAX_CATEGORICAL_CARDINALITY = DEFAULT_N_BINS

REFERENCE_FILENAME = "reference_profile.json"


class DriftError(ValueError):
    """Raised when drift cannot be computed from the given samples."""


@dataclass(frozen=True)
class FeatureReference:
    """How one feature was distributed in the training data.

    Exactly one of `bin_edges` (numeric) or `categories` (categorical) is set.
    `proportions` aligns positionally with whichever is set.
    """

    name: str
    kind: Literal["numeric", "categorical"]
    proportions: tuple[float, ...]
    bin_edges: tuple[float, ...] | None = None
    categories: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "proportions": list(self.proportions),
        }
        if self.bin_edges is not None:
            payload["bin_edges"] = list(self.bin_edges)
        if self.categories is not None:
            payload["categories"] = list(self.categories)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureReference:
        return cls(
            name=payload["name"],
            kind=payload["kind"],
            proportions=tuple(payload["proportions"]),
            bin_edges=tuple(payload["bin_edges"]) if "bin_edges" in payload else None,
            categories=tuple(payload["categories"]) if "categories" in payload else None,
        )


@dataclass(frozen=True)
class ReferenceProfile:
    """The training distribution of every feature. Built once, at training time."""

    features: dict[str, FeatureReference] = field(default_factory=dict)
    n_rows: int = 0

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "n_rows": self.n_rows,
                    "features": [ref.as_dict() for ref in self.features.values()],
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: Path) -> ReferenceProfile:
        if not path.exists():
            raise DriftError(
                f"{path} not found. Run: uv run python -m src.training.train --backend local"
            )
        payload = json.loads(path.read_text())
        return cls(
            features={
                item["name"]: FeatureReference.from_dict(item) for item in payload["features"]
            },
            n_rows=payload["n_rows"],
        )


def _is_categorical(series: pd.Series) -> bool:
    """Bools and low-cardinality features must not be quantile-binned."""
    if series.dtype == bool or series.dtype == object:
        return True
    return series.nunique(dropna=True) <= MAX_CATEGORICAL_CARDINALITY


def _as_categories(series: pd.Series) -> np.ndarray:
    """Stringify so bools, ints, and strings share one comparable key space."""
    return series.astype(str).to_numpy()


def _numeric_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    """Quantile bin edges, open at both ends.

    Duplicate quantiles (a spiked distribution) collapse to fewer bins rather
    than producing zero-width bins. The outer edges are infinite so that values
    in the current sample beyond the training range are still counted, instead
    of vanishing and silently deflating the PSI.
    """
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(series.to_numpy(dtype=float), quantiles))
    if len(edges) < 2:
        raise DriftError(f"feature {series.name!r} is constant; cannot bin it as numeric")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _proportions(counts: np.ndarray) -> np.ndarray:
    """Normalise counts to proportions, floored at EPSILON."""
    total = counts.sum()
    if total == 0:
        raise DriftError("cannot compute proportions of an empty sample")
    return np.maximum(counts / total, EPSILON)


def build_reference(frame: pd.DataFrame, *, n_bins: int = DEFAULT_N_BINS) -> ReferenceProfile:
    """Capture the training distribution of every column in `frame`."""
    if frame.empty:
        raise DriftError("cannot build a reference profile from an empty frame")

    references: dict[str, FeatureReference] = {}
    for name in frame.columns:
        series = frame[name].dropna()
        if series.empty:
            raise DriftError(f"feature {name!r} is entirely null in the reference sample")

        if _is_categorical(series):
            values = _as_categories(series)
            categories, counts = np.unique(values, return_counts=True)
            references[name] = FeatureReference(
                name=name,
                kind="categorical",
                categories=tuple(categories.tolist()),
                proportions=tuple(_proportions(counts.astype(float)).tolist()),
            )
        else:
            edges = _numeric_edges(series, n_bins)
            counts, _ = np.histogram(series.to_numpy(dtype=float), bins=edges)
            references[name] = FeatureReference(
                name=name,
                kind="numeric",
                bin_edges=tuple(edges.tolist()),
                proportions=tuple(_proportions(counts.astype(float)).tolist()),
            )

    return ReferenceProfile(features=references, n_rows=len(frame))


def _current_proportions(reference: FeatureReference, series: pd.Series) -> np.ndarray:
    """Bin the current sample into the reference's bins."""
    series = series.dropna()
    if series.empty:
        raise DriftError(f"feature {reference.name!r} has no values in the current sample")

    if reference.kind == "numeric":
        assert reference.bin_edges is not None
        counts, _ = np.histogram(series.to_numpy(dtype=float), bins=np.array(reference.bin_edges))
        return _proportions(counts.astype(float))

    assert reference.categories is not None
    values = _as_categories(series)
    counts = np.array([float((values == category).sum()) for category in reference.categories])
    # Anything unseen in training is drift; fold it into the smallest reference
    # bin so it contributes to PSI rather than being silently dropped.
    unseen = float(len(values) - counts.sum())
    if unseen > 0:
        counts[int(np.argmin(counts))] += unseen
    return _proportions(counts)


def population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI between two aligned proportion vectors.

    `Σ (current - reference) * ln(current / reference)`. Symmetric, zero when the
    distributions match, and always finite because both vectors are floored at
    EPSILON before the logarithm.
    """
    reference = np.maximum(np.asarray(reference, dtype=float), EPSILON)
    current = np.maximum(np.asarray(current, dtype=float), EPSILON)
    if reference.shape != current.shape:
        raise DriftError(f"shape mismatch: {reference.shape} vs {current.shape}")
    return float(np.sum((current - reference) * np.log(current / reference)))


def classify(psi: float) -> Severity:
    """Map a PSI value onto the conventional severity bands."""
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "none"


@dataclass(frozen=True)
class FeatureDrift:
    """One feature's drift verdict."""

    feature: str
    psi: float
    severity: Severity
    ks_statistic: float | None = None
    ks_pvalue: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": self.psi,
            "severity": self.severity,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
        }


@dataclass(frozen=True)
class DriftReport:
    """The full drift verdict across every feature."""

    features: tuple[FeatureDrift, ...]
    n_current_rows: int
    importance_shift: float | None = None

    @property
    def drifted(self) -> bool:
        """True when any feature has drifted significantly.

        Deliberately `any`, not a mean. A single feature moving hard is exactly
        the fraud-ring signature; averaging it across thirteen stable features
        would hide it.
        """
        return any(feature.severity == "significant" for feature in self.features)

    @property
    def significant(self) -> tuple[FeatureDrift, ...]:
        return tuple(f for f in self.features if f.severity == "significant")

    @property
    def worst(self) -> FeatureDrift | None:
        return max(self.features, key=lambda f: f.psi, default=None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "drifted": self.drifted,
            "n_current_rows": self.n_current_rows,
            "importance_shift": self.importance_shift,
            "worst_feature": self.worst.feature if self.worst else None,
            "worst_psi": self.worst.psi if self.worst else None,
            "features": [f.as_dict() for f in self.features],
        }

    def summary(self) -> str:
        if not self.features:
            return "no features compared"
        worst = self.worst
        verdict = "DRIFT" if self.drifted else "stable"
        return (
            f"{verdict}: {len(self.significant)}/{len(self.features)} features drifted "
            f"significantly; worst {worst.feature} psi={worst.psi:.4f} "
            f"(n={self.n_current_rows})"
        )


def detect_drift(
    reference: ReferenceProfile,
    current: pd.DataFrame,
    *,
    importance_shift: float | None = None,
) -> DriftReport:
    """Compare a current batch of features against the training reference.

    Args:
        reference: Profile captured at training time.
        current: Engineered features observed in production.
        importance_shift: Optional SHAP-importance drift from
            `src.evaluation.explainer.importance_shift`, carried through for
            reporting. It answers a different question -- *what drives the
            model* rather than *what the model sees*.
    """
    if current.empty:
        raise DriftError("cannot detect drift on an empty current sample")

    missing = set(reference.features) - set(current.columns)
    if missing:
        raise DriftError(f"current sample is missing features: {', '.join(sorted(missing))}")

    drifts: list[FeatureDrift] = []
    for name, feature_reference in reference.features.items():
        series = current[name]
        current_proportions = _current_proportions(feature_reference, series)
        psi = population_stability_index(
            np.array(feature_reference.proportions), current_proportions
        )

        ks_statistic: float | None = None
        ks_pvalue: float | None = None
        if feature_reference.kind == "numeric":
            # KS needs the raw reference sample, which we do not persist. Compare
            # the current sample against the reference's binned CDF instead: draw
            # the reference's bin midpoints weighted by their proportions.
            ks_statistic, ks_pvalue = _ks_against_reference(feature_reference, series)

        drifts.append(
            FeatureDrift(
                feature=name,
                psi=psi,
                severity=classify(psi),
                ks_statistic=ks_statistic,
                ks_pvalue=ks_pvalue,
            )
        )

    return DriftReport(
        features=tuple(drifts),
        n_current_rows=len(current),
        importance_shift=importance_shift,
    )


def _ks_against_reference(
    reference: FeatureReference, series: pd.Series
) -> tuple[float | None, float | None]:
    """Two-sample KS between the current values and the reference's binned shape.

    We persist bins, not raw training values -- a reference profile must stay
    small enough to ship as a JSON artefact. Reconstructing a sample from bin
    midpoints is an approximation, so the KS statistic here is a *secondary*
    signal. PSI, computed on the same bins for both samples, is the one that
    gates retraining.
    """
    assert reference.bin_edges is not None
    edges = np.array(reference.bin_edges)
    interior = edges[1:-1]
    if len(interior) < 2:
        return None, None

    # Midpoints of the finite interior bins; the infinite outer bins are
    # represented by their finite edge.
    midpoints = np.concatenate(
        [[interior[0]], (interior[:-1] + interior[1:]) / 2.0, [interior[-1]]]
    )
    proportions = np.array(reference.proportions)
    if len(midpoints) != len(proportions):
        return None, None

    counts = np.maximum((proportions * 1000).round().astype(int), 1)
    synthetic = np.repeat(midpoints, counts)

    try:
        result = ks_2samp(series.to_numpy(dtype=float), synthetic)
    except ValueError:  # pragma: no cover -- degenerate input
        return None, None
    return float(result.statistic), float(result.pvalue)
