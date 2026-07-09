"""SHAP explainability for the fraud models.

A regulated lender must be able to explain an adverse automated decision. Every
prediction therefore carries the signed contribution of each feature, computed
with `shap.TreeExplainer` -- exact for tree ensembles, and fast enough to sit in
the request path.

Two facts about SHAP that this module exists to encapsulate:

**Attributions live in log-odds space, not probability space.** The additivity
guarantee is `base_value + sum(shap_values) == raw margin`, where the margin is
the ensemble's pre-sigmoid output. Summing attributions and expecting a
probability is a common and silent error, so `Explanation` names the space it is
in and `verify_additivity` asserts the identity.

**The shape of `shap_values` is not stable across shap versions or model types.**
Older releases return a two-element list (one array per class) for LightGBM
binary classifiers, and some return `(n, n_features, n_classes)`. Current
releases return `(n, n_features)`. `normalise_shap_values` collapses all three
onto the positive (fraud) class, because taking the wrong element silently
inverts the sign of every explanation.

The explainer is built once at training time and persisted next to the model, so
no explainer construction happens per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

#: Index of the positive (fraud) class wherever SHAP emits a per-class axis.
POSITIVE_CLASS = 1

#: Tolerance for the additivity identity. Tree ensembles accumulate float32
#: error; XGBoost lands around 1e-6 while LightGBM is near machine epsilon.
ADDITIVITY_TOLERANCE = 1e-4


class ExplainerError(ValueError):
    """Raised when SHAP output cannot be interpreted as binary attributions."""


@dataclass(frozen=True)
class FeatureAttribution:
    """One feature's signed contribution to one prediction."""

    feature: str
    value: float
    shap_value: float

    @property
    def direction(self) -> str:
        """Whether this feature pushed the prediction toward or away from fraud."""
        if self.shap_value > 0:
            return "toward_fraud"
        if self.shap_value < 0:
            return "toward_genuine"
        return "neutral"


@dataclass(frozen=True)
class Explanation:
    """A single prediction's full attribution set.

    `base_value` and every `shap_value` are in **log-odds space**. They sum to
    the model's raw margin, not to `probability`.
    """

    base_value: float
    attributions: tuple[FeatureAttribution, ...]
    probability: float

    @property
    def margin(self) -> float:
        """The reconstructed raw (pre-sigmoid) model output."""
        return self.base_value + sum(a.shap_value for a in self.attributions)

    def top_contributions(self, k: int = 5) -> tuple[FeatureAttribution, ...]:
        """The `k` features that moved this prediction most, by absolute effect.

        Magnitude, not signedness: an auditor asking "why was this blocked?"
        needs the features that argued *against* the decision too.
        """
        ranked = sorted(self.attributions, key=lambda a: abs(a.shap_value), reverse=True)
        return tuple(ranked[:k])

    def as_dict(self, k: int = 5) -> dict[str, Any]:
        """Audit-log shaped payload for BigQuery and Vertex AI Experiments."""
        return {
            "probability": self.probability,
            "base_value": self.base_value,
            "margin": self.margin,
            "top_features": [
                {
                    "feature": a.feature,
                    "value": a.value,
                    "shap_value": a.shap_value,
                    "direction": a.direction,
                }
                for a in self.top_contributions(k)
            ],
        }


def normalise_shap_values(raw: Any) -> np.ndarray:
    """Collapse any SHAP output shape onto `(n_samples, n_features)` for fraud.

    Handles the three shapes `TreeExplainer.shap_values` emits for binary
    classifiers across versions and libraries:

    * `[array(n, f), array(n, f)]` -- one array per class (older LightGBM)
    * `(n, f, 2)` -- class on the trailing axis
    * `(n, f)` -- already collapsed (current shap for both variants)
    """
    if isinstance(raw, list):
        if len(raw) != 2:
            raise ExplainerError(f"expected 2 per-class arrays; got {len(raw)}")
        return np.asarray(raw[POSITIVE_CLASS], dtype=float)

    values = np.asarray(raw, dtype=float)
    if values.ndim == 3:
        if values.shape[2] != 2:
            raise ExplainerError(f"expected a binary class axis; got shape {values.shape}")
        return values[:, :, POSITIVE_CLASS]
    if values.ndim == 2:
        return values
    raise ExplainerError(f"cannot interpret SHAP values of shape {values.shape}")


def normalise_base_value(raw: Any) -> float:
    """Collapse `expected_value` onto the positive class as a scalar."""
    value = np.asarray(raw, dtype=float)
    if value.ndim == 0:
        return float(value)
    if value.ndim == 1:
        if value.size == 1:
            return float(value[0])
        if value.size == 2:
            return float(value[POSITIVE_CLASS])
    raise ExplainerError(f"cannot interpret expected_value of shape {value.shape}")


class FraudExplainer:
    """A fitted model paired with its SHAP explainer.

    Constructed once at training time and persisted alongside the model, so the
    inference service pays no explainer-construction cost per request.
    """

    def __init__(self, model: Any, feature_names: list[str], explainer: Any) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.explainer = explainer

    @classmethod
    def from_model(cls, model: Any, feature_names: list[str]) -> FraudExplainer:
        """Build a `TreeExplainer` over a fitted tree ensemble.

        Both A/B variants are tree ensembles, which is what makes the *exact*
        TreeExplainer applicable to each. That is a real constraint on the
        variant choice, not a coincidence.
        """
        import shap

        return cls(model, feature_names, shap.TreeExplainer(model))

    @property
    def base_value(self) -> float:
        """The model's expected log-odds output over the background data."""
        return normalise_base_value(self.explainer.expected_value)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Signed per-feature attributions, `(n_samples, n_features)`, log-odds."""
        self._check_columns(X)
        values = normalise_shap_values(self.explainer.shap_values(X))
        if values.shape != X.shape:
            raise ExplainerError(
                f"SHAP returned {values.shape} for input of shape {X.shape}; "
                "the class axis was probably collapsed incorrectly"
            )
        return values

    def explain(self, X: pd.DataFrame) -> list[Explanation]:
        """Explain every row of `X`."""
        values = self.shap_values(X)
        probabilities = np.asarray(self.model.predict_proba(X))[:, POSITIVE_CLASS]
        base = self.base_value

        return [
            Explanation(
                base_value=base,
                attributions=tuple(
                    FeatureAttribution(
                        feature=name,
                        value=float(X.iloc[row][name]),
                        shap_value=float(values[row, col]),
                    )
                    for col, name in enumerate(self.feature_names)
                ),
                probability=float(probabilities[row]),
            )
            for row in range(len(X))
        ]

    def explain_one(self, row: pd.DataFrame | pd.Series) -> Explanation:
        """Explain a single transaction. Accepts a one-row frame or a Series.

        A DataFrame is passed through with its dtypes intact. Casting it to float
        would make the explainer score a different matrix than the caller scored
        with `predict_proba`, so the explanation could disagree with the
        probability it is meant to justify. Only the Series path is cast, because
        `Series.to_frame().T` yields object columns that no model can consume.
        """
        if isinstance(row, pd.Series):
            frame = row.to_frame().T.infer_objects()
        else:
            frame = row
        if len(frame) != 1:
            raise ExplainerError(f"explain_one expects exactly one row; got {len(frame)}")
        return self.explain(frame)[0]

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        """Mean absolute SHAP value per feature, descending.

        This is the model's own account of what drives it. Comparing it across
        training runs is itself a drift signal -- see `importance_shift`.
        """
        values = np.abs(self.shap_values(X)).mean(axis=0)
        return pd.Series(values, index=self.feature_names).sort_values(ascending=False)

    def verify_additivity(self, X: pd.DataFrame, margins: np.ndarray) -> None:
        """Assert `base_value + sum(shap) == raw margin` for every row.

        Cheap insurance against a version bump quietly changing the output
        convention (probability space, or the wrong class).
        """
        reconstructed = self.shap_values(X).sum(axis=1) + self.base_value
        error = np.abs(reconstructed - np.asarray(margins, dtype=float)).max()
        if error > ADDITIVITY_TOLERANCE:
            raise ExplainerError(
                f"SHAP additivity violated: max error {error:.2e} exceeds "
                f"{ADDITIVITY_TOLERANCE:.0e}. Attributions may be in the wrong "
                "space or for the wrong class."
            )

    def _check_columns(self, X: pd.DataFrame) -> None:
        if list(X.columns) != self.feature_names:
            raise ExplainerError(
                "feature columns do not match the explainer's; "
                f"expected {self.feature_names}, got {list(X.columns)}"
            )

    def save(self, path: Path) -> Path:
        """Persist the explainer next to its model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> FraudExplainer:
        """Load a persisted explainer."""
        return joblib.load(path)


def importance_shift(previous: pd.Series, current: pd.Series) -> float:
    """Total variation distance between two global-importance profiles.

    Both are normalised to sum to 1, then compared. Returns a value in `[0, 1]`:
    0 means the model attributes importance identically, 1 means it has moved to
    an entirely disjoint set of features.

    A large shift between consecutive training runs means *what drives the
    model* has changed, even if AUC has not. That is an early drift warning, and
    it is why Phase 6's monitor watches this alongside feature distributions.
    """
    features = sorted(set(previous.index) | set(current.index))
    prev = previous.reindex(features, fill_value=0.0).to_numpy(dtype=float)
    curr = current.reindex(features, fill_value=0.0).to_numpy(dtype=float)

    if prev.sum() <= 0 or curr.sum() <= 0:
        raise ValueError("importance profiles must have positive total mass")

    return float(np.abs(prev / prev.sum() - curr / curr.sum()).sum() / 2.0)
