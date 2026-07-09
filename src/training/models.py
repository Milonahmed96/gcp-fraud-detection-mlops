"""The two A/B model variants.

For the A/B test to say anything about *algorithms*, everything else must be
held constant: same features, same temporal split, same class weighting, same
seed, comparable capacity (depth, trees, learning rate). Otherwise a win could
just mean one variant got more trees.

Both are gradient-boosted tree ensembles, which is also what makes SHAP's exact
`TreeExplainer` usable on both in Phase 4 -- a genuine constraint on this
choice, not an afterthought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

Variant = Literal["xgboost", "lightgbm"]

VARIANTS: tuple[Variant, ...] = ("xgboost", "lightgbm")

#: Held identical across variants so the comparison isolates the algorithm.
SHARED_HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


class FittedModel(Protocol):
    """The slice of the sklearn estimator API we depend on."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class UnknownVariantError(ValueError):
    """Raised when a variant name is not one of `VARIANTS`."""


@dataclass(frozen=True)
class ModelSpec:
    """A variant's identity and its resolved hyperparameters."""

    variant: Variant
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    @property
    def cloud_run_revision(self) -> str:
        """The Cloud Run revision suffix this variant is served under."""
        return {"xgboost": "xgb", "lightgbm": "lgbm"}[self.variant]


def _check_variant(variant: str) -> Variant:
    if variant not in VARIANTS:
        raise UnknownVariantError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    return variant  # type: ignore[return-value]


def hyperparameters_for(variant: str, *, scale_pos_weight: float) -> dict[str, Any]:
    """Resolve the full hyperparameter set for a variant.

    `scale_pos_weight` is computed from the training split and passed to both
    libraries under the same name -- they happen to agree on it, which is one
    fewer confounder.
    """
    variant = _check_variant(variant)
    if scale_pos_weight <= 0:
        raise ValueError(f"scale_pos_weight must be positive; got {scale_pos_weight}")

    params = dict(SHARED_HYPERPARAMETERS)
    params["scale_pos_weight"] = scale_pos_weight

    if variant == "xgboost":
        params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",  # PR-AUC, not accuracy: the class is rare
                "tree_method": "hist",
                "n_jobs": -1,
            }
        )
    else:
        params.update(
            {
                "objective": "binary",
                "metric": "average_precision",
                "n_jobs": -1,
                "verbose": -1,  # LightGBM is chatty by default
                # LightGBM grows leaf-wise; cap the leaves so its effective
                # capacity matches XGBoost's depth-wise 2**max_depth.
                "num_leaves": 2 ** SHARED_HYPERPARAMETERS["max_depth"] - 1,
                "min_child_samples": 20,
            }
        )
    return params


def build_model(variant: str, *, scale_pos_weight: float) -> FittedModel:
    """Construct an unfitted classifier for the given variant.

    Libraries are imported lazily so that `metrics.py` and `dataset.py` stay
    importable without xgboost/lightgbm on the path.
    """
    variant = _check_variant(variant)
    params = hyperparameters_for(variant, scale_pos_weight=scale_pos_weight)

    if variant == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**params)

    from lightgbm import LGBMClassifier

    return LGBMClassifier(**params)


def predict_fraud_probability(model: FittedModel, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive (fraud) class, as a 1-D array.

    Both libraries return an `(n, 2)` array from `predict_proba`; column 1 is
    the positive class. Wrapping this once stops the index being wrong somewhere.
    """
    proba = np.asarray(model.predict_proba(X))
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"expected (n, 2) probabilities; got shape {proba.shape}")
    return proba[:, 1]
