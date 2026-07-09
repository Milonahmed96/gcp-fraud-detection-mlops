"""Tests for the two A/B model variants.

The A/B test only isolates the *algorithm* if everything else is held constant.
`TestFairComparison` is what enforces that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.models import (
    SHARED_HYPERPARAMETERS,
    VARIANTS,
    UnknownVariantError,
    build_model,
    hyperparameters_for,
    predict_fraud_probability,
)


@pytest.fixture(scope="module")
def toy_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + rng.normal(scale=0.3, size=n) > 1.0).astype(int).to_numpy()
    return X, y


class TestVariantRegistry:
    def test_exactly_two_variants(self):
        assert VARIANTS == ("xgboost", "lightgbm")

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_known_variants_build(self, variant):
        assert build_model(variant, scale_pos_weight=10.0) is not None

    def test_unknown_variant_is_rejected(self):
        with pytest.raises(UnknownVariantError, match="unknown variant 'catboost'"):
            build_model("catboost", scale_pos_weight=1.0)

    def test_hyperparameters_reject_a_bad_class_weight(self):
        with pytest.raises(ValueError, match="scale_pos_weight must be positive"):
            hyperparameters_for("xgboost", scale_pos_weight=0.0)


class TestFairComparison:
    """Anything that differs between variants beyond the algorithm is a confounder."""

    @pytest.mark.parametrize("key", ["n_estimators", "learning_rate", "max_depth", "random_state"])
    def test_shared_capacity_parameters_are_identical(self, key):
        xgb = hyperparameters_for("xgboost", scale_pos_weight=10.0)
        lgbm = hyperparameters_for("lightgbm", scale_pos_weight=10.0)
        assert xgb[key] == lgbm[key] == SHARED_HYPERPARAMETERS[key]

    def test_both_variants_get_the_same_class_weight(self):
        xgb = hyperparameters_for("xgboost", scale_pos_weight=42.5)
        lgbm = hyperparameters_for("lightgbm", scale_pos_weight=42.5)
        assert xgb["scale_pos_weight"] == lgbm["scale_pos_weight"] == 42.5

    def test_both_variants_are_seeded(self):
        for variant in VARIANTS:
            assert hyperparameters_for(variant, scale_pos_weight=1.0)["random_state"] == 42

    def test_lightgbm_leaves_are_capped_to_match_xgboost_depth(self):
        """LightGBM grows leaf-wise; uncapped it would be far higher-capacity."""
        lgbm = hyperparameters_for("lightgbm", scale_pos_weight=1.0)
        assert lgbm["num_leaves"] == 2 ** SHARED_HYPERPARAMETERS["max_depth"] - 1

    def test_neither_variant_optimises_accuracy(self):
        """On a 2% base rate, accuracy is a broken objective."""
        assert hyperparameters_for("xgboost", scale_pos_weight=1.0)["eval_metric"] == "aucpr"
        assert (
            hyperparameters_for("lightgbm", scale_pos_weight=1.0)["metric"] == "average_precision"
        )

    def test_hyperparameters_are_not_shared_mutable_state(self):
        """A caller mutating the returned dict must not poison the next call."""
        first = hyperparameters_for("xgboost", scale_pos_weight=1.0)
        first["n_estimators"] = 99999
        assert (
            hyperparameters_for("xgboost", scale_pos_weight=1.0)["n_estimators"]
            == (SHARED_HYPERPARAMETERS["n_estimators"])
        )


class TestPrediction:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_fit_then_predict_returns_calibrated_range(self, variant, toy_data):
        X, y = toy_data
        model = build_model(variant, scale_pos_weight=3.0)
        model.fit(X, y)
        proba = predict_fraud_probability(model, X)

        assert proba.shape == (len(X),)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_learns_the_signal(self, variant, toy_data):
        X, y = toy_data
        model = build_model(variant, scale_pos_weight=3.0)
        model.fit(X, y)
        proba = predict_fraud_probability(model, X)
        assert proba[y == 1].mean() > proba[y == 0].mean()

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_is_reproducible(self, variant, toy_data):
        X, y = toy_data
        first = build_model(variant, scale_pos_weight=3.0)
        second = build_model(variant, scale_pos_weight=3.0)
        first.fit(X, y)
        second.fit(X, y)
        np.testing.assert_allclose(
            predict_fraud_probability(first, X), predict_fraud_probability(second, X)
        )

    def test_returns_the_positive_class_column(self):
        """Silently taking column 0 would invert every prediction."""

        class TwoColumnModel:
            def fit(self, X, y): ...

            def predict_proba(self, X):
                return np.array([[0.9, 0.1], [0.2, 0.8]])

        proba = predict_fraud_probability(TwoColumnModel(), pd.DataFrame(index=[0, 1]))
        np.testing.assert_allclose(proba, [0.1, 0.8])

    def test_rejects_a_non_binary_probability_matrix(self):
        class ThreeClassModel:
            def fit(self, X, y): ...

            def predict_proba(self, X):
                return np.zeros((2, 3))

        with pytest.raises(ValueError, match=r"expected \(n, 2\) probabilities"):
            predict_fraud_probability(ThreeClassModel(), pd.DataFrame(index=[0, 1]))
