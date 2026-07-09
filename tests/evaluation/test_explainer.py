"""Tests for SHAP explainability.

Two classes carry the weight:

* `TestNormalisation` -- pure-data tests over every SHAP output shape. Taking the
  wrong class axis inverts the sign of every explanation and nothing raises.
* `TestAdditivityAgainstRealModels` -- fits both real variants and asserts the
  SHAP identity `base + sum(shap) == raw margin` holds. This is what catches a
  shap version bump that changes the output convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.explainer import (
    ADDITIVITY_TOLERANCE,
    Explanation,
    ExplainerError,
    FeatureAttribution,
    FraudExplainer,
    importance_shift,
    normalise_base_value,
    normalise_shap_values,
)
from src.training.models import VARIANTS, build_model


@pytest.fixture(scope="module")
def toy() -> tuple[pd.DataFrame, np.ndarray]:
    """Three features so the feature axis cannot be confused with a binary class axis."""
    rng = np.random.default_rng(7)
    n = 300
    X = pd.DataFrame(
        {
            "amount_log": rng.normal(size=n),
            "is_foreign": rng.integers(0, 2, size=n).astype(float),
            "txn_count_24h": rng.normal(size=n),
        }
    )
    # is_foreign is the dominant driver, by construction.
    logit = 2.5 * X["is_foreign"] + 0.4 * X["amount_log"]
    y = (logit + rng.normal(scale=0.3, size=n) > 1.2).astype(int).to_numpy()
    return X, y


@pytest.fixture(scope="module", params=VARIANTS)
def fitted(request, toy):
    X, y = toy
    model = build_model(request.param, scale_pos_weight=2.0)
    model.fit(X, y)
    return request.param, model, X, y


@pytest.fixture(scope="module")
def explainer(fitted) -> FraudExplainer:
    _, model, X, _ = fitted
    return FraudExplainer.from_model(model, list(X.columns))


def raw_margin(variant: str, model, X: pd.DataFrame) -> np.ndarray:
    """The ensemble's pre-sigmoid output, however the library spells it."""
    if variant == "xgboost":
        return np.asarray(model.predict(X, output_margin=True), dtype=float)
    return np.asarray(model.predict_proba(X, raw_score=True), dtype=float)


class TestNormalisation:
    def test_two_dimensional_output_passes_through(self):
        values = np.arange(6, dtype=float).reshape(2, 3)
        np.testing.assert_array_equal(normalise_shap_values(values), values)

    def test_per_class_list_takes_the_fraud_class(self):
        """Older LightGBM returns [class_0, class_1]. Taking [0] inverts every sign."""
        genuine = np.zeros((2, 3))
        fraud = np.ones((2, 3))
        np.testing.assert_array_equal(normalise_shap_values([genuine, fraud]), fraud)

    def test_three_dimensional_output_takes_the_trailing_fraud_axis(self):
        values = np.zeros((4, 3, 2))
        values[:, :, 1] = 5.0
        np.testing.assert_array_equal(normalise_shap_values(values), np.full((4, 3), 5.0))

    def test_rejects_a_multiclass_list(self):
        with pytest.raises(ExplainerError, match="expected 2 per-class arrays; got 3"):
            normalise_shap_values([np.zeros((2, 2))] * 3)

    def test_rejects_a_multiclass_trailing_axis(self):
        with pytest.raises(ExplainerError, match="expected a binary class axis"):
            normalise_shap_values(np.zeros((4, 3, 5)))

    def test_rejects_a_one_dimensional_array(self):
        with pytest.raises(ExplainerError, match="cannot interpret SHAP values"):
            normalise_shap_values(np.zeros(5))

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(0.5, 0.5), (np.float32(-1.25), -1.25), ([2.0], 2.0), ([-1.0, 3.0], 3.0)],
    )
    def test_base_value_collapses_to_the_fraud_class(self, raw, expected):
        assert normalise_base_value(raw) == pytest.approx(expected)

    def test_rejects_a_multiclass_base_value(self):
        with pytest.raises(ExplainerError, match="cannot interpret expected_value"):
            normalise_base_value([0.1, 0.2, 0.3])


class TestFeatureAttribution:
    @pytest.mark.parametrize(
        ("shap_value", "direction"),
        [(1.5, "toward_fraud"), (-1.5, "toward_genuine"), (0.0, "neutral")],
    )
    def test_direction_reads_the_sign(self, shap_value, direction):
        assert FeatureAttribution("f", 1.0, shap_value).direction == direction


class TestExplanation:
    def _explanation(self) -> Explanation:
        return Explanation(
            base_value=-2.0,
            attributions=(
                FeatureAttribution("a", 1.0, 0.5),
                FeatureAttribution("b", 2.0, -3.0),
                FeatureAttribution("c", 3.0, 1.0),
            ),
            probability=0.3,
        )

    def test_margin_is_base_plus_the_attributions(self):
        assert self._explanation().margin == pytest.approx(-2.0 + 0.5 - 3.0 + 1.0)

    def test_top_contributions_rank_by_absolute_effect(self):
        """A strongly exculpatory feature matters to an auditor as much as an
        incriminating one."""
        top = self._explanation().top_contributions(2)
        assert [a.feature for a in top] == ["b", "c"]

    def test_top_contributions_are_capped_at_k(self):
        assert len(self._explanation().top_contributions(2)) == 2

    def test_k_larger_than_the_feature_set_returns_everything(self):
        assert len(self._explanation().top_contributions(99)) == 3

    def test_as_dict_is_audit_shaped(self):
        payload = self._explanation().as_dict(k=2)
        assert payload["probability"] == 0.3
        assert payload["margin"] == pytest.approx(-3.5)
        assert payload["top_features"][0] == {
            "feature": "b",
            "value": 2.0,
            "shap_value": -3.0,
            "direction": "toward_genuine",
        }


class TestAdditivityAgainstRealModels:
    """The identity that proves the attributions are in the right space and class."""

    def test_shap_values_have_one_column_per_feature(self, explainer, fitted):
        _, _, X, _ = fitted
        assert explainer.shap_values(X).shape == X.shape

    def test_base_value_is_a_scalar(self, explainer):
        assert isinstance(explainer.base_value, float)

    def test_additivity_holds(self, explainer, fitted):
        variant, model, X, _ = fitted
        explainer.verify_additivity(X, raw_margin(variant, model, X))

    def test_additivity_is_actually_checked(self, explainer, fitted):
        """The guard must fail on a wrong margin, or it proves nothing."""
        _, _, X, _ = fitted
        with pytest.raises(ExplainerError, match="additivity violated"):
            explainer.verify_additivity(X, np.full(len(X), 999.0))

    def test_attributions_sum_to_the_margin_per_row(self, explainer, fitted):
        variant, model, X, _ = fitted
        margins = raw_margin(variant, model, X)
        for i, explanation in enumerate(explainer.explain(X.head(10))):
            assert explanation.margin == pytest.approx(margins[i], abs=ADDITIVITY_TOLERANCE)

    def test_the_dominant_feature_is_attributed_the_most(self, explainer, fitted):
        """`is_foreign` drives the toy label; SHAP must say so."""
        _, _, X, _ = fitted
        assert explainer.global_importance(X).index[0] == "is_foreign"

    def test_attributions_point_toward_fraud_for_fraudulent_looking_rows(self, explainer, fitted):
        """Sign check. If the class axis were inverted this would flip."""
        _, _, X, _ = fitted
        foreign = X[X["is_foreign"] == 1.0]
        values = explainer.shap_values(foreign)
        column = list(X.columns).index("is_foreign")
        assert values[:, column].mean() > 0


class TestExplainAPI:
    def test_explain_returns_one_explanation_per_row(self, explainer, fitted):
        _, _, X, _ = fitted
        assert len(explainer.explain(X.head(5))) == 5

    def test_every_feature_is_attributed(self, explainer, fitted):
        _, _, X, _ = fitted
        explanation = explainer.explain(X.head(1))[0]
        assert [a.feature for a in explanation.attributions] == list(X.columns)

    def test_attribution_values_match_the_input_row(self, explainer, fitted):
        _, _, X, _ = fitted
        row = X.head(1)
        explanation = explainer.explain(row)[0]
        for attribution in explanation.attributions:
            assert attribution.value == pytest.approx(row.iloc[0][attribution.feature])

    def test_probability_matches_the_model(self, explainer, fitted):
        _, model, X, _ = fitted
        expected = model.predict_proba(X.head(3))[:, 1]
        actual = [e.probability for e in explainer.explain(X.head(3))]
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_explain_one_accepts_a_single_row_frame(self, explainer, fitted):
        _, _, X, _ = fitted
        assert isinstance(explainer.explain_one(X.head(1)), Explanation)

    def test_explain_one_accepts_a_series(self, explainer, fitted):
        _, _, X, _ = fitted
        assert isinstance(explainer.explain_one(X.iloc[0]), Explanation)

    def test_explain_one_rejects_multiple_rows(self, explainer, fitted):
        _, _, X, _ = fitted
        with pytest.raises(ExplainerError, match="expects exactly one row"):
            explainer.explain_one(X.head(2))

    def test_mismatched_columns_are_rejected(self, explainer, fitted):
        """Silently reordering columns would scramble every attribution."""
        _, _, X, _ = fitted
        scrambled = X[list(X.columns)[::-1]]
        with pytest.raises(ExplainerError, match="feature columns do not match"):
            explainer.shap_values(scrambled)

    def test_global_importance_is_non_negative_and_sorted(self, explainer, fitted):
        _, _, X, _ = fitted
        importance = explainer.global_importance(X)
        assert (importance >= 0).all()
        assert importance.is_monotonic_decreasing
        assert list(importance.index.sort_values()) == sorted(X.columns)


class TestPersistence:
    def test_round_trips_and_still_explains_identically(self, explainer, fitted, tmp_path):
        _, _, X, _ = fitted
        path = explainer.save(tmp_path / "nested" / "explainer.joblib")
        assert path.exists()

        reloaded = FraudExplainer.load(path)
        np.testing.assert_allclose(
            reloaded.shap_values(X.head(5)), explainer.shap_values(X.head(5))
        )
        assert reloaded.feature_names == explainer.feature_names


class TestImportanceShift:
    def test_identical_profiles_have_zero_shift(self):
        profile = pd.Series({"a": 1.0, "b": 3.0})
        assert importance_shift(profile, profile) == pytest.approx(0.0)

    def test_scale_invariant(self):
        """Only the *distribution* of importance matters, not its magnitude."""
        a = pd.Series({"a": 1.0, "b": 3.0})
        assert importance_shift(a, a * 100) == pytest.approx(0.0)

    def test_disjoint_profiles_shift_by_one(self):
        a = pd.Series({"a": 1.0, "b": 0.0})
        b = pd.Series({"a": 0.0, "b": 1.0})
        assert importance_shift(a, b) == pytest.approx(1.0)

    def test_partial_shift_is_between_zero_and_one(self):
        a = pd.Series({"a": 0.5, "b": 0.5})
        b = pd.Series({"a": 0.75, "b": 0.25})
        assert importance_shift(a, b) == pytest.approx(0.25)

    def test_features_absent_from_one_profile_are_treated_as_zero(self):
        a = pd.Series({"a": 1.0})
        b = pd.Series({"a": 1.0, "b": 1.0})
        assert importance_shift(a, b) == pytest.approx(0.5)

    def test_is_symmetric(self):
        a = pd.Series({"a": 2.0, "b": 1.0, "c": 1.0})
        b = pd.Series({"a": 1.0, "b": 1.0, "c": 2.0})
        assert importance_shift(a, b) == pytest.approx(importance_shift(b, a))

    def test_rejects_an_empty_profile(self):
        with pytest.raises(ValueError, match="positive total mass"):
            importance_shift(pd.Series({"a": 0.0}), pd.Series({"a": 1.0}))
