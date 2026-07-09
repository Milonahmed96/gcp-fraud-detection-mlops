"""Tests for the evaluation metrics.

The business cost metric decides the A/B test, so it gets hand-computed
expected values rather than golden numbers copied from a run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.metrics import (
    CostModel,
    bootstrap_cost_difference,
    business_cost,
    evaluate,
    optimal_threshold,
)


class TestCostModel:
    def test_rejects_negative_costs(self):
        with pytest.raises(ValueError, match="costs must be non-negative"):
            CostModel(false_positive_cost=-1.0)

    def test_is_immutable(self):
        with pytest.raises(Exception):
            CostModel().false_positive_cost = 99.0  # type: ignore[misc]


class TestBusinessCost:
    def test_perfect_predictions_cost_nothing(self):
        y = np.array([0, 1, 0, 1])
        assert business_cost(y, y, amounts=np.array([10.0, 500.0, 20.0, 300.0])) == 0.0

    def test_missed_fraud_costs_the_transaction_amount(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 0])  # missed the fraud
        amounts = np.array([10.0, 500.0])
        assert business_cost(y_true, y_pred, amounts=amounts) == pytest.approx(500.0)

    def test_blocked_genuine_costs_the_fixed_fee(self):
        y_true = np.array([0, 1])
        y_pred = np.array([1, 1])  # blocked the genuine one
        amounts = np.array([10.0, 500.0])
        cost = business_cost(y_true, y_pred, amounts=amounts, cost_model=CostModel(7.0))
        assert cost == pytest.approx(7.0)

    def test_the_two_errors_are_priced_asymmetrically(self):
        """The whole reason this metric exists: a missed 500 fraud is not a
        blocked 500 payment."""
        amounts = np.array([500.0, 500.0])
        miss = business_cost(np.array([0, 1]), np.array([0, 0]), amounts=amounts)
        block = business_cost(np.array([0, 1]), np.array([1, 1]), amounts=amounts)
        assert miss > block

    def test_falls_back_to_a_flat_cost_without_amounts(self):
        cost = business_cost(
            np.array([1, 1, 0]),
            np.array([0, 0, 1]),
            cost_model=CostModel(false_positive_cost=2.0, false_negative_cost=100.0),
        )
        assert cost == pytest.approx(2 * 100.0 + 1 * 2.0)

    def test_rejects_mismatched_amounts(self):
        with pytest.raises(ValueError, match="amounts must have the same shape"):
            business_cost(np.array([0, 1]), np.array([0, 1]), amounts=np.array([1.0]))

    def test_predicting_all_fraud_costs_only_false_positives(self):
        y_true = np.array([0, 0, 0, 1])
        cost = business_cost(
            y_true, np.ones(4, dtype=int), amounts=np.array([1.0, 2.0, 3.0, 900.0])
        )
        assert cost == pytest.approx(3 * 5.0)


class TestOptimalThreshold:
    def test_picks_the_cost_minimising_threshold(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        amounts = np.array([10.0, 10.0, 1000.0, 1000.0])
        threshold = optimal_threshold(y_true, y_score, amounts=amounts)
        predictions = (y_score >= threshold).astype(int)
        assert business_cost(y_true, predictions, amounts=amounts) == 0.0

    def test_expensive_fraud_pushes_the_threshold_down(self):
        """When misses are costly, block more aggressively."""
        y_true = np.array([0, 0, 0, 1])
        y_score = np.array([0.1, 0.4, 0.45, 0.5])

        cheap = optimal_threshold(
            y_true, y_score, cost_model=CostModel(false_positive_cost=50.0, false_negative_cost=1.0)
        )
        costly = optimal_threshold(
            y_true,
            y_score,
            cost_model=CostModel(false_positive_cost=1.0, false_negative_cost=500.0),
        )
        assert costly <= cheap

    def test_when_fraud_is_worthless_it_blocks_nothing(self):
        y_true = np.array([0, 0, 1])
        y_score = np.array([0.2, 0.3, 0.9])
        threshold = optimal_threshold(
            y_true, y_score, cost_model=CostModel(false_positive_cost=10.0, false_negative_cost=0.0)
        )
        assert (y_score >= threshold).sum() == 0

    def test_returned_threshold_is_achievable(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=50)
        y_score = rng.random(50)
        threshold = optimal_threshold(y_true, y_score)
        assert 0.0 <= threshold <= np.nextafter(1.0, np.inf)


class TestEvaluate:
    def test_perfect_separation_scores_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.01, 0.02, 0.98, 0.99])
        result = evaluate(y_true, y_score, threshold=0.5)
        assert result.roc_auc == pytest.approx(1.0)
        assert result.pr_auc == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.false_negatives == 0 and result.false_positives == 0

    def test_confusion_counts_are_right(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.1, 0.8, 0.2])
        result = evaluate(y_true, y_score, threshold=0.5)
        assert (result.true_positives, result.false_negatives, result.false_positives) == (1, 1, 1)

    def test_cost_per_1000_scales_with_the_population(self):
        y_true = np.array([0, 1])
        y_score = np.array([0.9, 0.1])  # one FP, one FN
        result = evaluate(y_true, y_score, threshold=0.5, amounts=np.array([10.0, 90.0]))
        assert result.total_cost == pytest.approx(90.0 + 5.0)
        assert result.cost_per_1000 == pytest.approx(95.0 / 2 * 1000)

    def test_single_class_gives_nan_auc_not_a_crash(self):
        """Rare, but a degenerate slice must not take the training job down."""
        result = evaluate(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), threshold=0.5)
        assert np.isnan(result.roc_auc) and np.isnan(result.pr_auc)

    def test_as_dict_is_flat_and_numeric(self):
        result = evaluate(np.array([0, 1]), np.array([0.1, 0.9]), threshold=0.5)
        payload = result.as_dict()
        assert all(isinstance(v, float) for v in payload.values())
        assert "cost_per_1000" in payload and "pr_auc" in payload

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty prediction set"):
            evaluate(np.array([]), np.array([]), threshold=0.5)

    def test_rejects_non_binary_labels(self):
        with pytest.raises(ValueError, match="only 0 and 1"):
            evaluate(np.array([0, 2]), np.array([0.1, 0.9]), threshold=0.5)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate(np.array([0, 1]), np.array([0.5]), threshold=0.5)


class TestBootstrapCostDifference:
    def test_identical_models_have_zero_difference(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=200)
        score = rng.random(200)
        point, low, high = bootstrap_cost_difference(
            y_true, score, score, threshold_a=0.5, threshold_b=0.5, n_resamples=100
        )
        assert point == pytest.approx(0.0)
        assert low == pytest.approx(0.0) and high == pytest.approx(0.0)

    def test_a_clearly_worse_model_has_a_positive_difference(self):
        """`point > 0` means A costs more than B, so B wins."""
        y_true = np.array([0, 1] * 100)
        good = np.tile([0.1, 0.9], 100)  # perfectly separates
        bad = np.tile([0.9, 0.1], 100)  # perfectly inverted
        point, low, high = bootstrap_cost_difference(
            y_true, bad, good, threshold_a=0.5, threshold_b=0.5, n_resamples=200
        )
        assert point > 0
        assert low > 0  # unambiguous: interval excludes zero

    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, size=150)
        point, low, high = bootstrap_cost_difference(
            y_true,
            rng.random(150),
            rng.random(150),
            threshold_a=0.5,
            threshold_b=0.5,
            n_resamples=200,
        )
        assert low <= point <= high

    def test_is_deterministic_under_a_fixed_seed(self):
        rng = np.random.default_rng(4)
        y_true, a, b = rng.integers(0, 2, size=80), rng.random(80), rng.random(80)
        kwargs = dict(threshold_a=0.5, threshold_b=0.5, n_resamples=50, seed=7)
        assert bootstrap_cost_difference(y_true, a, b, **kwargs) == bootstrap_cost_difference(
            y_true, a, b, **kwargs
        )

    def test_wider_confidence_gives_a_wider_interval(self):
        rng = np.random.default_rng(5)
        y_true, a, b = rng.integers(0, 2, size=120), rng.random(120), rng.random(120)
        base = dict(threshold_a=0.5, threshold_b=0.5, n_resamples=300, seed=1)
        _, low_80, high_80 = bootstrap_cost_difference(y_true, a, b, confidence=0.80, **base)
        _, low_99, high_99 = bootstrap_cost_difference(y_true, a, b, confidence=0.99, **base)
        assert (high_99 - low_99) >= (high_80 - low_80)
