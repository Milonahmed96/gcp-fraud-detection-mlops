"""Tests for the training orchestrator.

These actually fit both variants against the committed sample, so they are the
slowest tests in the suite -- and the ones that would catch a real regression in
the pipeline. Bootstrap resamples are kept low; `test_metrics.py` covers the
statistics properly.
"""

from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pytest

from src.features.config import GCPConfig
from src.training.dataset import load_features, temporal_split
from src.training.metrics import CostModel
from src.training.models import VARIANTS, predict_fraud_probability
from src.training.train import (
    compare_variants,
    run_local_training,
    save_model,
    train_variant,
)


@pytest.fixture(scope="module")
def dataset():
    return temporal_split(load_features("sample"))


@pytest.fixture(scope="module")
def trained(dataset):
    model, result = train_variant(dataset, "xgboost")
    return model, result


@pytest.fixture(scope="module")
def comparison(dataset):
    """Trains both variants once; reused across the comparison assertions."""
    return compare_variants(dataset, n_resamples=50)


class TestTrainVariant:
    def test_produces_a_fitted_model_and_a_result(self, trained):
        model, result = trained
        assert result.variant == "xgboost"
        assert hasattr(model, "predict_proba")

    def test_result_records_the_split_sizes(self, trained, dataset):
        _, result = trained
        assert result.n_train == len(dataset.train)
        assert result.n_test == len(dataset.test)
        assert 0 < result.train_fraud_rate < 0.1

    def test_the_model_beats_random(self, trained):
        _, result = trained
        assert result.evaluation.roc_auc > 0.6
        assert result.evaluation.pr_auc > result.train_fraud_rate  # beats the base rate

    def test_the_problem_is_not_trivially_separable(self, trained):
        """A perfect score would mean the sample leaks the label."""
        _, result = trained
        assert result.evaluation.roc_auc < 0.999

    def test_threshold_comes_from_validation_not_test(self, dataset):
        """Refitting the threshold on test can only ever lower the test cost.
        If it does not, the threshold was already tuned on test -- i.e. leaked."""
        from src.training.metrics import evaluate, optimal_threshold

        model, result = train_variant(dataset, "xgboost")
        test_scores = predict_fraud_probability(model, dataset.test.X)

        cheating_threshold = optimal_threshold(
            dataset.test.y, test_scores, amounts=dataset.test.amounts
        )
        cheating_cost = evaluate(
            dataset.test.y,
            test_scores,
            threshold=cheating_threshold,
            amounts=dataset.test.amounts,
        ).total_cost

        assert cheating_cost <= result.evaluation.total_cost

    def test_hyperparameters_are_recorded_for_the_experiment_log(self, trained):
        _, result = trained
        assert result.hyperparameters["n_estimators"] == 300
        assert result.hyperparameters["scale_pos_weight"] > 1

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_both_variants_train(self, dataset, variant):
        _, result = train_variant(dataset, variant)
        assert result.evaluation.roc_auc > 0.5

    def test_a_harsher_false_positive_cost_blocks_fewer_customers(self, dataset):
        """Sanity: the cost model actually steers the decision threshold."""
        _, cheap = train_variant(dataset, "xgboost", cost_model=CostModel(false_positive_cost=0.5))
        _, dear = train_variant(dataset, "xgboost", cost_model=CostModel(false_positive_cost=500.0))
        assert dear.evaluation.threshold >= cheap.evaluation.threshold
        assert dear.evaluation.false_positives <= cheap.evaluation.false_positives


class TestSaveModel:
    def test_round_trips_through_joblib(self, trained, tmp_path, dataset):
        model, _ = trained
        path = save_model(model, "xgboost", tmp_path)
        assert path.exists()

        reloaded = joblib.load(path)
        np.testing.assert_allclose(
            predict_fraud_probability(model, dataset.test.X),
            predict_fraud_probability(reloaded, dataset.test.X),
        )

    def test_creates_the_output_directory(self, trained, tmp_path):
        model, _ = trained
        path = save_model(model, "xgboost", tmp_path / "nested" / "dir")
        assert path.exists()


class TestCompareVariants:
    def test_trains_every_variant(self, comparison):
        assert set(comparison.results) == set(VARIANTS)

    def test_winner_is_one_of_the_variants(self, comparison):
        assert comparison.winner in VARIANTS

    def test_confidence_interval_brackets_the_point_estimate(self, comparison):
        low, high = comparison.confidence_interval
        assert low <= comparison.cost_difference_per_1000 <= high

    def test_significance_is_interval_excludes_zero(self, comparison):
        low, high = comparison.confidence_interval
        assert comparison.significant == (not (low <= 0.0 <= high))

    def test_an_inconclusive_result_keeps_the_incumbent(self, comparison):
        """When the interval straddles zero we must not ship a coin flip."""
        if not comparison.significant:
            assert comparison.winner == "xgboost"

    def test_winner_has_the_lower_cost_when_significant(self, comparison):
        if comparison.significant:
            costs = {k: v.evaluation.cost_per_1000 for k, v in comparison.results.items()}
            assert comparison.winner == min(costs, key=costs.get)

    def test_summary_mentions_the_verdict(self, comparison):
        summary = comparison.summary()
        assert comparison.winner in summary
        assert "significant" in summary

    def test_both_variants_saw_identical_training_data(self, comparison):
        """If the splits differed, the comparison would be meaningless."""
        sizes = {(r.n_train, r.n_validation, r.n_test) for r in comparison.results.values()}
        assert len(sizes) == 1

    def test_saves_model_artefacts_when_given_an_output_dir(self, dataset, tmp_path):
        result = compare_variants(dataset, output_dir=tmp_path, n_resamples=10)
        for variant in VARIANTS:
            assert (tmp_path / f"model_{variant}.joblib").exists()
            assert result.results[variant].model_path is not None

    def test_skips_artefacts_without_an_output_dir(self, comparison):
        assert all(r.model_path is None for r in comparison.results.values())


class TestRunLocalTraining:
    def test_writes_a_metrics_json_with_the_verdict(self, tmp_path):
        comparison = run_local_training(output_dir=tmp_path, n_resamples=10)
        payload = json.loads((tmp_path / "metrics.json").read_text())

        assert payload["winner"] == comparison.winner
        assert set(payload["variants"]) == set(VARIANTS)
        assert len(payload["confidence_interval"]) == 2
        assert "cost_per_1000" in payload["variants"]["xgboost"]["evaluation"]

    def test_metrics_json_is_serialisable_without_numpy_types(self, tmp_path):
        run_local_training(output_dir=tmp_path, n_resamples=10)
        json.loads((tmp_path / "metrics.json").read_text())  # would raise on np.float64


class TestVertexBackendCLI:
    """The `--backend vertex` branch of main(), with the SDK stubbed out."""

    @pytest.fixture
    def submitted(self, monkeypatch):
        from src.features import config as config_module
        from src.training import train as train_module
        from src.training import vertex as vertex_module

        captured: dict = {}

        def fake_submit(config, *, source="bigquery", args=None, **kwargs):
            captured["source"] = source
            captured["args"] = args
            return "fake-job"

        monkeypatch.setattr(vertex_module, "submit_training_job", fake_submit)
        monkeypatch.setattr(
            config_module,
            "load_config",
            lambda: GCPConfig("p", "europe-west2", "b", "d", "f"),
        )
        return captured, train_module

    def test_forwards_flags_when_argv_is_passed_explicitly(self, submitted):
        captured, train_module = submitted
        assert train_module.main(["--backend", "vertex", "--bootstrap-resamples", "50"]) == 0
        assert "--bootstrap-resamples" in captured["args"]

    def test_forwards_flags_when_argv_comes_from_the_real_command_line(
        self, submitted, monkeypatch
    ):
        """Regression: `main()` used to forward `argv or []`, so a real terminal
        invocation (argv=None) silently dropped every flag on the remote run."""
        captured, train_module = submitted
        monkeypatch.setattr(
            sys, "argv", ["train.py", "--backend", "vertex", "--bootstrap-resamples", "50"]
        )

        assert train_module.main() == 0
        assert "--bootstrap-resamples" in captured["args"]
        assert "50" in captured["args"]

    def test_does_not_train_locally_when_submitting_to_vertex(self, submitted, monkeypatch):
        captured, train_module = submitted
        monkeypatch.setattr(
            train_module,
            "run_local_training",
            lambda **kw: pytest.fail("must not train locally on the vertex backend"),
        )
        assert train_module.main(["--backend", "vertex"]) == 0
