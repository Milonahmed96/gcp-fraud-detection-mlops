"""Tests for Vertex AI Experiments logging.

The governing rule: **experiment tracking must never fail a training job.** A
model that trained successfully but could not be logged is still a model. These
tests pin that behaviour, because the obvious implementation lets an SDK
exception escape and destroy an hour of GPU time.
"""

from __future__ import annotations

import pytest

from src.features.config import GCPConfig
from src.training import experiments
from src.training.metrics import evaluate
from src.training.train import ComparisonResult, TrainingResult

import numpy as np


@pytest.fixture
def config() -> GCPConfig:
    return GCPConfig("test-project", "europe-west2", "test-bucket", "fraud_features", "store")


@pytest.fixture
def result() -> TrainingResult:
    evaluation = evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), threshold=0.5)
    return TrainingResult(
        variant="xgboost",
        evaluation=evaluation,
        hyperparameters={"n_estimators": 300, "n_jobs": -1, "tree_method": "hist"},
        n_train=100,
        n_validation=20,
        n_test=30,
        train_fraud_rate=0.02,
    )


class FakeRun:
    def __init__(self):
        self.params: dict = {}
        self.metrics: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)


class FakeAIPlatform:
    def __init__(self, *, fail_on_init: bool = False):
        self.init_kwargs: dict | None = None
        self.run = FakeRun()
        self.run_name: str | None = None
        self._fail_on_init = fail_on_init

    def init(self, **kwargs):
        if self._fail_on_init:
            raise RuntimeError("vertex is unreachable")
        self.init_kwargs = kwargs

    def start_run(self, run):
        self.run_name = run
        return self.run


class TestStringifyParams:
    def test_scalars_pass_through(self):
        result = experiments._stringify_params({"a": 1, "b": 2.5, "c": "x", "d": True})
        assert result == {"a": 1, "b": 2.5, "c": "x", "d": True}

    def test_non_scalars_are_stringified(self):
        """Vertex AI rejects list/dict parameter values."""
        result = experiments._stringify_params({"leaves": [1, 2], "cfg": {"k": 1}})
        assert result == {"leaves": "[1, 2]", "cfg": "{'k': 1}"}


class TestLogTrainingRun:
    def test_logs_params_and_metrics(self, config, result, monkeypatch):
        fake = FakeAIPlatform()
        monkeypatch.setattr(experiments, "_aiplatform", lambda: fake)

        run_name = experiments.log_training_run(config, result)

        assert run_name == "xgboost-run"
        assert fake.run.params["n_estimators"] == 300
        assert fake.run.metrics["roc_auc"] == pytest.approx(1.0)
        assert "cost_per_1000" in fake.run.metrics

    def test_targets_the_configured_project_and_experiment(self, config, result, monkeypatch):
        fake = FakeAIPlatform()
        monkeypatch.setattr(experiments, "_aiplatform", lambda: fake)

        experiments.log_training_run(config, result, experiment_name="my-experiment")

        assert fake.init_kwargs["project"] == "test-project"
        assert fake.init_kwargs["location"] == "europe-west2"
        assert fake.init_kwargs["experiment"] == "my-experiment"

    def test_run_name_is_overridable(self, config, result, monkeypatch):
        fake = FakeAIPlatform()
        monkeypatch.setattr(experiments, "_aiplatform", lambda: fake)
        assert experiments.log_training_run(config, result, run_name="custom") == "custom"
        assert fake.run_name == "custom"

    def test_a_tracking_failure_never_raises(self, config, result, monkeypatch, caplog):
        """The load-bearing behaviour: a dead experiment tracker must not kill
        a training job that already succeeded."""
        monkeypatch.setattr(experiments, "_aiplatform", lambda: FakeAIPlatform(fail_on_init=True))

        assert experiments.log_training_run(config, result) is None
        assert "failed to log run" in caplog.text


class TestLogComparison:
    def test_logs_every_variant(self, config, result, monkeypatch):
        fake = FakeAIPlatform()
        monkeypatch.setattr(experiments, "_aiplatform", lambda: fake)

        comparison = ComparisonResult(
            winner="xgboost",
            cost_difference_per_1000=-12.0,
            confidence_interval=(-30.0, 5.0),
            significant=False,
            results={"xgboost": result, "lightgbm": result},
        )
        logged = experiments.log_comparison(config, comparison)
        assert len(logged) == 2

    def test_returns_only_the_runs_that_succeeded(self, config, result, monkeypatch):
        monkeypatch.setattr(experiments, "_aiplatform", lambda: FakeAIPlatform(fail_on_init=True))

        comparison = ComparisonResult(
            winner="xgboost",
            cost_difference_per_1000=0.0,
            confidence_interval=(-1.0, 1.0),
            significant=False,
            results={"xgboost": result},
        )
        assert experiments.log_comparison(config, comparison) == []
