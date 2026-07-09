"""Tests for SHAP logging to Vertex AI Experiments and BigQuery.

The governing rule mirrors `src/training/experiments.py`: **logging must never
fail the thing it is logging.** By the time an explanation is written the
customer's transaction has already been approved or blocked.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.evaluation import experiments
from src.evaluation.explainer import Explanation, FeatureAttribution
from src.features.config import GCPConfig


@pytest.fixture
def config() -> GCPConfig:
    return GCPConfig("test-project", "europe-west2", "test-bucket", "fraud_features", "store")


@pytest.fixture
def importance() -> pd.Series:
    return pd.Series({"is_foreign": 0.8, "amount_log": 0.5, "hour_of_day": 0.1})


def make_explanation(probability: float = 0.9) -> Explanation:
    return Explanation(
        base_value=-3.0,
        attributions=(
            FeatureAttribution("is_foreign", 1.0, 2.5),
            FeatureAttribution("amount_log", 6.2, 1.1),
            FeatureAttribution("hour_of_day", 3.0, -0.4),
        ),
        probability=probability,
    )


class FakeRun:
    def __init__(self):
        self.metrics: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def log_metrics(self, metrics):
        self.metrics.update(metrics)


class FakeAIPlatform:
    def __init__(self, *, fail: bool = False):
        self.init_kwargs: dict | None = None
        self.start_run_kwargs: dict | None = None
        self.run = FakeRun()
        self._fail = fail

    def init(self, **kwargs):
        if self._fail:
            raise RuntimeError("vertex unreachable")
        self.init_kwargs = kwargs

    def start_run(self, **kwargs):
        self.start_run_kwargs = kwargs
        return self.run


class FakeJob:
    def result(self):
        return self


class FakeBQClient:
    def __init__(self, *, fail: bool = False):
        self.loaded: list = []
        self._fail = fail

    def load_table_from_dataframe(self, df, table_ref):
        if self._fail:
            raise RuntimeError("bigquery unreachable")
        self.loaded.append((df, table_ref))
        return FakeJob()


class TestLogGlobalImportance:
    def test_logs_one_metric_per_feature(self, config, importance, monkeypatch):
        fake = FakeAIPlatform()
        monkeypatch.setattr(experiments, "_aiplatform", lambda: fake)

        assert experiments.log_global_importance(config, "xgboost", importance) == "xgboost-run"
        assert fake.run.metrics == {
            "shap_importance__is_foreign": 0.8,
            "shap_importance__amount_log": 0.5,
            "shap_importance__hour_of_day": 0.1,
        }

    def test_metrics_are_prefixed_to_avoid_colliding_with_evaluation_metrics(
        self, config, importance, monkeypatch
    ):
        """`src/training/experiments.py` logs `roc_auc` etc. onto the same run."""
        fake = FakeAIPlatform()
        monkeypatch.setattr(experiments, "_aiplatform", lambda: fake)
        experiments.log_global_importance(config, "xgboost", importance)
        assert all(k.startswith("shap_importance__") for k in fake.run.metrics)

    def test_resumes_the_run_created_by_the_training_job(self, config, importance, monkeypatch):
        """SHAP artefacts must land on the run holding the metrics they explain."""
        fake = FakeAIPlatform()
        monkeypatch.setattr(experiments, "_aiplatform", lambda: fake)
        experiments.log_global_importance(config, "lightgbm", importance)
        assert fake.start_run_kwargs == {"run": "lightgbm-run", "resume": True}

    def test_targets_the_configured_project_and_experiment(self, config, importance, monkeypatch):
        fake = FakeAIPlatform()
        monkeypatch.setattr(experiments, "_aiplatform", lambda: fake)
        experiments.log_global_importance(config, "xgboost", importance, experiment_name="exp")
        assert fake.init_kwargs == {
            "project": "test-project",
            "location": "europe-west2",
            "experiment": "exp",
        }

    def test_a_tracking_failure_never_raises(self, config, importance, monkeypatch, caplog):
        monkeypatch.setattr(experiments, "_aiplatform", lambda: FakeAIPlatform(fail=True))
        assert experiments.log_global_importance(config, "xgboost", importance) is None
        assert "failed to log SHAP importance" in caplog.text


class TestExplanationRows:
    def test_one_row_per_explanation(self):
        rows = experiments.explanation_rows(["t1", "t2"], "xgboost", [make_explanation()] * 2)
        assert len(rows) == 2
        assert list(rows["transaction_id"]) == ["t1", "t2"]

    def test_carries_the_serving_variant_for_the_ab_test(self):
        rows = experiments.explanation_rows(["t1"], "lightgbm", [make_explanation()])
        assert rows["variant"].iloc[0] == "lightgbm"

    def test_top_features_are_json_and_ranked_by_absolute_effect(self):
        rows = experiments.explanation_rows(["t1"], "xgboost", [make_explanation()], top_k=2)
        top = json.loads(rows["top_features"].iloc[0])
        assert [f["feature"] for f in top] == ["is_foreign", "amount_log"]
        assert top[0]["direction"] == "toward_fraud"

    def test_top_k_bounds_the_row_size(self):
        rows = experiments.explanation_rows(["t1"], "xgboost", [make_explanation()], top_k=1)
        assert len(json.loads(rows["top_features"].iloc[0])) == 1

    def test_probability_and_base_value_are_persisted(self):
        rows = experiments.explanation_rows(["t1"], "xgboost", [make_explanation(0.77)])
        assert rows["fraud_probability"].iloc[0] == pytest.approx(0.77)
        assert rows["base_value"].iloc[0] == pytest.approx(-3.0)

    def test_mismatched_id_count_is_rejected(self):
        with pytest.raises(ValueError, match="2 transaction ids for 1 explanations"):
            experiments.explanation_rows(["t1", "t2"], "xgboost", [make_explanation()])


class TestLogPredictionsToBigQuery:
    def test_writes_rows_to_the_prediction_log(self, config):
        client = FakeBQClient()
        written = experiments.log_predictions_to_bigquery(
            client, config, ["t1", "t2"], "xgboost", [make_explanation()] * 2
        )
        assert written == 2

        frame, table_ref = client.loaded[0]
        assert table_ref == "test-project.fraud_features.prediction_log"
        assert len(frame) == 2

    def test_a_write_failure_never_raises(self, config, caplog):
        """The prediction has already been served; losing the audit row must not
        take down the request that produced it."""
        client = FakeBQClient(fail=True)
        written = experiments.log_predictions_to_bigquery(
            client, config, ["t1"], "xgboost", [make_explanation()]
        )
        assert written == 0
        assert "failed to write explanations" in caplog.text

    def test_a_bad_payload_still_raises_before_any_write(self, config):
        """Programmer error, not an outage: fail loudly."""
        client = FakeBQClient()
        with pytest.raises(ValueError, match="transaction ids"):
            experiments.log_predictions_to_bigquery(
                client, config, ["t1", "t2"], "xgboost", [make_explanation()]
            )
        assert client.loaded == []
