"""Tests for the scheduled drift check.

These run against real training artefacts: a real reference profile captured
from the training split, and a real SHAP explainer. The retraining trigger is
stubbed -- submitting a Vertex AI job from a unit test would be a bad day.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring import monitor as monitor_module
from src.monitoring.drift import REFERENCE_FILENAME, DriftError, ReferenceProfile
from src.monitoring.monitor import (
    IMPORTANCE_FILENAME,
    MonitorError,
    check_and_maybe_retrain,
    current_importance,
    load_reference_importance,
    run_drift_check,
    save_reference,
    trigger_retraining,
)


@pytest.fixture(scope="module")
def training_features(dataset) -> pd.DataFrame:
    return dataset.train.X


@pytest.fixture(scope="module")
def drifted_features(dataset) -> pd.DataFrame:
    """The training features, violently perturbed on the strongest signal."""
    frame = dataset.train.X.copy()
    frame["amount_vs_customer_mean"] = frame["amount_vs_customer_mean"] * 50.0 + 100.0
    frame["is_foreign"] = True
    return frame


class TestTrainingWritesTheBaselines:
    def test_the_reference_profile_is_written(self, artifacts_dir):
        assert (artifacts_dir / REFERENCE_FILENAME).exists()

    def test_the_importance_baseline_is_written(self, artifacts_dir):
        assert (artifacts_dir / IMPORTANCE_FILENAME).exists()

    def test_the_profile_covers_every_model_feature(self, artifacts_dir):
        from src.features.schema import feature_names

        profile = ReferenceProfile.load(artifacts_dir / REFERENCE_FILENAME)
        assert set(profile.features) == set(feature_names())

    def test_the_profile_is_built_from_the_training_split(self, artifacts_dir, dataset):
        """Not test -- the model never saw test, so it is not the baseline."""
        profile = ReferenceProfile.load(artifacts_dir / REFERENCE_FILENAME)
        assert profile.n_rows == len(dataset.train)

    def test_the_importance_baseline_covers_every_feature(self, artifacts_dir):
        from src.features.schema import feature_names

        importance = load_reference_importance(artifacts_dir)
        assert set(importance.index) == set(feature_names())
        assert (importance >= 0).all()


class TestSaveReference:
    def test_writes_both_artefacts(self, tmp_path):
        features = pd.DataFrame({"a": np.random.default_rng(0).normal(size=200)})
        importance = pd.Series({"a": 1.5})
        profile_path, importance_path = save_reference(features, importance, tmp_path)

        assert profile_path.exists() and importance_path.exists()
        assert json.loads(importance_path.read_text()) == {"a": 1.5}

    def test_creates_the_output_directory(self, tmp_path):
        features = pd.DataFrame({"a": np.random.default_rng(0).normal(size=100)})
        save_reference(features, pd.Series({"a": 1.0}), tmp_path / "nested")
        assert (tmp_path / "nested" / REFERENCE_FILENAME).exists()


class TestLoadReferenceImportance:
    def test_a_missing_baseline_names_the_fix(self, tmp_path):
        with pytest.raises(MonitorError, match="src.training.train"):
            load_reference_importance(tmp_path)


class TestRunDriftCheck:
    def test_the_training_distribution_shows_no_drift_against_itself(
        self, artifacts_dir, training_features
    ):
        """The sanity check. If this fails, the reference is not what we trained on."""
        report = run_drift_check(training_features, artifacts_dir=artifacts_dir)
        assert not report.drifted

    def test_a_perturbed_distribution_is_flagged(self, artifacts_dir, drifted_features):
        report = run_drift_check(drifted_features, artifacts_dir=artifacts_dir)
        assert report.drifted
        drifted = {f.feature for f in report.significant}
        assert "amount_vs_customer_mean" in drifted
        assert "is_foreign" in drifted

    def test_a_held_out_slice_of_the_same_period_does_not_drift(self, artifacts_dir, dataset):
        """Validation comes from the same distribution as training; a monitor
        that flags it would page someone every night for nothing."""
        report = run_drift_check(dataset.validation.X, artifacts_dir=artifacts_dir)
        assert not report.drifted

    def test_every_psi_is_finite_on_the_real_feature_matrix(self, artifacts_dir, dataset):
        report = run_drift_check(dataset.test.X, artifacts_dir=artifacts_dir)
        assert all(np.isfinite(f.psi) for f in report.features)

    def test_explanation_drift_is_computed_when_an_explainer_is_given(
        self, artifacts_dir, training_features, explainer
    ):
        report = run_drift_check(
            training_features, artifacts_dir=artifacts_dir, explainer=explainer
        )
        assert report.importance_shift is not None
        assert 0.0 <= report.importance_shift <= 1.0

    def test_explanation_drift_is_near_zero_against_the_training_data(
        self, artifacts_dir, training_features, explainer
    ):
        """Same model, same data -> the same attribution profile."""
        report = run_drift_check(
            training_features, artifacts_dir=artifacts_dir, explainer=explainer
        )
        assert report.importance_shift == pytest.approx(0.0, abs=0.05)

    def test_explanation_drift_rises_on_perturbed_data(
        self, artifacts_dir, training_features, drifted_features, explainer
    ):
        baseline = run_drift_check(
            training_features, artifacts_dir=artifacts_dir, explainer=explainer
        ).importance_shift
        perturbed = run_drift_check(
            drifted_features, artifacts_dir=artifacts_dir, explainer=explainer
        ).importance_shift
        assert perturbed > baseline

    def test_explanation_drift_is_skipped_without_an_explainer(
        self, artifacts_dir, training_features
    ):
        assert (
            run_drift_check(training_features, artifacts_dir=artifacts_dir).importance_shift is None
        )

    def test_a_broken_importance_baseline_does_not_fail_the_check(
        self, artifacts_dir, training_features, explainer, tmp_path, caplog
    ):
        """Feature drift is the signal that gates retraining. Losing the
        secondary signal must not take the whole check down."""
        import shutil

        shutil.copy(artifacts_dir / REFERENCE_FILENAME, tmp_path / REFERENCE_FILENAME)
        # No importance baseline in tmp_path.
        report = run_drift_check(training_features, artifacts_dir=tmp_path, explainer=explainer)
        assert report.importance_shift is None
        assert not report.drifted  # feature drift still computed

    def test_a_missing_reference_profile_names_the_fix(self, tmp_path, training_features):
        with pytest.raises(DriftError, match="src.training.train"):
            run_drift_check(training_features, artifacts_dir=tmp_path)

    def test_a_frame_without_model_features_is_rejected(self, artifacts_dir):
        with pytest.raises(MonitorError, match="none of the model's features"):
            run_drift_check(pd.DataFrame({"unrelated": [1, 2, 3]}), artifacts_dir=artifacts_dir)

    def test_extra_columns_are_ignored(self, artifacts_dir, training_features):
        frame = training_features.copy()
        frame["is_fraud"] = 0  # the label must not be profiled
        report = run_drift_check(frame, artifacts_dir=artifacts_dir)
        assert "is_fraud" not in {f.feature for f in report.features}


class TestCurrentImportance:
    def test_is_capped_for_cost(self, explainer, training_features, monkeypatch):
        """Exact TreeExplainer is linear in rows; the daily batch can be large."""
        monkeypatch.setattr(monitor_module, "IMPORTANCE_SAMPLE_ROWS", 50)
        captured = {}

        original = explainer.global_importance

        def spy(frame):
            captured["rows"] = len(frame)
            return original(frame)

        monkeypatch.setattr(explainer, "global_importance", spy)
        current_importance(explainer, training_features)
        assert captured["rows"] == 50


class TestTriggerRetraining:
    def test_dry_run_never_submits(self, caplog):
        assert trigger_retraining(config=None, dry_run=True) is False
        assert "SKIPPED" in caplog.text

    def test_submits_when_not_a_dry_run(self, monkeypatch):
        submitted = {}

        import src.training.vertex as vertex_module

        def fake_submit(config, *, source="bigquery", args=None, **kwargs):
            submitted["source"] = source
            return "job"

        monkeypatch.setattr(vertex_module, "submit_training_job", fake_submit)
        assert trigger_retraining(config=object(), dry_run=False) is True
        assert submitted["source"] == "bigquery"

    def test_a_submission_failure_is_logged_not_raised(self, monkeypatch, caplog):
        """A failed Cloud Scheduler invocation would be retried, submitting a
        thundering herd of training jobs. The check already did its job."""
        import src.training.vertex as vertex_module

        def boom(*args, **kwargs):
            raise RuntimeError("vertex is down")

        monkeypatch.setattr(vertex_module, "submit_training_job", boom)
        assert trigger_retraining(config=object(), dry_run=False) is False
        assert "retraining submission failed" in caplog.text


class TestCheckAndMaybeRetrain:
    def test_no_drift_means_no_retraining(self, artifacts_dir, training_features):
        result = check_and_maybe_retrain(training_features, artifacts_dir=artifacts_dir)
        assert not result.report.drifted
        assert result.retraining_triggered is False

    def test_drift_triggers_retraining(self, artifacts_dir, drifted_features, monkeypatch):
        import src.training.vertex as vertex_module

        monkeypatch.setattr(vertex_module, "submit_training_job", lambda *a, **k: "job")
        result = check_and_maybe_retrain(
            drifted_features, artifacts_dir=artifacts_dir, config=object(), dry_run=False
        )
        assert result.report.drifted
        assert result.retraining_triggered is True

    def test_drift_in_a_dry_run_reports_but_does_not_retrain(self, artifacts_dir, drifted_features):
        result = check_and_maybe_retrain(
            drifted_features, artifacts_dir=artifacts_dir, dry_run=True
        )
        assert result.report.drifted
        assert result.retraining_triggered is False
        assert result.dry_run is True

    def test_the_result_serialises_for_the_scheduler_response(
        self, artifacts_dir, training_features
    ):
        result = check_and_maybe_retrain(training_features, artifacts_dir=artifacts_dir)
        payload = json.loads(json.dumps(result.as_dict()))  # would raise on numpy types
        assert payload["drifted"] is False
        assert payload["retraining_triggered"] is False
        assert "features" in payload
