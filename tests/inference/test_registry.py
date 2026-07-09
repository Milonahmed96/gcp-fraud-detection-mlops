"""Tests for loading the serving artefacts.

The threshold assertions matter most. `train.py` fits a cost-minimising
threshold on validation; if the service quietly defaults to 0.5 it blocks a
different set of customers than the A/B test measured, and every business-cost
number in the dashboard becomes a lie.
"""

from __future__ import annotations

import json

import pytest

from src.inference.registry import (
    ArtifactError,
    load_bundle,
    load_threshold,
    resolve_artifacts_dir,
    resolve_variant,
)
from src.training.models import VARIANTS


class TestResolveVariant:
    def test_defaults_to_the_incumbent(self):
        assert resolve_variant(env={}) == "xgboost"

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_reads_the_environment(self, variant):
        assert resolve_variant(env={"SERVING_VARIANT": variant}) == variant

    def test_is_case_and_whitespace_tolerant(self):
        assert resolve_variant(env={"SERVING_VARIANT": "  LightGBM \n"}) == "lightgbm"

    def test_an_unknown_variant_fails_at_startup(self):
        """Better a failed health check than a 500 on every request."""
        with pytest.raises(ArtifactError, match="is not one of"):
            resolve_variant(env={"SERVING_VARIANT": "catboost"})


class TestResolveArtifactsDir:
    def test_defaults_to_artifacts(self):
        assert str(resolve_artifacts_dir(env={})) == "artifacts"

    def test_reads_the_environment(self):
        assert str(resolve_artifacts_dir(env={"MODEL_ARTIFACTS_DIR": "/app/x"})) == "/app/x"


class TestLoadThreshold:
    def test_reads_the_trained_threshold(self, artifacts_dir):
        payload = json.loads((artifacts_dir / "metrics.json").read_text())
        for variant in VARIANTS:
            expected = payload["variants"][variant]["evaluation"]["threshold"]
            assert load_threshold(artifacts_dir, variant) == pytest.approx(expected)

    def test_the_threshold_is_not_naively_one_half(self, artifacts_dir):
        """The whole point of the cost model is that 0.5 is the wrong threshold."""
        assert load_threshold(artifacts_dir, "xgboost") != 0.5

    def test_missing_metrics_file_fails_loudly(self, tmp_path):
        with pytest.raises(ArtifactError, match="metrics.json not found"):
            load_threshold(tmp_path, "xgboost")

    def test_malformed_metrics_file_fails_loudly(self, tmp_path):
        (tmp_path / "metrics.json").write_text("{not json")
        with pytest.raises(ArtifactError, match="cannot read threshold"):
            load_threshold(tmp_path, "xgboost")

    def test_a_missing_variant_entry_fails_loudly(self, tmp_path):
        (tmp_path / "metrics.json").write_text(json.dumps({"variants": {}}))
        with pytest.raises(ArtifactError, match="cannot read threshold for 'xgboost'"):
            load_threshold(tmp_path, "xgboost")

    def test_an_out_of_range_threshold_is_rejected(self, tmp_path):
        (tmp_path / "metrics.json").write_text(
            json.dumps({"variants": {"xgboost": {"evaluation": {"threshold": 1.7}}}})
        )
        with pytest.raises(ArtifactError, match="is not a probability"):
            load_threshold(tmp_path, "xgboost")


class TestLoadBundle:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_loads_model_explainer_and_threshold(self, artifacts_dir, variant):
        bundle = load_bundle(variant=variant, artifacts_dir=artifacts_dir)
        assert bundle.variant == variant
        assert hasattr(bundle.model, "predict_proba")
        assert 0.0 <= bundle.threshold <= 1.0

    def test_feature_names_come_from_the_explainer(self, artifacts_dir):
        from src.features.schema import feature_names

        bundle = load_bundle(variant="xgboost", artifacts_dir=artifacts_dir)
        assert bundle.feature_names == list(feature_names())

    def test_the_explainer_matches_its_model(self, artifacts_dir):
        bundle = load_bundle(variant="lightgbm", artifacts_dir=artifacts_dir)
        assert type(bundle.explainer.model) is type(bundle.model)

    def test_a_mismatched_explainer_is_rejected(self, artifacts_dir, tmp_path):
        """An explainer paired with the wrong model would justify a prediction
        the service never made -- exactly what the audit log must prevent."""
        import shutil

        shutil.copy(artifacts_dir / "metrics.json", tmp_path / "metrics.json")
        shutil.copy(artifacts_dir / "model_xgboost.joblib", tmp_path / "model_xgboost.joblib")
        # Deliberately pair the XGBoost model with the LightGBM explainer.
        shutil.copy(
            artifacts_dir / "explainer_lightgbm.joblib", tmp_path / "explainer_xgboost.joblib"
        )

        with pytest.raises(ArtifactError, match="explainer wraps"):
            load_bundle(variant="xgboost", artifacts_dir=tmp_path)

    def test_missing_artefacts_name_the_fix(self, tmp_path):
        with pytest.raises(ArtifactError, match="src.training.train"):
            load_bundle(variant="xgboost", artifacts_dir=tmp_path)

    def test_the_variant_can_come_from_the_environment(self, artifacts_dir):
        bundle = load_bundle(artifacts_dir=artifacts_dir, env={"SERVING_VARIANT": "lightgbm"})
        assert bundle.variant == "lightgbm"
