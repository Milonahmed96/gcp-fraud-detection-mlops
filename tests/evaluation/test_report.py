"""Tests for loading the A/B test result."""

from __future__ import annotations

import json

import pytest

from src.evaluation.report import RANKING_METRICS, ReportError, load_report


@pytest.fixture(scope="module")
def report(artifacts_dir):
    return load_report(artifacts_dir)


class TestLoadReport:
    def test_loads_both_variants(self, report):
        assert {v.variant for v in report.variants} == {"xgboost", "lightgbm"}

    def test_labels_are_human_readable(self, report):
        assert {v.label for v in report.variants} == {"XGBoost", "LightGBM"}

    def test_carries_every_ranking_metric(self, report):
        for variant in report.variants:
            for key, _ in RANKING_METRICS:
                assert 0.0 <= variant.metric(key) <= 1.0

    def test_carries_confusion_counts_as_integers(self, report):
        for variant in report.variants:
            assert isinstance(variant.false_positives, int)
            assert isinstance(variant.true_positives, int)

    def test_carries_the_trained_threshold(self, report):
        for variant in report.variants:
            assert 0.0 <= variant.threshold <= 1.0

    def test_carries_per_variant_shap_importance(self, report):
        """Each variant needs its own profile; the drift baseline is incumbent-only."""
        from src.features.schema import feature_names

        for variant in report.variants:
            assert set(variant.importance) == set(feature_names())
            assert all(value >= 0 for value in variant.importance.values())

    def test_the_winner_is_one_of_the_variants(self, report):
        assert report.winner in {v.variant for v in report.variants}

    def test_confidence_interval_brackets_the_point_estimate(self, report):
        low, high = report.confidence_interval
        assert low <= report.cost_difference_per_1000 <= high


class TestVerdict:
    def test_an_inconclusive_result_says_so_in_the_headline(self, report):
        """A reader who skims must not think a coin flip was a result."""
        if not report.significant:
            assert report.verdict == "No significant difference"
            assert "straddles zero" in report.verdict_detail

    def test_a_conclusive_result_names_the_winner(self, monkeypatch, report):
        from dataclasses import replace

        conclusive = replace(report, significant=True)
        assert conclusive.winner_result.label in conclusive.verdict
        assert "excludes zero" in conclusive.verdict_detail

    def test_cheaper_returns_the_lower_cost_variant(self, report):
        costs = {v.variant: v.cost_per_1000 for v in report.variants}
        assert report.cheaper().variant == min(costs, key=costs.get)

    def test_best_on_returns_the_higher_metric_variant(self, report):
        f1s = {v.variant: v.f1 for v in report.variants}
        assert report.best_on("f1").variant == max(f1s, key=f1s.get)

    def test_metrics_disagree_is_detected(self, report):
        """On the current sample LightGBM wins F1 while XGBoost wins on cost."""
        expected = report.best_on("f1").variant != report.cheaper().variant
        assert report.metrics_disagree is expected


class TestErrors:
    def test_a_missing_metrics_file_names_the_fix(self, tmp_path):
        with pytest.raises(ReportError, match="src.training.train"):
            load_report(tmp_path)

    def test_malformed_json_is_rejected(self, tmp_path):
        (tmp_path / "metrics.json").write_text("{not json")
        with pytest.raises(ReportError, match="cannot parse"):
            load_report(tmp_path)

    def test_a_missing_key_is_rejected(self, tmp_path):
        (tmp_path / "metrics.json").write_text(json.dumps({"variants": {}}))
        with pytest.raises(ReportError, match="cannot parse"):
            load_report(tmp_path)

    def test_a_winner_not_among_the_variants_is_rejected(self, tmp_path, artifacts_dir):
        payload = json.loads((artifacts_dir / "metrics.json").read_text())
        payload["winner"] = "catboost"
        (tmp_path / "metrics.json").write_text(json.dumps(payload))
        with pytest.raises(ReportError, match="is not among the variants"):
            load_report(tmp_path)

    def test_absent_importance_files_degrade_gracefully(self, tmp_path, artifacts_dir):
        """An older training run has no per-variant profiles; the dashboard says so
        rather than crashing."""
        import shutil

        shutil.copy(artifacts_dir / "metrics.json", tmp_path / "metrics.json")
        loaded = load_report(tmp_path)
        assert all(v.importance == {} for v in loaded.variants)
