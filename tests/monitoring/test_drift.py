"""Tests for feature distribution drift.

Most PSI bugs are not wrong formulas -- they are `inf` and `nan` produced by
empty bins, constant features, and bools handed to a quantile binner. Those
cases get the most attention here, because they are what a real feature matrix
contains: `is_night` is a bool and `txn_count_1h` is almost always 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    EPSILON,
    PSI_MODERATE_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD,
    DriftError,
    ReferenceProfile,
    build_reference,
    classify,
    detect_drift,
    population_stability_index,
)

RNG = np.random.default_rng(42)


def normal_frame(n: int = 2000, loc: float = 0.0, scale: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"x": RNG.normal(loc, scale, size=n)})


class TestPopulationStabilityIndex:
    def test_identical_distributions_score_zero(self):
        proportions = np.array([0.2, 0.3, 0.5])
        assert population_stability_index(proportions, proportions) == pytest.approx(0.0)

    def test_is_symmetric(self):
        a, b = np.array([0.2, 0.8]), np.array([0.5, 0.5])
        assert population_stability_index(a, b) == pytest.approx(population_stability_index(b, a))

    def test_is_non_negative(self):
        a, b = np.array([0.1, 0.9]), np.array([0.7, 0.3])
        assert population_stability_index(a, b) > 0

    def test_grows_with_divergence(self):
        reference = np.array([0.5, 0.5])
        near = population_stability_index(reference, np.array([0.45, 0.55]))
        far = population_stability_index(reference, np.array([0.05, 0.95]))
        assert far > near

    def test_an_empty_bin_does_not_produce_inf(self):
        """log(0) is the single most common PSI bug."""
        psi = population_stability_index(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
        assert np.isfinite(psi)
        assert psi > PSI_SIGNIFICANT_THRESHOLD

    def test_both_bins_empty_does_not_produce_nan(self):
        psi = population_stability_index(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert np.isfinite(psi)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_proportions_are_floored_at_epsilon(self):
        psi = population_stability_index(np.array([EPSILON, 1.0]), np.array([0.0, 1.0]))
        assert np.isfinite(psi)

    def test_rejects_a_shape_mismatch(self):
        with pytest.raises(DriftError, match="shape mismatch"):
            population_stability_index(np.array([0.5, 0.5]), np.array([1.0]))


class TestClassify:
    @pytest.mark.parametrize(
        ("psi", "severity"),
        [
            (0.0, "none"),
            (0.09, "none"),
            (PSI_MODERATE_THRESHOLD, "moderate"),
            (0.2, "moderate"),
            (PSI_SIGNIFICANT_THRESHOLD, "significant"),
            (1.5, "significant"),
        ],
    )
    def test_bands(self, psi, severity):
        assert classify(psi) == severity


class TestBuildReference:
    def test_numeric_features_are_quantile_binned(self):
        profile = build_reference(normal_frame())
        reference = profile.features["x"]
        assert reference.kind == "numeric"
        assert reference.categories is None
        assert np.isneginf(reference.bin_edges[0])
        assert np.isposinf(reference.bin_edges[-1])

    def test_outer_edges_are_infinite_so_new_extremes_are_counted(self):
        """A value beyond the training range must land in a bin, not vanish."""
        profile = build_reference(normal_frame())
        report = detect_drift(profile, pd.DataFrame({"x": np.full(500, 1e6)}))
        assert report.features[0].psi > PSI_SIGNIFICANT_THRESHOLD

    def test_bools_are_treated_as_categorical(self):
        frame = pd.DataFrame({"is_night": RNG.random(1000) < 0.2})
        reference = build_reference(frame).features["is_night"]
        assert reference.kind == "categorical"
        assert set(reference.categories) == {"True", "False"}

    def test_a_spiked_count_feature_is_categorical_not_quantile_binned(self):
        """`txn_count_1h` is almost always 1. Quantile bins would collapse."""
        frame = pd.DataFrame({"txn_count_1h": np.concatenate([np.ones(990), np.full(10, 2)])})
        reference = build_reference(frame).features["txn_count_1h"]
        assert reference.kind == "categorical"

    def test_proportions_sum_to_one(self):
        profile = build_reference(normal_frame())
        assert sum(profile.features["x"].proportions) == pytest.approx(1.0, abs=1e-6)

    def test_records_the_reference_size(self):
        assert build_reference(normal_frame(n=777)).n_rows == 777

    def test_rejects_an_empty_frame(self):
        with pytest.raises(DriftError, match="empty frame"):
            build_reference(pd.DataFrame())

    def test_rejects_an_all_null_feature(self):
        with pytest.raises(DriftError, match="entirely null"):
            build_reference(pd.DataFrame({"x": [np.nan, np.nan]}))

    def test_a_constant_high_cardinality_feature_falls_back_to_categorical(self):
        """A constant column has one distinct value, so it is categorical and
        binning never raises."""
        reference = build_reference(pd.DataFrame({"x": np.ones(500)})).features["x"]
        assert reference.kind == "categorical"


class TestDetectDrift:
    def test_the_same_distribution_shows_no_drift(self):
        profile = build_reference(normal_frame(n=5000))
        report = detect_drift(profile, normal_frame(n=5000))
        assert not report.drifted
        assert report.worst.psi < PSI_MODERATE_THRESHOLD

    def test_a_shifted_distribution_is_flagged(self):
        profile = build_reference(normal_frame(n=5000))
        report = detect_drift(profile, normal_frame(n=5000, loc=3.0))
        assert report.drifted
        assert report.features[0].severity == "significant"

    def test_a_widened_distribution_is_flagged(self):
        profile = build_reference(normal_frame(n=5000))
        report = detect_drift(profile, normal_frame(n=5000, scale=5.0))
        assert report.drifted

    def test_every_psi_is_finite(self):
        """The property that matters across a real, mixed-dtype feature matrix."""
        reference = pd.DataFrame(
            {
                "amount_log": RNG.normal(3, 1, 2000),
                "is_night": RNG.random(2000) < 0.2,
                "txn_count_1h": np.ones(2000),
                "hour_of_day": RNG.integers(0, 24, 2000),
            }
        )
        current = pd.DataFrame(
            {
                "amount_log": RNG.normal(6, 2, 500),
                "is_night": np.ones(500, dtype=bool),  # every bin but one is empty
                "txn_count_1h": np.full(500, 7),  # a value never seen in training
                "hour_of_day": np.full(500, 3),
            }
        )
        report = detect_drift(build_reference(reference), current)
        assert all(np.isfinite(f.psi) for f in report.features)
        assert report.drifted

    def test_an_unseen_category_counts_as_drift(self):
        """A value absent from training must not be silently dropped."""
        reference = pd.DataFrame({"c": ["a"] * 900 + ["b"] * 100})
        current = pd.DataFrame({"c": ["z"] * 500})
        report = detect_drift(build_reference(reference), current)
        assert report.drifted

    def test_drift_is_any_not_mean(self):
        """One feature moving hard is the fraud-ring signature; a mean would hide it."""
        reference = pd.DataFrame({f"f{i}": RNG.normal(size=2000) for i in range(10)})
        current = pd.DataFrame({f"f{i}": RNG.normal(size=2000) for i in range(10)})
        current["f3"] = RNG.normal(8.0, 1.0, size=2000)  # one feature blown out

        report = detect_drift(build_reference(reference), current)
        assert report.drifted
        assert [f.feature for f in report.significant] == ["f3"]

    def test_the_worst_feature_is_identified(self):
        reference = pd.DataFrame({"a": RNG.normal(size=2000), "b": RNG.normal(size=2000)})
        current = pd.DataFrame({"a": RNG.normal(size=2000), "b": RNG.normal(5.0, size=2000)})
        assert detect_drift(build_reference(reference), current).worst.feature == "b"

    def test_ks_is_reported_for_numeric_features(self):
        profile = build_reference(normal_frame(n=3000))
        drift = detect_drift(profile, normal_frame(n=3000, loc=2.0)).features[0]
        assert drift.ks_statistic is not None and 0.0 <= drift.ks_statistic <= 1.0
        assert drift.ks_pvalue is not None

    def test_ks_is_none_for_categorical_features(self):
        """KS on a boolean is meaningless."""
        frame = pd.DataFrame({"is_night": RNG.random(1000) < 0.3})
        drift = detect_drift(build_reference(frame), frame).features[0]
        assert drift.ks_statistic is None

    def test_importance_shift_is_carried_through(self):
        profile = build_reference(normal_frame())
        report = detect_drift(profile, normal_frame(), importance_shift=0.42)
        assert report.importance_shift == 0.42
        assert report.as_dict()["importance_shift"] == 0.42

    def test_rejects_an_empty_current_sample(self):
        with pytest.raises(DriftError, match="empty current sample"):
            detect_drift(build_reference(normal_frame()), pd.DataFrame({"x": []}))

    def test_rejects_a_current_sample_missing_features(self):
        profile = build_reference(
            pd.DataFrame({"a": RNG.normal(size=100), "b": RNG.normal(size=100)})
        )
        with pytest.raises(DriftError, match="missing features: b"):
            detect_drift(profile, pd.DataFrame({"a": RNG.normal(size=100)}))

    def test_extra_columns_in_the_current_sample_are_ignored(self):
        profile = build_reference(normal_frame())
        current = normal_frame()
        current["unrelated"] = 1.0
        assert len(detect_drift(profile, current).features) == 1

    def test_summary_names_the_verdict_and_worst_feature(self):
        profile = build_reference(normal_frame())
        summary = detect_drift(profile, normal_frame(loc=4.0)).summary()
        assert "DRIFT" in summary and "x" in summary

    def test_summary_says_stable_when_no_drift(self):
        profile = build_reference(normal_frame(n=5000))
        assert "stable" in detect_drift(profile, normal_frame(n=5000)).summary()


class TestReferenceProfilePersistence:
    def test_round_trips_through_json(self, tmp_path):
        reference = pd.DataFrame(
            {
                "amount_log": RNG.normal(size=1000),
                "is_night": RNG.random(1000) < 0.3,
            }
        )
        profile = build_reference(reference)
        path = profile.save(tmp_path / "nested" / "reference_profile.json")
        assert path.exists()

        reloaded = ReferenceProfile.load(path)
        assert reloaded.n_rows == profile.n_rows
        assert set(reloaded.features) == set(profile.features)
        assert reloaded.features["is_night"].categories == profile.features["is_night"].categories
        np.testing.assert_allclose(
            reloaded.features["amount_log"].bin_edges, profile.features["amount_log"].bin_edges
        )

    def test_a_reloaded_profile_produces_identical_drift(self, tmp_path):
        profile = build_reference(normal_frame(n=2000))
        path = profile.save(tmp_path / "reference_profile.json")
        current = normal_frame(n=800, loc=1.5)

        original = detect_drift(profile, current).features[0].psi
        reloaded = detect_drift(ReferenceProfile.load(path), current).features[0].psi
        assert original == pytest.approx(reloaded)

    def test_a_missing_profile_names_the_fix(self, tmp_path):
        with pytest.raises(DriftError, match="src.training.train"):
            ReferenceProfile.load(tmp_path / "absent.json")
