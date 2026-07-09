"""Tests for training data loading and the temporal split.

The critical property is that the split respects time. A random split would
quietly inflate every metric downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.schema import feature_names
from src.training.dataset import (
    DatasetError,
    load_features,
    temporal_split,
)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return load_features("sample")


class TestLoadFeatures:
    def test_sample_source_returns_engineered_features(self, features):
        assert set(feature_names()) <= set(features.columns)
        assert len(features) > 0

    def test_sample_source_needs_no_gcp_config(self, features):
        assert "is_fraud" in features.columns  # got here without credentials

    def test_unknown_source_is_rejected(self):
        with pytest.raises(DatasetError, match="unknown source 'redis'"):
            load_features("redis")  # type: ignore[arg-type]

    def test_bigquery_source_requires_config_and_client(self):
        with pytest.raises(DatasetError, match="bigquery source requires"):
            load_features("bigquery")

    def test_bigquery_source_names_every_missing_argument(self):
        with pytest.raises(DatasetError) as excinfo:
            load_features("bigquery")
        message = str(excinfo.value)
        for name in ("config", "client", "start_date", "end_date"):
            assert name in message

    def test_bigquery_source_delegates_to_fetch_training_set(self, monkeypatch):
        from src.features.config import GCPConfig
        from src.training import dataset as ds

        expected = pd.DataFrame({"is_fraud": [0, 1]})
        captured = {}

        def fake_fetch(client, config, *, start_date, end_date):
            captured.update(start_date=start_date, end_date=end_date)
            return expected

        monkeypatch.setattr(ds, "fetch_training_set", fake_fetch)
        config = GCPConfig("p", "europe-west2", "b", "d", "f")

        result = ds.load_features(
            "bigquery",
            config=config,
            client=object(),
            start_date="2024-01-01",
            end_date="2024-02-01",
        )
        pd.testing.assert_frame_equal(result, expected)
        assert captured == {"start_date": "2024-01-01", "end_date": "2024-02-01"}


class TestTemporalSplit:
    def test_splits_are_disjoint_and_exhaustive(self, features):
        d = temporal_split(features)
        assert len(d.train) + len(d.validation) + len(d.test) == len(features)

    def test_every_train_row_precedes_every_test_row(self, features):
        """The load-bearing property. Any overlap here is leakage."""
        d = temporal_split(features)
        assert d.train.timestamps.max() <= d.validation.timestamps.min()
        assert d.validation.timestamps.max() <= d.test.timestamps.min()

    def test_test_fraction_is_respected(self, features):
        d = temporal_split(features, test_fraction=0.3)
        assert len(d.test) == pytest.approx(len(features) * 0.3, abs=1)

    def test_validation_can_be_disabled(self, features):
        d = temporal_split(features, validation_fraction=0.0)
        assert len(d.validation) == 0
        assert len(d.train) + len(d.test) == len(features)

    def test_split_carries_features_labels_and_amounts(self, features):
        d = temporal_split(features)
        assert list(d.train.X.columns) == list(feature_names())
        assert d.train.y.dtype == int
        assert len(d.train.amounts) == len(d.train)
        assert (d.train.amounts > 0).all()

    def test_the_label_is_not_a_feature(self, features):
        d = temporal_split(features)
        assert "is_fraud" not in d.train.X.columns
        assert "amount" not in d.train.X.columns  # only amount_log is a feature

    def test_scale_pos_weight_is_negatives_over_positives(self, features):
        d = temporal_split(features)
        expected = (len(d.train) - d.train.n_fraud) / d.train.n_fraud
        assert d.scale_pos_weight == pytest.approx(expected)
        assert d.scale_pos_weight > 1  # fraud is the minority

    def test_all_splits_contain_fraud(self, features):
        d = temporal_split(features)
        assert d.train.n_fraud > 0 and d.test.n_fraud > 0

    def test_unlabelled_rows_are_dropped(self, features):
        frame = features.copy()
        frame.loc[frame.index[:100], "is_fraud"] = np.nan
        d = temporal_split(frame)
        assert len(d.train) + len(d.validation) + len(d.test) == len(features) - 100

    def test_rejects_a_frame_with_no_label_column(self, features):
        with pytest.raises(DatasetError, match="no is_fraud column"):
            temporal_split(features.drop(columns=["is_fraud"]))

    def test_rejects_too_little_data(self, features):
        with pytest.raises(DatasetError, match="at least 10 labelled rows"):
            temporal_split(features.head(5))

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_invalid_test_fraction(self, features, fraction):
        with pytest.raises(DatasetError, match="test_fraction must be in"):
            temporal_split(features, test_fraction=fraction)

    def test_rejects_invalid_validation_fraction(self, features):
        with pytest.raises(DatasetError, match="validation_fraction must be in"):
            temporal_split(features, validation_fraction=1.0)

    def test_rejects_a_test_split_with_no_fraud(self, features):
        """Metrics would be undefined; fail loudly rather than emit nan."""
        frame = features.copy().sort_values("timestamp")
        tail = frame.index[-int(len(frame) * 0.2) :]
        frame.loc[tail, "is_fraud"] = 0
        with pytest.raises(DatasetError, match="test split contains no fraud"):
            temporal_split(frame)

    def test_scale_pos_weight_errors_when_train_has_no_fraud(self, features):
        frame = features.copy().sort_values("timestamp").reset_index(drop=True)
        head = frame.index[: int(len(frame) * 0.6)]
        frame.loc[head, "is_fraud"] = 0
        d = temporal_split(frame)
        with pytest.raises(DatasetError, match="no fraud; cannot weight"):
            _ = d.scale_pos_weight

    def test_split_is_deterministic(self, features):
        a, b = temporal_split(features), temporal_split(features)
        np.testing.assert_array_equal(a.test.y, b.test.y)
        pd.testing.assert_frame_equal(a.train.X, b.train.X)
