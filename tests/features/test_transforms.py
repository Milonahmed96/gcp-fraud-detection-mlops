"""Tests for feature engineering.

The heart of this file is `TestCausality`: if those tests pass, the offline
training features and the online serving features can agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.schema import feature_names
from src.features.transforms import (
    NO_PRIOR_TRANSACTION,
    SchemaValidationError,
    add_channel_and_geo_features,
    add_customer_profile_features,
    add_temporal_features,
    add_velocity_features,
    build_feature_frame,
    validate_raw_transactions,
)


def _c1(frame: pd.DataFrame) -> pd.DataFrame:
    """Customer c1's rows, in time order, indexed by transaction_id."""
    return frame[frame["customer_id"] == "c1"].set_index("transaction_id")


class TestValidation:
    def test_accepts_a_well_formed_frame(self, raw_transactions):
        validate_raw_transactions(raw_transactions)  # must not raise

    def test_rejects_missing_column(self, raw_transactions):
        with pytest.raises(SchemaValidationError, match="missing required columns: amount"):
            validate_raw_transactions(raw_transactions.drop(columns=["amount"]))

    def test_rejects_nulls_in_required_column(self, raw_transactions):
        broken = raw_transactions.copy()
        broken.loc[0, "customer_id"] = None
        with pytest.raises(SchemaValidationError, match="nulls in required columns: customer_id=1"):
            validate_raw_transactions(broken)

    def test_rejects_negative_amount(self, raw_transactions):
        broken = raw_transactions.copy()
        broken.loc[0, "amount"] = -1.0
        with pytest.raises(SchemaValidationError, match="found 1 negative rows"):
            validate_raw_transactions(broken)

    def test_rejects_non_datetime_timestamp(self, raw_transactions):
        broken = raw_transactions.copy()
        broken["timestamp"] = broken["timestamp"].astype(str)
        with pytest.raises(SchemaValidationError, match="must be datetime64"):
            validate_raw_transactions(broken)

    def test_rejects_non_binary_label(self, raw_transactions):
        broken = raw_transactions.copy()
        broken.loc[0, "is_fraud"] = 7
        with pytest.raises(SchemaValidationError, match=r"must be 0/1; found \[7\]"):
            validate_raw_transactions(broken)

    def test_tolerates_unlabelled_rows(self, raw_transactions):
        unlabelled = raw_transactions.copy()
        unlabelled["is_fraud"] = np.nan
        validate_raw_transactions(unlabelled)  # inference-time frames carry no label


class TestTemporalFeatures:
    @pytest.mark.parametrize(
        ("hour", "expected_night"),
        [(0, True), (5, True), (6, False), (12, False), (22, False), (23, True)],
    )
    def test_night_window_spans_midnight(self, hour, expected_night):
        df = pd.DataFrame({"timestamp": [pd.Timestamp(f"2024-01-01T{hour:02d}:30:00")]})
        assert bool(add_temporal_features(df)["is_night"].iloc[0]) is expected_night

    @pytest.mark.parametrize(
        ("date", "dow", "weekend"),
        [
            ("2024-01-01", 0, False),
            ("2024-01-05", 4, False),
            ("2024-01-06", 5, True),
            ("2024-01-07", 6, True),
        ],
    )
    def test_weekday_and_weekend(self, date, dow, weekend):
        df = pd.DataFrame({"timestamp": [pd.Timestamp(f"{date}T12:00:00")]})
        out = add_temporal_features(df)
        assert int(out["day_of_week"].iloc[0]) == dow
        assert bool(out["is_weekend"].iloc[0]) is weekend


class TestChannelAndGeoFeatures:
    def test_foreign_and_card_not_present_flags(self, raw_transactions):
        out = add_channel_and_geo_features(raw_transactions).set_index("transaction_id")
        # t4 is the RO / card-not-present fraud row.
        assert bool(out.loc["t4", "is_foreign"]) is True
        assert bool(out.loc["t4", "card_not_present"]) is True
        # t1 is a domestic, card-present grocery run.
        assert bool(out.loc["t1", "is_foreign"]) is False
        assert bool(out.loc["t1", "card_not_present"]) is False


class TestVelocityFeatures:
    def test_counts_align_with_the_right_customer(self, raw_transactions):
        """Regression guard: groupby().rolling() must align positionally with the
        sorted frame. A misalignment here silently attributes one customer's
        velocity to another -- the kind of bug that never raises."""
        out = add_velocity_features(raw_transactions)
        c1 = _c1(out)

        # c1's transactions are at 09:00, 10:00, 12:00, 12:30 -- all within 24h.
        assert list(c1["txn_count_24h"]) == [1, 2, 3, 4]
        # 1h windows: 09:00 alone; 10:00 alone (09:00 is exactly 1h back but the
        # window is left-open in pandas' time-based rolling); 12:00 alone; 12:30 pairs with 12:00.
        assert list(c1["txn_count_1h"]) == [1, 1, 1, 2]

        # c2 has a single transaction and must not inherit any of c1's history.
        c2 = out[out["customer_id"] == "c2"]
        assert list(c2["txn_count_24h"]) == [1]
        assert float(c2["amount_sum_24h"].iloc[0]) == pytest.approx(500.0)

    def test_amount_sum_is_cumulative_within_window(self, raw_transactions):
        c1 = _c1(add_velocity_features(raw_transactions))
        assert list(c1["amount_sum_24h"]) == pytest.approx([10.0, 30.0, 60.0, 960.0])

    def test_seconds_since_prev_txn(self, raw_transactions):
        c1 = _c1(add_velocity_features(raw_transactions))
        assert c1.loc["t1", "seconds_since_prev_txn"] == NO_PRIOR_TRANSACTION
        assert c1.loc["t2", "seconds_since_prev_txn"] == pytest.approx(3600.0)
        assert c1.loc["t4", "seconds_since_prev_txn"] == pytest.approx(1800.0)

    def test_first_transaction_of_each_customer_has_no_prior(self, raw_transactions):
        out = add_velocity_features(raw_transactions)
        firsts = out.groupby("customer_id").first()
        assert (firsts["seconds_since_prev_txn"] == NO_PRIOR_TRANSACTION).all()


class TestCustomerProfileFeatures:
    def test_baseline_excludes_the_current_transaction(self, raw_transactions):
        """`customer_amount_mean_prior` must be the mean of *strictly earlier*
        amounts. c1 spends 10, 20, 30, 900."""
        c1 = _c1(add_customer_profile_features(raw_transactions))
        # t1: no prior -> falls back to own amount (10).
        # t2: prior = [10] -> 10.  t3: prior = [10,20] -> 15.  t4: prior = [10,20,30] -> 20.
        assert list(c1["customer_amount_mean_prior"]) == pytest.approx([10.0, 10.0, 15.0, 20.0])

    def test_ratio_flags_the_anomalous_transaction(self, raw_transactions):
        c1 = _c1(add_customer_profile_features(raw_transactions))
        assert c1.loc["t1", "amount_vs_customer_mean"] == pytest.approx(1.0)  # first txn
        assert c1.loc["t2", "amount_vs_customer_mean"] == pytest.approx(2.0)  # 20 / 10
        assert c1.loc["t4", "amount_vs_customer_mean"] == pytest.approx(45.0)  # 900 / 20

    def test_zero_amount_history_does_not_divide_by_zero(self):
        df = pd.DataFrame(
            {
                "customer_id": ["c9", "c9"],
                "timestamp": pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T01:00:00"]),
                "amount": [0.0, 50.0],
            }
        )
        out = add_customer_profile_features(df)
        assert np.isfinite(out["amount_vs_customer_mean"]).all()
        assert out["amount_vs_customer_mean"].iloc[1] == pytest.approx(1.0)


class TestCausality:
    def test_features_do_not_depend_on_future_transactions(self, raw_transactions):
        """The load-bearing property. Truncating the customer's history after a
        transaction must not change that transaction's features."""
        full = build_feature_frame(raw_transactions)
        cutoff = pd.Timestamp("2024-01-01T10:00:00")
        truncated = build_feature_frame(raw_transactions[raw_transactions["timestamp"] <= cutoff])

        shared = full[full["timestamp"] <= cutoff].set_index("transaction_id")
        recomputed = truncated.set_index("transaction_id")

        for column in feature_names():
            pd.testing.assert_series_equal(
                shared[column].sort_index(),
                recomputed[column].sort_index(),
                check_dtype=False,
                obj=f"feature {column!r} changed when future rows were removed",
            )

    def test_row_order_of_input_does_not_change_features(self, raw_transactions):
        """Ingestion order is not guaranteed by BigQuery; features must not depend on it."""
        forward = build_feature_frame(raw_transactions).set_index("transaction_id")
        shuffled = raw_transactions.sample(frac=1.0, random_state=17)
        backward = build_feature_frame(shuffled).set_index("transaction_id")

        pd.testing.assert_frame_equal(
            forward[list(feature_names())].sort_index(),
            backward[list(feature_names())].sort_index(),
        )


class TestBuildFeatureFrame:
    def test_produces_exactly_the_declared_features(self, raw_transactions):
        out = build_feature_frame(raw_transactions)
        assert set(feature_names()).issubset(out.columns)

    def test_preserves_every_input_row(self, raw_transactions):
        out = build_feature_frame(raw_transactions)
        assert len(out) == len(raw_transactions)
        assert set(out["transaction_id"]) == set(raw_transactions["transaction_id"])

    def test_output_is_sorted_by_customer_then_time(self, raw_transactions):
        out = build_feature_frame(raw_transactions)
        expected = out.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(out, expected)

    def test_validation_runs_by_default(self, raw_transactions):
        broken = raw_transactions.drop(columns=["amount"])
        with pytest.raises(SchemaValidationError):
            build_feature_frame(broken)

    def test_no_nulls_in_engineered_features(self, raw_transactions):
        out = build_feature_frame(raw_transactions)
        assert not out[list(feature_names())].isna().any().any()
