"""Tests for the synthetic sample data.

Beyond "does it generate", these assert the sample is *fit for purpose*: it must
be reproducible, schema-conformant, imbalanced, and carry the signal a fraud
model is supposed to find. A sample where fraud looks exactly like non-fraud
would make Phase 3's model metrics meaningless.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features.sample_data import (
    SAMPLE_PATH,
    generate_transactions,
    load_sample,
    write_sample,
)
from src.features.schema import raw_column_names
from src.features.transforms import build_feature_frame, validate_raw_transactions

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return generate_transactions()


class TestGeneration:
    def test_is_deterministic(self):
        pd.testing.assert_frame_equal(generate_transactions(), generate_transactions())

    def test_a_different_seed_gives_different_data(self, sample):
        other = generate_transactions(seed=99)
        assert not sample["amount"].equals(other["amount"])

    def test_has_exactly_the_raw_schema_columns(self, sample):
        assert list(sample.columns) == list(raw_column_names())

    def test_satisfies_the_ingestion_contract(self, sample):
        validate_raw_transactions(sample)

    def test_transaction_ids_are_unique(self, sample):
        assert sample["transaction_id"].is_unique

    def test_is_sorted_by_timestamp(self, sample):
        assert sample["timestamp"].is_monotonic_increasing

    def test_amounts_are_positive(self, sample):
        assert (sample["amount"] > 0).all()

    def test_countries_are_two_letter_codes(self, sample):
        assert sample["country"].str.len().eq(2).all()
        assert sample["customer_home_country"].str.len().eq(2).all()


class TestClassBalance:
    def test_fraud_is_rare(self, sample):
        """Fraud must be a small minority -- that imbalance is the whole problem."""
        rate = sample["is_fraud"].mean()
        assert 0.005 < rate < 0.05, f"fraud rate {rate:.3%} is not realistically imbalanced"

    def test_both_classes_are_present(self, sample):
        assert set(sample["is_fraud"].unique()) == {0, 1}

    def test_enough_fraud_rows_to_be_useful(self, sample):
        assert sample["is_fraud"].sum() >= 10


class TestSignal:
    """The injected correlations must actually survive into the data."""

    def test_fraud_is_more_often_foreign(self, sample):
        foreign = sample["country"] != sample["customer_home_country"]
        assert foreign[sample["is_fraud"] == 1].mean() > foreign[sample["is_fraud"] == 0].mean()

    def test_fraud_is_more_often_card_not_present(self, sample):
        cnp = ~sample["card_present"]
        assert cnp[sample["is_fraud"] == 1].mean() > cnp[sample["is_fraud"] == 0].mean()

    def test_fraud_amounts_are_larger(self, sample):
        fraud_median = sample.loc[sample["is_fraud"] == 1, "amount"].median()
        genuine_median = sample.loc[sample["is_fraud"] == 0, "amount"].median()
        assert fraud_median > genuine_median * 1.5

    def test_fraud_skews_to_the_small_hours(self, sample):
        night = sample["timestamp"].dt.hour <= 5
        assert night[sample["is_fraud"] == 1].mean() > night[sample["is_fraud"] == 0].mean()

    def test_the_classes_overlap_on_amount(self, sample):
        """Perfectly separable classes make the A/B test degenerate.

        Some genuine transactions must be larger than the median fraud, or both
        variants score a meaningless AUC of 1.0.
        """
        fraud_median = sample.loc[sample["is_fraud"] == 1, "amount"].median()
        genuine = sample.loc[sample["is_fraud"] == 0, "amount"]
        assert (genuine > fraud_median).mean() > 0.05


class TestStealthFraud:
    """A fraction of fraud must carry no signal at all -- it sets the recall ceiling."""

    def _stealth(self, sample: pd.DataFrame) -> pd.Series:
        fraud = sample["is_fraud"] == 1
        domestic = sample["country"] == sample["customer_home_country"]
        daytime = sample["timestamp"].dt.hour > 5
        return fraud & domestic & sample["card_present"] & daytime

    def test_some_fraud_looks_entirely_ordinary(self, sample):
        assert self._stealth(sample).sum() >= 5

    def test_stealth_fraud_is_a_minority_of_fraud(self, sample):
        share = self._stealth(sample).sum() / (sample["is_fraud"] == 1).sum()
        assert 0.0 < share < 0.5, f"stealth share {share:.2%} is implausible"

    def test_genuine_traffic_also_goes_abroad_and_card_not_present(self, sample):
        """Otherwise `is_foreign` or `card_not_present` alone would be a perfect classifier."""
        genuine = sample["is_fraud"] == 0
        foreign = sample["country"] != sample["customer_home_country"]
        assert foreign[genuine].sum() > 0
        assert (~sample["card_present"])[genuine].sum() > 0

    def test_engineered_ratio_separates_the_classes(self, sample):
        """`amount_vs_customer_mean` is the feature this sample exists to exercise."""
        features = build_feature_frame(sample)
        fraud = features.loc[features["is_fraud"] == 1, "amount_vs_customer_mean"].median()
        genuine = features.loc[features["is_fraud"] == 0, "amount_vs_customer_mean"].median()
        assert fraud > genuine


class TestPipelineIntegration:
    def test_sample_flows_through_the_full_feature_pipeline(self, sample):
        features = build_feature_frame(sample)
        assert len(features) == len(sample)
        assert not features.isna().any().any()

    def test_multiple_customers_have_transaction_history(self, sample):
        """Velocity features are vacuous if every customer has one transaction."""
        counts = sample["customer_id"].value_counts()
        assert (counts > 1).sum() >= 10


class TestCommittedSample:
    def test_the_committed_csv_exists(self):
        assert (REPO_ROOT / SAMPLE_PATH).exists(), "run: uv run python -m src.features.sample_data"

    def test_the_committed_csv_is_current(self, sample):
        """Fails if someone edits the generator without regenerating the CSV."""
        on_disk = load_sample(REPO_ROOT / SAMPLE_PATH)
        pd.testing.assert_frame_equal(on_disk, sample, check_dtype=False)

    def test_the_committed_csv_round_trips_through_the_pipeline(self):
        on_disk = load_sample(REPO_ROOT / SAMPLE_PATH)
        validate_raw_transactions(on_disk)
        assert len(build_feature_frame(on_disk)) == len(on_disk)

    def test_write_sample_creates_parent_directories(self, tmp_path):
        target = tmp_path / "nested" / "dir" / "sample.csv"
        assert write_sample(target) == target
        assert target.exists()
