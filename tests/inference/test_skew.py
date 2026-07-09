"""Train/serve skew: the online feature path must equal the offline one.

This is the most important test file in the repository.

`src/features/transforms.py` engineers features over a whole history with pandas
rolling windows. `src/inference/features.py` engineers them for one transaction
from a `CustomerState` lookup. They are separate implementations of the same
function. If they disagree, the model receives inputs at serving time it never
saw in training, and nothing raises -- the predictions are simply wrong, and the
offline metrics keep looking fine.

So: replay the sample transaction by transaction, and assert the two paths agree
exactly, value and dtype.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.sample_data import load_sample
from src.features.schema import feature_names
from src.features.transforms import build_feature_frame
from src.inference.features import build_serving_features
from src.inference.state import CustomerState

REPO_SAMPLE = "data/sample/transactions_sample.csv"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_sample(REPO_SAMPLE)


@pytest.fixture(scope="module")
def offline(raw) -> pd.DataFrame:
    """The offline pipeline's features, indexed by transaction id."""
    return build_feature_frame(raw).set_index("transaction_id")


def serve_one(raw: pd.DataFrame, transaction_id: str) -> pd.DataFrame:
    """Compute serving features for one transaction from its customer's priors."""
    row = raw.set_index("transaction_id").loc[transaction_id]
    history = raw[
        (raw["customer_id"] == row["customer_id"]) & (raw["timestamp"] < row["timestamp"])
    ]
    return build_serving_features(
        timestamp=row["timestamp"],
        amount=float(row["amount"]),
        country=row["country"],
        customer_home_country=row["customer_home_country"],
        card_present=bool(row["card_present"]),
        state=CustomerState.from_history(history),
    )


def transactions_with_at_least(raw: pd.DataFrame, n_priors: int) -> pd.Series:
    """Transaction ids whose customer had at least `n_priors` earlier transactions."""
    ordered = raw.sort_values(["customer_id", "timestamp"], kind="mergesort")
    rank = ordered.groupby("customer_id").cumcount()
    return ordered.loc[rank >= n_priors, "transaction_id"]


class TestParity:
    """The load-bearing assertions."""

    def test_a_broad_replay_matches_the_offline_pipeline(self, raw, offline):
        """Replay 150 transactions spread across the timeline and every customer."""
        sampled = raw["transaction_id"].sample(150, random_state=11)

        mismatches: list[str] = []
        for transaction_id in sampled:
            served = serve_one(raw, transaction_id).iloc[0]
            expected = offline.loc[transaction_id]
            for feature in feature_names():
                if not np.isclose(float(served[feature]), float(expected[feature]), rtol=1e-9):
                    mismatches.append(
                        f"{transaction_id}.{feature}: served={served[feature]!r} "
                        f"offline={expected[feature]!r}"
                    )

        assert not mismatches, "online/offline skew:\n" + "\n".join(mismatches[:20])

    def test_elapsed_time_is_exact_to_the_nanosecond(self, raw, offline):
        """`Timedelta.total_seconds()` truncates to microseconds and would pass a
        loose tolerance while still feeding the model a different number than it
        trained on. Demand exact equality."""
        sampled = transactions_with_at_least(raw, 1).sample(60, random_state=3)
        for transaction_id in sampled:
            served = serve_one(raw, transaction_id).iloc[0]
            assert (
                served["seconds_since_prev_txn"]
                == offline.loc[transaction_id, "seconds_since_prev_txn"]
            )

    def test_dtypes_match_the_offline_pipeline(self, raw, offline):
        """A float where the model trained on an int is a silent input change."""
        served = serve_one(raw, raw["transaction_id"].iloc[500])
        for feature in feature_names():
            assert served[feature].dtype == offline[feature].dtype, (
                f"{feature}: serving {served[feature].dtype} vs offline {offline[feature].dtype}"
            )

    def test_column_order_matches_the_training_matrix(self, raw):
        """Both libraries index features positionally once fitted."""
        served = serve_one(raw, raw["transaction_id"].iloc[0])
        assert list(served.columns) == list(feature_names())


class TestEdgeCases:
    """The rows where the two implementations are most likely to diverge."""

    def test_a_customers_first_transaction(self, raw, offline):
        first_ids = raw.sort_values("timestamp").groupby("customer_id").head(1)["transaction_id"]
        for transaction_id in first_ids.head(10):
            served = serve_one(raw, transaction_id).iloc[0]
            expected = offline.loc[transaction_id]
            assert served["seconds_since_prev_txn"] == expected["seconds_since_prev_txn"] == -1.0
            assert served["txn_count_24h"] == expected["txn_count_24h"] == 1
            assert served["amount_vs_customer_mean"] == pytest.approx(1.0)
            assert served["customer_amount_mean_prior"] == pytest.approx(
                expected["customer_amount_mean_prior"]
            )

    def test_transactions_deep_into_a_customers_history(self, raw, offline):
        """The expanding-mean baseline is where a naive implementation drifts."""
        deep = transactions_with_at_least(raw, 50)
        assert len(deep) > 0, "sample has no customer with 50+ priors"
        for transaction_id in deep.head(10):
            served = serve_one(raw, transaction_id).iloc[0]
            expected = offline.loc[transaction_id]
            assert served["customer_amount_mean_prior"] == pytest.approx(
                expected["customer_amount_mean_prior"]
            )
            assert served["amount_vs_customer_mean"] == pytest.approx(
                expected["amount_vs_customer_mean"]
            )

    def test_velocity_features_agree_on_busy_customers(self, raw, offline):
        busy = offline.sort_values("txn_count_24h", ascending=False).head(10)
        for transaction_id in busy.index:
            served = serve_one(raw, transaction_id).iloc[0]
            expected = offline.loc[transaction_id]
            assert served["txn_count_1h"] == expected["txn_count_1h"]
            assert served["txn_count_24h"] == expected["txn_count_24h"]
            assert served["amount_sum_24h"] == pytest.approx(expected["amount_sum_24h"])


class TestWindowBoundary:
    """`transforms.py` uses pandas rolling with a half-open `(t - w, t]` window.

    An event exactly `w` before the current transaction is excluded. Getting this
    off by one second changes the velocity features on exactly the transactions
    where velocity matters.
    """

    def _state(self, offsets_seconds: list[float], now: pd.Timestamp) -> CustomerState:
        history = pd.DataFrame(
            {
                "timestamp": [now - pd.Timedelta(seconds=s) for s in offsets_seconds],
                "amount": [10.0] * len(offsets_seconds),
            }
        )
        return CustomerState.from_history(history)

    def test_an_event_exactly_one_hour_back_is_excluded(self):
        now = pd.Timestamp("2024-06-01T12:00:00")
        state = self._state([3600.0], now)
        assert state.txn_count(now, 3600.0) == 1  # only the current transaction

    def test_an_event_just_inside_one_hour_is_included(self):
        now = pd.Timestamp("2024-06-01T12:00:00")
        state = self._state([3599.0], now)
        assert state.txn_count(now, 3600.0) == 2

    def test_the_boundary_matches_pandas_rolling(self):
        """Cross-check the rule against the real offline implementation."""
        from src.features.transforms import add_velocity_features

        now = pd.Timestamp("2024-06-01T12:00:00")
        frame = pd.DataFrame(
            {
                "customer_id": ["c"] * 3,
                "timestamp": [now - pd.Timedelta(seconds=3600), now - pd.Timedelta(seconds=1), now],
                "amount": [10.0, 10.0, 10.0],
            }
        )
        offline_count = add_velocity_features(frame)["txn_count_1h"].iloc[-1]
        online_count = self._state([3600.0, 1.0], now).txn_count(now, 3600.0)
        assert online_count == offline_count == 2

    def test_a_24h_boundary_event_is_excluded(self):
        now = pd.Timestamp("2024-06-01T12:00:00")
        state = self._state([86400.0], now)
        assert state.txn_count(now, 86400.0) == 1
        assert state.amount_sum(now, 86400.0, 50.0) == pytest.approx(50.0)


class TestTruncation:
    """`recent_events` is capped; the baseline must survive the cap."""

    def test_the_baseline_is_exact_despite_truncation(self):
        """`prior_amount_mean` is stored separately precisely so a truncated
        event log does not corrupt the customer's spending baseline."""
        now = pd.Timestamp("2024-06-01T12:00:00")
        history = pd.DataFrame(
            {
                "timestamp": [now - pd.Timedelta(days=200 - i) for i in range(200)],
                "amount": [float(i) for i in range(200)],
            }
        )
        state = CustomerState.from_history(history, max_events=10)

        assert state.truncated is True
        assert state.prior_count == 200
        assert len(state.recent_events) == 10
        assert state.amount_mean_prior(1.0) == pytest.approx(np.mean(range(200)))

    def test_truncation_does_not_affect_the_24h_window(self):
        """Only events inside 24h matter for velocity; the cap keeps the newest."""
        now = pd.Timestamp("2024-06-01T12:00:00")
        old = [now - pd.Timedelta(days=d) for d in range(30, 0, -1)]
        recent = [now - pd.Timedelta(hours=h) for h in (5, 3, 1)]
        history = pd.DataFrame(
            {"timestamp": old + recent, "amount": [1.0] * (len(old) + len(recent))}
        )
        state = CustomerState.from_history(history, max_events=5)
        assert state.txn_count(now, 86400.0) == 4  # 3 recent + the current one
