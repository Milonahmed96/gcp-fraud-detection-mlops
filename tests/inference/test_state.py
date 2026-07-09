"""Tests for customer state and the online lookup."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.transforms import NO_PRIOR_TRANSACTION
from src.inference.state import (
    CustomerState,
    InMemoryStateStore,
    TransactionEvent,
    elapsed_seconds,
)

NOW = pd.Timestamp("2024-06-01T12:00:00")


def history(*offsets_and_amounts: tuple[float, float]) -> pd.DataFrame:
    """Build a prior-transaction frame from (seconds_before_now, amount) pairs."""
    return pd.DataFrame(
        {
            "timestamp": [NOW - pd.Timedelta(seconds=s) for s, _ in offsets_and_amounts],
            "amount": [a for _, a in offsets_and_amounts],
        }
    )


class TestElapsedSeconds:
    def test_preserves_nanosecond_precision(self):
        """`Timedelta.total_seconds()` truncates to microseconds; this must not."""
        earlier = pd.Timestamp("2024-01-01T00:00:00.000000000")
        later = pd.Timestamp("2024-01-01T00:00:00.123456789")
        assert elapsed_seconds(later, earlier) == pytest.approx(0.123456789, abs=1e-12)

    def test_beats_the_naive_implementation(self):
        earlier = pd.Timestamp("2024-01-01T00:00:00.000000000")
        later = pd.Timestamp("2024-01-01T00:03:40.472044709")
        naive = (later - earlier).total_seconds()
        exact = elapsed_seconds(later, earlier)
        assert exact != naive  # the bug this function exists to fix
        assert exact == pytest.approx(220.472044709, abs=1e-9)


class TestEmptyState:
    def test_a_new_customer_has_no_history(self):
        state = CustomerState.empty()
        assert state.is_new_customer
        assert state.prior_count == 0
        assert state.recent_events == ()

    def test_no_prior_transaction_sentinel(self):
        assert CustomerState.empty().seconds_since_prev_txn(NOW) == NO_PRIOR_TRANSACTION

    def test_counts_only_the_current_transaction(self):
        state = CustomerState.empty()
        assert state.txn_count(NOW, 3600.0) == 1
        assert state.amount_sum(NOW, 86400.0, 42.0) == pytest.approx(42.0)

    def test_baseline_falls_back_to_the_current_amount(self):
        """Makes amount_vs_customer_mean exactly 1.0 -- 'unremarkable'."""
        assert CustomerState.empty().amount_mean_prior(99.0) == pytest.approx(99.0)


class TestFromHistory:
    def test_orders_events_ascending(self):
        state = CustomerState.from_history(history((10.0, 1.0), (100.0, 2.0), (50.0, 3.0)))
        timestamps = [event.timestamp for event in state.recent_events]
        assert timestamps == sorted(timestamps)

    def test_computes_the_expanding_baseline_over_all_priors(self):
        state = CustomerState.from_history(history((10.0, 10.0), (20.0, 20.0), (30.0, 60.0)))
        assert state.prior_amount_mean == pytest.approx(30.0)
        assert state.prior_count == 3

    def test_seconds_since_prev_txn_uses_the_most_recent_event(self):
        state = CustomerState.from_history(history((900.0, 1.0), (60.0, 1.0), (3600.0, 1.0)))
        assert state.seconds_since_prev_txn(NOW) == pytest.approx(60.0)

    def test_empty_history_yields_empty_state(self):
        empty = pd.DataFrame({"timestamp": [], "amount": []})
        assert CustomerState.from_history(empty).is_new_customer

    def test_truncation_is_flagged_and_keeps_the_newest_events(self):
        state = CustomerState.from_history(
            history(*[(float(600 - i), 1.0) for i in range(20)]), max_events=5
        )
        assert state.truncated is True
        assert state.prior_count == 20
        assert len(state.recent_events) == 5
        # Newest kept: the smallest offsets, i.e. the latest timestamps.
        assert state.seconds_since_prev_txn(NOW) == pytest.approx(581.0)


class TestWindows:
    def test_window_is_half_open_on_the_left(self):
        state = CustomerState.from_history(history((3600.0, 5.0)))
        assert state.txn_count(NOW, 3600.0) == 1  # boundary event excluded
        assert state.amount_sum(NOW, 3600.0, 1.0) == pytest.approx(1.0)

    def test_events_inside_the_window_are_counted(self):
        state = CustomerState.from_history(history((60.0, 5.0), (120.0, 7.0)))
        assert state.txn_count(NOW, 3600.0) == 3  # 2 priors + current
        assert state.amount_sum(NOW, 3600.0, 1.0) == pytest.approx(13.0)

    def test_events_outside_the_window_are_ignored(self):
        state = CustomerState.from_history(history((100_000.0, 500.0)))
        assert state.txn_count(NOW, 86400.0) == 1
        assert state.amount_sum(NOW, 86400.0, 1.0) == pytest.approx(1.0)

    def test_future_events_are_never_counted(self):
        """Causality guard: state must never contain transactions after `now`."""
        state = CustomerState(recent_events=(TransactionEvent(NOW + pd.Timedelta(hours=1), 50.0),))
        assert state.txn_count(NOW, 86400.0) == 1
        assert state.amount_sum(NOW, 86400.0, 1.0) == pytest.approx(1.0)


class TestBaseline:
    def test_zero_prior_spend_does_not_divide_by_zero(self):
        state = CustomerState.from_history(history((60.0, 0.0), (120.0, 0.0)))
        assert state.amount_mean_prior(50.0) == pytest.approx(0.0)  # caller guards the division

    def test_baseline_survives_truncation(self):
        """`prior_amount_mean` is stored separately from the capped event log."""
        state = CustomerState.from_history(
            history(*[(float(i + 1), float(i)) for i in range(50)]), max_events=3
        )
        assert state.amount_mean_prior(1.0) == pytest.approx(sum(range(50)) / 50)


@pytest.fixture(scope="module")
def store() -> InMemoryStateStore:
    transactions = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2"],
            "timestamp": pd.to_datetime(
                ["2024-01-01T09:00:00", "2024-01-01T10:00:00", "2024-01-01T11:00:00"]
            ),
            "amount": [10.0, 20.0, 300.0],
        }
    )
    return InMemoryStateStore.from_transactions(transactions)


class TestInMemoryStateStore:
    def test_indexes_every_customer(self, store):
        assert set(store.histories) == {"c1", "c2"}

    def test_customers_do_not_share_history(self, store):
        assert store.lookup("c1").prior_count == 2
        assert store.lookup("c2").prior_count == 1

    def test_unknown_customer_gets_empty_state_not_an_error(self, store):
        """A first-time cardholder is normal; failing would decline a good customer."""
        state = store.lookup("never-seen")
        assert state.is_new_customer
        assert state.prior_count == 0

    def test_baselines_are_per_customer(self, store):
        assert store.lookup("c1").prior_amount_mean == pytest.approx(15.0)
        assert store.lookup("c2").prior_amount_mean == pytest.approx(300.0)


class TestCausalLookup:
    """State is only defined relative to the transaction being scored."""

    def test_as_of_excludes_the_transaction_being_scored(self, store):
        """Scoring c1's 10:00 transaction must not see the 10:00 row itself."""
        state = store.lookup("c1", as_of=pd.Timestamp("2024-01-01T10:00:00"))
        assert state.prior_count == 1
        assert state.prior_amount_mean == pytest.approx(10.0)

    def test_as_of_excludes_the_future(self, store):
        state = store.lookup("c1", as_of=pd.Timestamp("2024-01-01T09:30:00"))
        assert state.prior_count == 1  # only the 09:00 transaction

    def test_as_of_before_all_history_yields_a_new_customer(self, store):
        assert store.lookup("c1", as_of=pd.Timestamp("2023-01-01")).is_new_customer

    def test_without_as_of_the_full_history_is_returned(self, store):
        """The live-serving case: the store holds only the past anyway."""
        assert store.lookup("c1").prior_count == 2

    def test_a_future_event_never_yields_a_negative_elapsed_time(self):
        """Defensive: a duplicated or late-arriving write must not produce a
        `seconds_since_prev_txn` the model never saw in training."""
        now = pd.Timestamp("2024-06-01T12:00:00")
        state = CustomerState(
            recent_events=(
                TransactionEvent(now - pd.Timedelta(seconds=30), 5.0),
                TransactionEvent(now + pd.Timedelta(seconds=30), 5.0),
            ),
            prior_amount_mean=5.0,
            prior_count=2,
        )
        assert state.seconds_since_prev_txn(now) == pytest.approx(30.0)

    def test_an_event_exactly_now_is_not_a_prior(self):
        now = pd.Timestamp("2024-06-01T12:00:00")
        state = CustomerState(
            recent_events=(TransactionEvent(now, 5.0),), prior_amount_mean=5.0, prior_count=1
        )
        assert state.seconds_since_prev_txn(now) == NO_PRIOR_TRANSACTION
