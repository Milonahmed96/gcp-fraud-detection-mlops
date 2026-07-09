"""Customer state for online feature computation.

This module exists to defeat train/serve skew.

Offline, `src/features/transforms.py` computes a customer's velocity features
with pandas time-based rolling windows over their full history. Online, we have
one incoming transaction and whatever the Feature Store can hand us. If the two
paths disagree by even a boundary condition, the model sees inputs at serving
time that it never saw in training, and its calibration silently rots.

The naive fix -- store precomputed window aggregates in the Feature Store -- does
not work: a trailing 24h window depends on *when you ask*, and the stored value
was computed when the customer last transacted. It would be stale by exactly the
gap between transactions, which is precisely when velocity matters most.

So the online store holds the customer's **recent event log** (timestamps and
amounts) plus their prior spending baseline, and windows are computed exactly at
request time against the incoming transaction's own timestamp. `tests/inference/
test_skew.py` asserts, transaction by transaction over the sample, that this
produces byte-identical features to the offline pipeline.

Boundary note: pandas time-based rolling uses a half-open interval `(t - w, t]`.
An event exactly `w` before the current transaction is **excluded**. `_within`
replicates that with a strict `<`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.features.transforms import NO_PRIOR_TRANSACTION

ONE_HOUR_SECONDS = 3600.0
ONE_DAY_SECONDS = 86_400.0

#: How many of the customer's recent transactions the online store keeps.
#: Bounds the payload while comfortably covering a 24h window -- a customer
#: making more than this many transactions in a day is itself anomalous, and
#: `truncated` records when the cap bit.
MAX_RECENT_EVENTS = 100

#: Nanoseconds per second. See `elapsed_seconds`.
NANOSECONDS_PER_SECOND = 1_000_000_000


def elapsed_seconds(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    """Seconds between two timestamps, at nanosecond resolution.

    `pd.Timedelta.total_seconds()` inherits `datetime.timedelta`'s **microsecond**
    resolution and silently truncates. The offline pipeline uses
    `Series.dt.total_seconds()`, which divides in nanoseconds. Using the scalar
    method here produced `220.472044` where training saw `220.472044709` -- a
    real train/serve skew in `seconds_since_prev_txn`, caught by
    `tests/inference/test_skew.py`.
    """
    return (later - earlier).value / NANOSECONDS_PER_SECOND


@dataclass(frozen=True)
class TransactionEvent:
    """One of a customer's prior transactions, as held in the online store."""

    timestamp: pd.Timestamp
    amount: float


@dataclass(frozen=True)
class CustomerState:
    """A customer's history, as of just before the transaction being scored.

    Attributes:
        recent_events: Prior transactions, ascending by timestamp. Only those
            inside the widest window (24h) affect velocity features, but the
            spending baseline needs the count and mean of *all* priors.
        prior_amount_mean: Expanding mean of every prior amount. Carried
            separately because `recent_events` is truncated.
        prior_count: How many prior transactions the customer has, ever.
        truncated: True if `recent_events` was capped at `MAX_RECENT_EVENTS`.
    """

    recent_events: tuple[TransactionEvent, ...] = ()
    prior_amount_mean: float | None = None
    prior_count: int = 0
    truncated: bool = False

    @property
    def is_new_customer(self) -> bool:
        """No prior transactions at all."""
        return self.prior_count == 0

    @classmethod
    def empty(cls) -> CustomerState:
        """State for a customer the online store has never seen."""
        return cls()

    @classmethod
    def from_history(
        cls, history: pd.DataFrame, *, max_events: int = MAX_RECENT_EVENTS
    ) -> CustomerState:
        """Build state from a customer's prior transactions.

        Args:
            history: Rows for one customer, strictly earlier than the
                transaction being scored. Needs `timestamp` and `amount`.
        """
        if history.empty:
            return cls.empty()

        ordered = history.sort_values("timestamp", kind="mergesort")
        amounts = ordered["amount"].astype(float)

        recent = ordered.tail(max_events)
        events = tuple(
            TransactionEvent(timestamp=pd.Timestamp(row.timestamp), amount=float(row.amount))
            for row in recent.itertuples()
        )

        return cls(
            recent_events=events,
            prior_amount_mean=float(amounts.mean()),
            prior_count=len(ordered),
            truncated=len(ordered) > max_events,
        )

    def _within(self, now: pd.Timestamp, window_seconds: float) -> list[TransactionEvent]:
        """Prior events inside the half-open window `(now - w, now]`.

        The strict `<` mirrors pandas' `closed="right"` rolling semantics: an
        event exactly `w` seconds before `now` falls outside the window.
        """
        return [
            event
            for event in self.recent_events
            if 0.0 <= elapsed_seconds(now, event.timestamp) < window_seconds
        ]

    def txn_count(self, now: pd.Timestamp, window_seconds: float) -> int:
        """Transactions in the trailing window, **including** the current one.

        Causal: at serving time the transaction being scored is known.
        """
        return len(self._within(now, window_seconds)) + 1

    def amount_sum(self, now: pd.Timestamp, window_seconds: float, current_amount: float) -> float:
        """Summed amounts in the trailing window, including the current transaction."""
        return sum(event.amount for event in self._within(now, window_seconds)) + current_amount

    def seconds_since_prev_txn(self, now: pd.Timestamp) -> float:
        """Seconds since this customer's most recent prior transaction.

        Events at or after `now` are ignored. The online store should never
        return them, but a late-arriving or duplicated write would otherwise
        yield a negative elapsed time -- a value the model never saw in training.
        """
        past = [event.timestamp for event in self.recent_events if event.timestamp < now]
        if not past:
            return NO_PRIOR_TRANSACTION
        return elapsed_seconds(now, max(past))

    def amount_mean_prior(self, current_amount: float) -> float:
        """The customer's prior spending baseline.

        With no history there is no baseline, so we fall back to the current
        amount -- which makes `amount_vs_customer_mean` exactly 1.0, the correct
        "unremarkable" prior when nothing is known. This mirrors
        `transforms.add_customer_profile_features`.
        """
        if self.is_new_customer or self.prior_amount_mean is None:
            return float(current_amount)
        return float(self.prior_amount_mean)


@dataclass
class InMemoryStateStore:
    """A local stand-in for the Vertex AI Feature Store.

    Lets the inference service run end-to-end with no GCP project -- which is
    what the tests and `docker run` use. Production swaps in the Feature Store
    reader; both satisfy the same `lookup(customer_id, as_of)` signature.

    Holds each customer's raw history rather than a precomputed `CustomerState`,
    because state is only meaningful relative to a point in time. Recomputing
    per lookup is cheap here and keeps the store honest.
    """

    histories: dict[str, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def from_transactions(cls, transactions: pd.DataFrame) -> InMemoryStateStore:
        """Index every customer's history from a raw transaction frame."""
        return cls(
            histories={
                str(customer_id): group[["timestamp", "amount"]].reset_index(drop=True)
                for customer_id, group in transactions.groupby("customer_id")
            }
        )

    def lookup(self, customer_id: str, as_of: pd.Timestamp | None = None) -> CustomerState:
        """Fetch a customer's state as of `as_of`, exclusive.

        `as_of` is not optional in spirit: a customer's state is only defined
        relative to the transaction being scored. Passing the incoming
        timestamp is what stops a replayed historical transaction from seeing
        its own future -- and what stops a duplicated write from doing the same
        in production.

        Unknown customers get empty state, not an error. A first-time cardholder
        is a completely normal event at a payment gateway; failing the request
        would decline a legitimate new customer.
        """
        history = self.histories.get(customer_id)
        if history is None:
            return CustomerState.empty()
        if as_of is not None:
            history = history[history["timestamp"] < as_of]
        return CustomerState.from_history(history)
