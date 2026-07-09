"""Feature engineering for fraud detection.

Design constraint: **every feature is causal.** A feature may depend on the
current transaction and on that customer's strictly-earlier transactions, never
on later ones. Violating this leaks the future into training, inflates offline
metrics, and produces a model that collapses in production because the online
Feature Store cannot supply the same values.

The two places this bites, and how they are handled here:

* Customer aggregates (`customer_amount_mean_prior`) use an *expanding* mean over
  `shift(1)`, so the current amount never contributes to its own baseline.
* Trailing windows (`txn_count_24h`, `amount_sum_24h`) are time-based rolling
  windows that include the current transaction -- which is legitimate, since at
  serving time the transaction being scored is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.schema import (
    EVENT_TIMESTAMP_COLUMN,
    LABEL_COLUMN,
    feature_names,
    required_raw_columns,
)

NIGHT_START_HOUR = 23
NIGHT_END_HOUR = 5  # inclusive; night is [23:00, 05:59]
SATURDAY = 5

#: Sentinel for "this customer has no earlier transaction". A negative value is
#: unreachable for a real elapsed time, so tree models can split it off cleanly.
NO_PRIOR_TRANSACTION = -1.0


class SchemaValidationError(ValueError):
    """Raised when a raw transaction frame does not satisfy the ingestion contract."""


def validate_raw_transactions(df: pd.DataFrame) -> None:
    """Check a raw transaction frame against the ingestion contract.

    Raises:
        SchemaValidationError: on missing columns, nulls in required columns,
            negative amounts, or a non-datetime timestamp column.
    """
    missing = [col for col in required_raw_columns() if col not in df.columns]
    if missing:
        raise SchemaValidationError(f"missing required columns: {', '.join(sorted(missing))}")

    null_counts = {
        col: int(df[col].isna().sum()) for col in required_raw_columns() if df[col].isna().any()
    }
    if null_counts:
        detail = ", ".join(f"{col}={n}" for col, n in sorted(null_counts.items()))
        raise SchemaValidationError(f"nulls in required columns: {detail}")

    if not pd.api.types.is_datetime64_any_dtype(df[EVENT_TIMESTAMP_COLUMN]):
        raise SchemaValidationError(
            f"{EVENT_TIMESTAMP_COLUMN!r} must be datetime64; got {df[EVENT_TIMESTAMP_COLUMN].dtype}"
        )

    if (df["amount"] < 0).any():
        n_negative = int((df["amount"] < 0).sum())
        raise SchemaValidationError(
            f"amount must be non-negative; found {n_negative} negative rows"
        )

    if LABEL_COLUMN in df.columns:
        # Cast through int() so numpy scalars render as `7`, not `np.int64(7)`.
        labels = {int(v) for v in df[LABEL_COLUMN].dropna().unique()}
        invalid = sorted(labels - {0, 1})
        if invalid:
            raise SchemaValidationError(f"{LABEL_COLUMN} must be 0/1; found {invalid}")


def _sorted_by_customer_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by (customer, time) -- the precondition for all causal aggregates."""
    return df.sort_values(["customer_id", EVENT_TIMESTAMP_COLUMN], kind="mergesort").reset_index(
        drop=True
    )


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, weekday, and the night/weekend flags. Row-local, no ordering needed."""
    out = df.copy()
    ts = out[EVENT_TIMESTAMP_COLUMN].dt
    out["hour_of_day"] = ts.hour.astype("int64")
    out["day_of_week"] = ts.dayofweek.astype("int64")
    out["is_night"] = (out["hour_of_day"] >= NIGHT_START_HOUR) | (
        out["hour_of_day"] <= NIGHT_END_HOUR
    )
    out["is_weekend"] = out["day_of_week"] >= SATURDAY
    return out


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """log1p the amount. Fraud amounts are heavy-tailed; the raw scale is unusable."""
    out = df.copy()
    out["amount_log"] = np.log1p(out["amount"].astype("float64"))
    return out


def add_channel_and_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Card-not-present and cross-border flags -- both classic fraud signals."""
    out = df.copy()
    out["card_not_present"] = ~out["card_present"].astype(bool)
    out["is_foreign"] = out["country"] != out["customer_home_country"]
    return out


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing-window transaction velocity per customer.

    Windows include the current transaction, which is causal: when scoring a
    live transaction its own amount and timestamp are known. Requires the frame
    to be sorted by (customer_id, timestamp).
    """
    out = _sorted_by_customer_time(df)

    # Time-based rolling needs the timestamp on the index. groupby(...).rolling()
    # preserves within-group row order, and the frame is sorted by customer then
    # time, so the result aligns positionally with `out`.
    indexed = out.set_index(EVENT_TIMESTAMP_COLUMN)
    grouped = indexed.groupby("customer_id")["amount"]

    out["txn_count_1h"] = grouped.rolling("1h").count().to_numpy().astype("int64")
    out["txn_count_24h"] = grouped.rolling("24h").count().to_numpy().astype("int64")
    out["amount_sum_24h"] = grouped.rolling("24h").sum().to_numpy().astype("float64")

    elapsed = out.groupby("customer_id")[EVENT_TIMESTAMP_COLUMN].diff().dt.total_seconds()
    out["seconds_since_prev_txn"] = elapsed.fillna(NO_PRIOR_TRANSACTION).astype("float64")

    return out


def add_customer_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compare this amount against the customer's own spending baseline.

    The baseline is an expanding mean over *strictly prior* transactions
    (`shift(1)`), so a transaction never contributes to the baseline it is
    measured against. For a customer's first transaction there is no baseline;
    we fall back to the current amount, which makes the ratio exactly 1.0 --
    i.e. "unremarkable", the correct prior when nothing is known.
    """
    out = _sorted_by_customer_time(df)

    prior_mean = out.groupby("customer_id")["amount"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    prior_mean = prior_mean.fillna(out["amount"]).astype("float64")
    out["customer_amount_mean_prior"] = prior_mean

    # A customer whose entire prior history is zero-amount transactions would
    # divide by zero. Treat that as "no useful baseline" -> ratio 1.0.
    ratio = np.where(prior_mean > 0, out["amount"] / prior_mean.replace(0, np.nan), 1.0)
    out["amount_vs_customer_mean"] = pd.Series(ratio, index=out.index).fillna(1.0).astype("float64")

    return out


def build_feature_frame(df: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
    """Run the full feature pipeline over a raw transaction frame.

    Args:
        df: Raw transactions matching `RAW_TRANSACTION_SCHEMA`.
        validate: Whether to enforce the ingestion contract first.

    Returns:
        A new frame sorted by (customer_id, timestamp) carrying the original
        columns plus every column in `FEATURE_SPECS`.
    """
    if validate:
        validate_raw_transactions(df)

    out = _sorted_by_customer_time(df)
    out = add_temporal_features(out)
    out = add_amount_features(out)
    out = add_channel_and_geo_features(out)
    out = add_velocity_features(out)
    out = add_customer_profile_features(out)

    produced = set(out.columns)
    expected = set(feature_names())
    missing = expected - produced
    if missing:  # pragma: no cover -- guards against schema/transform drift
        raise SchemaValidationError(
            f"pipeline did not produce declared features: {', '.join(sorted(missing))}"
        )

    return out
