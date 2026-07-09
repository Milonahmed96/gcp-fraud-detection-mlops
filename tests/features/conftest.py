"""Shared fixtures for feature engineering tests."""

from __future__ import annotations

import pandas as pd
import pytest


def _txn(
    transaction_id: str,
    customer_id: str,
    timestamp: str,
    amount: float,
    *,
    country: str = "GB",
    home: str = "GB",
    card_present: bool = True,
    merchant_category: str = "grocery",
    is_fraud: int = 0,
) -> dict:
    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "timestamp": pd.Timestamp(timestamp),
        "amount": amount,
        "merchant_id": "m_001",
        "merchant_category": merchant_category,
        "country": country,
        "customer_home_country": home,
        "card_present": card_present,
        "is_fraud": is_fraud,
    }


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    """Two customers, deliberately interleaved and out of time order.

    Customer c1 has four transactions; the last is a large, foreign,
    card-not-present transaction 30 minutes after the third -- the shape of a
    real fraud event. Customer c2 has a single transaction, exercising the
    no-prior-history path.
    """
    rows = [
        _txn("t3", "c1", "2024-01-01T12:00:00", 30.0),
        _txn("t1", "c1", "2024-01-01T09:00:00", 10.0),
        _txn("t5", "c2", "2024-01-01T08:00:00", 500.0),
        _txn(
            "t4", "c1", "2024-01-01T12:30:00", 900.0, country="RO", card_present=False, is_fraud=1
        ),
        _txn("t2", "c1", "2024-01-01T10:00:00", 20.0),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def gcp_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """A fully-populated, non-placeholder GCP environment."""
    env = {
        "GCP_PROJECT_ID": "test-project",
        "GCP_REGION": "europe-west2",
        "GCP_BUCKET_NAME": "test-bucket",
        "BIGQUERY_DATASET": "fraud_features",
        "FEATURE_STORE_ID": "fraud_online_store",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env
