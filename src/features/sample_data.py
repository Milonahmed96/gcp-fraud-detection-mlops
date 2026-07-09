"""Deterministic synthetic transactions for local development and tests.

This is *not* the training data. It exists so that the feature pipeline can be
exercised end-to-end without a GCP project, and so the repository carries a
committed sample (`data/sample/transactions_sample.csv`) that anyone can run.

Fraud is injected with the correlations a real fraud model relies on -- foreign,
card-not-present, out-of-hours, and large relative to the customer's baseline --
so a model trained on it learns something rather than noise. The base rate is
deliberately low (~1.5%) to reproduce the class imbalance that makes accuracy a
useless metric here.

Regenerate with:  uv run python -m src.features.sample_data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_PATH = Path("data/sample/transactions_sample.csv")

N_CUSTOMERS = 40
N_TRANSACTIONS = 1_200
FRAUD_RATE = 0.015
SEED = 20240101

MERCHANT_CATEGORIES = ("grocery", "fuel", "restaurant", "electronics", "travel", "gambling")
HOME_COUNTRIES = ("GB", "GB", "GB", "IE", "FR")  # weighted toward GB
FOREIGN_COUNTRIES = ("RO", "NG", "US", "RU", "BR")

START = pd.Timestamp("2024-01-01T00:00:00")
DAYS_SPANNED = 60


def generate_transactions(
    *,
    n_customers: int = N_CUSTOMERS,
    n_transactions: int = N_TRANSACTIONS,
    fraud_rate: float = FRAUD_RATE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate a synthetic transaction frame matching `RAW_TRANSACTION_SCHEMA`."""
    rng = np.random.default_rng(seed)

    customer_ids = np.array([f"c_{i:03d}" for i in range(n_customers)])
    home_country = {c: rng.choice(HOME_COUNTRIES) for c in customer_ids}
    # Each customer has their own spending scale, so `amount_vs_customer_mean`
    # carries signal rather than just tracking the global amount distribution.
    customer_scale = {c: rng.uniform(15.0, 120.0) for c in customer_ids}

    customer = rng.choice(customer_ids, size=n_transactions)
    offsets = rng.uniform(0, DAYS_SPANNED * 24 * 3600, size=n_transactions)
    timestamp = START + pd.to_timedelta(np.sort(offsets), unit="s")

    is_fraud = (rng.random(n_transactions) < fraud_rate).astype(int)
    fraud = is_fraud.astype(bool)

    scales = np.array([customer_scale[c] for c in customer])
    amount = rng.gamma(shape=2.0, scale=scales / 2.0)
    # Fraudulent transactions are large relative to that customer's baseline.
    amount[fraud] *= rng.uniform(6.0, 25.0, size=fraud.sum())
    amount = np.round(amount, 2)

    home = np.array([home_country[c] for c in customer])
    country = home.copy()
    # ~70% of fraud is cross-border; ~3% of genuine traffic is too (holidays).
    foreign_mask = np.where(
        fraud, rng.random(n_transactions) < 0.70, rng.random(n_transactions) < 0.03
    )
    country[foreign_mask] = rng.choice(FOREIGN_COUNTRIES, size=foreign_mask.sum())

    # Fraud skews card-not-present; genuine traffic is mostly card-present.
    card_present = np.where(
        fraud, rng.random(n_transactions) < 0.15, rng.random(n_transactions) < 0.88
    )

    # Fraud skews into the small hours; genuine traffic keeps its natural time.
    moved_to_night = fraud & (rng.random(n_transactions) < 0.60)
    night_hour = pd.to_timedelta(rng.integers(0, 6, size=n_transactions), unit="h")
    timestamp = pd.DatetimeIndex(
        np.where(moved_to_night, timestamp.normalize() + night_hour, timestamp)
    )

    # Fraud over-indexes on the resale-friendly and high-risk categories.
    high_risk = fraud & (rng.random(n_transactions) < 0.40)
    merchant_category = np.where(
        high_risk,
        rng.choice(("electronics", "gambling", "travel"), size=n_transactions),
        rng.choice(MERCHANT_CATEGORIES, size=n_transactions),
    )

    df = pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:06d}" for i in range(n_transactions)],
            "customer_id": customer,
            "timestamp": timestamp,
            "amount": amount,
            "merchant_id": [f"m_{i:03d}" for i in rng.integers(0, 60, size=n_transactions)],
            "merchant_category": merchant_category,
            "country": country,
            "customer_home_country": home,
            "card_present": card_present,
            "is_fraud": is_fraud,
        }
    )

    # Re-sort after the night-hour shift perturbed the ordering.
    return df.sort_values("timestamp").reset_index(drop=True)


def write_sample(path: Path = SAMPLE_PATH) -> Path:
    """Generate the sample and write it to CSV, creating parent dirs as needed."""
    df = generate_transactions()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    """Read the committed sample back with the correct dtypes."""
    return pd.read_csv(path, parse_dates=["timestamp"])


if __name__ == "__main__":  # pragma: no cover
    written = write_sample()
    frame = load_sample(written)
    fraud_rate = frame["is_fraud"].mean()
    print(f"wrote {len(frame)} transactions to {written} (fraud rate {fraud_rate:.2%})")
