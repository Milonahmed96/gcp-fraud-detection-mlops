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
N_TRANSACTIONS = 6_000
FRAUD_RATE = 0.02
SEED = 20240101

#: Fraction of fraud that carries **no** injected signal: a normal-sized,
#: domestic, card-present, daytime transaction that happens to be fraudulent.
#: Real fraud includes stolen cards used carefully. Without this cohort both
#: variants score a perfect AUC on the sample, the A/B test becomes degenerate,
#: and the dashboard reports a number no reviewer would believe.
STEALTH_FRAUD_FRACTION = 0.35

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

    # Split fraud into "loud" (carries the classic signals) and "stealth"
    # (indistinguishable from genuine traffic on features alone). Stealth fraud
    # sets the ceiling on achievable recall -- exactly as it does in production.
    stealth = fraud & (rng.random(n_transactions) < STEALTH_FRAUD_FRACTION)
    loud = fraud & ~stealth

    scales = np.array([customer_scale[c] for c in customer])
    amount = rng.gamma(shape=2.0, scale=scales / 2.0)
    # Loud fraud is large relative to that customer's baseline, but the multiplier
    # overlaps the upper tail of genuine spending -- so the classes are not
    # linearly separable on amount alone.
    amount[loud] *= rng.uniform(1.8, 7.0, size=loud.sum())
    amount = np.round(amount, 2)

    home = np.array([home_country[c] for c in customer])
    country = home.copy()
    # ~55% of loud fraud is cross-border; ~6% of genuine traffic is too (holidays).
    foreign_mask = np.where(
        loud, rng.random(n_transactions) < 0.55, rng.random(n_transactions) < 0.06
    )
    country[foreign_mask] = rng.choice(FOREIGN_COUNTRIES, size=foreign_mask.sum())

    # Loud fraud skews card-not-present; genuine traffic is mostly card-present.
    # The 20% genuine CNP rate (online shopping) blunts the signal.
    card_present = np.where(
        loud, rng.random(n_transactions) < 0.30, rng.random(n_transactions) < 0.80
    )

    # Loud fraud skews into the small hours; genuine traffic keeps its natural time.
    moved_to_night = loud & (rng.random(n_transactions) < 0.45)
    night_hour = pd.to_timedelta(rng.integers(0, 6, size=n_transactions), unit="h")
    timestamp = pd.DatetimeIndex(
        np.where(moved_to_night, timestamp.normalize() + night_hour, timestamp)
    )

    # Loud fraud over-indexes on the resale-friendly and high-risk categories.
    high_risk = loud & (rng.random(n_transactions) < 0.40)
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
