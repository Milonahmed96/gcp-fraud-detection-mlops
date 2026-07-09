"""Build the model's feature vector for a single incoming transaction.

The offline pipeline (`src/features/transforms.py`) engineers features over a
whole DataFrame of history. Here we have one transaction plus a `CustomerState`
lookup, and we must produce a feature vector *identical* to what the offline
pipeline would have produced for that same transaction.

Every constant and boundary condition below is deliberately duplicated from
`transforms.py` rather than re-derived, and `tests/inference/test_skew.py`
asserts equality against the real pipeline over the sample data. If those two
implementations ever drift, that test fails rather than the model quietly
degrading in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.schema import feature_names
from src.features.transforms import NIGHT_END_HOUR, NIGHT_START_HOUR, SATURDAY
from src.inference.state import ONE_DAY_SECONDS, ONE_HOUR_SECONDS, CustomerState


def build_serving_features(
    *,
    timestamp: pd.Timestamp,
    amount: float,
    country: str,
    customer_home_country: str,
    card_present: bool,
    state: CustomerState,
) -> pd.DataFrame:
    """Engineer the model's features for one transaction.

    Returns a single-row DataFrame whose columns are exactly `feature_names()`,
    in the order the model was trained on. Column order matters: XGBoost and
    LightGBM both index features positionally once fitted.
    """
    timestamp = pd.Timestamp(timestamp)
    amount = float(amount)

    hour_of_day = int(timestamp.hour)
    day_of_week = int(timestamp.dayofweek)

    mean_prior = state.amount_mean_prior(amount)
    # Guard the same division `transforms.add_customer_profile_features` guards:
    # a customer whose entire prior history is zero-amount transactions.
    ratio = amount / mean_prior if mean_prior > 0 else 1.0

    row = {
        "amount_log": float(np.log1p(amount)),
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_night": hour_of_day >= NIGHT_START_HOUR or hour_of_day <= NIGHT_END_HOUR,
        "is_weekend": day_of_week >= SATURDAY,
        "is_foreign": country != customer_home_country,
        "card_not_present": not card_present,
        "seconds_since_prev_txn": state.seconds_since_prev_txn(timestamp),
        "txn_count_1h": state.txn_count(timestamp, ONE_HOUR_SECONDS),
        "txn_count_24h": state.txn_count(timestamp, ONE_DAY_SECONDS),
        "amount_sum_24h": state.amount_sum(timestamp, ONE_DAY_SECONDS, amount),
        "customer_amount_mean_prior": mean_prior,
        "amount_vs_customer_mean": float(ratio),
    }

    frame = pd.DataFrame([row])[list(feature_names())]
    return _match_training_dtypes(frame)


def _match_training_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the dtypes the offline pipeline emits.

    The models were fitted on int64 counts, float64 amounts, and bool flags.
    Handing XGBoost a float where it trained on an int is harmless; handing it
    an object column is not. Being explicit costs nothing and removes a class of
    Heisenbug.
    """
    integer_columns = ["hour_of_day", "day_of_week", "txn_count_1h", "txn_count_24h"]
    float_columns = [
        "amount_log",
        "seconds_since_prev_txn",
        "amount_sum_24h",
        "customer_amount_mean_prior",
        "amount_vs_customer_mean",
    ]
    bool_columns = ["is_night", "is_weekend", "is_foreign", "card_not_present"]

    return frame.astype(
        {
            **{name: "int64" for name in integer_columns},
            **{name: "float64" for name in float_columns},
            **{name: "bool" for name in bool_columns},
        }
    )
