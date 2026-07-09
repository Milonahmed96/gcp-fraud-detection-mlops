"""Training data loading and splitting.

Two sources, selected by flag:

* `sample` -- the committed synthetic CSV. Fast, free, no GCP, no credentials.
  This is what CI and the test suite use.
* `bigquery` -- the real offline feature store. What the portfolio demo uses.

Both return the same engineered feature frame, because both run through
`transforms.build_feature_frame`. The BigQuery path reads features that were
already materialised by the Phase 2 ingestion job; the sample path computes them
on the fly from raw transactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.features.bigquery import fetch_training_set
from src.features.config import GCPConfig
from src.features.sample_data import SAMPLE_PATH, load_sample
from src.features.schema import EVENT_TIMESTAMP_COLUMN, LABEL_COLUMN, feature_names
from src.features.transforms import build_feature_frame

DataSource = Literal["sample", "bigquery"]

#: Fraction of the timeline held out for testing. A temporal split, not a random
#: one -- see `temporal_split`.
DEFAULT_TEST_FRACTION = 0.2

#: Of the training portion, how much is carved off to fit the decision threshold.
DEFAULT_VALIDATION_FRACTION = 0.25


class DatasetError(ValueError):
    """Raised when the loaded data cannot support training."""


@dataclass(frozen=True)
class Split:
    """One temporal slice of the data, with features, labels, and amounts.

    `amounts` is carried alongside because the business cost metric prices a
    missed fraud at the transaction's own value.
    """

    X: pd.DataFrame
    y: np.ndarray
    amounts: np.ndarray
    timestamps: pd.Series

    def __len__(self) -> int:
        return len(self.X)

    @property
    def fraud_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else 0.0

    @property
    def n_fraud(self) -> int:
        return int(self.y.sum())


@dataclass(frozen=True)
class Dataset:
    """A temporally-ordered train / validation / test partition."""

    train: Split
    validation: Split
    test: Split

    @property
    def scale_pos_weight(self) -> float:
        """Ratio of negatives to positives in the training split.

        Fed to both XGBoost and LightGBM so the rare positive class is not
        drowned out. Identical for both variants, so it cannot confound the A/B
        comparison.
        """
        positives = self.train.n_fraud
        if positives == 0:
            raise DatasetError("training split contains no fraud; cannot weight the positive class")
        return float(len(self.train) - positives) / float(positives)


def load_features(
    source: DataSource = "sample",
    *,
    config: GCPConfig | None = None,
    client: Any | None = None,
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
    sample_path: Path = SAMPLE_PATH,
) -> pd.DataFrame:
    """Load engineered features from the chosen source.

    Args:
        source: `"sample"` for the committed CSV, `"bigquery"` for the offline store.
        config: Required for the BigQuery path.
        client: Required for the BigQuery path. Injected, never constructed here.
        start_date, end_date: Required date bounds for the BigQuery path.
        sample_path: Override for the sample CSV location.
    """
    if source == "sample":
        raw = load_sample(sample_path)
        return build_feature_frame(raw)

    if source == "bigquery":
        missing = [
            name
            for name, value in [
                ("config", config),
                ("client", client),
                ("start_date", start_date),
                ("end_date", end_date),
            ]
            if value is None
        ]
        if missing:
            raise DatasetError(f"bigquery source requires: {', '.join(missing)}")
        assert config is not None  # narrowed by the check above
        return fetch_training_set(client, config, start_date=start_date, end_date=end_date)

    raise DatasetError(f"unknown source {source!r}; expected 'sample' or 'bigquery'")


def _slice(frame: pd.DataFrame) -> Split:
    return Split(
        X=frame[list(feature_names())].reset_index(drop=True),
        y=frame[LABEL_COLUMN].to_numpy().astype(int),
        amounts=frame["amount"].to_numpy().astype(float)
        if "amount" in frame.columns
        else np.zeros(len(frame)),
        timestamps=frame[EVENT_TIMESTAMP_COLUMN].reset_index(drop=True),
    )


def temporal_split(
    frame: pd.DataFrame,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> Dataset:
    """Split by time, never at random.

    Fraud is non-stationary: attack patterns emerge, get detected, and die. A
    random split lets the model see next month's fraud ring while training on
    this month's, which inflates offline AUC and produces a model that decays
    the moment it ships. Splitting on the timeline reproduces the real task --
    predict tomorrow from yesterday.

    The validation slice sits between train and test, and exists to fit the
    decision threshold. Fitting the threshold on test would leak.
    """
    if not 0 < test_fraction < 1:
        raise DatasetError(f"test_fraction must be in (0, 1); got {test_fraction}")
    if not 0 <= validation_fraction < 1:
        raise DatasetError(f"validation_fraction must be in [0, 1); got {validation_fraction}")
    if LABEL_COLUMN not in frame.columns:
        raise DatasetError(f"frame has no {LABEL_COLUMN} column; cannot train")

    labelled = frame[frame[LABEL_COLUMN].notna()]
    if len(labelled) < 10:
        raise DatasetError(f"need at least 10 labelled rows to split; got {len(labelled)}")

    ordered = labelled.sort_values(EVENT_TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)

    n = len(ordered)
    n_test = int(round(n * test_fraction))
    n_train_pool = n - n_test
    n_validation = int(round(n_train_pool * validation_fraction))
    n_train = n_train_pool - n_validation

    if min(n_train, n_test) < 1:
        raise DatasetError("split produced an empty train or test slice; adjust the fractions")

    dataset = Dataset(
        train=_slice(ordered.iloc[:n_train]),
        validation=_slice(ordered.iloc[n_train:n_train_pool]),
        test=_slice(ordered.iloc[n_train_pool:]),
    )

    if dataset.test.n_fraud == 0:
        raise DatasetError(
            "test split contains no fraud; metrics would be undefined. "
            "Widen the date range or lower test_fraction."
        )
    return dataset
