"""Schemas for raw transactions and engineered features.

This module is intentionally free of GCP imports: it is the single source of
truth for column names and types, consumed by `bigquery.py` (which converts the
specs into `bigquery.SchemaField`) and by `feature_store.py` (which converts
them into Vertex AI feature definitions). Tests assert that the engineered
frame produced by `transforms.py` matches `FEATURE_SPECS` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FieldType = Literal["STRING", "INTEGER", "FLOAT", "BOOLEAN", "TIMESTAMP"]
Mode = Literal["REQUIRED", "NULLABLE"]


@dataclass(frozen=True)
class FieldSpec:
    """One column: name, BigQuery type, nullability, and what it means."""

    name: str
    field_type: FieldType
    mode: Mode = "REQUIRED"
    description: str = ""


# The entity we key the online store on. One row per transaction at serving
# time, but customer-level aggregates are looked up by customer_id.
ENTITY_ID_COLUMN = "customer_id"
EVENT_TIMESTAMP_COLUMN = "timestamp"
TRANSACTION_ID_COLUMN = "transaction_id"
LABEL_COLUMN = "is_fraud"

RAW_TRANSACTIONS_TABLE = "raw_transactions"
FEATURES_TABLE = "transaction_features"
PREDICTIONS_TABLE = "prediction_log"


RAW_TRANSACTION_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("transaction_id", "STRING", "REQUIRED", "Unique transaction identifier"),
    FieldSpec("customer_id", "STRING", "REQUIRED", "Cardholder identifier (entity key)"),
    FieldSpec("timestamp", "TIMESTAMP", "REQUIRED", "Transaction event time, UTC"),
    FieldSpec("amount", "FLOAT", "REQUIRED", "Transaction amount in minor-unit-normalised GBP"),
    FieldSpec("merchant_id", "STRING", "REQUIRED", "Merchant identifier"),
    FieldSpec("merchant_category", "STRING", "REQUIRED", "Merchant category code group"),
    FieldSpec("country", "STRING", "REQUIRED", "ISO-3166 alpha-2 country of the transaction"),
    FieldSpec(
        "customer_home_country", "STRING", "REQUIRED", "ISO-3166 alpha-2 country of the customer"
    ),
    FieldSpec(
        "card_present",
        "BOOLEAN",
        "REQUIRED",
        "True for chip-and-PIN / contactless, False for card-not-present",
    ),
    FieldSpec(
        "is_fraud",
        "INTEGER",
        "NULLABLE",
        "Label: 1 if confirmed fraudulent, 0 otherwise, NULL if unlabelled",
    ),
)


# Engineered features. Every one of these is causal -- computed only from the
# current transaction and the customer's *prior* transactions -- so the same
# code can run offline over history and online over a Feature Store lookup
# without train/serve skew.
FEATURE_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "amount_log", "FLOAT", "REQUIRED", "log1p of transaction amount; tames the heavy right tail"
    ),
    FieldSpec("hour_of_day", "INTEGER", "REQUIRED", "Hour 0-23 of the transaction, UTC"),
    FieldSpec("day_of_week", "INTEGER", "REQUIRED", "Monday=0 .. Sunday=6"),
    FieldSpec("is_night", "BOOLEAN", "REQUIRED", "True between 23:00 and 05:59 inclusive"),
    FieldSpec("is_weekend", "BOOLEAN", "REQUIRED", "True on Saturday or Sunday"),
    FieldSpec(
        "is_foreign",
        "BOOLEAN",
        "REQUIRED",
        "True when transaction country != customer home country",
    ),
    FieldSpec(
        "card_not_present",
        "BOOLEAN",
        "REQUIRED",
        "Inverse of card_present; the higher-risk channel",
    ),
    FieldSpec(
        "seconds_since_prev_txn",
        "FLOAT",
        "REQUIRED",
        "Seconds since this customer's previous transaction; -1 if none",
    ),
    FieldSpec(
        "txn_count_1h",
        "INTEGER",
        "REQUIRED",
        "Customer transactions in the trailing 1h, inclusive of this one",
    ),
    FieldSpec(
        "txn_count_24h",
        "INTEGER",
        "REQUIRED",
        "Customer transactions in the trailing 24h, inclusive of this one",
    ),
    FieldSpec(
        "amount_sum_24h",
        "FLOAT",
        "REQUIRED",
        "Sum of customer amounts in the trailing 24h, inclusive of this one",
    ),
    FieldSpec(
        "customer_amount_mean_prior",
        "FLOAT",
        "REQUIRED",
        "Expanding mean of the customer's prior amounts; equals this amount if none",
    ),
    FieldSpec(
        "amount_vs_customer_mean",
        "FLOAT",
        "REQUIRED",
        "amount / customer_amount_mean_prior; 1.0 for a first transaction",
    ),
)


# The audit log. One row per served prediction, carrying the SHAP attributions
# that justified it. This is the table a regulator queries: "why was transaction
# X blocked on date Y?" It is written by src/evaluation/experiments.py and by
# the inference service.
PREDICTION_LOG_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("transaction_id", "STRING", "REQUIRED", "Transaction that was scored"),
    FieldSpec("customer_id", "STRING", "REQUIRED", "Cardholder the transaction belongs to"),
    FieldSpec("timestamp", "TIMESTAMP", "REQUIRED", "Time the prediction was served, UTC"),
    FieldSpec("variant", "STRING", "REQUIRED", "Serving model variant: xgboost or lightgbm"),
    FieldSpec("fraud_probability", "FLOAT", "REQUIRED", "Predicted probability of fraud"),
    FieldSpec("threshold", "FLOAT", "REQUIRED", "Decision threshold applied to the probability"),
    FieldSpec("is_flagged", "BOOLEAN", "REQUIRED", "Whether the transaction was flagged as fraud"),
    FieldSpec("base_value", "FLOAT", "REQUIRED", "SHAP base value, log-odds space"),
    FieldSpec(
        "top_features",
        "STRING",
        "REQUIRED",
        "JSON array of top SHAP attributions; query with JSON_VALUE",
    ),
    FieldSpec("latency_ms", "FLOAT", "NULLABLE", "Server-side handler latency in milliseconds"),
)


def feature_names() -> tuple[str, ...]:
    """Ordered names of the engineered features fed to the model."""
    return tuple(spec.name for spec in FEATURE_SPECS)


def prediction_log_columns() -> tuple[str, ...]:
    """Ordered column names of the prediction audit log."""
    return tuple(spec.name for spec in PREDICTION_LOG_SCHEMA)


def raw_column_names() -> tuple[str, ...]:
    """Ordered names of the raw transaction columns."""
    return tuple(spec.name for spec in RAW_TRANSACTION_SCHEMA)


def required_raw_columns() -> tuple[str, ...]:
    """Raw columns that must be present and non-null on ingestion (excludes the label)."""
    return tuple(
        spec.name
        for spec in RAW_TRANSACTION_SCHEMA
        if spec.mode == "REQUIRED" and spec.name != LABEL_COLUMN
    )
