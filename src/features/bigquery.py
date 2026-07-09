"""BigQuery as the offline feature store and audit log.

GCP imports are deferred to call time so that `schema.py` and `transforms.py`
stay importable (and testable) without the cloud SDK on the path.

Nothing here creates a client implicitly: callers pass one in. That keeps every
function unit-testable against a fake and makes the credential boundary explicit.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.features.config import GCPConfig
from src.features.schema import (
    EVENT_TIMESTAMP_COLUMN,
    FEATURE_SPECS,
    FEATURES_TABLE,
    LABEL_COLUMN,
    RAW_TRANSACTION_SCHEMA,
    RAW_TRANSACTIONS_TABLE,
    FieldSpec,
    feature_names,
)

if TYPE_CHECKING:  # pragma: no cover
    from google.cloud import bigquery


def _bq():
    """Import the BigQuery SDK lazily."""
    from google.cloud import bigquery

    return bigquery


def to_bigquery_schema(specs: tuple[FieldSpec, ...]) -> list[bigquery.SchemaField]:
    """Convert our dependency-free FieldSpecs into BigQuery SchemaFields."""
    bigquery = _bq()
    return [
        bigquery.SchemaField(
            name=spec.name,
            field_type=spec.field_type,
            mode=spec.mode,
            description=spec.description or None,
        )
        for spec in specs
    ]


def features_table_schema() -> list[bigquery.SchemaField]:
    """Schema of the engineered feature table: keys + label + every feature."""
    keys = tuple(
        spec
        for spec in RAW_TRANSACTION_SCHEMA
        if spec.name in {"transaction_id", "customer_id", EVENT_TIMESTAMP_COLUMN, LABEL_COLUMN}
    )
    return to_bigquery_schema(keys + FEATURE_SPECS)


def create_client(config: GCPConfig) -> bigquery.Client:
    """Build an authenticated BigQuery client from Application Default Credentials."""
    return _bq().Client(project=config.project_id)


def ensure_dataset(client: Any, config: GCPConfig) -> str:
    """Create the dataset in the configured region if it does not already exist.

    Returns the fully-qualified dataset reference.
    """
    bigquery = _bq()
    dataset = bigquery.Dataset(config.dataset_ref)
    dataset.location = config.region
    client.create_dataset(dataset, exists_ok=True)
    return config.dataset_ref


def ensure_table(
    client: Any,
    config: GCPConfig,
    table_name: str,
    schema: list[bigquery.SchemaField],
    *,
    partition_field: str | None = EVENT_TIMESTAMP_COLUMN,
    cluster_fields: tuple[str, ...] = ("customer_id",),
) -> str:
    """Create a partitioned, clustered table if absent. Returns its table ref.

    Partitioning on the event timestamp and clustering on `customer_id` is what
    keeps the training-set query inside the BigQuery free tier: a date-bounded
    scan of one customer's history touches a handful of blocks rather than the
    whole table.
    """
    bigquery = _bq()
    table_ref = config.table_ref(table_name)
    table = bigquery.Table(table_ref, schema=schema)

    if partition_field:
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field,
        )
    if cluster_fields:
        table.clustering_fields = list(cluster_fields)

    client.create_table(table, exists_ok=True)
    return table_ref


def load_dataframe(
    client: Any,
    config: GCPConfig,
    df: pd.DataFrame,
    table_name: str,
    schema: list[bigquery.SchemaField],
    *,
    write_disposition: str = "WRITE_APPEND",
) -> int:
    """Load a DataFrame into BigQuery and block until the job finishes.

    Returns the number of rows submitted. Raises whatever the load job raises --
    a partially-loaded table is worse than a loud failure.
    """
    bigquery = _bq()
    table_ref = config.table_ref(table_name)
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition=write_disposition)

    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # blocks; raises on failure
    return len(df)


def ingest_raw_transactions(client: Any, config: GCPConfig, df: pd.DataFrame) -> int:
    """Append raw transactions to the landing table."""
    return load_dataframe(
        client, config, df, RAW_TRANSACTIONS_TABLE, to_bigquery_schema(RAW_TRANSACTION_SCHEMA)
    )


def ingest_features(client: Any, config: GCPConfig, df: pd.DataFrame) -> int:
    """Append engineered features to the offline feature table.

    Only the key columns, the label, and the declared features are written --
    the raw columns they were derived from stay in `raw_transactions`.
    """
    columns = [
        "transaction_id",
        "customer_id",
        EVENT_TIMESTAMP_COLUMN,
        LABEL_COLUMN,
        *feature_names(),
    ]
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"feature frame is missing columns: {', '.join(missing)}")

    return load_dataframe(client, config, df[columns], FEATURES_TABLE, features_table_schema())


def training_set_query(config: GCPConfig) -> str:
    """SQL for a labelled, date-bounded training set.

    Dates arrive as bound query parameters (`@start_date`, `@end_date`), never
    interpolated into the string. The date bound is not cosmetic: it prunes
    partitions, and it is how we build a *temporal* train/test split. Random
    splits leak future fraud patterns into the training set and overstate AUC.

    The table reference is interpolated because BigQuery cannot parameterise
    identifiers; it comes from validated config, not from user input.
    """
    feature_columns = ",\n    ".join(feature_names())
    return f"""
SELECT
    transaction_id,
    customer_id,
    {EVENT_TIMESTAMP_COLUMN},
    {feature_columns},
    {LABEL_COLUMN}
FROM `{config.table_ref(FEATURES_TABLE)}`
WHERE {EVENT_TIMESTAMP_COLUMN} >= @start_date
  AND {EVENT_TIMESTAMP_COLUMN} <  @end_date
  AND {LABEL_COLUMN} IS NOT NULL
ORDER BY {EVENT_TIMESTAMP_COLUMN}
""".strip()


def _as_timestamp(name: str, value: datetime | str) -> datetime:
    """Normalise a date bound to a `datetime`.

    BigQuery's `ScalarQueryParameter` serialises TIMESTAMP values to RFC3339 and
    parses them back on read, so a bare `"2024-01-01"` round-trips badly. Coerce
    up front, and reject anything that is not a timestamp -- which incidentally
    stops a malformed bound long before it reaches BigQuery.
    """
    try:
        parsed = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} is not a valid timestamp: {value!r}") from exc
    if pd.isna(parsed):
        raise ValueError(f"{name} is not a valid timestamp: {value!r}")
    return parsed.to_pydatetime()


def fetch_training_set(
    client: Any, config: GCPConfig, *, start_date: datetime | str, end_date: datetime | str
) -> pd.DataFrame:
    """Run `training_set_query` over `[start_date, end_date)` and return a DataFrame.

    Args:
        start_date: Inclusive lower bound. A `datetime`, or any string pandas can
            parse as one (e.g. `"2024-01-01"`).
        end_date: Exclusive upper bound.

    Raises:
        ValueError: If either bound is not a parseable timestamp.
    """
    bigquery = _bq()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "start_date", "TIMESTAMP", _as_timestamp("start_date", start_date)
            ),
            bigquery.ScalarQueryParameter(
                "end_date", "TIMESTAMP", _as_timestamp("end_date", end_date)
            ),
        ]
    )
    return client.query(training_set_query(config), job_config=job_config).to_dataframe()
