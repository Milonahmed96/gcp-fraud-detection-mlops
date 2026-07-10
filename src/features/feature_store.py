"""Vertex AI Feature Store: online serving of the engineered features.

The entity type is keyed on `customer_id`. At serving time the FastAPI handler
looks up the customer's aggregate features (velocity, spending baseline) from
the online store and combines them with the row-local features it can compute
from the incoming transaction itself.

Feature definitions are derived from `FEATURE_SPECS`, so the online store and
the offline BigQuery table can never disagree about what a feature is called or
what type it holds. That single source of truth is the main defence against
train/serve skew.

As with `bigquery.py`, the Vertex AI SDK is imported lazily and clients are
passed in rather than constructed implicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

from src.features.config import GCPConfig
from src.features.schema import (
    ENTITY_ID_COLUMN,
    FEATURE_SPECS,
    FieldSpec,
)

#: The entity type registered in the Feature Store.
ENTITY_TYPE_ID = "customer"

#: Our BigQuery field types mapped onto Vertex AI Feature Store value types.
VALUE_TYPE_BY_FIELD_TYPE: dict[str, str] = {
    "STRING": "STRING",
    "INTEGER": "INT64",
    "FLOAT": "DOUBLE",
    "BOOLEAN": "BOOL",
}

#: Features that describe the *customer* and therefore belong in the online
#: store. The remainder are row-local (hour of day, is_foreign, ...) and are
#: computed from the request payload at serving time -- storing them online
#: would be both wasteful and wrong, since they change per transaction.
ONLINE_FEATURE_NAMES: tuple[str, ...] = (
    "seconds_since_prev_txn",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_24h",
    "customer_amount_mean_prior",
)


class FeatureStoreConfigError(ValueError):
    """Raised when the declared feature set cannot be mapped onto Vertex AI."""


def _aiplatform():
    """Import the Vertex AI SDK lazily."""
    from google.cloud import aiplatform

    return aiplatform


def value_type_for(spec: FieldSpec) -> str:
    """Map a FieldSpec onto a Vertex AI Feature Store value type."""
    try:
        return VALUE_TYPE_BY_FIELD_TYPE[spec.field_type]
    except KeyError as exc:  # pragma: no cover -- guards against a new FieldType
        raise FeatureStoreConfigError(
            f"no Vertex AI value type for {spec.name!r} of type {spec.field_type!r}"
        ) from exc


def online_feature_specs() -> tuple[FieldSpec, ...]:
    """The subset of FEATURE_SPECS served from the online store, in declared order."""
    by_name = {spec.name: spec for spec in FEATURE_SPECS}

    unknown = [name for name in ONLINE_FEATURE_NAMES if name not in by_name]
    if unknown:
        raise FeatureStoreConfigError(
            f"ONLINE_FEATURE_NAMES references features absent from FEATURE_SPECS: "
            f"{', '.join(sorted(unknown))}"
        )

    return tuple(by_name[name] for name in ONLINE_FEATURE_NAMES)


def feature_definitions() -> list[dict[str, str]]:
    """Feature Store definitions for every online feature.

    Returned as plain dicts so this is assertable in tests without the SDK.
    """
    return [
        {
            "feature_id": spec.name,
            "value_type": value_type_for(spec),
            "description": spec.description,
        }
        for spec in online_feature_specs()
    ]


def init_vertex_ai(config: GCPConfig) -> None:
    """Point the Vertex AI SDK at the configured project, region, and staging bucket."""
    _aiplatform().init(
        project=config.project_id,
        location=config.region,
        staging_bucket=f"gs://{config.bucket_name}",
    )


def create_feature_store(config: GCPConfig, *, online_store_fixed_node_count: int = 1) -> Any:
    """Create (or fetch) the Feature Store instance.

    `online_store_fixed_node_count` is the dominant cost lever in this project --
    a provisioned node bills whether or not anything reads from it. One node is
    ample for a demo. Tear the store down when not demoing (see README).
    """
    aiplatform = _aiplatform()
    return aiplatform.Featurestore.create(
        featurestore_id=config.feature_store_id,
        online_store_fixed_node_count=online_store_fixed_node_count,
        project=config.project_id,
        location=config.region,
        sync=True,
    )


def create_entity_type(feature_store: Any) -> Any:
    """Create the `customer` entity type and register every online feature on it."""
    entity_type = feature_store.create_entity_type(
        entity_type_id=ENTITY_TYPE_ID,
        description=f"Fraud detection features keyed on {ENTITY_ID_COLUMN}",
    )
    for definition in feature_definitions():
        entity_type.create_feature(
            feature_id=definition["feature_id"],
            value_type=definition["value_type"],
            description=definition["description"],
        )
    return entity_type


def latest_customer_state(features: "pd.DataFrame") -> "pd.DataFrame":
    """Reduce an engineered feature frame to one row per customer: their latest.

    The online store answers "what is true about this customer *now*", so only
    the most recent row per customer is meaningful. Feeding it the full history
    would store the same key many times and serve whichever version won the race.

    Returns a frame with `customer_id`, `timestamp`, and the online features.
    """
    if features.empty:
        raise FeatureStoreConfigError("cannot ingest an empty feature frame")

    missing = [name for name in ONLINE_FEATURE_NAMES if name not in features.columns]
    if missing:
        raise FeatureStoreConfigError(
            f"feature frame is missing online features: {', '.join(sorted(missing))}"
        )

    ordered = features.sort_values([ENTITY_ID_COLUMN, "timestamp"], kind="mergesort")
    latest = ordered.groupby(ENTITY_ID_COLUMN, as_index=False).last()
    columns = [ENTITY_ID_COLUMN, "timestamp", *ONLINE_FEATURE_NAMES]
    out = latest[columns].copy()

    # Vertex AI wants int64 counts as Python ints and a timezone-aware event time.
    for spec in online_feature_specs():
        if spec.field_type == "INTEGER":
            out[spec.name] = out[spec.name].astype("int64")
        elif spec.field_type == "FLOAT":
            out[spec.name] = out[spec.name].astype("float64")

    # `ingest_from_df` stages through BigQuery, whose TIMESTAMP is microsecond
    # resolution. Nanoseconds raise `ArrowInvalid: ... would lose data`, exactly
    # as on the offline ingestion path. Reuse the same flooring rather than
    # writing a second implementation of it.
    from src.features.bigquery import truncate_timestamps

    out = truncate_timestamps(out)

    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize("UTC")
    return out


def ingest_online_features(entity_type: Any, latest: "pd.DataFrame") -> int:
    """Write one row per customer into the online store. Returns rows written.

    `feature_time_field` matters: Vertex AI keeps the value with the newest
    event time, so a late-arriving backfill cannot overwrite fresher state.
    """
    entity_type.ingest_from_df(
        feature_ids=list(ONLINE_FEATURE_NAMES),
        feature_time="timestamp",
        df_source=latest,
        entity_id_field=ENTITY_ID_COLUMN,
    )
    return len(latest)


def read_online_features(entity_type: Any, customer_ids: list[str]) -> Any:
    """Low-latency lookup of the online features for a batch of customers."""
    return entity_type.read(
        entity_ids=customer_ids,
        feature_ids=list(ONLINE_FEATURE_NAMES),
    )
