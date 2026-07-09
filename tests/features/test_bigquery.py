"""Tests for the BigQuery offline store layer.

No network and no credentials: a fake client records the calls, and real
`SchemaField` / `Table` objects are constructed locally (which the SDK permits
without authenticating).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.features import bigquery as bq
from src.features.config import GCPConfig
from src.features.schema import (
    FEATURES_TABLE,
    RAW_TRANSACTION_SCHEMA,
    RAW_TRANSACTIONS_TABLE,
    feature_names,
    raw_column_names,
)
from src.features.transforms import build_feature_frame


@pytest.fixture
def config() -> GCPConfig:
    return GCPConfig(
        project_id="test-project",
        region="europe-west2",
        bucket_name="test-bucket",
        bigquery_dataset="fraud_features",
        feature_store_id="fraud_online_store",
    )


class FakeJob:
    def __init__(self, frame: pd.DataFrame | None = None):
        self._frame = frame if frame is not None else pd.DataFrame()
        self.result_called = False

    def result(self):
        self.result_called = True
        return self

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame


class FakeClient:
    """Records every call so tests can assert on what would have hit GCP."""

    def __init__(self, query_result: pd.DataFrame | None = None):
        self.created_datasets: list = []
        self.created_tables: list = []
        self.loaded: list[tuple[pd.DataFrame, str, object]] = []
        self.queries: list[tuple[str, object]] = []
        self._query_result = query_result
        self.last_load_job = FakeJob()

    def create_dataset(self, dataset, exists_ok=False):
        self.created_datasets.append((dataset, exists_ok))
        return dataset

    def create_table(self, table, exists_ok=False):
        self.created_tables.append((table, exists_ok))
        return table

    def load_table_from_dataframe(self, df, table_ref, job_config=None):
        self.loaded.append((df, table_ref, job_config))
        self.last_load_job = FakeJob()
        return self.last_load_job

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        return FakeJob(self._query_result)


class TestSchemaConversion:
    def test_raw_schema_round_trips_names_types_and_modes(self):
        fields = bq.to_bigquery_schema(RAW_TRANSACTION_SCHEMA)
        assert [f.name for f in fields] == list(raw_column_names())

        by_name = {f.name: f for f in fields}
        assert by_name["amount"].field_type == "FLOAT"
        assert by_name["timestamp"].field_type == "TIMESTAMP"
        assert by_name["card_present"].field_type == "BOOLEAN"
        assert by_name["is_fraud"].mode == "NULLABLE"
        assert by_name["customer_id"].mode == "REQUIRED"

    def test_descriptions_are_carried_through(self):
        fields = bq.to_bigquery_schema(RAW_TRANSACTION_SCHEMA)
        assert all(f.description for f in fields)

    def test_features_table_schema_has_keys_label_and_every_feature(self):
        names = [f.name for f in bq.features_table_schema()]
        assert set(feature_names()) <= set(names)
        for key in ("transaction_id", "customer_id", "timestamp", "is_fraud"):
            assert key in names
        # Raw inputs stay in raw_transactions and must not be duplicated here.
        assert "merchant_id" not in names
        assert "country" not in names

    def test_features_table_schema_has_no_duplicates(self):
        names = [f.name for f in bq.features_table_schema()]
        assert len(names) == len(set(names))


class TestTimestampTruncation:
    """BigQuery TIMESTAMP is microsecond-resolution.

    pyarrow refuses to cast `timestamp[ns]` to `timestamp[us]` and raises
    `ArrowInvalid: ... would lose data`. The fake client never serialised
    anything, so this only surfaced on the first live ingestion.
    """

    def test_nanoseconds_are_floored_to_microseconds(self):
        frame = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2024-01-01 00:13:20.491923226"]), "amount": [1.0]}
        )
        assert frame["timestamp"].iloc[0].nanosecond == 226

        out = bq.truncate_timestamps(frame)
        assert out["timestamp"].iloc[0].nanosecond == 0
        assert out["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:13:20.491923")

    def test_the_callers_frame_is_not_mutated(self):
        frame = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 00:00:00.123456789"])})
        bq.truncate_timestamps(frame)
        assert frame["timestamp"].iloc[0].nanosecond == 789

    def test_non_datetime_columns_are_untouched(self):
        frame = pd.DataFrame({"amount": [1.5], "name": ["x"]})
        pd.testing.assert_frame_equal(bq.truncate_timestamps(frame), frame)

    def test_a_frame_without_datetimes_is_returned_unchanged(self):
        frame = pd.DataFrame({"a": [1, 2]})
        assert bq.truncate_timestamps(frame) is frame  # no needless copy

    def test_every_datetime_column_is_floored(self):
        frame = pd.DataFrame(
            {
                "created": pd.to_datetime(["2024-01-01 00:00:00.111111111"]),
                "updated": pd.to_datetime(["2024-01-01 00:00:00.222222222"]),
            }
        )
        out = bq.truncate_timestamps(frame)
        assert all(out[c].iloc[0].nanosecond == 0 for c in ("created", "updated"))

    def test_ingestion_floors_timestamps_before_loading(self, config, raw_transactions):
        """The load path must never hand pyarrow a nanosecond timestamp."""
        client = FakeClient()
        nanosecond = raw_transactions.copy()
        nanosecond["timestamp"] = nanosecond["timestamp"] + pd.Timedelta("123ns")

        bq.ingest_raw_transactions(client, config, nanosecond)

        loaded, _, _ = client.loaded[0]
        assert (loaded["timestamp"].dt.nanosecond == 0).all()

    def test_the_sample_data_actually_has_nanoseconds(self):
        """Guards the premise: if the sample lost its nanoseconds, this test file
        would be asserting nothing."""
        from src.features.sample_data import load_sample

        sample = load_sample()
        assert (sample["timestamp"].dt.nanosecond != 0).any()


class TestPredictionLog:
    """The audit log table. Phase 4 wrote to it before it had a schema."""

    def test_schema_covers_the_audit_question(self):
        """'Why was transaction X blocked on date Y, and by which variant?'"""
        names = [f.name for f in bq.prediction_log_schema()]
        for required in (
            "transaction_id",
            "customer_id",
            "timestamp",
            "variant",
            "fraud_probability",
            "threshold",
            "is_flagged",
            "base_value",
            "top_features",
        ):
            assert required in names

    def test_attributions_are_stored_as_a_json_string(self):
        field = next(f for f in bq.prediction_log_schema() if f.name == "top_features")
        assert field.field_type == "STRING"

    def test_latency_is_nullable(self):
        """A prediction is still auditable if the latency probe failed."""
        field = next(f for f in bq.prediction_log_schema() if f.name == "latency_ms")
        assert field.mode == "NULLABLE"

    def test_ensure_prediction_log_partitions_and_clusters_for_ab_queries(self, config):
        client = FakeClient()
        ref = bq.ensure_prediction_log(client, config)

        assert ref == "test-project.fraud_features.prediction_log"
        table, exists_ok = client.created_tables[0]
        assert exists_ok is True
        assert table.time_partitioning.field == "timestamp"
        # The A/B dashboard groups by variant; clustering keeps that scan cheap.
        assert table.clustering_fields == ["variant", "customer_id"]

    def test_the_explanation_writer_matches_this_schema(self):
        """`log_predictions_to_bigquery` must not write columns the table lacks."""
        from src.evaluation.experiments import explanation_rows
        from src.evaluation.explainer import Explanation, FeatureAttribution

        explanation = Explanation(
            base_value=-1.0,
            attributions=(FeatureAttribution("is_foreign", 1.0, 2.0),),
            probability=0.9,
        )
        written = set(explanation_rows(["t1"], "xgboost", [explanation]).columns)
        declared = {f.name for f in bq.prediction_log_schema()}
        assert written <= declared, f"writer emits undeclared columns: {written - declared}"


class TestEnsureDataset:
    def test_creates_dataset_in_the_configured_region(self, config):
        client = FakeClient()
        ref = bq.ensure_dataset(client, config)

        assert ref == "test-project.fraud_features"
        dataset, exists_ok = client.created_datasets[0]
        assert dataset.location == "europe-west2"
        assert exists_ok is True  # idempotent: safe to re-run


class TestEnsureTable:
    def test_partitions_on_timestamp_and_clusters_on_customer(self, config):
        client = FakeClient()
        ref = bq.ensure_table(
            client, config, RAW_TRANSACTIONS_TABLE, bq.to_bigquery_schema(RAW_TRANSACTION_SCHEMA)
        )

        assert ref == "test-project.fraud_features.raw_transactions"
        table, exists_ok = client.created_tables[0]
        assert exists_ok is True
        assert table.time_partitioning.field == "timestamp"
        assert table.clustering_fields == ["customer_id"]

    def test_partitioning_can_be_disabled(self, config):
        client = FakeClient()
        bq.ensure_table(
            client,
            config,
            "unpartitioned",
            bq.to_bigquery_schema(RAW_TRANSACTION_SCHEMA),
            partition_field=None,
            cluster_fields=(),
        )
        table, _ = client.created_tables[0]
        assert table.time_partitioning is None


class TestIngestion:
    def test_ingest_raw_transactions_targets_the_landing_table(self, config, raw_transactions):
        client = FakeClient()
        rows = bq.ingest_raw_transactions(client, config, raw_transactions)

        assert rows == len(raw_transactions)
        df, table_ref, _ = client.loaded[0]
        assert table_ref == "test-project.fraud_features.raw_transactions"
        assert len(df) == len(raw_transactions)

    def test_load_blocks_on_the_job_so_failures_surface(self, config, raw_transactions):
        client = FakeClient()
        bq.ingest_raw_transactions(client, config, raw_transactions)
        assert client.last_load_job.result_called is True

    def test_ingest_features_writes_keys_label_cost_basis_and_features(
        self, config, raw_transactions
    ):
        """`amount` must travel with the features.

        It is not a model input -- `amount_log` is -- but the business cost
        metric prices a missed fraud at the amount stolen. Omitting it zeroes
        every false-negative cost, and the first live Vertex run did exactly
        that: cost/1k = 0.00, zero customers blocked, every fraud missed."""
        client = FakeClient()
        features = build_feature_frame(raw_transactions)
        rows = bq.ingest_features(client, config, features)

        assert rows == len(features)
        df, table_ref, _ = client.loaded[0]
        assert table_ref == f"test-project.fraud_features.{FEATURES_TABLE}"

        expected = {
            "transaction_id",
            "customer_id",
            "timestamp",
            "is_fraud",
            "amount",
            *feature_names(),
        }
        assert set(df.columns) == expected

    def test_the_raw_inputs_are_not_duplicated_into_the_feature_table(
        self, config, raw_transactions
    ):
        """Only the cost basis crosses over; merchant/country stay in raw_transactions."""
        client = FakeClient()
        bq.ingest_features(client, config, build_feature_frame(raw_transactions))
        df, _, _ = client.loaded[0]
        assert "merchant_id" not in df.columns
        assert "country" not in df.columns

    def test_ingest_features_column_order_matches_the_table_schema(self, config, raw_transactions):
        client = FakeClient()
        bq.ingest_features(client, config, build_feature_frame(raw_transactions))
        df, _, _ = client.loaded[0]
        assert list(df.columns) == [f.name for f in bq.features_table_schema()]

    def test_ingest_features_rejects_an_incomplete_frame(self, config, raw_transactions):
        client = FakeClient()
        features = build_feature_frame(raw_transactions).drop(columns=["amount_log"])
        with pytest.raises(ValueError, match="missing columns: amount_log"):
            bq.ingest_features(client, config, features)


class TestTrainingSetQuery:
    def test_query_selects_every_feature_and_the_label(self, config):
        sql = bq.training_set_query(config)
        for name in feature_names():
            assert name in sql
        assert "is_fraud IS NOT NULL" in sql

    def test_query_uses_bound_parameters_not_interpolation(self, config):
        """Dates must never be formatted into the SQL string."""
        sql = bq.training_set_query(config)
        assert "@start_date" in sql and "@end_date" in sql
        assert "TIMESTAMP('" not in sql

    def test_query_targets_the_feature_table(self, config):
        assert "`test-project.fraud_features.transaction_features`" in bq.training_set_query(config)

    def test_query_is_ordered_temporally(self, config):
        assert bq.training_set_query(config).rstrip().endswith("ORDER BY timestamp")

    def test_fetch_binds_the_date_range_as_parameters(self, config):
        expected = pd.DataFrame({"transaction_id": ["t1"], "is_fraud": [0]})
        client = FakeClient(query_result=expected)

        result = bq.fetch_training_set(
            client, config, start_date="2024-01-01", end_date="2024-02-01"
        )

        pd.testing.assert_frame_equal(result, expected)
        _, job_config = client.queries[0]
        params = {p.name: p.value for p in job_config.query_parameters}
        assert params == {
            "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_date": datetime(2024, 2, 1, tzinfo=timezone.utc),
        }
        assert all(p.type_ == "TIMESTAMP" for p in job_config.query_parameters)

    def test_fetch_accepts_datetime_bounds_directly(self, config):
        client = FakeClient(query_result=pd.DataFrame())
        bq.fetch_training_set(
            client, config, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 2, 1)
        )
        _, job_config = client.queries[0]
        assert job_config.query_parameters[0].value == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_a_malicious_date_is_rejected_before_it_reaches_bigquery(self, config):
        """Two layers: the bound is a parameter, *and* it must parse as a timestamp."""
        client = FakeClient(query_result=pd.DataFrame())
        injection = "2024-01-01') OR TRUE --"

        with pytest.raises(ValueError, match="start_date is not a valid timestamp"):
            bq.fetch_training_set(client, config, start_date=injection, end_date="2024-02-01")

        assert client.queries == []  # nothing was ever sent

    def test_a_non_date_string_is_rejected(self, config):
        client = FakeClient(query_result=pd.DataFrame())
        with pytest.raises(ValueError, match="end_date is not a valid timestamp"):
            bq.fetch_training_set(client, config, start_date="2024-01-01", end_date="not-a-date")
