"""Tests for the Vertex AI Feature Store layer.

The Vertex AI SDK is never touched: `init_vertex_ai` and `create_feature_store`
are exercised by monkeypatching the lazy `_aiplatform()` import, and everything
else is pure data.
"""

from __future__ import annotations

import pytest

from src.features import feature_store as fs
from src.features.config import GCPConfig
from src.features.schema import FEATURE_SPECS, feature_names


@pytest.fixture
def config() -> GCPConfig:
    return GCPConfig(
        project_id="test-project",
        region="europe-west2",
        bucket_name="test-bucket",
        bigquery_dataset="fraud_features",
        feature_store_id="fraud_online_store",
    )


class TestValueTypeMapping:
    @pytest.mark.parametrize(
        ("field_type", "expected"),
        [("STRING", "STRING"), ("INTEGER", "INT64"), ("FLOAT", "DOUBLE"), ("BOOLEAN", "BOOL")],
    )
    def test_known_types_map_to_vertex_value_types(self, field_type, expected):
        from src.features.schema import FieldSpec

        assert fs.value_type_for(FieldSpec("x", field_type)) == expected

    def test_every_online_feature_has_a_value_type(self):
        for spec in fs.online_feature_specs():
            assert fs.value_type_for(spec) in {"STRING", "INT64", "DOUBLE", "BOOL"}


class TestOnlineFeatureSelection:
    def test_online_features_are_a_subset_of_declared_features(self):
        """Guards against a rename in schema.py silently orphaning the online store."""
        assert set(fs.ONLINE_FEATURE_NAMES) <= set(feature_names())

    def test_online_features_are_customer_level_not_row_local(self):
        """Row-local features are derived from the request, never stored online."""
        row_local = {
            "hour_of_day",
            "day_of_week",
            "is_night",
            "is_weekend",
            "is_foreign",
            "card_not_present",
            "amount_log",
            "amount_vs_customer_mean",
        }
        assert not set(fs.ONLINE_FEATURE_NAMES) & row_local

    def test_online_feature_specs_preserve_declaration_order(self):
        assert tuple(s.name for s in fs.online_feature_specs()) == fs.ONLINE_FEATURE_NAMES

    def test_unknown_online_feature_is_rejected(self, monkeypatch):
        monkeypatch.setattr(fs, "ONLINE_FEATURE_NAMES", ("txn_count_24h", "not_a_real_feature"))
        with pytest.raises(fs.FeatureStoreConfigError, match="not_a_real_feature"):
            fs.online_feature_specs()


class TestFeatureDefinitions:
    def test_definitions_cover_every_online_feature(self):
        definitions = fs.feature_definitions()
        assert [d["feature_id"] for d in definitions] == list(fs.ONLINE_FEATURE_NAMES)

    def test_definitions_carry_types_and_descriptions(self):
        for definition in fs.feature_definitions():
            assert definition["value_type"]
            assert definition["description"]

    def test_types_agree_with_the_offline_schema(self):
        """The whole point of deriving from FEATURE_SPECS: no train/serve skew."""
        by_name = {s.name: s for s in FEATURE_SPECS}
        for definition in fs.feature_definitions():
            spec = by_name[definition["feature_id"]]
            assert definition["value_type"] == fs.VALUE_TYPE_BY_FIELD_TYPE[spec.field_type]

    def test_counts_are_int64_and_amounts_are_double(self):
        types = {d["feature_id"]: d["value_type"] for d in fs.feature_definitions()}
        assert types["txn_count_1h"] == "INT64"
        assert types["txn_count_24h"] == "INT64"
        assert types["amount_sum_24h"] == "DOUBLE"
        assert types["customer_amount_mean_prior"] == "DOUBLE"


class FakeAIPlatform:
    def __init__(self):
        self.init_kwargs: dict | None = None
        self.create_kwargs: dict | None = None
        self.Featurestore = self  # type: ignore[assignment]

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return "featurestore"


class TestVertexAIWiring:
    def test_init_passes_project_region_and_staging_bucket(self, config, monkeypatch):
        fake = FakeAIPlatform()
        monkeypatch.setattr(fs, "_aiplatform", lambda: fake)

        fs.init_vertex_ai(config)

        assert fake.init_kwargs == {
            "project": "test-project",
            "location": "europe-west2",
            "staging_bucket": "gs://test-bucket",
        }

    def test_create_feature_store_defaults_to_one_node_for_cost(self, config, monkeypatch):
        fake = FakeAIPlatform()
        monkeypatch.setattr(fs, "_aiplatform", lambda: fake)

        fs.create_feature_store(config)

        assert fake.create_kwargs["online_store_fixed_node_count"] == 1
        assert fake.create_kwargs["featurestore_id"] == "fraud_online_store"
        assert fake.create_kwargs["location"] == "europe-west2"


class FakeEntityType:
    def __init__(self):
        self.features: list[dict] = []
        self.read_kwargs: dict | None = None

    def create_feature(self, **kwargs):
        self.features.append(kwargs)

    def read(self, **kwargs):
        self.read_kwargs = kwargs
        return "rows"


class FakeFeatureStore:
    def __init__(self):
        self.entity_type = FakeEntityType()
        self.create_kwargs: dict | None = None

    def create_entity_type(self, **kwargs):
        self.create_kwargs = kwargs
        return self.entity_type


class TestEntityType:
    def test_entity_type_registers_every_online_feature(self):
        store = FakeFeatureStore()
        entity_type = fs.create_entity_type(store)

        assert store.create_kwargs["entity_type_id"] == "customer"
        registered = [f["feature_id"] for f in entity_type.features]
        assert registered == list(fs.ONLINE_FEATURE_NAMES)

    def test_entity_type_is_keyed_on_customer_id(self):
        store = FakeFeatureStore()
        fs.create_entity_type(store)
        assert "customer_id" in store.create_kwargs["description"]

    def test_read_online_features_requests_the_declared_feature_ids(self):
        entity_type = FakeEntityType()
        fs.read_online_features(entity_type, ["c1", "c2"])

        assert entity_type.read_kwargs == {
            "entity_ids": ["c1", "c2"],
            "feature_ids": list(fs.ONLINE_FEATURE_NAMES),
        }


class FakeEntityTypeWithIngest:
    def __init__(self):
        self.ingested: dict | None = None

    def ingest_from_df(self, **kwargs):
        self.ingested = kwargs
        return self


class TestLatestCustomerState:
    """The online store answers 'what is true about this customer *now*'."""

    def _frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "customer_id": ["c1", "c1", "c2"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01T09:00:00.111111111",
                        "2024-01-01T10:00:00.222222222",
                        "2024-01-01T11:00:00.333333333",
                    ]
                ),
                "seconds_since_prev_txn": [-1.0, 3600.0, -1.0],
                "txn_count_1h": [1, 1, 1],
                "txn_count_24h": [1, 2, 1],
                "amount_sum_24h": [10.0, 30.0, 5.0],
                "customer_amount_mean_prior": [10.0, 10.0, 5.0],
            }
        )

    def test_keeps_only_the_latest_row_per_customer(self):
        latest = fs.latest_customer_state(self._frame())
        assert len(latest) == 2
        c1 = latest[latest["customer_id"] == "c1"].iloc[0]
        assert c1["txn_count_24h"] == 2  # the 10:00 row, not the 09:00 one

    def test_timestamps_are_floored_to_microseconds(self):
        """`ingest_from_df` stages through BigQuery, which cannot hold nanoseconds.
        Same ArrowInvalid as the offline path -- found on the first real ingest."""
        latest = fs.latest_customer_state(self._frame())
        assert (latest["timestamp"].dt.nanosecond == 0).all()

    def test_timestamps_are_timezone_aware(self):
        latest = fs.latest_customer_state(self._frame())
        assert latest["timestamp"].dt.tz is not None

    def test_only_the_online_features_are_carried(self):
        latest = fs.latest_customer_state(self._frame())
        assert set(latest.columns) == {"customer_id", "timestamp", *fs.ONLINE_FEATURE_NAMES}

    def test_an_empty_frame_is_rejected(self):
        import pandas as pd

        with pytest.raises(fs.FeatureStoreConfigError, match="empty feature frame"):
            fs.latest_customer_state(pd.DataFrame())

    def test_a_frame_missing_online_features_is_rejected(self):
        frame = self._frame().drop(columns=["txn_count_1h"])
        with pytest.raises(fs.FeatureStoreConfigError, match="missing online features"):
            fs.latest_customer_state(frame)


class TestIngestOnlineFeatures:
    def test_writes_every_online_feature_keyed_on_customer_id(self):
        import pandas as pd

        entity_type = FakeEntityTypeWithIngest()
        latest = pd.DataFrame(
            {
                "customer_id": ["c1"],
                "timestamp": pd.to_datetime(["2024-01-01T10:00:00"]).tz_localize("UTC"),
                **{name: [1.0] for name in fs.ONLINE_FEATURE_NAMES},
            }
        )
        assert fs.ingest_online_features(entity_type, latest) == 1
        assert entity_type.ingested["feature_ids"] == list(fs.ONLINE_FEATURE_NAMES)
        assert entity_type.ingested["entity_id_field"] == "customer_id"

    def test_the_event_time_column_is_passed_through(self):
        """Vertex keeps the newest event time, so a backfill cannot clobber
        fresher state."""
        import pandas as pd

        entity_type = FakeEntityTypeWithIngest()
        latest = pd.DataFrame(
            {
                "customer_id": ["c1"],
                "timestamp": pd.to_datetime(["2024-01-01T10:00:00"]).tz_localize("UTC"),
                **{name: [1.0] for name in fs.ONLINE_FEATURE_NAMES},
            }
        )
        fs.ingest_online_features(entity_type, latest)
        assert entity_type.ingested["feature_time"] == "timestamp"
