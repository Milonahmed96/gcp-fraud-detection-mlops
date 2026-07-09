"""Tests for the schema contract.

`schema.py` is the single source of truth shared by BigQuery, the Feature Store,
and the transforms. These tests keep the three from drifting apart.
"""

from __future__ import annotations

import pandas as pd

from src.features import schema
from src.features.transforms import build_feature_frame


def test_no_duplicate_column_names():
    raw = schema.raw_column_names()
    feats = schema.feature_names()
    assert len(set(raw)) == len(raw)
    assert len(set(feats)) == len(feats)


def test_features_do_not_shadow_raw_columns():
    assert not set(schema.feature_names()) & set(schema.raw_column_names())


def test_entity_and_label_columns_exist_in_the_raw_schema():
    raw = set(schema.raw_column_names())
    assert schema.ENTITY_ID_COLUMN in raw
    assert schema.EVENT_TIMESTAMP_COLUMN in raw
    assert schema.TRANSACTION_ID_COLUMN in raw
    assert schema.LABEL_COLUMN in raw


def test_label_is_nullable_so_unlabelled_rows_can_be_ingested():
    label = next(f for f in schema.RAW_TRANSACTION_SCHEMA if f.name == schema.LABEL_COLUMN)
    assert label.mode == "NULLABLE"


def test_required_raw_columns_exclude_the_label():
    assert schema.LABEL_COLUMN not in schema.required_raw_columns()


def test_every_field_is_documented():
    for field in (*schema.RAW_TRANSACTION_SCHEMA, *schema.FEATURE_SPECS):
        assert field.description, f"{field.name} has no description"


def test_field_specs_are_immutable():
    import pytest

    field = schema.FEATURE_SPECS[0]
    with pytest.raises(Exception):
        field.name = "renamed"  # type: ignore[misc]


def test_declared_features_match_what_the_pipeline_produces(raw_transactions):
    """The contract test: `FEATURE_SPECS` and `transforms.py` must agree exactly."""
    produced = build_feature_frame(raw_transactions)
    engineered = set(produced.columns) - set(schema.raw_column_names())
    assert engineered == set(schema.feature_names())


def test_declared_feature_types_match_the_produced_dtypes(raw_transactions):
    produced = build_feature_frame(raw_transactions)
    kind_for = {"FLOAT": "f", "INTEGER": "i", "BOOLEAN": "b", "STRING": "O"}

    for spec in schema.FEATURE_SPECS:
        actual = produced[spec.name].dtype.kind
        expected = kind_for[spec.field_type]
        assert actual == expected, (
            f"{spec.name}: schema declares {spec.field_type} (kind {expected!r}) "
            f"but the pipeline produced dtype kind {actual!r}"
        )


def test_raw_schema_types_match_the_sample_fixture(raw_transactions: pd.DataFrame):
    kind_for = {"FLOAT": "f", "INTEGER": "i", "BOOLEAN": "b", "STRING": "O", "TIMESTAMP": "M"}
    for spec in schema.RAW_TRANSACTION_SCHEMA:
        actual = raw_transactions[spec.name].dtype.kind
        assert actual == kind_for[spec.field_type], f"{spec.name} dtype {actual!r}"
