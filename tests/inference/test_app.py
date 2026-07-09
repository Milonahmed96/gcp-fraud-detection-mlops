"""Tests for the FastAPI inference service.

These drive the real app through its startup lifespan with real fitted models
and real SHAP explainers. Nothing about the prediction path is mocked.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.inference.app import create_app, to_naive_utc

GENUINE_TRANSACTION = {
    "transaction_id": "txn_test_001",
    "customer_id": "c_000",
    "timestamp": "2024-03-01T14:30:00",
    "amount": 42.50,
    "merchant_id": "m_001",
    "merchant_category": "grocery",
    "country": "GB",
    "customer_home_country": "GB",
    "card_present": True,
}

FRAUDULENT_TRANSACTION = {
    **GENUINE_TRANSACTION,
    "transaction_id": "txn_test_002",
    "timestamp": "2024-03-01T03:15:00",
    "amount": 4800.00,
    "merchant_category": "electronics",
    "country": "RO",
    "card_present": False,
}


@pytest.fixture
def client(artifacts_dir, monkeypatch) -> TestClient:
    """A client whose app has loaded real artefacts via its lifespan.

    Each client gets its own `create_app()` instance: state lives on
    `app.state`, so two clients (e.g. one per variant) cannot clobber each
    other's loaded model.
    """
    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("SERVING_VARIANT", "xgboost")
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def broken_client(tmp_path, monkeypatch) -> TestClient:
    """A client whose artefacts are absent, to exercise the unhealthy path."""
    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(tmp_path))
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


class TestHealth:
    def test_reports_ok_when_artefacts_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["variant"] == "xgboost"
        assert body["model_loaded"] and body["explainer_loaded"]

    def test_reports_503_when_artefacts_are_missing(self, broken_client):
        """Cloud Run must see an unhealthy revision, not a crash loop."""
        response = broken_client.get("/health")
        assert response.status_code == 503
        assert "src.training.train" in response.json()["detail"]

    def test_the_process_stays_up_after_a_failed_startup(self, broken_client):
        assert broken_client.get("/health").status_code == 503  # responding, not dead


class TestPredict:
    def test_scores_a_genuine_transaction(self, client):
        response = client.post("/predict", json=GENUINE_TRANSACTION)
        assert response.status_code == 200

        body = response.json()
        assert body["transaction_id"] == "txn_test_001"
        assert body["variant"] == "xgboost"
        assert 0.0 <= body["fraud_probability"] <= 1.0

    def test_a_fraudulent_looking_transaction_scores_higher(self, client):
        """Foreign, card-not-present, 100x the baseline, at 03:15."""
        genuine = client.post("/predict", json=GENUINE_TRANSACTION).json()
        fraudulent = client.post("/predict", json=FRAUDULENT_TRANSACTION).json()
        assert fraudulent["fraud_probability"] > genuine["fraud_probability"]

    def test_the_flag_follows_the_trained_threshold(self, client):
        body = client.post("/predict", json=FRAUDULENT_TRANSACTION).json()
        assert body["is_flagged"] == (body["fraud_probability"] >= body["threshold"])

    def test_the_threshold_is_the_trained_one_not_one_half(self, client, artifacts_dir):
        from src.inference.registry import load_threshold

        body = client.post("/predict", json=GENUINE_TRANSACTION).json()
        assert body["threshold"] == pytest.approx(load_threshold(artifacts_dir, "xgboost"))

    def test_returns_shap_attributions(self, client):
        body = client.post("/predict", json=FRAUDULENT_TRANSACTION).json()
        assert len(body["top_features"]) == 5
        for attribution in body["top_features"]:
            assert set(attribution) == {"feature", "value", "shap_value", "direction"}
            assert attribution["direction"] in {"toward_fraud", "toward_genuine", "neutral"}

    def test_attributions_are_ranked_by_absolute_effect(self, client):
        body = client.post("/predict", json=FRAUDULENT_TRANSACTION).json()
        magnitudes = [abs(a["shap_value"]) for a in body["top_features"]]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_the_explanation_justifies_the_prediction(self, client):
        """base_value + sum(all shap) == margin. We only return the top-k, so
        assert the weaker property: a high-probability fraud is driven by
        features pointing toward fraud."""
        body = client.post("/predict", json=FRAUDULENT_TRANSACTION).json()
        toward_fraud = [a for a in body["top_features"] if a["direction"] == "toward_fraud"]
        assert len(toward_fraud) >= 3

    def test_a_foreign_card_not_present_transaction_cites_those_features(self, client):
        body = client.post("/predict", json=FRAUDULENT_TRANSACTION).json()
        cited = {a["feature"] for a in body["top_features"]}
        assert cited & {"is_foreign", "card_not_present", "amount_vs_customer_mean"}

    def test_reports_latency(self, client):
        body = client.post("/predict", json=GENUINE_TRANSACTION).json()
        assert body["latency_ms"] > 0.0

    def test_sets_the_latency_header(self, client):
        response = client.post("/predict", json=GENUINE_TRANSACTION)
        assert float(response.headers["X-Response-Time-Ms"]) > 0.0

    def test_an_unknown_customer_is_served_not_rejected(self, client):
        """A first-time cardholder is a normal event at a payment gateway."""
        payload = {**GENUINE_TRANSACTION, "customer_id": "never-seen-before"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["new_customer"] is True

    def test_a_known_customer_is_not_flagged_as_new(self, client):
        assert client.post("/predict", json=GENUINE_TRANSACTION).json()["new_customer"] is False

    def test_returns_503_when_artefacts_are_missing(self, broken_client):
        response = broken_client.post("/predict", json=GENUINE_TRANSACTION)
        assert response.status_code == 503


class TestServingUsesTheRealCustomerHistory:
    def test_the_same_transaction_scores_differently_for_different_customers(self, client):
        """Proves the online state lookup actually reaches the model: the
        `amount_vs_customer_mean` feature depends on who is transacting."""
        big_spender = {**GENUINE_TRANSACTION, "customer_id": "c_000", "amount": 500.0}
        other = {**GENUINE_TRANSACTION, "customer_id": "c_001", "amount": 500.0}

        a = client.post("/predict", json=big_spender).json()["fraud_probability"]
        b = client.post("/predict", json=other).json()["fraud_probability"]
        assert a != b


class TestValidation:
    def test_rejects_a_negative_amount(self, client):
        response = client.post("/predict", json={**GENUINE_TRANSACTION, "amount": -1.0})
        assert response.status_code == 422

    def test_rejects_a_missing_field(self, client):
        payload = {k: v for k, v in GENUINE_TRANSACTION.items() if k != "amount"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_rejects_an_unknown_field(self, client):
        """extra='forbid': a typo'd field name is a bug, not something to ignore."""
        payload = {**GENUINE_TRANSACTION, "amont": 50.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_rejects_a_bad_country_code(self, client):
        response = client.post("/predict", json={**GENUINE_TRANSACTION, "country": "GBR"})
        assert response.status_code == 422

    def test_rejects_a_non_boolean_card_present(self, client):
        response = client.post("/predict", json={**GENUINE_TRANSACTION, "card_present": "maybe"})
        assert response.status_code == 422

    def test_rejects_an_empty_customer_id(self, client):
        assert (
            client.post("/predict", json={**GENUINE_TRANSACTION, "customer_id": ""}).status_code
            == 422
        )

    def test_rejects_an_absurd_amount(self, client):
        response = client.post("/predict", json={**GENUINE_TRANSACTION, "amount": 1e12})
        assert response.status_code == 422

    def test_accepts_an_iso_timestamp_with_timezone(self, client):
        payload = {**GENUINE_TRANSACTION, "timestamp": "2024-03-01T14:30:00+00:00"}
        assert client.post("/predict", json=payload).status_code == 200


class TestTimezoneNormalisation:
    """An offset must be *converted* to UTC, not discarded.

    Dropping `+02:00` would shift the transaction two hours and silently change
    `hour_of_day`, `is_night`, and every velocity window.
    """

    def test_a_naive_timestamp_passes_through(self):
        assert to_naive_utc("2024-03-01T14:30:00") == pd.Timestamp("2024-03-01T14:30:00")

    def test_an_offset_is_converted_not_truncated(self):
        assert to_naive_utc("2024-03-01T14:30:00+02:00") == pd.Timestamp("2024-03-01T12:30:00")

    def test_the_result_is_always_naive(self):
        assert to_naive_utc("2024-03-01T14:30:00-05:00").tzinfo is None

    def test_an_offset_that_crosses_midnight_changes_is_night(self, client):
        """00:30+02:00 is 22:30 UTC -- not night. The naive reading would be."""
        utc_night = {**GENUINE_TRANSACTION, "timestamp": "2024-03-01T00:30:00+00:00"}
        same_instant_offset = {**GENUINE_TRANSACTION, "timestamp": "2024-03-01T02:30:00+02:00"}

        a = client.post("/predict", json=utc_night).json()["fraud_probability"]
        b = client.post("/predict", json=same_instant_offset).json()["fraud_probability"]
        assert a == pytest.approx(b)  # same instant -> same features -> same score


class TestVariantSelection:
    def test_the_lightgbm_revision_serves_lightgbm(self, artifacts_dir, monkeypatch):
        """Each Cloud Run revision serves exactly one variant; the split is
        Cloud Run's job, not the application's."""
        monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(artifacts_dir))
        monkeypatch.setenv("SERVING_VARIANT", "lightgbm")

        with TestClient(create_app()) as lgbm_client:
            assert lgbm_client.get("/health").json()["variant"] == "lightgbm"
            body = lgbm_client.post("/predict", json=GENUINE_TRANSACTION).json()
            assert body["variant"] == "lightgbm"

    def test_the_two_variants_disagree_somewhere(self, artifacts_dir, monkeypatch, client):
        """If they agreed everywhere the A/B test would be pointless."""
        xgb = client.post("/predict", json=FRAUDULENT_TRANSACTION).json()["fraud_probability"]

        monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(artifacts_dir))
        monkeypatch.setenv("SERVING_VARIANT", "lightgbm")
        with TestClient(create_app()) as lgbm_client:
            lgbm = lgbm_client.post("/predict", json=FRAUDULENT_TRANSACTION).json()[
                "fraud_probability"
            ]
        assert xgb != lgbm


class TestServingMatchesOfflineScoring:
    def test_the_api_reproduces_the_offline_prediction(self, client, artifacts_dir, dataset):
        """End-to-end skew check: the probability the API returns for a real
        sample transaction must equal what the trained model produces offline
        for the same row."""
        from src.features.sample_data import load_sample
        from src.inference.registry import load_bundle
        from src.inference.state import CustomerState
        from src.inference.features import build_serving_features

        raw = load_sample()
        row = raw.iloc[4000]
        history = raw[
            (raw["customer_id"] == row["customer_id"]) & (raw["timestamp"] < row["timestamp"])
        ]
        bundle = load_bundle(variant="xgboost", artifacts_dir=artifacts_dir)
        offline_features = build_serving_features(
            timestamp=row["timestamp"],
            amount=float(row["amount"]),
            country=row["country"],
            customer_home_country=row["customer_home_country"],
            card_present=bool(row["card_present"]),
            state=CustomerState.from_history(history),
        )
        expected = float(bundle.model.predict_proba(offline_features)[0, 1])

        payload = {
            "transaction_id": str(row["transaction_id"]),
            "customer_id": str(row["customer_id"]),
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "amount": float(row["amount"]),
            "merchant_id": str(row["merchant_id"]),
            "merchant_category": str(row["merchant_category"]),
            "country": str(row["country"]),
            "customer_home_country": str(row["customer_home_country"]),
            "card_present": bool(row["card_present"]),
        }
        served = client.post("/predict", json=payload).json()["fraud_probability"]
        assert served == pytest.approx(expected, rel=1e-9)
