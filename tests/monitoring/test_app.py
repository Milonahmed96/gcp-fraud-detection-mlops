"""Tests for the drift-monitor Cloud Run service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.monitoring import app as app_module
from src.monitoring.app import create_app


@pytest.fixture
def client(artifacts_dir, monkeypatch) -> TestClient:
    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("DRIFT_SOURCE", "sample")
    with TestClient(create_app()) as test_client:
        yield test_client


class TestHealth:
    def test_ok_when_the_reference_profile_exists(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["source"] == "sample"

    def test_503_without_a_reference_profile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(tmp_path))
        with TestClient(create_app()) as broken:
            response = broken.get("/health")
            assert response.status_code == 503
            assert "reference_profile.json" in response.json()["detail"]


class TestDriftCheck:
    def test_an_empty_body_runs_the_check(self, client):
        """Cloud Scheduler posts no body."""
        response = client.post("/drift-check")
        assert response.status_code == 200
        assert response.json()["drifted"] is False

    def test_the_sample_does_not_drift_against_its_own_reference(self, client):
        body = client.post("/drift-check", json={"dry_run": True}).json()
        assert body["drifted"] is False
        assert body["retraining_triggered"] is False
        assert body["significant_features"] == []
        assert body["n_current_rows"] > 0

    def test_reports_the_worst_feature(self, client):
        body = client.post("/drift-check", json={"dry_run": True}).json()
        assert body["worst_feature"] is not None
        assert body["worst_psi"] >= 0.0

    def test_explanation_drift_is_reported(self, client):
        body = client.post("/drift-check", json={"dry_run": True}).json()
        assert body["importance_shift"] is not None

    def test_drift_triggers_retraining(self, client, monkeypatch):
        """Substitute a drifted feature batch and stub the Vertex submission."""
        import src.training.vertex as vertex_module

        submitted = {}
        monkeypatch.setattr(
            vertex_module, "submit_training_job", lambda *a, **k: submitted.setdefault("yes", True)
        )

        from src.features.sample_data import load_sample
        from src.features.transforms import build_feature_frame

        drifted = build_feature_frame(load_sample()).tail(1000).copy()
        drifted["is_foreign"] = True
        drifted["amount_vs_customer_mean"] *= 30.0

        monkeypatch.setattr(app_module, "_load_current_features", lambda s, r: (drifted, object()))

        body = client.post("/drift-check", json={"dry_run": False}).json()
        assert body["drifted"] is True
        assert body["retraining_triggered"] is True
        assert "is_foreign" in body["significant_features"]
        assert submitted == {"yes": True}

    def test_dry_run_reports_drift_without_retraining(self, client, monkeypatch):
        from src.features.sample_data import load_sample
        from src.features.transforms import build_feature_frame

        drifted = build_feature_frame(load_sample()).tail(1000).copy()
        drifted["is_foreign"] = True
        drifted["amount_vs_customer_mean"] *= 30.0
        monkeypatch.setattr(app_module, "_load_current_features", lambda s, r: (drifted, None))

        body = client.post("/drift-check", json={"dry_run": True}).json()
        assert body["drifted"] is True
        assert body["retraining_triggered"] is False

    def test_a_missing_reference_profile_returns_500(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(tmp_path))
        monkeypatch.setenv("DRIFT_SOURCE", "sample")
        with TestClient(create_app(), raise_server_exceptions=False) as broken:
            assert broken.post("/drift-check").status_code == 500

    def test_the_service_starts_without_an_explainer(self, tmp_path, monkeypatch):
        """Explanation drift is the secondary signal; its absence must not
        prevent the service from coming up."""
        monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(tmp_path))
        with TestClient(create_app()) as degraded:
            assert degraded.app.state.explainer is None
            assert degraded.app.state.startup_error is not None
