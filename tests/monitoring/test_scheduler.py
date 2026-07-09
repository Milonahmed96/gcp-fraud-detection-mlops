"""Tests for the Cloud Scheduler drift-check job.

The SDK is replaced with a fake. What matters is that provisioning is idempotent
(a redeploy must not fail with AlreadyExists, nor leave a stale schedule) and
that authentication is OIDC rather than a shared secret.
"""

from __future__ import annotations

import pytest

from src.features.config import GCPConfig
from src.monitoring import scheduler as scheduler_module
from src.monitoring.scheduler import (
    DEFAULT_ATTEMPT_DEADLINE_SECONDS,
    DEFAULT_SCHEDULE,
    SchedulerJobSpec,
    build_job,
    delete_drift_check_job,
    ensure_drift_check_job,
)

TARGET = "https://fraud-inference-api-abc123-nw.a.run.app/drift-check"


@pytest.fixture
def spec() -> SchedulerJobSpec:
    return SchedulerJobSpec(
        config=GCPConfig("test-project", "europe-west2", "b", "d", "f"),
        target_uri=TARGET,
        service_account_email="drift@test-project.iam.gserviceaccount.com",
    )


class FakeClient:
    def __init__(self, *, exists: bool = False, delete_fails: bool = False):
        self.created: list = []
        self.updated: list = []
        self.deleted: list = []
        self._exists = exists
        self._delete_fails = delete_fails

    def get_job(self, name):
        if not self._exists:
            raise RuntimeError("NotFound")
        return {"name": name}

    def create_job(self, parent, job):
        self.created.append((parent, job))
        return job

    def update_job(self, job):
        self.updated.append(job)
        return job

    def delete_job(self, name):
        if self._delete_fails:
            raise RuntimeError("NotFound")
        self.deleted.append(name)


class TestSpec:
    def test_parent_is_the_project_location_path(self, spec):
        assert spec.parent == "projects/test-project/locations/europe-west2"

    def test_full_name_includes_the_job(self, spec):
        assert spec.full_name.endswith("/jobs/fraud-drift-check")

    def test_defaults_to_a_daily_off_peak_schedule(self, spec):
        assert spec.schedule == DEFAULT_SCHEDULE == "0 2 * * *"


class TestBuildJob:
    def test_posts_to_the_target(self, spec):
        job = build_job(spec)
        assert job["http_target"]["uri"] == TARGET
        assert job["http_target"]["http_method"] == "POST"

    def test_authenticates_with_oidc_not_a_shared_secret(self, spec):
        job = build_job(spec)
        oidc = job["http_target"]["oidc_token"]
        assert oidc["service_account_email"] == "drift@test-project.iam.gserviceaccount.com"
        assert "headers" not in oidc
        # Nothing resembling an API key anywhere in the definition.
        assert "api_key" not in str(job).lower()

    def test_the_oidc_audience_is_the_bare_service_url(self, spec):
        """Cloud Run rejects a token whose audience carries the request path."""
        audience = build_job(spec)["http_target"]["oidc_token"]["audience"]
        assert audience == "https://fraud-inference-api-abc123-nw.a.run.app"
        assert "/drift-check" not in audience

    def test_the_attempt_deadline_exceeds_the_scheduler_default(self, spec):
        """The drift check reads a day of features and runs SHAP; 180s is not enough."""
        assert build_job(spec)["attempt_deadline"]["seconds"] == DEFAULT_ATTEMPT_DEADLINE_SECONDS
        assert DEFAULT_ATTEMPT_DEADLINE_SECONDS > 180

    def test_carries_the_schedule_and_timezone(self, spec):
        job = build_job(spec)
        assert job["schedule"] == "0 2 * * *"
        assert job["time_zone"] == "Europe/London"

    def test_an_insecure_target_is_rejected(self, spec):
        from dataclasses import replace

        with pytest.raises(ValueError, match="must be https"):
            build_job(replace(spec, target_uri="http://insecure.example.com/x"))


class TestEnsureDriftCheckJob:
    def test_creates_the_job_when_absent(self, spec, monkeypatch):
        monkeypatch.setattr(scheduler_module, "_scheduler", lambda: None)
        client = FakeClient(exists=False)

        ensure_drift_check_job(spec, client=client)

        assert len(client.created) == 1
        assert not client.updated
        parent, job = client.created[0]
        assert parent == spec.parent
        assert job["schedule"] == DEFAULT_SCHEDULE

    def test_updates_the_job_when_present(self, spec, monkeypatch):
        """A redeploy must not fail with AlreadyExists."""
        monkeypatch.setattr(scheduler_module, "_scheduler", lambda: None)
        client = FakeClient(exists=True)

        ensure_drift_check_job(spec, client=client)

        assert not client.created
        assert len(client.updated) == 1

    def test_a_schedule_change_is_applied_in_place(self, spec, monkeypatch):
        """Otherwise a stale schedule survives the deploy."""
        from dataclasses import replace

        monkeypatch.setattr(scheduler_module, "_scheduler", lambda: None)
        client = FakeClient(exists=True)

        ensure_drift_check_job(replace(spec, schedule="30 4 * * *"), client=client)
        assert client.updated[0]["schedule"] == "30 4 * * *"

    def test_is_idempotent_across_repeated_deploys(self, spec, monkeypatch):
        monkeypatch.setattr(scheduler_module, "_scheduler", lambda: None)
        client = FakeClient(exists=True)

        ensure_drift_check_job(spec, client=client)
        ensure_drift_check_job(spec, client=client)

        assert not client.created
        assert len(client.updated) == 2  # both no-ops against the same definition


class TestDeleteDriftCheckJob:
    def test_deletes_an_existing_job(self, spec, monkeypatch):
        monkeypatch.setattr(scheduler_module, "_scheduler", lambda: None)
        client = FakeClient()
        assert delete_drift_check_job(spec, client=client) is True
        assert client.deleted == [spec.full_name]

    def test_deleting_an_absent_job_is_not_an_error(self, spec, monkeypatch):
        """Teardown must be safe to run twice."""
        monkeypatch.setattr(scheduler_module, "_scheduler", lambda: None)
        assert delete_drift_check_job(spec, client=FakeClient(delete_fails=True)) is False
