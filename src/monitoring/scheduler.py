"""Cloud Scheduler job that triggers the daily drift check.

Scheduler fires an authenticated HTTP POST at a Cloud Run endpoint. Two details
that are easy to get wrong and expensive to debug:

* **OIDC, not a shared secret.** The job carries an OIDC token minted for a
  service account, and Cloud Run verifies it. No API key in the job definition,
  nothing to rotate, nothing to leak.
* **Idempotent provisioning.** `ensure_drift_check_job` creates the job or
  updates it in place. Re-running the deploy must not fail with `AlreadyExists`,
  and must not silently leave a stale schedule behind.

As elsewhere, the SDK is imported lazily so this module stays importable without
the `gcp` extra installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.features.config import GCPConfig

logger = logging.getLogger(__name__)

#: 02:00 every day. Off-peak, and late enough that the previous day's
#: transactions have all landed in BigQuery.
DEFAULT_SCHEDULE = "0 2 * * *"
DEFAULT_TIMEZONE = "Europe/London"
DEFAULT_JOB_NAME = "fraud-drift-check"

#: The drift check reads a day of features and runs SHAP; it is not a 30-second
#: job. Cloud Scheduler's default deadline is 180s.
DEFAULT_ATTEMPT_DEADLINE_SECONDS = 900


def _scheduler():
    """Import the Cloud Scheduler SDK lazily."""
    from google.cloud import scheduler_v1

    return scheduler_v1


@dataclass(frozen=True)
class SchedulerJobSpec:
    """Everything needed to describe the drift-check job."""

    config: GCPConfig
    target_uri: str
    service_account_email: str
    schedule: str = DEFAULT_SCHEDULE
    timezone: str = DEFAULT_TIMEZONE
    job_name: str = DEFAULT_JOB_NAME
    attempt_deadline_seconds: int = DEFAULT_ATTEMPT_DEADLINE_SECONDS

    @property
    def parent(self) -> str:
        return f"projects/{self.config.project_id}/locations/{self.config.region}"

    @property
    def full_name(self) -> str:
        return f"{self.parent}/jobs/{self.job_name}"


def build_job(spec: SchedulerJobSpec) -> dict:
    """The job definition, as a plain dict the SDK accepts.

    Returned as a dict rather than a proto so it is assertable in tests without
    the SDK on the path.
    """
    if not spec.target_uri.startswith("https://"):
        raise ValueError(f"target_uri must be https; got {spec.target_uri!r}")

    return {
        "name": spec.full_name,
        "description": "Daily feature-drift check; triggers retraining when drift is significant.",
        "schedule": spec.schedule,
        "time_zone": spec.timezone,
        "attempt_deadline": {"seconds": spec.attempt_deadline_seconds},
        "http_target": {
            "uri": spec.target_uri,
            "http_method": "POST",
            "headers": {"Content-Type": "application/json"},
            # The audience must be the bare service URL, not the path, or
            # Cloud Run rejects the token.
            "oidc_token": {
                "service_account_email": spec.service_account_email,
                "audience": _audience(spec.target_uri),
            },
        },
    }


def _audience(target_uri: str) -> str:
    """The OIDC audience: scheme + host, without the path."""
    from urllib.parse import urlsplit

    parts = urlsplit(target_uri)
    return f"{parts.scheme}://{parts.netloc}"


def ensure_drift_check_job(spec: SchedulerJobSpec, client=None):
    """Create the Cloud Scheduler job, or update it if it already exists.

    Idempotent: safe to call on every deploy. Returns the SDK's job handle.
    """
    scheduler_v1 = _scheduler()
    client = client or scheduler_v1.CloudSchedulerClient()
    job = build_job(spec)

    try:
        existing = client.get_job(name=spec.full_name)
    except Exception:  # noqa: BLE001 -- NotFound is the expected first-run path
        existing = None

    if existing is None:
        logger.info("creating Cloud Scheduler job %s (%s)", spec.job_name, spec.schedule)
        return client.create_job(parent=spec.parent, job=job)

    logger.info("updating Cloud Scheduler job %s (%s)", spec.job_name, spec.schedule)
    return client.update_job(job=job)


def delete_drift_check_job(spec: SchedulerJobSpec, client=None) -> bool:
    """Delete the job. Returns False if it did not exist. Used when tearing down."""
    scheduler_v1 = _scheduler()
    client = client or scheduler_v1.CloudSchedulerClient()
    try:
        client.delete_job(name=spec.full_name)
    except Exception:  # noqa: BLE001 -- deleting an absent job is not an error
        logger.info("Cloud Scheduler job %s did not exist", spec.job_name)
        return False
    return True
