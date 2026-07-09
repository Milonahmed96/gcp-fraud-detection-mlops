"""Submit training to Vertex AI Custom Training.

This is the `--backend vertex` path. It does not reimplement training: it ships
`src/training/train.py` to a managed machine and runs it there with
`--backend local`, so the fitting code is byte-identical in both environments.

The SDK is imported lazily and never at module import, keeping `train.py`
runnable with no GCP libraries configured.
"""

from __future__ import annotations

import logging
from typing import Any

from src.features.config import GCPConfig

logger = logging.getLogger(__name__)

#: Cheapest machine that comfortably fits the sample. The README's cost estimate
#: assumes this; changing it changes the bill.
DEFAULT_MACHINE_TYPE = "n1-standard-4"

#: Prebuilt Vertex AI container. sklearn/xgboost/lightgbm are pip-installed on
#: top via `requirements`, so the image does not need rebuilding per commit.
DEFAULT_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest"

TRAINING_SCRIPT = "src/training/train.py"

#: Mirrors pyproject's runtime deps. Vertex installs these into the container.
TRAINING_REQUIREMENTS = (
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.5",
    "xgboost>=2.1",
    "lightgbm>=4.5",
    "joblib>=1.4",
    "google-cloud-bigquery>=3.25",
    "python-dotenv>=1.0",
)


def _aiplatform():
    """Import the Vertex AI SDK lazily."""
    from google.cloud import aiplatform

    return aiplatform


def strip_backend_flag(args: list[str]) -> list[str]:
    """Remove `--backend vertex` from argv before forwarding it to the remote run.

    Without this the remote process would resubmit itself to Vertex AI, forever.
    Handles both `--backend vertex` and `--backend=vertex`.
    """
    cleaned: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--backend":
            skip_next = True
            continue
        if arg.startswith("--backend="):
            continue
        cleaned.append(arg)
    return cleaned


def build_job_args(source: str, args: list[str]) -> list[str]:
    """Arguments handed to the remote `train.py`, forced onto the local backend."""
    forwarded = strip_backend_flag(args)
    has_source = any(a == "--source" or a.startswith("--source=") for a in forwarded)
    if not has_source:
        forwarded += ["--source", source]
    return ["--backend", "local", *forwarded]


def submit_training_job(
    config: GCPConfig,
    *,
    source: str = "bigquery",
    args: list[str] | None = None,
    machine_type: str = DEFAULT_MACHINE_TYPE,
    container_uri: str = DEFAULT_CONTAINER_URI,
    display_name: str = "fraud-detection-training",
    sync: bool = True,
) -> Any:
    """Package `train.py` and run it as a Vertex AI Custom Training job.

    Returns the SDK's job handle. Raises whatever the SDK raises -- a training
    job that silently fails to submit is worse than a stack trace.
    """
    aiplatform = _aiplatform()
    aiplatform.init(
        project=config.project_id,
        location=config.region,
        staging_bucket=f"gs://{config.bucket_name}",
    )

    job = aiplatform.CustomTrainingJob(
        display_name=display_name,
        script_path=TRAINING_SCRIPT,
        container_uri=container_uri,
        requirements=list(TRAINING_REQUIREMENTS),
    )

    job_args = build_job_args(source, args or [])
    logger.info("submitting %s to Vertex AI with args: %s", display_name, job_args)

    return job.run(
        args=job_args,
        machine_type=machine_type,
        replica_count=1,
        sync=sync,
    )
