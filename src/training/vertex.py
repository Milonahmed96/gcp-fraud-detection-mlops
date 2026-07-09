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

#: The training image. This is the *monitoring* image: it already carries the
#: `gcp` extra (BigQuery, Vertex AI SDK) plus xgboost, lightgbm and shap, and it
#: is built and smoke-tested by CI on every commit. Reusing it means the code
#: that trains in the cloud is the same code, in the same image, that CI ran.
DEFAULT_IMAGE_NAME = "fraud-drift-monitor"

#: Vertex AI mounts every GCS bucket the job can see under /gcs via Cloud
#: Storage FUSE. Writing artefacts there is how they survive the container.
GCS_FUSE_ROOT = "/gcs"

TRAINING_MODULE = "src.training.train"


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


def strip_output_dir_flag(args: list[str]) -> list[str]:
    """Remove any `--output-dir`, so the remote run always writes to GCS."""
    cleaned: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--output-dir":
            skip_next = True
            continue
        if arg.startswith("--output-dir="):
            continue
        cleaned.append(arg)
    return cleaned


def build_job_args(source: str, args: list[str], *, output_dir: str | None = None) -> list[str]:
    """Arguments handed to the remote `train.py`, forced onto the local backend."""
    forwarded = strip_output_dir_flag(strip_backend_flag(args))
    has_source = any(a == "--source" or a.startswith("--source=") for a in forwarded)
    if not has_source:
        forwarded += ["--source", source]
    if output_dir:
        forwarded += ["--output-dir", output_dir]
    return ["--backend", "local", *forwarded]


def default_image_uri(
    config: GCPConfig, *, repository: str = "fraud-detection", tag: str = "latest"
) -> str:
    """Artifact Registry URI of the training image."""
    return (
        f"{config.region}-docker.pkg.dev/{config.project_id}/"
        f"{repository}/{DEFAULT_IMAGE_NAME}:{tag}"
    )


def artifact_output_dir(config: GCPConfig) -> str:
    """Where the remote job writes models, explainers and metrics.

    Vertex mounts GCS under /gcs via Cloud Storage FUSE, so a plain filesystem
    path reaches the bucket and the artefacts outlive the container.
    """
    return f"{GCS_FUSE_ROOT}/{config.bucket_name}/artifacts"


def submit_training_job(
    config: GCPConfig,
    *,
    source: str = "bigquery",
    args: list[str] | None = None,
    machine_type: str = DEFAULT_MACHINE_TYPE,
    image_uri: str | None = None,
    display_name: str = "fraud-detection-training",
    sync: bool = True,
) -> Any:
    """Run `train.py` on Vertex AI as a container job.

    Uses our own image rather than `CustomTrainingJob(script_path=...)`. That
    API packages a *single file* and a generated `setup.py`; `train.py` imports
    `src.features`, `src.evaluation` and `src.monitoring`, none of which would
    exist in the remote container. It also required `setuptools` in the local
    venv, which uv does not install. Both failures showed up on the first real
    submission and neither could have shown up against a fake client.

    Shipping the image CI already builds and smoke-tests means the code that
    trains in the cloud is byte-identical to the code that trains locally.

    Returns the SDK's job handle. Raises whatever the SDK raises -- a training
    job that silently fails to submit is worse than a stack trace.
    """
    aiplatform = _aiplatform()
    aiplatform.init(
        project=config.project_id,
        location=config.region,
        staging_bucket=f"gs://{config.bucket_name}",
    )

    image_uri = image_uri or default_image_uri(config)
    job_args = build_job_args(source, args or [], output_dir=artifact_output_dir(config))

    logger.info("submitting %s: image=%s args=%s", display_name, image_uri, job_args)

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": machine_type},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image_uri,
                    "command": ["python", "-m", TRAINING_MODULE],
                    "args": job_args,
                    "env": [
                        {"name": "GCP_PROJECT_ID", "value": config.project_id},
                        {"name": "GCP_REGION", "value": config.region},
                        {"name": "GCP_BUCKET_NAME", "value": config.bucket_name},
                        {"name": "BIGQUERY_DATASET", "value": config.bigquery_dataset},
                        {"name": "FEATURE_STORE_ID", "value": config.feature_store_id},
                    ],
                },
            }
        ],
        staging_bucket=f"gs://{config.bucket_name}",
    )

    job.run(sync=sync)
    return job
