"""Environment-driven GCP configuration.

Every GCP identifier is read from the environment (populated from `.env` via
python-dotenv). Nothing in this repository hardcodes a project ID, bucket, or
credential.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

REQUIRED_KEYS = (
    "GCP_PROJECT_ID",
    "GCP_REGION",
    "GCP_BUCKET_NAME",
    "BIGQUERY_DATASET",
    "FEATURE_STORE_ID",
)

# Values shipped in .env.example. If one of these survives into a real run the
# user copied the template but never filled it in, which fails far away from the
# cause -- so we reject them here.
_PLACEHOLDER_PREFIX = "your-"


class ConfigError(RuntimeError):
    """Raised when required GCP configuration is missing or still a placeholder."""


@dataclass(frozen=True)
class GCPConfig:
    """Resolved GCP settings for the feature pipeline."""

    project_id: str
    region: str
    bucket_name: str
    bigquery_dataset: str
    feature_store_id: str

    @property
    def dataset_ref(self) -> str:
        """Fully-qualified BigQuery dataset, e.g. `my-project.fraud_features`."""
        return f"{self.project_id}.{self.bigquery_dataset}"

    def table_ref(self, table_name: str) -> str:
        """Fully-qualified BigQuery table reference."""
        return f"{self.dataset_ref}.{table_name}"


def load_config(*, env_file: str | None = None, override: bool = False) -> GCPConfig:
    """Load and validate GCP configuration from the environment.

    Args:
        env_file: Optional explicit path to a `.env` file. Defaults to discovery
            of `.env` from the current working directory upward.
        override: Whether values in the `.env` file take precedence over
            variables already exported in the shell. Defaults to False so that
            CI-injected environment variables win over any stray local file.

    Raises:
        ConfigError: If a required key is absent, empty, or still a placeholder.
    """
    load_dotenv(dotenv_path=env_file, override=override)

    values: dict[str, str] = {}
    missing: list[str] = []
    placeholders: list[str] = []

    for key in REQUIRED_KEYS:
        raw = os.environ.get(key, "").strip()
        if not raw:
            missing.append(key)
        elif raw.startswith(_PLACEHOLDER_PREFIX):
            placeholders.append(key)
        else:
            values[key] = raw

    problems = []
    if missing:
        problems.append(f"missing or empty: {', '.join(sorted(missing))}")
    if placeholders:
        problems.append(
            f"still set to .env.example placeholders: {', '.join(sorted(placeholders))}"
        )
    if problems:
        raise ConfigError(
            "Invalid GCP configuration (" + "; ".join(problems) + "). "
            "Copy .env.example to .env and fill in your own values."
        )

    return GCPConfig(
        project_id=values["GCP_PROJECT_ID"],
        region=values["GCP_REGION"],
        bucket_name=values["GCP_BUCKET_NAME"],
        bigquery_dataset=values["BIGQUERY_DATASET"],
        feature_store_id=values["FEATURE_STORE_ID"],
    )
