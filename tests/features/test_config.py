"""Tests for environment-driven GCP configuration."""

from __future__ import annotations

import pytest

from src.features.config import REQUIRED_KEYS, ConfigError, load_config


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Strip GCP vars and stop dotenv discovering the repository's real `.env`.

    `chdir` alone is not enough: with no explicit path, python-dotenv's
    `find_dotenv` walks up from the *calling module's* directory, not the cwd —
    so it finds `<repo>/.env` and silently satisfies the very keys these tests
    assert are missing. That only breaks once a developer actually has a `.env`,
    which is why CI never caught it.

    Discovery is neutered; an explicit `env_file=` still loads for real.
    """
    from src.features import config as config_module

    for key in REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)

    real_load_dotenv = config_module.load_dotenv

    def load_dotenv_without_discovery(dotenv_path=None, **kwargs):
        if dotenv_path is None:
            return False  # never walk the filesystem looking for a .env
        return real_load_dotenv(dotenv_path=dotenv_path, **kwargs)

    monkeypatch.setattr(config_module, "load_dotenv", load_dotenv_without_discovery)
    monkeypatch.chdir(tmp_path)


def test_loads_a_complete_environment(gcp_env):
    config = load_config()
    assert config.project_id == "test-project"
    assert config.region == "europe-west2"
    assert config.bigquery_dataset == "fraud_features"
    assert config.feature_store_id == "fraud_online_store"


def test_dataset_and_table_refs_are_fully_qualified(gcp_env):
    config = load_config()
    assert config.dataset_ref == "test-project.fraud_features"
    assert config.table_ref("raw_transactions") == "test-project.fraud_features.raw_transactions"


def test_config_is_immutable(gcp_env):
    config = load_config()
    with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
        config.project_id = "someone-elses-project"  # type: ignore[misc]


def test_missing_key_is_rejected(gcp_env, monkeypatch):
    monkeypatch.delenv("GCP_PROJECT_ID")
    with pytest.raises(ConfigError, match="missing or empty: GCP_PROJECT_ID"):
        load_config()


def test_empty_key_is_rejected(gcp_env, monkeypatch):
    monkeypatch.setenv("BIGQUERY_DATASET", "   ")
    with pytest.raises(ConfigError, match="missing or empty: BIGQUERY_DATASET"):
        load_config()


def test_unfilled_placeholder_is_rejected(gcp_env, monkeypatch):
    """Copying .env.example without editing it must fail loudly and immediately."""
    monkeypatch.setenv("GCP_PROJECT_ID", "your-gcp-project-id")
    with pytest.raises(ConfigError, match="still set to .env.example placeholders: GCP_PROJECT_ID"):
        load_config()


def test_all_problems_are_reported_together(gcp_env, monkeypatch):
    monkeypatch.delenv("GCP_REGION")
    monkeypatch.setenv("GCP_BUCKET_NAME", "your-gcs-bucket-name")
    with pytest.raises(ConfigError) as excinfo:
        load_config()
    message = str(excinfo.value)
    assert "GCP_REGION" in message and "GCP_BUCKET_NAME" in message


def test_shell_env_wins_over_dotenv_by_default(monkeypatch, tmp_path):
    """CI injects real values as env vars; a stray local .env must not override them."""
    env_file = tmp_path / ".env"
    env_file.write_text("GCP_PROJECT_ID=from-dotenv\n")
    for key, value in {
        "GCP_PROJECT_ID": "from-shell",
        "GCP_REGION": "europe-west2",
        "GCP_BUCKET_NAME": "b",
        "BIGQUERY_DATASET": "d",
        "FEATURE_STORE_ID": "f",
    }.items():
        monkeypatch.setenv(key, value)

    assert load_config(env_file=str(env_file)).project_id == "from-shell"
    assert load_config(env_file=str(env_file), override=True).project_id == "from-dotenv"


def test_env_example_covers_every_required_key():
    """`.env.example` must stay in step with REQUIRED_KEYS -- CLAUDE.md mandates it."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    example = (repo_root / ".env.example").read_text()
    documented = {
        line.split("=", 1)[0].strip()
        for line in example.splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert set(REQUIRED_KEYS) <= documented
