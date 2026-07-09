"""Tests for the Vertex AI Custom Training submission path.

The SDK is replaced with a fake. What matters is that the remote job is told to
run the *local* backend -- otherwise it would resubmit itself to Vertex AI in an
infinite loop, billing all the way.
"""

from __future__ import annotations

import pytest

from src.features.config import GCPConfig
from src.training import vertex


@pytest.fixture
def config() -> GCPConfig:
    return GCPConfig("test-project", "europe-west2", "test-bucket", "fraud_features", "store")


class FakeJob:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.run_kwargs: dict | None = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return "model-artifact"


class FakeAIPlatform:
    def __init__(self):
        self.init_kwargs: dict | None = None
        self.job: FakeJob | None = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def CustomTrainingJob(self, **kwargs):  # noqa: N802 -- mirrors the SDK's name
        self.job = FakeJob(**kwargs)
        return self.job


@pytest.fixture
def fake_sdk(monkeypatch) -> FakeAIPlatform:
    fake = FakeAIPlatform()
    monkeypatch.setattr(vertex, "_aiplatform", lambda: fake)
    return fake


class TestStripBackendFlag:
    def test_removes_space_separated_flag(self):
        assert vertex.strip_backend_flag(["--backend", "vertex", "--source", "bigquery"]) == [
            "--source",
            "bigquery",
        ]

    def test_removes_equals_separated_flag(self):
        assert vertex.strip_backend_flag(["--backend=vertex", "--source", "sample"]) == [
            "--source",
            "sample",
        ]

    def test_leaves_other_arguments_alone(self):
        args = ["--source", "sample", "--bootstrap-resamples", "10"]
        assert vertex.strip_backend_flag(args) == args

    def test_tolerates_a_trailing_backend_flag(self):
        assert vertex.strip_backend_flag(["--source", "sample", "--backend"]) == [
            "--source",
            "sample",
        ]

    def test_empty_args(self):
        assert vertex.strip_backend_flag([]) == []


class TestBuildJobArgs:
    def test_remote_run_is_forced_onto_the_local_backend(self):
        """Without this the remote job resubmits itself to Vertex AI, forever."""
        args = vertex.build_job_args("bigquery", ["--backend", "vertex"])
        assert args[:2] == ["--backend", "local"]
        assert "vertex" not in args

    def test_only_one_backend_flag_survives(self):
        args = vertex.build_job_args("sample", ["--backend", "vertex"])
        assert args.count("--backend") == 1

    def test_source_is_injected_when_absent(self):
        assert vertex.build_job_args("bigquery", []) == [
            "--backend",
            "local",
            "--source",
            "bigquery",
        ]

    def test_an_explicit_source_is_not_duplicated(self):
        args = vertex.build_job_args("bigquery", ["--source", "sample"])
        assert args.count("--source") == 1
        assert "sample" in args and "bigquery" not in args

    def test_an_equals_form_source_is_not_duplicated(self):
        args = vertex.build_job_args("bigquery", ["--source=sample"])
        assert args.count("--source=sample") == 1
        assert "--source" not in args

    def test_other_flags_are_forwarded(self):
        args = vertex.build_job_args("sample", ["--bootstrap-resamples", "50"])
        assert "--bootstrap-resamples" in args and "50" in args


class TestSubmitTrainingJob:
    def test_initialises_the_sdk_from_config(self, config, fake_sdk):
        vertex.submit_training_job(config)
        assert fake_sdk.init_kwargs == {
            "project": "test-project",
            "location": "europe-west2",
            "staging_bucket": "gs://test-bucket",
        }

    def test_packages_the_training_script(self, config, fake_sdk):
        vertex.submit_training_job(config)
        assert fake_sdk.job.init_kwargs["script_path"] == "src/training/train.py"

    def test_installs_the_model_libraries_in_the_container(self, config, fake_sdk):
        vertex.submit_training_job(config)
        requirements = " ".join(fake_sdk.job.init_kwargs["requirements"])
        assert "xgboost" in requirements and "lightgbm" in requirements

    def test_runs_on_the_documented_machine_type(self, config, fake_sdk):
        """The README's cost estimate assumes this machine."""
        vertex.submit_training_job(config)
        assert fake_sdk.job.run_kwargs["machine_type"] == "n1-standard-4"
        assert fake_sdk.job.run_kwargs["replica_count"] == 1

    def test_forwards_local_backend_args_to_the_remote_run(self, config, fake_sdk):
        vertex.submit_training_job(config, source="bigquery", args=["--backend", "vertex"])
        assert fake_sdk.job.run_kwargs["args"][:2] == ["--backend", "local"]

    def test_returns_the_sdk_handle(self, config, fake_sdk):
        assert vertex.submit_training_job(config) == "model-artifact"

    def test_machine_type_is_overridable(self, config, fake_sdk):
        vertex.submit_training_job(config, machine_type="n1-highmem-8")
        assert fake_sdk.job.run_kwargs["machine_type"] == "n1-highmem-8"
