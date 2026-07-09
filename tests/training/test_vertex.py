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

    def CustomJob(self, **kwargs):  # noqa: N802 -- mirrors the SDK's name
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
    """Submission runs our own container image, not a packaged script.

    `CustomTrainingJob(script_path=...)` ships a single file plus a generated
    setup.py. `train.py` imports src.features / src.evaluation / src.monitoring,
    none of which would exist remotely -- and it needs setuptools locally, which
    uv does not install. Both failures appeared on the first real submission.
    """

    def _spec(self, fake_sdk) -> dict:
        return fake_sdk.job.init_kwargs["worker_pool_specs"][0]

    def test_initialises_the_sdk_from_config(self, config, fake_sdk):
        vertex.submit_training_job(config)
        assert fake_sdk.init_kwargs == {
            "project": "test-project",
            "location": "europe-west2",
            "staging_bucket": "gs://test-bucket",
        }

    def test_runs_our_own_artifact_registry_image(self, config, fake_sdk):
        vertex.submit_training_job(config)
        image = self._spec(fake_sdk)["container_spec"]["image_uri"]
        assert image == (
            "europe-west2-docker.pkg.dev/test-project/fraud-detection/fraud-drift-monitor:latest"
        )

    def test_never_packages_a_bare_script(self, config, fake_sdk):
        """The regression this rewrite exists to prevent."""
        vertex.submit_training_job(config)
        assert "script_path" not in fake_sdk.job.init_kwargs
        assert "requirements" not in fake_sdk.job.init_kwargs

    def test_invokes_the_training_module_not_a_file(self, config, fake_sdk):
        vertex.submit_training_job(config)
        assert self._spec(fake_sdk)["container_spec"]["command"] == [
            "python",
            "-m",
            "src.training.train",
        ]

    def test_runs_on_the_documented_machine_type(self, config, fake_sdk):
        """The README's cost estimate assumes this machine."""
        vertex.submit_training_job(config)
        spec = self._spec(fake_sdk)
        assert spec["machine_spec"]["machine_type"] == "n1-standard-4"
        assert spec["replica_count"] == 1

    def test_machine_type_is_overridable(self, config, fake_sdk):
        vertex.submit_training_job(config, machine_type="n1-highmem-8")
        assert self._spec(fake_sdk)["machine_spec"]["machine_type"] == "n1-highmem-8"

    def test_forwards_local_backend_args_to_the_remote_run(self, config, fake_sdk):
        vertex.submit_training_job(config, source="bigquery", args=["--backend", "vertex"])
        assert self._spec(fake_sdk)["container_spec"]["args"][:2] == ["--backend", "local"]

    def test_artefacts_are_written_to_the_gcs_fuse_mount(self, config, fake_sdk):
        """A container filesystem is ephemeral; /gcs survives the job."""
        vertex.submit_training_job(config)
        args = self._spec(fake_sdk)["container_spec"]["args"]
        assert "--output-dir" in args
        assert "/gcs/test-bucket/artifacts" in args

    def test_a_caller_supplied_output_dir_is_overridden(self, config, fake_sdk):
        """A local path would silently discard every artefact."""
        vertex.submit_training_job(config, args=["--output-dir", "artifacts"])
        args = self._spec(fake_sdk)["container_spec"]["args"]
        assert args.count("--output-dir") == 1
        assert "artifacts" not in args or "/gcs/" in " ".join(args)
        assert "/gcs/test-bucket/artifacts" in args

    def test_gcp_config_is_passed_as_environment(self, config, fake_sdk):
        """The container has no .env file; config must arrive as env vars."""
        vertex.submit_training_job(config)
        env = {e["name"]: e["value"] for e in self._spec(fake_sdk)["container_spec"]["env"]}
        assert env["GCP_PROJECT_ID"] == "test-project"
        assert env["GCP_REGION"] == "europe-west2"
        assert env["BIGQUERY_DATASET"] == "fraud_features"

    def test_returns_the_job_handle(self, config, fake_sdk):
        assert vertex.submit_training_job(config) is fake_sdk.job


class TestImageUri:
    def test_points_at_artifact_registry_in_the_configured_region(self, config):
        assert vertex.default_image_uri(config).startswith("europe-west2-docker.pkg.dev/")

    def test_tag_is_overridable(self, config):
        assert vertex.default_image_uri(config, tag="abc123").endswith(":abc123")


class TestStripOutputDirFlag:
    def test_removes_space_separated_flag(self):
        assert vertex.strip_output_dir_flag(
            ["--output-dir", "artifacts", "--source", "sample"]
        ) == [
            "--source",
            "sample",
        ]

    def test_removes_equals_separated_flag(self):
        assert vertex.strip_output_dir_flag(["--output-dir=artifacts", "--source", "sample"]) == [
            "--source",
            "sample",
        ]

    def test_leaves_other_arguments_alone(self):
        args = ["--source", "sample"]
        assert vertex.strip_output_dir_flag(args) == args
