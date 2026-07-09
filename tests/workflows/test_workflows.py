"""Tests for the CI/CD workflows.

A broken workflow is discovered when you push to main, which is the worst
possible moment. These assert the invariants that matter:

* No service-account key, anywhere. Authentication is Workload Identity
  Federation, which needs `id-token: write`.
* `main` never deploys code that has not passed the same gate a PR does.
* Traffic never shifts to a revision that was not smoke-tested first.
* The test job installs the `gcp` extra, without which the suite cannot import
  `bigquery.SchemaField`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def steps_of(workflow: dict, job: str) -> list[dict]:
    return workflow["jobs"][job].get("steps", [])


def run_script(workflow: dict, job: str) -> str:
    """All shell in a job, concatenated."""
    return "\n".join(step.get("run", "") for step in steps_of(workflow, job))


@pytest.fixture(scope="module")
def ci() -> dict:
    return load("ci.yml")


@pytest.fixture(scope="module")
def deploy() -> dict:
    return load("deploy.yml")


class TestWorkflowsAreValid:
    @pytest.mark.parametrize("name", ["ci.yml", "deploy.yml"])
    def test_parses_as_yaml(self, name):
        assert load(name)["jobs"]

    @pytest.mark.parametrize("name", ["ci.yml", "deploy.yml"])
    def test_has_a_name_and_triggers(self, name):
        workflow = load(name)
        assert workflow["name"]
        # PyYAML parses a bare `on:` key as the boolean True. Accept either.
        assert workflow.get("on") or workflow.get(True)


class TestNoCredentialsInGit:
    """The single most important property of this pipeline."""

    @pytest.mark.parametrize("name", ["ci.yml", "deploy.yml"])
    def test_no_service_account_key_is_referenced(self, name):
        text = (WORKFLOWS / name).read_text().lower()
        for forbidden in ("credentials_json", "service_account_key", "gcp_sa_key", "private_key"):
            assert forbidden not in text, f"{name} references {forbidden}"

    def test_deploy_requests_an_oidc_token(self, deploy):
        """Workload Identity Federation needs `id-token: write`."""
        assert deploy["jobs"]["deploy"]["permissions"]["id-token"] == "write"

    def test_deploy_uses_workload_identity_federation(self, deploy):
        auth = next(step for step in steps_of(deploy, "deploy") if step.get("id") == "auth")
        assert auth["uses"].startswith("google-github-actions/auth@")
        assert "workload_identity_provider" in auth["with"]
        assert "credentials_json" not in auth["with"]

    def test_contents_permission_is_read_only(self, deploy):
        assert deploy["jobs"]["deploy"]["permissions"]["contents"] == "read"

    def test_the_bootstrap_script_creates_no_key(self):
        script = (REPO_ROOT / "infrastructure" / "setup_gcp.sh").read_text()
        assert "keys create" not in script
        assert "No service-account key was created" in script

    def test_the_bootstrap_script_scopes_wif_to_this_repository(self):
        """Without an attribute-condition, any repo on GitHub could assume the SA."""
        script = (REPO_ROOT / "infrastructure" / "setup_gcp.sh").read_text()
        assert "attribute-condition" in script
        assert "assertion.repository ==" in script

    def test_the_bootstrap_script_avoids_owner_and_editor_roles(self):
        script = (REPO_ROOT / "infrastructure" / "setup_gcp.sh").read_text()
        assert "roles/owner" not in script
        assert "roles/editor" not in script


class TestCIGate:
    def test_runs_on_pull_requests_into_develop_and_main(self, ci):
        triggers = ci.get("on") or ci.get(True)
        assert set(triggers["pull_request"]["branches"]) == {"develop", "main"}

    def test_is_reusable_so_deploy_can_call_it(self, ci):
        triggers = ci.get("on") or ci.get(True)
        assert "workflow_call" in triggers

    def test_lints_and_format_checks(self, ci):
        script = run_script(ci, "lint")
        assert "ruff check" in script
        assert "ruff format --check" in script

    def test_runs_the_test_suite(self, ci):
        assert "pytest tests/" in run_script(ci, "test")

    @pytest.mark.parametrize("job", ["lint", "test", "build-images"])
    def test_every_job_installs_the_gcp_extra(self, ci, job):
        """Without it the suite cannot import bigquery.SchemaField."""
        assert "--extra gcp" in run_script(ci, job)

    def test_images_build_only_after_lint_and_tests_pass(self, ci):
        assert set(ci["jobs"]["build-images"]["needs"]) == {"lint", "test"}

    def test_images_are_built_but_not_pushed(self, ci):
        for step in steps_of(ci, "build-images"):
            if "docker/build-push-action" in step.get("uses", ""):
                assert step["with"]["push"] is False

    def test_both_images_are_built(self, ci):
        files = {
            step["with"]["file"]
            for step in steps_of(ci, "build-images")
            if "docker/build-push-action" in step.get("uses", "")
        }
        assert files == {"Dockerfile", "Dockerfile.monitoring"}

    def test_training_runs_before_the_image_build(self, ci):
        """Both Dockerfiles COPY artifacts/, which only training produces."""
        script = run_script(ci, "build-images")
        assert "src.training.train" in script

    def test_the_containers_are_smoke_tested_not_merely_built(self, ci):
        script = run_script(ci, "build-images")
        assert "/health" in script
        assert "/predict" in script

    def test_the_sample_data_freshness_check_runs(self, ci):
        assert "test_sample_data.py" in run_script(ci, "test")


class TestDeploySafety:
    def test_only_main_deploys(self, deploy):
        triggers = deploy.get("on") or deploy.get(True)
        assert triggers["push"]["branches"] == ["main"]

    def test_deploy_reuses_the_ci_gate(self, deploy):
        """main must never deploy code that has not passed the PR gate."""
        assert deploy["jobs"]["verify"]["uses"] == "./.github/workflows/ci.yml"
        assert deploy["jobs"]["deploy"]["needs"] == "verify"

    def test_concurrency_prevents_racing_deploys(self, deploy):
        """Two deploys racing would leave the traffic split indeterminate."""
        assert deploy["concurrency"]["group"] == "deploy-main"
        assert deploy["concurrency"]["cancel-in-progress"] is False

    def test_new_revisions_take_no_traffic_on_deploy(self, deploy):
        script = run_script(deploy, "deploy")
        assert script.count("--no-traffic") == 2  # one per variant

    def test_revisions_are_smoke_tested_before_traffic_shifts(self, deploy):
        """The ordering IS the rollback strategy: if the smoke test fails, the
        job stops and the previous revision keeps serving 100%."""
        script = run_script(deploy, "deploy")
        smoke = script.index("smoke-testing")
        shift = script.index("update-traffic")
        assert smoke < shift

    def test_the_smoke_test_fails_the_job_on_error(self, deploy):
        smoke = next(
            step
            for step in steps_of(deploy, "deploy")
            if "smoke-test" in step.get("name", "").lower()
        )
        assert "set -euo pipefail" in smoke["run"]
        assert "curl -sf" in smoke["run"]  # -f makes curl exit non-zero on 4xx/5xx
        assert "jq -er" in smoke["run"]  # -e makes jq exit non-zero on a null result

    def test_the_smoke_test_impersonates_to_mint_an_id_token(self, deploy):
        """A Workload Identity federated credential cannot mint an ID token
        directly. Bare `gcloud auth print-identity-token` would 403 against the
        private service on every deploy."""
        smoke = next(
            step
            for step in steps_of(deploy, "deploy")
            if "smoke-test" in step.get("name", "").lower()
        )
        assert "--impersonate-service-account" in smoke["run"]
        assert "--audiences=" in smoke["run"]

    def test_the_bootstrap_grants_token_creator_for_that_impersonation(self):
        """Without this binding the smoke test 403s and nothing ever deploys."""
        script = (REPO_ROOT / "infrastructure" / "setup_gcp.sh").read_text()
        assert "roles/iam.serviceAccountTokenCreator" in script

    def test_both_variants_are_deployed_as_separate_revisions(self, deploy):
        script = run_script(deploy, "deploy")
        assert "SERVING_VARIANT=xgboost" in script
        assert "SERVING_VARIANT=lightgbm" in script

    def test_traffic_is_split_between_the_two_variants(self, deploy):
        assert "--to-tags=" in run_script(deploy, "deploy")

    def test_images_are_tagged_with_the_commit_sha(self, deploy):
        """A revision's model must be reproducible from its image tag."""
        assert "${GITHUB_SHA}" in run_script(deploy, "deploy")

    def test_services_are_private(self, deploy):
        script = run_script(deploy, "deploy")
        assert "--allow-unauthenticated" not in script.replace("--no-allow-unauthenticated", "")

    def test_the_monitor_gets_a_long_timeout(self, deploy):
        """The drift check is a minutes-long batch job."""
        assert "--timeout=900" in run_script(deploy, "deploy")

    def test_the_scheduler_job_is_provisioned_by_our_tested_code(self, deploy):
        """Not a hand-rolled `gcloud scheduler jobs create || update`."""
        script = run_script(deploy, "deploy")
        assert "src.monitoring.scheduler" in script
        assert "gcloud scheduler jobs create" not in script

    def test_the_deploy_targets_a_protected_environment(self, deploy):
        assert deploy["jobs"]["deploy"]["environment"] == "production"


class TestActionsArePinned:
    @pytest.mark.parametrize("name", ["ci.yml", "deploy.yml"])
    def test_every_action_pins_a_major_version(self, name):
        """`uses: foo/bar@main` silently changes under you."""
        workflow = load(name)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                uses = step.get("uses")
                if not uses:
                    continue
                assert "@" in uses, f"{uses} is not pinned"
                ref = uses.split("@", 1)[1]
                assert ref not in {"main", "master", "latest"}, f"{uses} floats"
