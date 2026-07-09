"""Fixtures for the evaluation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.train import run_local_training


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory) -> Path:
    """Train once so the report and dashboard run against real artefacts."""
    output = tmp_path_factory.mktemp("eval_artifacts")
    run_local_training(output_dir=output, n_resamples=10)
    return output
