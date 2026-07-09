"""Shared fixtures for the inference service tests.

Training both variants is expensive, so it happens once per session into a temp
directory that then stands in for `artifacts/`. These are real fitted models and
real SHAP explainers -- the API tests exercise the genuine prediction path, not a
mock of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.dataset import load_features, temporal_split
from src.training.train import run_local_training


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory) -> Path:
    """Train both variants once and return the directory holding the artefacts."""
    output = tmp_path_factory.mktemp("artifacts")
    run_local_training(output_dir=output, n_resamples=10)
    return output


@pytest.fixture(scope="session")
def dataset():
    return temporal_split(load_features("sample"))
