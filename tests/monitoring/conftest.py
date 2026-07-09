"""Fixtures for the monitoring tests.

Trains once per session so the drift monitor runs against real artefacts: a real
reference profile, a real SHAP importance baseline, and a real explainer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.dataset import load_features, temporal_split
from src.training.train import run_local_training


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("monitor_artifacts")
    run_local_training(output_dir=output, n_resamples=10)
    return output


@pytest.fixture(scope="session")
def dataset():
    return temporal_split(load_features("sample"))


@pytest.fixture(scope="session")
def explainer(artifacts_dir):
    from src.inference.registry import load_bundle

    return load_bundle(variant="xgboost", artifacts_dir=artifacts_dir).explainer
