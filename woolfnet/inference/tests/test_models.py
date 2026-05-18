"""Unit tests for ``WoolfModel``."""

from pathlib import Path

import mlflow
import pytest
import yaml

from woolfnet.inference import WoolfModel
from woolfnet.paths import MLFLOW_LOCAL_URI


@pytest.fixture(autouse=True)
def _mlflow_tracking_uri():
    """Point MLflow at the local file store for the duration of each test."""
    mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)


@pytest.fixture
def custom_config(tmp_path, monkeypatch):
    """Build a one-off inference.yml at tmp_path and point WoolfModel at it."""

    def _make(models: dict) -> Path:
        path = tmp_path / "inference.yml"
        path.write_text(yaml.safe_dump({"models": models}))
        monkeypatch.setattr(WoolfModel, "CONFIG_PATH", path)
        return path

    return _make


def test_load_from_mlflow_and_generate():
    """Resolving ``gpt2-woolf`` by versioned name should produce a real model that generates text."""
    model = WoolfModel("gpt2-woolf")
    assert model.source == "huggingface"
    out = model.generate("Mrs Dalloway said", max_new_tokens=10, temperature=0.8)
    assert isinstance(out, str)
    assert out.startswith("Mrs Dalloway said")
    assert len(out) > len("Mrs Dalloway said")


def test_available_includes_mlflow_and_hub_models():
    """``available()`` should include entries reachable via the Hub or MLflow."""
    available = WoolfModel.available()
    assert "gpt2" in available
    assert "gpt2-woolf" in available
