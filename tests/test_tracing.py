import os

import pytest

from main import tracing_enabled_context


def test_tracing_disabled_sets_env_false() -> None:
    os.environ.pop("LANGSMITH_TRACING", None)
    with tracing_enabled_context(enabled=False, project=None):
        assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_enabled_sets_env_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    with tracing_enabled_context(enabled=True, project="demo-project"):
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_PROJECT"] == "demo-project"
