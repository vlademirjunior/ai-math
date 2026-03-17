import pytest
from typer.testing import CliRunner

from main import app

runner = CliRunner()


def test_models_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "planner" in result.stdout


def test_skills_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    result = runner.invoke(app, ["skills"])
    assert result.exit_code == 0


def test_skills_list_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0
