from pathlib import Path

import pytest
from pydantic import ValidationError

from main import AppSettings, Provider


@pytest.mark.parametrize(
    ("env", "expected_provider"),
    [
        (
            {
                "OPENROUTER_API_KEY": "test-key",
                "PLANNER__PROVIDER": "openrouter",
                "PLANNER__MODEL": "stepfun/step-3.5-flash:free",
            },
            Provider.OPENROUTER,
        ),
        (
            {
                "OPENROUTER_API_KEY": "test-key",
                "PLANNER__PROVIDER": "ollama",
                "PLANNER__MODEL": "stepfun/step-3.5-flash:free",
            },
            Provider.OLLAMA,
        ),
    ],
)
def test_settings_env_parse(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected_provider: Provider
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PLANNER__PROVIDER", raising=False)
    monkeypatch.delenv("PLANNER__MODEL", raising=False)

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    settings = AppSettings(project_root=Path.cwd())
    assert settings.planner.provider == expected_provider


def test_openrouter_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("PLANNER__PROVIDER", "openrouter")
    monkeypatch.setenv("GENERATOR__PROVIDER", "ollama")
    monkeypatch.setenv("IMPLEMENTER__PROVIDER", "ollama")

    with pytest.raises(ValidationError):
        AppSettings(project_root=Path.cwd())
