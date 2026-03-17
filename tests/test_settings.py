import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from main import (
    AppSettings,
    Provider,
    load_dotenv_into_environ,
    parse_dotenv_line,
    resolve_env_file,
)


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
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PLANNER__PROVIDER", "openrouter")
    monkeypatch.setenv("GENERATOR__PROVIDER", "ollama")
    monkeypatch.setenv("IMPLEMENTER__PROVIDER", "ollama")

    with pytest.raises(ValidationError):
        AppSettings(project_root=Path.cwd())


def test_model_only_env_defaults_provider_to_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("PLANNER__MODEL", "stepfun/step-3.5-flash:free")
    monkeypatch.setenv("GENERATOR__MODEL", "stepfun/step-3.5-flash:free")
    monkeypatch.setenv("IMPLEMENTER__MODEL", "stepfun/step-3.5-flash:free")

    settings = AppSettings(project_root=Path.cwd())

    assert settings.planner.provider == Provider.OPENROUTER
    assert settings.generator.provider == Provider.OPENROUTER
    assert settings.implementer.provider == Provider.OPENROUTER


def test_openai_api_key_works_as_openrouter_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PLANNER__PROVIDER", "openrouter")
    monkeypatch.setenv("PLANNER__MODEL", "stepfun/step-3.5-flash:free")
    monkeypatch.setenv("GENERATOR__PROVIDER", "openrouter")
    monkeypatch.setenv("GENERATOR__MODEL", "stepfun/step-3.5-flash:free")
    monkeypatch.setenv("IMPLEMENTER__PROVIDER", "openrouter")
    monkeypatch.setenv("IMPLEMENTER__MODEL", "stepfun/step-3.5-flash:free")

    settings = AppSettings(project_root=Path.cwd())

    assert settings.openrouter_effective_api_key == "test-key"


def test_resolve_env_file_prefers_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")

    resolve_env_file.cache_clear()
    assert resolve_env_file() == env_file.resolve()


def test_resolve_env_file_uses_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cwd = tmp_path / "runner"
    project = tmp_path / "project"
    cwd.mkdir()
    project.mkdir()

    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PROJECT_ROOT", str(project))

    env_file = project / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")

    resolve_env_file.cache_clear()
    assert resolve_env_file() == env_file.resolve()


def test_parse_dotenv_line_supports_export_syntax() -> None:
    assert parse_dotenv_line("export OPENROUTER_API_KEY=test-key") == (
        "OPENROUTER_API_KEY",
        "test-key",
    )


def test_load_dotenv_into_environ_does_not_override_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY=from-file\nOPENAI_API_KEY=from-file\n", encoding="utf-8"
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_dotenv_into_environ(env_file)

    assert loaded == 1
    assert os.getenv("OPENROUTER_API_KEY") == "from-shell"
    assert os.getenv("OPENAI_API_KEY") == "from-file"


def test_load_dotenv_overrides_empty_env_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from-file\n", encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    loaded = load_dotenv_into_environ(env_file)

    assert loaded == 1
    assert os.getenv("OPENROUTER_API_KEY") == "from-file"
