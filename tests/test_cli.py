import json

import pytest
from typer.testing import CliRunner

import main
from main import AppSettings, Provider, RoleModelConfig, app

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

    payload = json.loads(result.stdout)
    assert payload["skills_required"] is False
    assert isinstance(payload.get("skills"), list)
    assert any(skill.get("id") == "code-review" for skill in payload["skills"])


def test_skills_list_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["skills_required"] is False
    assert isinstance(payload.get("skills"), list)
    assert any(skill.get("id") == "code-review" for skill in payload["skills"])


def _litellm_settings() -> AppSettings:
    return AppSettings(
        planner=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
    )


def test_chat_prompt_auto_calls_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings) -> None:
            self.settings = settings

        def run_pipeline(
            self, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ) -> list:
            calls.append((prompt, auto))
            return []

        def run_chat(self, prompt: str, thread_id: str, on_chunk) -> str:
            return ""

    monkeypatch.setattr(main, "get_settings", _litellm_settings)
    monkeypatch.setattr(main, "AgentRuntime", StubRuntime)

    result = runner.invoke(app, ["chat", "--prompt", "Pipeline this", "--auto"])
    assert result.exit_code == 0
    assert calls == [("Pipeline this", True)]


def test_role_command_routes_manual_role(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings) -> None:
            self.settings = settings

        def run_manual_role(
            self, role, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ) -> list:
            calls.append((str(role), auto))
            return []

    monkeypatch.setattr(main, "get_settings", _litellm_settings)
    monkeypatch.setattr(main, "AgentRuntime", StubRuntime)

    result = runner.invoke(app, ["role", "planner", "Criar plano", "--auto"])
    assert result.exit_code == 0
    assert calls == [("planner", True)]


def test_chat_prompt_slash_routes_manual_role(monkeypatch: pytest.MonkeyPatch) -> None:
    manual_calls: list[tuple[str, str, bool]] = []
    pipeline_calls: list[str] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings) -> None:
            self.settings = settings

        def run_manual_role(
            self, role, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ) -> list:
            manual_calls.append((str(role), prompt, auto))
            return []

        def run_pipeline(
            self, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ) -> list:
            pipeline_calls.append(prompt)
            return []

        def run_chat(self, prompt: str, thread_id: str, on_chunk) -> str:
            return ""

    monkeypatch.setattr(main, "get_settings", _litellm_settings)
    monkeypatch.setattr(main, "AgentRuntime", StubRuntime)

    result = runner.invoke(app, ["chat", "--prompt", "/planner implementar feature x"])
    assert result.exit_code == 0
    assert manual_calls == [("planner", "implementar feature x", False)]
    assert pipeline_calls == []


def test_run_prompt_slash_routes_manual_role(monkeypatch: pytest.MonkeyPatch) -> None:
    manual_calls: list[tuple[str, str, bool]] = []
    pipeline_calls: list[str] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings) -> None:
            self.settings = settings

        def run_manual_role(
            self, role, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ) -> list:
            manual_calls.append((str(role), prompt, auto))
            return []

        def run_pipeline(
            self, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ) -> list:
            pipeline_calls.append(prompt)
            return []

        def run_chat(self, prompt: str, thread_id: str, on_chunk) -> str:
            return ""

    monkeypatch.setattr(main, "get_settings", _litellm_settings)
    monkeypatch.setattr(main, "AgentRuntime", StubRuntime)

    result = runner.invoke(app, ["run", "/implementer executar plano atual", "--auto"])
    assert result.exit_code == 0
    assert manual_calls == [("implementer", "executar plano atual", True)]
    assert pipeline_calls == []


def test_chat_oi_routes_to_natural_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_calls: list[str] = []
    chat_calls: list[str] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings) -> None:
            self.settings = settings

        def run_pipeline(
            self, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ) -> list:
            pipeline_calls.append(prompt)
            return []

        def run_chat(self, prompt: str, thread_id: str, on_chunk) -> str:
            chat_calls.append(prompt)
            return "oi"

    monkeypatch.setattr(main, "get_settings", _litellm_settings)
    monkeypatch.setattr(main, "AgentRuntime", StubRuntime)

    result = runner.invoke(app, ["chat", "--prompt", "oi"])
    assert result.exit_code == 0
    assert pipeline_calls == []
    assert chat_calls == ["oi"]
