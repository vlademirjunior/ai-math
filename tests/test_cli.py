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
        def __init__(self, settings: AppSettings, *args, **kwargs) -> None:
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


def test_chat_engineering_prompt_routes_to_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_calls: list[str] = []
    chat_calls: list[str] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings, *args, **kwargs) -> None:
            self.settings = settings

        def run_pipeline(
            self, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ) -> list:
            pipeline_calls.append(prompt)
            return []

        def run_chat(self, prompt: str, thread_id: str, on_chunk) -> str:
            chat_calls.append(prompt)
            return ""

    monkeypatch.setattr(main, "get_settings", _litellm_settings)
    monkeypatch.setattr(main, "AgentRuntime", StubRuntime)

    result = runner.invoke(
        app,
        [
            "chat",
            "--prompt",
            (
                "Crie uma API simples com FastAPI, Pydantic e SQLAlchemy, com Dockerfile e "
                "docker-compose"
            ),
        ],
    )
    assert result.exit_code == 0
    assert len(pipeline_calls) == 1
    assert chat_calls == []


def test_role_command_routes_manual_role(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings, *args, **kwargs) -> None:
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


def test_role_command_handles_clarification_followup(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings, *args, **kwargs) -> None:
            self.settings = settings

        def run_manual_role(
            self, role, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ) -> list:
            prompts.append(prompt)
            if len(prompts) == 1:
                on_chunk("Perguntas de Clarificacao:\n1. Qual arquivo devo ler?")
            else:
                on_chunk("Plano criado.")
            return []

    answers = iter(["Ler .vscode/settings.json"])

    monkeypatch.setattr(main, "get_settings", _litellm_settings)
    monkeypatch.setattr(main, "AgentRuntime", StubRuntime)
    monkeypatch.setattr(main.console, "input", lambda _prompt: next(answers, ""))

    result = runner.invoke(app, ["role", "planner", "Criar plano inicial"])
    assert result.exit_code == 0
    assert prompts == ["Criar plano inicial", "Ler .vscode/settings.json"]


def test_chat_prompt_slash_routes_manual_role(monkeypatch: pytest.MonkeyPatch) -> None:
    manual_calls: list[tuple[str, str, bool]] = []
    pipeline_calls: list[str] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings, *args, **kwargs) -> None:
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
        def __init__(self, settings: AppSettings, *args, **kwargs) -> None:
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
        def __init__(self, settings: AppSettings, *args, **kwargs) -> None:
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


def test_chat_pipeline_clarification_followup_stays_in_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_calls: list[str] = []
    chat_calls: list[str] = []

    class _FakePromptSession:
        def __init__(self, messages: list[str]) -> None:
            self._messages = messages
            self._index = 0

        def prompt(self, _message: str) -> str:
            if self._index >= len(self._messages):
                return "/exit"
            value = self._messages[self._index]
            self._index += 1
            return value

    class StubRuntime:
        def __init__(self, settings: AppSettings, *args, **kwargs) -> None:
            self.settings = settings
            self._pipeline_runs = 0

        def run_pipeline(
            self, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ) -> list:
            del thread_id, auto, request_continue
            self._pipeline_runs += 1
            pipeline_calls.append(prompt)
            if self._pipeline_runs == 1:
                on_chunk("Perguntas de Clarificacao:\n1. Qual framework devo usar?")
                return []
            on_chunk(f"{main.STATUS_EVENT_PREFIX}Starting planner phase")
            on_chunk(f"{main.STATUS_EVENT_PREFIX}Implementer phase completed")
            return []

        def run_chat(self, prompt: str, thread_id: str, on_chunk) -> str:
            del thread_id, on_chunk
            chat_calls.append(prompt)
            return "chat"

    monkeypatch.setattr(main, "get_settings", _litellm_settings)
    monkeypatch.setattr(main, "AgentRuntime", StubRuntime)
    monkeypatch.setattr(
        main,
        "build_chat_session_prompt",
        lambda _root: _FakePromptSession(
            [
                "Crie uma API com autenticação",
                "1. FastAPI 2. SQLAlchemy",
                "/exit",
            ]
        ),
    )

    result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    assert len(pipeline_calls) == 2
    assert "answering clarification questions" in pipeline_calls[1].lower()
    assert "clarification answers:" in pipeline_calls[1].lower()
    assert "1. FastAPI 2. SQLAlchemy" in pipeline_calls[1]
    assert chat_calls == []
    assert result.stdout.count("Pipeline concluido.") == 1
    assert "Pipeline aguardando clarificacao para continuar." in result.stdout
