from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from prompt_toolkit.document import Document
from typer.testing import CliRunner

import main
from main import (
    AppSettings,
    ChatCompleter,
    Provider,
    RoleModelConfig,
    _skills_for_metadata,
    app,
    build_contextual_prompt,
    extract_context_references,
    infer_context_window,
)

runner = CliRunner()


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


def _litellm_settings(project_root: Path) -> AppSettings:
    return AppSettings(
        project_root=project_root,
        planner=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
    )


def test_extract_context_references_splits_message() -> None:
    cleaned, refs = extract_context_references("ajusta isso #main.py #tests")
    assert cleaned == "ajusta isso"
    assert refs == ["main.py", "tests"]


def test_build_contextual_prompt_reads_file(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("hello world", encoding="utf-8")

    result = build_contextual_prompt("usar #sample.txt", tmp_path)

    assert "Contexto adicional" in result.prompt
    assert "hello world" in result.prompt
    assert result.sources == ["sample.txt"]
    assert result.warnings == []


def test_build_contextual_prompt_does_not_load_skills_unless_referenced(tmp_path: Path) -> None:
    # Ensure skill files are not loaded just because some context is requested.
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / ".agents" / "code-review" / "SKILL.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".agents" / "code-review" / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    result = build_contextual_prompt("fale sobre #README.md", tmp_path)

    assert ".agents" not in ";".join(result.sources)
    assert result.sources == ["README.md"]


def test_build_contextual_prompt_loads_referenced_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".agents" / "refactor"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    result = build_contextual_prompt("use #.agents/refactor/SKILL.md", tmp_path)

    assert ".agents/refactor/SKILL.md" in result.sources


def test_build_contextual_prompt_autoloads_skill_by_intent(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".agents" / "refactor"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    result = build_contextual_prompt(
        "please refactor this code",
        tmp_path,
        auto_load_skills=True,
    )

    assert ".agents/refactor/SKILL.md" in result.sources


def test_infer_context_window_uses_configured_max_tokens(tmp_path: Path) -> None:
    settings = AppSettings(
        project_root=tmp_path,
        planner=RoleModelConfig(
            provider=Provider.LITELLM,
            model="openai/gpt-4o-mini",
            max_tokens=8192,
        ),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
    )
    assert infer_context_window(settings) == 8192


def test_chat_reset_starts_new_session(monkeypatch, tmp_path: Path) -> None:
    printed: list[str] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings, *args: Any, **kwargs: Any) -> None:
            self.settings = settings
            self.skills: list[str] = []

        def run_chat(self, prompt: str, thread_id: str, on_chunk) -> str:
            return "ok"

        def run_pipeline(
            self, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ):
            return []

        def run_manual_role(
            self,
            role,
            prompt: str,
            thread_id: str,
            *,
            auto: bool,
            request_continue,
            on_chunk,
        ):
            return []

    def fake_print(*args, **kwargs) -> None:
        del kwargs
        if args:
            printed.append(str(args[0]))

    monkeypatch.setattr(main, "get_settings", lambda: _litellm_settings(tmp_path))
    monkeypatch.setattr(main, "AgentRuntime", StubRuntime)
    monkeypatch.setattr(
        main, "build_chat_session_prompt", lambda _root: _FakePromptSession(["/reset", "/exit"])
    )
    monkeypatch.setattr(main.console, "print", fake_print)

    result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    assert any("Nova sessao iniciada" in line for line in printed)


def test_chat_clear_calls_console_clear(monkeypatch, tmp_path: Path) -> None:
    clear_calls: list[bool] = []

    class StubRuntime:
        def __init__(self, settings: AppSettings, *args: Any, **kwargs: Any) -> None:
            self.settings = settings
            self.skills: list[str] = []

        def run_chat(self, prompt: str, thread_id: str, on_chunk) -> str:
            return "ok"

        def run_pipeline(
            self, prompt: str, thread_id: str, *, auto: bool, request_continue, on_chunk
        ):
            return []

        def run_manual_role(
            self,
            role,
            prompt: str,
            thread_id: str,
            *,
            auto: bool,
            request_continue,
            on_chunk,
        ):
            return []

    monkeypatch.setattr(main, "get_settings", lambda: _litellm_settings(tmp_path))
    monkeypatch.setattr(main, "AgentRuntime", StubRuntime)
    monkeypatch.setattr(
        main, "build_chat_session_prompt", lambda _root: _FakePromptSession(["/clear", "/exit"])
    )
    monkeypatch.setattr(main.console, "clear", lambda: clear_calls.append(True))

    result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    assert clear_calls == [True]


def test_chat_completer_suggests_role_commands(tmp_path: Path) -> None:
    completer = ChatCompleter(tmp_path)
    doc = Document(text="/pl", cursor_position=3)

    completions = list(completer.get_completions(doc, cast(Any, object())))
    values = {item.text for item in completions}
    assert "/planner" in values


def test_chat_completer_suggests_context_paths(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('ok')", encoding="utf-8")

    completer = ChatCompleter(tmp_path)
    doc = Document(text="#src/", cursor_position=5)

    completions = list(completer.get_completions(doc, cast(Any, object())))
    values = {item.text for item in completions}
    assert "#src/main.py" in values


def test_skills_for_metadata_pipeline_includes_builtin_and_external(tmp_path: Path) -> None:
    class StubRuntime:
        skills = ["/.agents/"]

    values = _skills_for_metadata(cast(Any, StubRuntime()), "pipeline")
    assert "builtin:planner_skill.md" in values
    assert "builtin:generator_skill.md" in values
    assert "builtin:implementer_skill.md" in values
    assert "/.agents/" in values
