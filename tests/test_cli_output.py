from pathlib import Path

import main
from main import (
    DUMP_EVENT_PREFIX,
    IMPLEMENTER_DONE_TOKEN,
    STATUS_EVENT_PREFIX,
    STOP_AND_COMMIT_SENTENCE,
    AgentRuntime,
    AppSettings,
    ModelRole,
    OutputStreamState,
    Provider,
    RoleModelConfig,
    build_output_handler,
)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        project_root=tmp_path,
        planner=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
    )


def test_output_handler_hides_intermediate_logs_when_not_verbose(monkeypatch) -> None:
    printed: list[str] = []

    def fake_print(*args, **kwargs) -> None:
        if args:
            printed.append(str(args[0]))

    monkeypatch.setattr(main.console, "print", fake_print)

    handler = build_output_handler(verbose=False)
    handler("plain-stream-log")
    handler(f"{STATUS_EVENT_PREFIX}=== Planner ===")
    handler(f"{DUMP_EVENT_PREFIX}{{'messages': []}}")

    assert "plain-stream-log" not in "\n".join(printed)
    assert "messages" not in "\n".join(printed)
    assert any("Planner" in item for item in printed)


def test_output_handler_shows_dump_when_verbose(monkeypatch) -> None:
    printed: list[str] = []

    def fake_print(*args, **kwargs) -> None:
        if args:
            printed.append(str(args[0]))

    monkeypatch.setattr(main.console, "print", fake_print)

    handler = build_output_handler(verbose=True)
    handler(f"{DUMP_EVENT_PREFIX}{{'messages': []}}")

    assert any("messages" in item for item in printed)


def test_output_handler_shows_clarification_when_not_verbose(monkeypatch) -> None:
    printed: list[str] = []

    def fake_print(*args, **kwargs) -> None:
        if args:
            printed.append(str(args[0]))

    monkeypatch.setattr(main.console, "print", fake_print)

    state = OutputStreamState()
    handler = build_output_handler(verbose=False, state=state)
    handler("Perguntas de Clarificacao:\n1. Qual pasta devo usar?")

    assert state.clarification_requested is True
    assert printed


def test_output_handler_detects_additional_clarification_phrases(monkeypatch) -> None:
    printed: list[str] = []

    def fake_print(*args, **kwargs) -> None:
        if args:
            printed.append(str(args[0]))

    monkeypatch.setattr(main.console, "print", fake_print)

    state = OutputStreamState()
    handler = build_output_handler(verbose=False, state=state)
    handler("Clarification needed: please provide your preferences.")

    assert state.clarification_requested is True
    assert printed


def test_output_handler_tracks_pipeline_completion_status(monkeypatch) -> None:
    printed: list[str] = []

    def fake_print(*args, **kwargs) -> None:
        if args:
            printed.append(str(args[0]))

    monkeypatch.setattr(main.console, "print", fake_print)

    state = OutputStreamState()
    handler = build_output_handler(verbose=False, state=state)
    handler(f"{STATUS_EVENT_PREFIX}Starting planner phase")
    handler(f"{STATUS_EVENT_PREFIX}Implementer phase completed")

    assert state.pipeline_started is True
    assert state.pipeline_completed is True
    assert state.pipeline_blocked is False
    assert printed


def test_assistant_text_extracts_questions_from_tool_call() -> None:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "name": "vscode_askQuestions",
                "args": {
                    "questions": [
                        {
                            "header": "Workspace",
                            "question": "Qual diretorio devo usar?",
                            "options": [{"label": "src"}, {"label": "app"}],
                        }
                    ]
                },
            }
        ],
    }

    text = AgentRuntime._assistant_message_to_text(message)
    assert "Perguntas de Clarificacao" in text
    assert "Qual diretorio devo usar?" in text


def test_run_pipeline_emits_phase_status_events(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        plans_dir = tmp_path / "plans" / "feature-x"
        plans_dir.mkdir(parents=True, exist_ok=True)
        if role is ModelRole.PLANNER:
            (plans_dir / "plan.md").write_text("# Plan", encoding="utf-8")
            return ["**File:** `plans/feature-x/plan.md`\n# Plan"]
        if role is ModelRole.GENERATOR:
            (plans_dir / "implementation.md").write_text(
                (f"# Implementation\n## Step 1\nDo work\n\n{STOP_AND_COMMIT_SENTENCE}\n"),
                encoding="utf-8",
            )
            return ["**File:** `plans/feature-x/implementation.md`\n# Implementation"]
        if role is ModelRole.IMPLEMENTER:
            return [f"done {IMPLEMENTER_DONE_TOKEN}"]
        return ["ok"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    seen_events: list[str] = []

    def on_chunk(text: str) -> None:
        if text.startswith(STATUS_EVENT_PREFIX):
            seen_events.append(text)

    runtime.run_pipeline(
        prompt="implementar feature x",
        thread_id="thread-status",
        auto=True,
        request_continue=None,
        on_chunk=on_chunk,
    )

    merged = "\n".join(seen_events)
    assert "Planner" in merged
    assert "Generator" in merged
    assert "Implementer" in merged
