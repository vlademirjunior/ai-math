from pathlib import Path

import main
from main import (
    IMPLEMENTER_DONE_TOKEN,
    STATUS_EVENT_PREFIX,
    AgentRuntime,
    AppSettings,
    ModelRole,
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

    assert "plain-stream-log" not in "\n".join(printed)
    assert any("Planner" in item for item in printed)


def test_run_pipeline_emits_phase_status_events(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
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
