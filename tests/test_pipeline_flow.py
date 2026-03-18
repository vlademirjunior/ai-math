from pathlib import Path

from main import (
    IMPLEMENTER_DONE_TOKEN,
    IMPLEMENTER_STOP_TOKEN,
    AgentRuntime,
    AppSettings,
    ModelRole,
    Provider,
    RoleModelConfig,
)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        project_root=tmp_path,
        planner=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
    )


def test_pipeline_runs_all_phases(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    implementer_calls = {"count": 0}

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        if role is ModelRole.PLANNER:
            return ["planner-output"]
        if role is ModelRole.GENERATOR:
            return ["generator-output"]
        implementer_calls["count"] += 1
        if implementer_calls["count"] == 1:
            return [f"checkpoint {IMPLEMENTER_STOP_TOKEN}"]
        return [f"done {IMPLEMENTER_DONE_TOKEN}"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    decisions = iter([True])
    generated_files = runtime.run_pipeline(
        prompt="Implement feature",
        thread_id="thread-1",
        auto=False,
        request_continue=lambda: next(decisions, False),
        on_chunk=None,
    )

    assert generated_files == []
    assert implementer_calls["count"] == 2


def test_pipeline_auto_mode_runs_implementer_once(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    calls: list[ModelRole] = []

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        calls.append(role)
        return ["ok"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    generated_files = runtime.run_pipeline(
        prompt="Implement feature",
        thread_id="thread-2",
        auto=True,
        request_continue=None,
        on_chunk=None,
    )

    assert calls.count(ModelRole.IMPLEMENTER) == 1
    assert generated_files == []
