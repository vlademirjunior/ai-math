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


def test_implementer_hitl_waits_for_continue(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))
    feature_dir = tmp_path / "plans" / "thread-4-task"
    feature_dir.mkdir(parents=True, exist_ok=True)
    implementation_file = feature_dir / "implementation.md"
    implementation_file.write_text("step 1", encoding="utf-8")
    state_dir = tmp_path / "plans" / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "thread-4.txt").write_text("thread-4-task", encoding="utf-8")

    call_count = {"count": 0}

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return [f"pause {IMPLEMENTER_STOP_TOKEN}"]
        return [f"done {IMPLEMENTER_DONE_TOKEN}"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    decisions = iter([True])
    artifacts = runtime.run_manual_role(
        role=ModelRole.IMPLEMENTER,
        prompt="Implement now",
        thread_id="thread-4",
        auto=False,
        request_continue=lambda: next(decisions, False),
        on_chunk=None,
    )

    assert artifacts == [implementation_file]
    assert call_count["count"] == 2


def test_implementer_auto_does_not_wait(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))
    feature_dir = tmp_path / "plans" / "thread-5-task"
    feature_dir.mkdir(parents=True, exist_ok=True)
    implementation_file = feature_dir / "implementation.md"
    implementation_file.write_text("step 1", encoding="utf-8")
    state_dir = tmp_path / "plans" / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "thread-5.txt").write_text("thread-5-task", encoding="utf-8")

    call_count = {"count": 0}

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        call_count["count"] += 1
        return ["done"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    artifacts = runtime.run_manual_role(
        role=ModelRole.IMPLEMENTER,
        prompt="Implement now",
        thread_id="thread-5",
        auto=True,
        request_continue=None,
        on_chunk=None,
    )

    assert artifacts == [implementation_file]
    assert call_count["count"] == 1
