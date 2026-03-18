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

    plan_path = tmp_path / "plans" / "implement-feature" / "plan.md"
    impl_path = tmp_path / "plans" / "implement-feature" / "implementation.md"

    assert plan_path.exists()
    assert impl_path.exists()
    assert plan_path in generated_files
    assert impl_path in generated_files
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

    plan_path = tmp_path / "plans" / "implement-feature" / "plan.md"
    impl_path = tmp_path / "plans" / "implement-feature" / "implementation.md"

    assert plan_path.exists()
    assert impl_path.exists()
    assert plan_path in generated_files
    assert impl_path in generated_files


def test_pipeline_writes_plan_and_implementation_files(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        if role is ModelRole.PLANNER:
            return ["**File:** `plans/my-feature/plan.md`\n# My Feature\nThis is a plan."]
        if role is ModelRole.GENERATOR:
            return [
                "**File:** `plans/my-feature/implementation.md`\nThis is the implementation guide."
            ]
        return [f"done {IMPLEMENTER_DONE_TOKEN}"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    generated_files = runtime.run_pipeline(
        prompt="Implement feature",
        thread_id="thread-3",
        auto=True,
        request_continue=None,
        on_chunk=None,
    )

    plan_path = tmp_path / "plans" / "my-feature" / "plan.md"
    impl_path = tmp_path / "plans" / "my-feature" / "implementation.md"

    assert plan_path.exists(), "Expected plan.md to be written"
    assert impl_path.exists(), "Expected implementation.md to be written"
    assert plan_path.read_text(encoding="utf-8").startswith("**File:**"), (
        "Plan file should contain the planner output"
    )
    assert impl_path.read_text(encoding="utf-8").startswith("**File:**"), (
        "Implementation file should contain the generator output"
    )

    assert plan_path in generated_files
    assert impl_path in generated_files
