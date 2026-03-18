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
        plans_dir = tmp_path / "plans" / "implement-feature"
        plans_dir.mkdir(parents=True, exist_ok=True)
        if role is ModelRole.PLANNER:
            (plans_dir / "plan.md").write_text("# Plan", encoding="utf-8")
            return ["**File:** `plans/implement-feature/plan.md`\n# Plan"]
        if role is ModelRole.GENERATOR:
            (plans_dir / "implementation.md").write_text("# Implementation", encoding="utf-8")
            return ["**File:** `plans/implement-feature/implementation.md`\n# Implementation"]
        implementer_calls["count"] += 1
        if implementer_calls["count"] == 1:
            return [f"checkpoint {IMPLEMENTER_STOP_TOKEN}"]
        return [f"done {IMPLEMENTER_DONE_TOKEN}"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    decisions = iter([True, True, True])
    generated_files = runtime.run_pipeline(
        prompt="Implement feature",
        thread_id="thread-1",
        auto=False,
        request_continue=lambda _phase: next(decisions, False),
        on_chunk=None,
    )

    assert len(generated_files) == 2
    assert implementer_calls["count"] == 2


def test_pipeline_auto_mode_runs_implementer_once(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    calls: list[ModelRole] = []

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        plans_dir = tmp_path / "plans" / "implement-feature"
        plans_dir.mkdir(parents=True, exist_ok=True)
        if role is ModelRole.PLANNER:
            (plans_dir / "plan.md").write_text("# Plan", encoding="utf-8")
            return ["**File:** `plans/implement-feature/plan.md`\n# Plan"]
        if role is ModelRole.GENERATOR:
            (plans_dir / "implementation.md").write_text("# Implementation", encoding="utf-8")
            return ["**File:** `plans/implement-feature/implementation.md`\n# Implementation"]
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

    assert len(generated_files) == 2


def test_pipeline_stops_if_planner_does_not_create_plan(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    calls: list[ModelRole] = []

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        calls.append(role)
        if role is ModelRole.PLANNER:
            return ["planner-output-without-file"]
        return ["should-not-run"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    generated_files = runtime.run_pipeline(
        prompt="Implement feature",
        thread_id="thread-missing-plan",
        auto=False,
        request_continue=lambda _phase: True,
        on_chunk=None,
    )

    assert generated_files == []
    assert calls == [ModelRole.PLANNER]


def test_pipeline_stops_if_generator_does_not_create_implementation(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    calls: list[ModelRole] = []

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        plans_dir = tmp_path / "plans" / "implement-feature"
        plans_dir.mkdir(parents=True, exist_ok=True)
        calls.append(role)
        if role is ModelRole.PLANNER:
            (plans_dir / "plan.md").write_text("# Plan", encoding="utf-8")
            return ["**File:** `plans/implement-feature/plan.md`\n# Plan"]
        if role is ModelRole.GENERATOR:
            return ["generator-output-without-file"]
        return ["should-not-run"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    generated_files = runtime.run_pipeline(
        prompt="Implement feature",
        thread_id="thread-missing-impl",
        auto=False,
        request_continue=lambda _phase: True,
        on_chunk=None,
    )

    assert len(generated_files) == 1
    assert generated_files[0].name == "plan.md"
    assert calls == [ModelRole.PLANNER, ModelRole.GENERATOR]


def test_pipeline_writes_plan_and_implementation_files(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        plans_dir = tmp_path / "plans" / "my-feature"
        plans_dir.mkdir(parents=True, exist_ok=True)
        if role is ModelRole.PLANNER:
            (plans_dir / "plan.md").write_text("# My Feature\nThis is a plan.", encoding="utf-8")
            return ["**File:** `plans/my-feature/plan.md`\n# My Feature\nThis is a plan."]
        if role is ModelRole.GENERATOR:
            (plans_dir / "implementation.md").write_text(
                "This is the implementation guide.", encoding="utf-8"
            )
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
    assert plan_path.read_text(encoding="utf-8").startswith("# My Feature")
    assert impl_path.read_text(encoding="utf-8").startswith("This is the implementation")

    assert plan_path in generated_files
    assert impl_path in generated_files


def test_pipeline_uses_filesystem_artifacts_even_without_output_path(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = AgentRuntime(_settings(tmp_path))
    calls: list[ModelRole] = []

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        plans_dir = tmp_path / "plans" / "feature-no-path"
        plans_dir.mkdir(parents=True, exist_ok=True)
        calls.append(role)
        if role is ModelRole.PLANNER:
            (plans_dir / "plan.md").write_text("# Plan", encoding="utf-8")
            return ["planner done"]
        if role is ModelRole.GENERATOR:
            (plans_dir / "implementation.md").write_text("# Implementation", encoding="utf-8")
            return ["generator done"]
        return [f"done {IMPLEMENTER_DONE_TOKEN}"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    generated_files = runtime.run_pipeline(
        prompt="Implement feature",
        thread_id="thread-no-path",
        auto=True,
        request_continue=None,
        on_chunk=None,
    )

    assert calls == [ModelRole.PLANNER, ModelRole.GENERATOR, ModelRole.IMPLEMENTER]
    names = {path.name for path in generated_files}
    assert "plan.md" in names
    assert "implementation.md" in names
