from pathlib import Path

from main import (
    IMPLEMENTER_DONE_TOKEN,
    IMPLEMENTER_STOP_TOKEN,
    STOP_AND_COMMIT_SENTENCE,
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
            (plans_dir / "implementation.md").write_text(
                (f"# Implementation\n## Step 1\nDo work\n\n{STOP_AND_COMMIT_SENTENCE}\n"),
                encoding="utf-8",
            )
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
            (plans_dir / "implementation.md").write_text(
                (f"# Implementation\n## Step 1\nDo work\n\n{STOP_AND_COMMIT_SENTENCE}\n"),
                encoding="utf-8",
            )
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


def test_pipeline_waits_for_clarification_before_completing_planner(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = AgentRuntime(_settings(tmp_path))
    calls: list[ModelRole] = []
    continue_calls: list[str] = []
    status_events: list[str] = []

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        del prompt, thread_id, auto
        calls.append(role)
        if role is ModelRole.PLANNER:
            plans_dir = tmp_path / "plans" / "clarification"
            plans_dir.mkdir(parents=True, exist_ok=True)
            (plans_dir / "plan.md").write_text("# Draft plan", encoding="utf-8")
            return ["Perguntas de Clarificacao:\n1. Qual estrategia de cache devo usar?"]
        return ["should-not-run"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    def request_continue(phase: str) -> bool:
        continue_calls.append(phase)
        return True

    runtime.run_pipeline(
        prompt="Criar API",
        thread_id="thread-clarification",
        auto=False,
        request_continue=request_continue,
        on_chunk=lambda text: status_events.append(text),
    )

    joined = "\n".join(status_events)
    assert calls == [ModelRole.PLANNER]
    assert continue_calls == []
    assert "Planner awaiting clarification. Phase not completed yet." in joined
    assert "Planner phase completed" not in joined
    assert "Pipeline paused after planner" not in joined


def test_pipeline_stops_on_planner_scope_violation(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))
    calls: list[ModelRole] = []
    status_events: list[str] = []

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        del prompt, thread_id, auto
        calls.append(role)
        if role is ModelRole.PLANNER:
            (tmp_path / "random-root-file.md").write_text("oops", encoding="utf-8")
            plans_dir = tmp_path / "plans" / "scoped"
            plans_dir.mkdir(parents=True, exist_ok=True)
            (plans_dir / "plan.md").write_text("# Plan", encoding="utf-8")
            return ["**File:** `plans/scoped/plan.md`\n# Plan"]
        return ["should-not-run"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    generated = runtime.run_pipeline(
        prompt="Implement feature",
        thread_id="thread-scope-violation",
        auto=True,
        request_continue=None,
        on_chunk=lambda text: status_events.append(text),
    )

    joined = "\n".join(status_events)
    assert generated == []
    assert calls == [ModelRole.PLANNER]
    assert "Planner changed files outside plans/{feature-name}/plan.md" in joined
    assert "Planner scope violation: random-root-file.md" in joined


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
            (plans_dir / "implementation.md").write_text(
                (f"# Implementation\n## Step 1\nDo work\n\n{STOP_AND_COMMIT_SENTENCE}\n"),
                encoding="utf-8",
            )
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


def test_pipeline_retries_generator_when_missing_stop_and_commit(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = AgentRuntime(_settings(tmp_path))
    calls = {"generator": 0, "implementer": 0}

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        plans_dir = tmp_path / "plans" / "feature-stop"
        plans_dir.mkdir(parents=True, exist_ok=True)
        if role is ModelRole.PLANNER:
            (plans_dir / "plan.md").write_text("# Plan", encoding="utf-8")
            return ["**File:** `plans/feature-stop/plan.md`\n# Plan"]
        if role is ModelRole.GENERATOR:
            calls["generator"] += 1
            if calls["generator"] == 1:
                (plans_dir / "implementation.md").write_text(
                    "# Guide\n## Step 1\nDo work", encoding="utf-8"
                )
                return ["**File:** `plans/feature-stop/implementation.md`\n# Guide"]
            (plans_dir / "implementation.md").write_text(
                (f"# Guide\n## Step 1\nDo work\n\n{STOP_AND_COMMIT_SENTENCE}\n"),
                encoding="utf-8",
            )
            return ["**File:** `plans/feature-stop/implementation.md`\n# Guide repaired"]
        calls["implementer"] += 1
        return [f"done {IMPLEMENTER_DONE_TOKEN}"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    generated_files = runtime.run_pipeline(
        prompt="Implement feature",
        thread_id="thread-stop-repair",
        auto=True,
        request_continue=None,
        on_chunk=None,
    )

    assert calls["generator"] == 2
    assert calls["implementer"] == 1
    assert any(path.name == "implementation.md" for path in generated_files)


def test_pipeline_stops_if_generator_never_adds_stop_and_commit(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = AgentRuntime(_settings(tmp_path))
    calls: list[ModelRole] = []

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        plans_dir = tmp_path / "plans" / "feature-stop-fail"
        plans_dir.mkdir(parents=True, exist_ok=True)
        calls.append(role)
        if role is ModelRole.PLANNER:
            (plans_dir / "plan.md").write_text("# Plan", encoding="utf-8")
            return ["**File:** `plans/feature-stop-fail/plan.md`\n# Plan"]
        if role is ModelRole.GENERATOR:
            (plans_dir / "implementation.md").write_text(
                "# Guide\n## Step 1\nDo work", encoding="utf-8"
            )
            return ["**File:** `plans/feature-stop-fail/implementation.md`\n# Guide"]
        return [f"done {IMPLEMENTER_DONE_TOKEN}"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    runtime.run_pipeline(
        prompt="Implement feature",
        thread_id="thread-stop-fail",
        auto=True,
        request_continue=None,
        on_chunk=None,
    )

    assert calls == [ModelRole.PLANNER, ModelRole.GENERATOR, ModelRole.GENERATOR]
