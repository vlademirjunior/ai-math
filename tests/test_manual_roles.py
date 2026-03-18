from pathlib import Path

from main import (
    AgentRuntime,
    AppSettings,
    ModelRole,
    Provider,
    RoleModelConfig,
    parse_manual_role_command,
    should_trigger_pipeline,
)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        project_root=tmp_path,
        planner=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
    )


def test_parse_manual_role_command() -> None:
    parsed = parse_manual_role_command("/planner implementar feature x")
    assert parsed == (ModelRole.PLANNER, "implementar feature x")
    assert parse_manual_role_command("/unknown algo") is None
    assert parse_manual_role_command("planner sem barra") is None


def test_manual_role_runs_single_shot_for_non_implementer(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    def fake_stream_role(self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False):
        plans_dir = tmp_path / "plans" / "generate-docs"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / "plan.md").write_text("# Plan", encoding="utf-8")
        return ["**File:** `plans/generate-docs/plan.md`\n# Plan"]

    monkeypatch.setattr(AgentRuntime, "stream_role", fake_stream_role)

    artifacts = runtime.run_manual_role(
        role=ModelRole.PLANNER,
        prompt="Generate docs",
        thread_id="thread-3",
        auto=False,
        request_continue=None,
        on_chunk=None,
    )

    plan_path = tmp_path / "plans" / "generate-docs" / "plan.md"
    assert plan_path.exists()
    assert plan_path in artifacts


def test_should_trigger_pipeline_for_greeting_is_false() -> None:
    assert should_trigger_pipeline("oi") is False


def test_should_trigger_pipeline_for_engineering_request_is_true() -> None:
    assert should_trigger_pipeline("implementar feature de auth") is True


def test_should_trigger_pipeline_for_long_api_build_prompt_is_true() -> None:
    prompt = (
        "Voce e um Engenheiro de Software Senior. Crie uma API simples com FastAPI, "
        "Pydantic, SQLAlchemy, Dockerfile e docker-compose."
    )
    assert should_trigger_pipeline(prompt) is True
