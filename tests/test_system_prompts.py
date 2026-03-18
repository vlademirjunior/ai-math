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


def test_system_prompt_includes_builtin_skill_text(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))
    monkeypatch.setattr(runtime, "_load_builtin_skill_text", lambda role: f"skill-{role.value}")

    planner_prompt = runtime._system_prompt_for_role(ModelRole.PLANNER, auto=False)
    generator_prompt = runtime._system_prompt_for_role(ModelRole.GENERATOR, auto=False)
    implementer_prompt = runtime._system_prompt_for_role(ModelRole.IMPLEMENTER, auto=False)

    assert "skill-planner" in planner_prompt
    assert "skill-generator" in generator_prompt
    assert "skill-implementer" in implementer_prompt


def test_implementer_prompt_manual_mode_contains_checkpoint_tokens(tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    prompt = runtime._system_prompt_for_role(ModelRole.IMPLEMENTER, auto=False)

    assert "MANUAL CHECKPOINT MODE" in prompt
    assert IMPLEMENTER_STOP_TOKEN in prompt
    assert IMPLEMENTER_DONE_TOKEN in prompt


def test_implementer_prompt_auto_mode_contains_done_token_only(tmp_path: Path) -> None:
    runtime = AgentRuntime(_settings(tmp_path))

    prompt = runtime._system_prompt_for_role(ModelRole.IMPLEMENTER, auto=True)

    assert "AUTO MODE" in prompt
    assert IMPLEMENTER_DONE_TOKEN in prompt
    assert IMPLEMENTER_STOP_TOKEN not in prompt
