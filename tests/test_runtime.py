from unittest.mock import MagicMock

import pytest

from main import AgentRuntime, AppSettings, Provider, RoleModelConfig


def test_runtime_initializes() -> None:
    settings = AppSettings(
        planner=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
    )
    runtime = AgentRuntime(settings)
    assert runtime.settings.thread_id_default


def test_stream_uses_thread_id(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings(
        planner=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
    )
    runtime = AgentRuntime(settings)

    fake_agent = MagicMock()
    fake_agent.stream.return_value = ["ok"]
    monkeypatch.setattr(runtime, "create_chat_agent", lambda: fake_agent)

    chunks = list(runtime.stream("hello", "abc-thread"))
    assert chunks == ["ok"]
    fake_agent.stream.assert_called_once()
