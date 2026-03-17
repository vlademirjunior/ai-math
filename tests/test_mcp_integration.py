from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import main
from main import (
    MAX_MCP_CONTEXT_CHARS,
    AppSettings,
    MCPManager,
    MCPResourceBlob,
    MCPServerConfig,
    Provider,
    RoleModelConfig,
    _run_async,
    build_contextual_prompt,
    render_mcp_context_update,
    render_mcp_status,
)


def _settings(
    project_root: Path, mcp_servers: dict[str, MCPServerConfig] | None = None
) -> AppSettings:
    return AppSettings(
        project_root=project_root,
        planner=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        mcp_servers=mcp_servers or {},
    )


def test_mcp_config_loaded_from_vscode_file(tmp_path: Path) -> None:
    (tmp_path / ".vscode").mkdir()
    (tmp_path / ".vscode" / "mcp.json").write_text(
        '{"servers": {"docs": {"type": "http", "url": "https://docs.example/mcp"}}}',
        encoding="utf-8",
    )

    settings = _settings(tmp_path)

    assert "docs" in settings.mcp_servers
    assert settings.mcp_servers["docs"].transport == "streamable_http"
    assert settings.mcp_servers["docs"].url == "https://docs.example/mcp"


def test_mcp_config_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MCP_SERVERS__DOCS__TRANSPORT", "streamable_http")
    monkeypatch.setenv("MCP_SERVERS__DOCS__URL", "https://docs.example/mcp")

    settings = _settings(tmp_path)

    assert "docs" in settings.mcp_servers
    assert settings.mcp_servers["docs"].transport == "streamable_http"


def test_invalid_mcp_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _settings(
            tmp_path,
            mcp_servers={
                "broken": MCPServerConfig(transport="stdio"),
            },
        )


class _FakeMCPClient:
    def __init__(self, payload: dict[str, dict[str, object]]) -> None:
        self.payload = payload

    async def get_tools(self, server_name: str | None = None):
        if server_name == "offline":
            raise ConnectionError("unreachable")
        return [{"name": f"tool-{server_name or 'global'}"}]

    async def get_resources(self, server_name: str | None = None):
        if server_name == "offline":
            raise TimeoutError("timeout")
        return [
            {
                "uri": f"mcp://{server_name}/resource-1",
                "content": f"payload-{server_name}",
            }
        ]


def test_mcp_manager_mixed_server_status(tmp_path: Path) -> None:
    servers = {
        "online": MCPServerConfig(transport="streamable_http", url="https://ok.example"),
        "offline": MCPServerConfig(transport="streamable_http", url="https://down.example"),
    }
    manager = MCPManager(_settings(tmp_path, mcp_servers=servers), client_factory=_FakeMCPClient)

    resources = _run_async(manager.get_resources())
    tools = _run_async(manager.get_all_tools())
    status = manager.get_server_status()

    assert any(item.server_name == "online" for item in resources)
    assert tools["online"]
    assert tools["offline"] == []
    assert status["online"].online is True
    assert status["offline"].online is False


def test_context_includes_mcp_resources(tmp_path: Path) -> None:
    servers = {
        "docs": MCPServerConfig(transport="streamable_http", url="https://docs.example"),
    }
    manager = MCPManager(_settings(tmp_path, mcp_servers=servers), client_factory=_FakeMCPClient)

    result = build_contextual_prompt("gerar plano", tmp_path, mcp_manager=manager)

    assert "[MCP: docs]" in result.prompt
    assert result.mcp_sources_used == ["docs"]
    assert result.mcp_resources_available == 1


def test_context_mcp_limit_is_separate(tmp_path: Path) -> None:
    class _LargeMCPManager:
        def has_servers(self) -> bool:
            return True

        async def get_all_tools(self):
            return {"docs": [{"name": "search"}]}

        async def get_resources(self):
            return [
                MCPResourceBlob(
                    server_name="docs",
                    resource_uri="mcp://docs/huge",
                    content="x" * (MAX_MCP_CONTEXT_CHARS + 500),
                )
            ]

        def get_server_status(self):
            return {"docs": main.MCPServerStatus(online=True, tools_count=1, resources_count=1)}

    sample = tmp_path / "file.txt"
    sample.write_text("a" * 2000, encoding="utf-8")

    result = build_contextual_prompt("usar #file.txt", tmp_path, mcp_manager=_LargeMCPManager())

    assert result.mcp_chars_used <= MAX_MCP_CONTEXT_CHARS
    assert "### file.txt" in result.prompt
    assert "[MCP: docs]" in result.prompt


def test_context_priority_order(tmp_path: Path) -> None:
    (tmp_path / ".agents" / "planner").mkdir(parents=True)
    (tmp_path / ".agents" / "planner" / "SKILL.md").write_text("skill docs", encoding="utf-8")
    (tmp_path / "src.txt").write_text("local source", encoding="utf-8")

    class _PriorityMCP:
        def has_servers(self) -> bool:
            return True

        async def get_all_tools(self):
            return {"docs": []}

        async def get_resources(self):
            return [
                MCPResourceBlob(
                    server_name="docs",
                    resource_uri="mcp://docs/r1",
                    content="mcp resource",
                )
            ]

        def get_server_status(self):
            return {"docs": main.MCPServerStatus(online=True)}

    result = build_contextual_prompt("analisar #src.txt", tmp_path, mcp_manager=_PriorityMCP())

    src_index = result.prompt.index("### src.txt")
    agents_index = result.prompt.index("### .agents/planner/SKILL.md")
    mcp_index = result.prompt.index("[MCP: docs]")
    assert src_index < agents_index < mcp_index


def test_render_mcp_status_offline_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    printed: list[str] = []
    monkeypatch.setattr(main.console, "print", lambda *args, **kwargs: printed.append(str(args[0])))

    servers = {
        "offline": MCPServerConfig(transport="streamable_http", url="https://down.example"),
    }
    manager = MCPManager(_settings(tmp_path, mcp_servers=servers), client_factory=_FakeMCPClient)
    _run_async(manager.get_resources())

    render_mcp_status(manager)

    assert any("All MCP servers offline" in line for line in printed)


def test_render_mcp_context_update_online(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []
    monkeypatch.setattr(main.console, "print", lambda *args, **kwargs: printed.append(str(args[0])))

    result = main.ContextBuildResult(
        prompt="ok",
        sources=[],
        warnings=[],
        mcp_sources_used=["docs"],
        mcp_chars_used=150,
        mcp_servers_offline=[],
        mcp_tools_available=3,
        mcp_resources_available=2,
    )

    render_mcp_context_update(result)

    assert any("Added 2 resources" in line for line in printed)
