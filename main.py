from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast
from uuid import uuid4

import typer
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.columns import Columns
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

console = Console()
error_console = Console(stderr=True)
app = typer.Typer(help="Helo CLI (AI)", no_args_is_help=True)
logger = logging.getLogger(__name__)

IMPLEMENTER_STOP_TOKEN = "STOP_FOR_COMMIT"
IMPLEMENTER_DONE_TOKEN = "IMPLEMENTATION_COMPLETE"
STOP_AND_COMMIT_SENTENCE = (
    "**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit "
    "the change."
)
STATUS_EVENT_PREFIX = "__STATUS__::"
DUMP_EVENT_PREFIX = "__DUMP__::"
META_CONTEXT_WINDOW_FALLBACK = 128_000
MAX_CONTEXT_FILE_CHARS = 12_000
MAX_MCP_CONTEXT_CHARS = 6_000
MAX_DIRECTORY_CONTEXT_ITEMS = 60
MAX_VERBOSE_DUMP_CHARS = 4_000
EXIT_COMMANDS = {"exit", "quit", "/exit"}
CHAT_COMMANDS = {
    "/planner",
    "/generator",
    "/implementer",
    "/clear",
    "/reset",
    "/help",
    "/exit",
}
ROLE_COMMANDS = {"/planner", "/generator", "/implementer"}
ROLE_ARGUMENT = typer.Argument(..., help="Role to execute manually: planner|generator|implementer")
PROMPT_ARGUMENT = typer.Argument(..., help="Role prompt.")

FALLBACK_BUILTIN_SKILLS: dict[str, str] = {
    "planner": (
        "You are the planner role. Do not write production code or edit source files. "
        "Your mandatory artifact is plans/{feature-name}/plan.md. "
        "Use tools to gather context, then create or update only plan.md with a concrete, "
        "testable, ordered plan and validation checkpoints."
    ),
    "generator": (
        "You are the generator role. Read plan.md and produce plans/{feature-name}/"
        "implementation.md. Do not implement source code changes in this phase. "
        "The implementation guide must be explicit, file-level, and checkpoint-driven."
    ),
    "implementer": (
        "You are the implementer role. Read implementation.md and execute only what the "
        "plan specifies, step by step, reporting progress and validations."
    ),
}


@dataclass(frozen=True)
class ContextBuildResult:
    prompt: str
    sources: list[str]
    warnings: list[str]
    mcp_sources_used: list[str]
    mcp_chars_used: int
    mcp_servers_offline: list[str]
    mcp_tools_available: int
    mcp_resources_available: int


@dataclass
class ChatSessionMetrics:
    total_tokens: int = 0
    last_tokens: int = 0


@dataclass
class OutputStreamState:
    clarification_requested: bool = False


@dataclass(frozen=True)
class ChatInteractionMetadata:
    role: str
    thread_id: str
    skills_loaded: list[str]
    sources: list[str]
    last_tokens: int
    total_tokens: int
    context_window: int
    mcp_sources_used: list[str]
    mcp_chars_used: int
    mcp_servers_offline: list[str]
    mcp_tools_available: int
    mcp_resources_available: int


class ChatCompleter(Completer):
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def get_completions(self, document: Document, complete_event: Any) -> Iterable[Completion]:
        del complete_event
        token = _current_input_token(document.text_before_cursor)
        if not token:
            return

        if token.startswith("/"):
            for command in sorted(CHAT_COMMANDS):
                if command.startswith(token):
                    yield Completion(
                        command,
                        start_position=-len(token),
                        display=command,
                        display_meta="command",
                    )
            return

        if token.startswith("#"):
            raw_prefix = token[1:]
            for candidate in list_context_candidates(self._project_root, raw_prefix):
                completion_value = f"#{candidate}"
                yield Completion(
                    completion_value,
                    start_position=-len(token),
                    display=completion_value,
                    display_meta="context",
                )


def _current_input_token(value: str) -> str:
    stripped = value.rstrip()
    if not stripped:
        return ""
    parts = stripped.split()
    return parts[-1] if parts else ""


@lru_cache(maxsize=4)
def _workspace_candidates(project_root: str) -> tuple[str, ...]:
    root = Path(project_root)
    candidates: list[str] = []
    for path in root.rglob("*"):
        if path == root:
            continue
        rel = path.relative_to(root)
        rel_parts = rel.parts
        if any(part.startswith(".") for part in rel_parts if part not in {".agents"}):
            continue
        display = rel.as_posix()
        if path.is_dir():
            display = f"{display}/"
        candidates.append(display)
    return tuple(sorted(candidates))


def list_context_candidates(project_root: Path, prefix: str, limit: int = 25) -> list[str]:
    normalized_prefix = prefix.lstrip("./")
    entries = _workspace_candidates(str(project_root.resolve()))
    matches = [entry for entry in entries if entry.startswith(normalized_prefix)]
    return matches[:limit]


def estimate_token_count(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, len(text) // 4)


def infer_context_window(settings: AppSettings) -> int:
    configured = [
        value
        for value in (
            settings.planner.max_tokens,
            settings.generator.max_tokens,
            settings.implementer.max_tokens,
        )
        if value is not None
    ]
    return max(configured) if configured else META_CONTEXT_WINDOW_FALLBACK


def _normalize_context_ref(raw_ref: str) -> str:
    return raw_ref.strip().rstrip(",.;:")


def extract_context_references(message: str) -> tuple[str, list[str]]:
    refs: list[str] = []
    kept_parts: list[str] = []
    for part in message.split():
        if part.startswith("#") and len(part) > 1:
            normalized = _normalize_context_ref(part[1:])
            if normalized:
                refs.append(normalized)
        else:
            kept_parts.append(part)
    return " ".join(kept_parts).strip(), refs


def _resolve_context_path(project_root: Path, raw_ref: str) -> Path | None:
    ref_path = Path(raw_ref)
    candidate = ref_path if ref_path.is_absolute() else (project_root / ref_path)
    try:
        resolved = candidate.resolve()
        resolved.relative_to(project_root.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def _read_file_for_context(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= MAX_CONTEXT_FILE_CHARS:
        return text
    return (
        f"{text[:MAX_CONTEXT_FILE_CHARS]}\n"
        "\n[truncated: file exceeded context limit for inline attachment]"
    )


def _summarize_directory_for_context(path: Path, project_root: Path) -> str:
    files: list[str] = []
    for child in sorted(path.rglob("*")):
        if child.is_file():
            rel = child.relative_to(project_root).as_posix()
            files.append(rel)
        if len(files) >= MAX_DIRECTORY_CONTEXT_ITEMS:
            break
    if not files:
        return "[empty directory]"
    lines = "\n".join(f"- {file_path}" for file_path in files)
    if len(files) >= MAX_DIRECTORY_CONTEXT_ITEMS:
        lines += "\n- ..."
    return lines


def _trim_to_remaining(text: str, remaining: int) -> tuple[str, int]:
    if remaining <= 0:
        return "", 0
    if len(text) <= remaining:
        return text, len(text)
    suffix = "\n\n[truncated: context budget exceeded]"
    safe = max(0, remaining - len(suffix))
    return f"{text[:safe]}{suffix}", remaining


class MCPServerConfig(BaseModel):
    transport: Literal["stdio", "streamable_http", "sse", "websocket"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: int = 30

    @model_validator(mode="after")
    def validate_required_transport_fields(self) -> MCPServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError("command is required when transport=stdio")
        if self.transport in {"streamable_http", "sse", "websocket"} and not self.url:
            raise ValueError("url is required for http/sse/websocket transports")
        return self


def _normalize_mcp_transport(raw: str | None) -> str:
    if raw is None:
        return ""
    lowered = raw.strip().lower()
    aliases = {
        "http": "streamable_http",
        "streamable-http": "streamable_http",
        "streamable_http": "streamable_http",
        "ws": "websocket",
    }
    return aliases.get(lowered, lowered)


def _load_mcp_servers_from_file(config_path: Path) -> dict[str, MCPServerConfig]:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    servers_raw = data.get("servers")
    if not isinstance(servers_raw, dict):
        return {}

    parsed: dict[str, MCPServerConfig] = {}
    for name, payload in servers_raw.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            continue
        transport = _normalize_mcp_transport(
            cast(str | None, payload.get("transport") or payload.get("type"))
        )
        if transport not in {"stdio", "streamable_http", "sse", "websocket"}:
            continue
        try:
            parsed[name] = MCPServerConfig(
                transport=cast(
                    Literal["stdio", "streamable_http", "sse", "websocket"],
                    transport,
                ),
                command=cast(str | None, payload.get("command")),
                args=[str(item) for item in payload.get("args", []) if isinstance(item, str)],
                url=cast(str | None, payload.get("url")),
                headers={
                    str(k): str(v)
                    for k, v in cast(dict[str, Any], payload.get("headers", {})).items()
                }
                if isinstance(payload.get("headers"), dict)
                else {},
                timeout=int(payload.get("timeout", 30)),
            )
        except (TypeError, ValueError):
            continue

    return parsed


def load_mcp_servers_from_workspace(project_root: Path) -> dict[str, MCPServerConfig]:
    merged: dict[str, MCPServerConfig] = {}
    for path in (
        project_root / ".agents" / "mcp.json",
        project_root / ".vscode" / "mcp.json",
    ):
        merged.update(_load_mcp_servers_from_file(path))
    return merged


@dataclass(frozen=True)
class MCPResourceBlob:
    server_name: str
    resource_uri: str
    content: str


@dataclass(frozen=True)
class MCPServerStatus:
    online: bool
    tools_count: int = 0
    resources_count: int = 0
    last_error: str | None = None


class MCPManagerLike(Protocol):
    def has_servers(self) -> bool: ...

    def get_server_status(self) -> dict[str, MCPServerStatus]: ...

    @property
    def cached_tools(self) -> dict[str, list[Any]] | None: ...

    @property
    def cached_resources(self) -> list[MCPResourceBlob] | None: ...

    async def refresh(self) -> None: ...

    async def get_all_tools(self) -> dict[str, list[Any]]: ...

    async def get_resources(self) -> list[MCPResourceBlob]: ...


class MCPManager:
    def __init__(
        self,
        settings: AppSettings,
        client_factory: Callable[[dict[str, dict[str, Any]]], Any] | None = None,
        debug: bool = False,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory
        self._client: Any | None = None
        self._debug = debug
        self._status: dict[str, MCPServerStatus] = {
            name: MCPServerStatus(online=False, last_error="not connected")
            for name in settings.mcp_servers
        }
        self._cached_tools: dict[str, list[Any]] | None = None
        self._cached_resources: list[MCPResourceBlob] | None = None

        # Start a background thread to refresh MCP status silently.
        if self.has_servers():
            thread = threading.Thread(target=self._background_refresh, daemon=True)
            thread.start()

    @property
    def cached_tools(self) -> dict[str, list[Any]] | None:
        return self._cached_tools

    @property
    def cached_resources(self) -> list[MCPResourceBlob] | None:
        return self._cached_resources

    def has_cached_tools(self) -> bool:
        return self._cached_tools is not None

    def has_cached_resources(self) -> bool:
        return self._cached_resources is not None

    async def refresh(self) -> None:
        await self.get_all_tools()
        await self.get_resources()

    def _background_refresh(self) -> None:
        # Continuously refresh MCP status in the background.
        while True:
            try:
                asyncio.run(self.refresh())
            except Exception:
                pass
            # Wait before refreshing again to avoid hammering MCP endpoints.
            time.sleep(30)

    def has_servers(self) -> bool:
        return bool(self._settings.mcp_servers)

    def is_available(self) -> bool:
        return any(status.online for status in self._status.values())

    def get_server_status(self) -> dict[str, MCPServerStatus]:
        return dict(self._status)

    def _client_config(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, cfg in self._settings.mcp_servers.items():
            entry: dict[str, Any] = {
                "transport": cfg.transport,
                "timeout": cfg.timeout,
                "headers": cfg.headers,
            }
            if cfg.command:
                entry["command"] = cfg.command
            if cfg.args:
                entry["args"] = cfg.args
            if cfg.url:
                entry["url"] = cfg.url
            payload[name] = entry
        return payload

    async def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        if not self._settings.mcp_servers:
            return None

        try:
            config = self._client_config()
            if self._client_factory is not None:
                self._client = self._client_factory(config)
            else:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                from langchain_mcp_adapters.sessions import (
                    SSEConnection,
                    StdioConnection,
                    StreamableHttpConnection,
                    WebsocketConnection,
                )

                typed: dict[
                    str,
                    StdioConnection
                    | SSEConnection
                    | StreamableHttpConnection
                    | WebsocketConnection,
                ] = cast(
                    dict[
                        str,
                        StdioConnection
                        | SSEConnection
                        | StreamableHttpConnection
                        | WebsocketConnection,
                    ],
                    config,
                )
                self._client = MultiServerMCPClient(typed)
            return self._client
        except Exception as error:
            for name in self._settings.mcp_servers:
                self._status[name] = MCPServerStatus(online=False, last_error=str(error))
                logger.warning(
                    "MCP server '%s' is offline: %s. Continuing without MCP context.",
                    name,
                    error,
                )
            return None

    async def _fetch_server_tools(self, server_name: str) -> list[Any]:
        client = await self._ensure_client()
        if client is None:
            return []

        method = getattr(client, "get_tools", None)
        if method is None:
            return []

        # Try different call signatures until one succeeds.
        try:
            value = method(server_name=server_name)
        except TypeError:
            try:
                value = method(server_name)
            except TypeError:
                try:
                    value = method()
                except TypeError:
                    return []

        if asyncio.iscoroutine(value):
            value = await value
        if isinstance(value, list):
            return value
        return []

    async def _fetch_server_resources(self, server_name: str) -> list[MCPResourceBlob]:
        client = await self._ensure_client()
        if client is None:
            return []

        for method_name in ("get_resources", "list_resources"):
            method = getattr(client, method_name, None)
            if method is None:
                continue
            # Try various call signatures for maximum compatibility.
            try:
                value = method(server_name=server_name)
            except TypeError:
                try:
                    value = method(server_name)
                except TypeError:
                    try:
                        value = method()
                    except TypeError:
                        continue

            if asyncio.iscoroutine(value):
                value = await value
            blobs = self._normalize_resources(server_name, value)
            if blobs:
                return blobs
        return []

    @staticmethod
    def _normalize_resources(server_name: str, value: Any) -> list[MCPResourceBlob]:
        if not isinstance(value, list):
            return []
        blobs: list[MCPResourceBlob] = []
        for item in value:
            if isinstance(item, MCPResourceBlob):
                blobs.append(item)
                continue
            if isinstance(item, dict):
                uri = str(item.get("uri") or item.get("resource_uri") or "resource")
                content = str(
                    item.get("content")
                    or item.get("text")
                    or item.get("blob")
                    or item.get("value")
                    or ""
                )
                if content:
                    blobs.append(
                        MCPResourceBlob(
                            server_name=server_name,
                            resource_uri=uri,
                            content=content,
                        )
                    )
        return blobs

    async def get_all_tools(self) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = {}
        for name in self._settings.mcp_servers:
            try:
                tools = await self._fetch_server_tools(name)
                result[name] = tools
                status = self._status.get(name, MCPServerStatus(online=False))
                self._status[name] = MCPServerStatus(
                    online=True,
                    tools_count=len(tools),
                    resources_count=status.resources_count,
                )
            except (ConnectionError, TimeoutError, ValueError, Exception) as error:
                http_status = _extract_http_status(error)
                self._status[name] = MCPServerStatus(
                    online=False,
                    last_error=_sanitize_error_message(error),
                )
                if http_status:
                    logger.debug("MCP server '%s' offline (%s).", name, http_status)
                else:
                    if self._debug:
                        logger.exception(
                            "MCP server '%s' is offline. Continuing without MCP context.",
                            name,
                            error,
                        )
                    else:
                        logger.debug(
                            "MCP server '%s' offline: %s.",
                            name,
                            _sanitize_error_message(error),
                        )
                result[name] = []
        self._cached_tools = result
        return result

    async def get_resources(self) -> list[MCPResourceBlob]:
        result: list[MCPResourceBlob] = []
        for name in self._settings.mcp_servers:
            try:
                resources = await self._fetch_server_resources(name)
                result.extend(resources)
                status = self._status.get(name, MCPServerStatus(online=False))
                self._status[name] = MCPServerStatus(
                    online=True,
                    tools_count=status.tools_count,
                    resources_count=len(resources),
                )
            except (ConnectionError, TimeoutError, ValueError, Exception) as error:
                http_status = _extract_http_status(error)
                sanitized = _sanitize_error_message(error)
                self._status[name] = MCPServerStatus(
                    online=False,
                    last_error=sanitized,
                )
                if http_status:
                    logger.debug("MCP server '%s' offline (%s).", name, http_status)
                else:
                    logger.debug("MCP server '%s' offline: %s.", name, sanitized)
        self._cached_resources = result
        return result


def _sanitize_error_message(error: BaseException) -> str:
    """Produce a compact, user-facing error message without full tracebacks."""

    status = _extract_http_status(error)
    if status:
        return status

    # ExceptionGroup often wraps errors; pick the first useful suberror.
    if isinstance(error, BaseExceptionGroup):
        for sub in error.exceptions:
            msg = _sanitize_error_message(sub)
            if msg:
                return msg

    text = str(error).strip()
    if not text:
        text = type(error).__name__
    # Only keep the first line; strip common 'unhandled errors' verbosity.
    first_line = text.splitlines()[0]
    if len(first_line) > 200:
        first_line = first_line[:197] + "..."
    return first_line


def _extract_http_status(error: BaseException) -> str | None:
    # Try to extract an HTTP status when using httpx / anyio TaskGroup wrapping.
    try:
        import httpx
    except ImportError:
        return None

    if isinstance(error, httpx.HTTPStatusError):
        resp = getattr(error, "response", None)
        if resp is not None:
            return f"HTTP {resp.status_code} {resp.reason_phrase}"
        return "HTTP error"

    if isinstance(error, BaseExceptionGroup):
        for sub in error.exceptions:
            found = _extract_http_status(sub)
            if found:
                return found
    return None


def _run_async(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _collect_agent_skill_context(
    project_root: Path, remaining_budget: int
) -> tuple[list[str], list[str], int]:
    if remaining_budget <= 0:
        return [], [], 0
    agents_dir = project_root / ".agents"
    if not agents_dir.exists() or not agents_dir.is_dir():
        return [], [], 0

    blocks: list[str] = []
    sources: list[str] = []
    consumed = 0
    for skill_path in sorted(agents_dir.glob("*/SKILL.md")):
        rel = skill_path.relative_to(project_root).as_posix()
        raw = _read_file_for_context(skill_path)
        formatted = f"### {rel}\n{raw}"
        remaining = remaining_budget - consumed
        clipped, used = _trim_to_remaining(formatted, remaining)
        if used <= 0:
            break
        blocks.append(clipped)
        sources.append(rel)
        consumed += used
        if consumed >= remaining_budget:
            break
    return blocks, sources, consumed


# Mapping from intent keywords to the corresponding skill directory name.
# If the user message includes any of these keywords (word boundaries),
# we auto-include that skill's SKILL.md into the context.
_SKILL_INTENT_TRIGGERS: dict[str, str] = {
    "refactor": "refactor",
    "review": "code-review",
}


def _collect_intent_skill_context(
    project_root: Path,
    remaining_budget: int,
    message: str,
    already_loaded: set[str],
) -> tuple[list[str], list[str], int]:
    if remaining_budget <= 0:
        return [], [], 0

    # Identify which skills should be auto-loaded based on intent keywords.
    triggers: set[str] = set()
    message_lower = message.lower()
    for keyword, skill_dir in _SKILL_INTENT_TRIGGERS.items():
        # Use word boundary checks to avoid partial matches.
        if f"{keyword}" in message_lower.split():
            triggers.add(skill_dir)

    blocks: list[str] = []
    sources: list[str] = []
    consumed = 0

    for skill_dir in sorted(triggers):
        skill_path = project_root / ".agents" / skill_dir / "SKILL.md"
        if not skill_path.exists():
            continue

        rel = skill_path.relative_to(project_root).as_posix()
        if rel in already_loaded:
            continue

        raw = _read_file_for_context(skill_path)
        formatted = f"### {rel}\n{raw}"
        remaining = remaining_budget - consumed
        clipped, used = _trim_to_remaining(formatted, remaining)
        if used <= 0:
            break

        blocks.append(clipped)
        sources.append(rel)
        consumed += used

    return blocks, sources, consumed


def build_contextual_prompt(
    message: str,
    project_root: Path,
    mcp_manager: MCPManagerLike | None = None,
    auto_load_skills: bool = False,
) -> ContextBuildResult:
    # Normalize to an absolute project root so relative path operations work
    # even when callers pass a relative path like '.'
    project_root = project_root.resolve()

    cleaned_message, refs = extract_context_references(message)
    warnings: list[str] = []
    sources: list[str] = []
    context_blocks: list[str] = []
    seen: set[Path] = set()
    local_budget_used = 0

    for ref in refs:
        if local_budget_used >= MAX_CONTEXT_FILE_CHARS:
            break
        resolved = _resolve_context_path(project_root, ref)
        if resolved is None:
            warnings.append(f"Contexto ignorado (fora do projeto): #{ref}")
            continue
        if not resolved.exists():
            warnings.append(f"Contexto nao encontrado: #{ref}")
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        rel = resolved.relative_to(project_root).as_posix()
        sources.append(rel)
        if resolved.is_file():
            file_context = _read_file_for_context(resolved)
            formatted = f"### {rel}\n{file_context}"
        else:
            directory_context = _summarize_directory_for_context(resolved, project_root)
            formatted = f"### {rel}/\n{directory_context}"

        remaining = MAX_CONTEXT_FILE_CHARS - local_budget_used
        clipped, used = _trim_to_remaining(formatted, remaining)
        if used <= 0:
            break
        context_blocks.append(clipped)
        local_budget_used += used

    if auto_load_skills and local_budget_used < MAX_CONTEXT_FILE_CHARS:
        already_loaded = set(sources)
        intent_blocks, srcs, used = _collect_intent_skill_context(
            project_root,
            MAX_CONTEXT_FILE_CHARS - local_budget_used,
            cleaned_message,
            already_loaded,
        )
        context_blocks.extend(intent_blocks)
        sources.extend(srcs)
        local_budget_used += used

    mcp_sources_used: list[str] = []
    mcp_chars_used = 0
    mcp_servers_offline: list[str] = []
    mcp_tools_available = 0
    mcp_resources_available = 0

    if mcp_manager is not None and mcp_manager.has_servers():
        tools = getattr(mcp_manager, "cached_tools", None)
        if tools is None:
            tools = cast(dict[str, list[Any]], _run_async(mcp_manager.get_all_tools()))
        mcp_tools_available = sum(len(items) for items in tools.values())

        resources = getattr(mcp_manager, "cached_resources", None)
        if resources is None:
            resources = cast(list[MCPResourceBlob], _run_async(mcp_manager.get_resources()))
        mcp_resources_available = len(resources)

        for resource in resources:
            if mcp_chars_used >= MAX_MCP_CONTEXT_CHARS:
                break
            block: str = (
                f"[MCP: {resource.server_name}] {resource.resource_uri}\n\n{resource.content}"
            )
            clipped, used = _trim_to_remaining(block, MAX_MCP_CONTEXT_CHARS - mcp_chars_used)
            if used <= 0:
                break
            context_blocks.append(clipped)
            mcp_chars_used += used
            if resource.server_name not in mcp_sources_used:
                mcp_sources_used.append(resource.server_name)

        statuses = mcp_manager.get_server_status()
        mcp_servers_offline = []
        for name, status in statuses.items():
            if not status.online:
                if status.last_error:
                    mcp_servers_offline.append(f"{name} ({status.last_error})")
                else:
                    mcp_servers_offline.append(name)

    if not context_blocks:
        return ContextBuildResult(
            prompt=cleaned_message or message,
            sources=sources,
            warnings=warnings,
            mcp_sources_used=mcp_sources_used,
            mcp_chars_used=mcp_chars_used,
            mcp_servers_offline=mcp_servers_offline,
            mcp_tools_available=mcp_tools_available,
            mcp_resources_available=mcp_resources_available,
        )

    user_prompt = (
        cleaned_message if cleaned_message else "Use o contexto anexado e responda o pedido."
    )
    prompt = (
        "Contexto adicional fornecido pelo usuario via #:\n\n"
        + "\n\n".join(context_blocks)
        + "\n\nPedido do usuario:\n"
        + user_prompt
    )
    logger.info(
        "Context built: files_agents=%s, mcp=%s, total=%s",
        local_budget_used,
        mcp_chars_used,
        len(prompt),
    )
    logger.debug(
        "MCP servers: online=%s offline=%s",
        mcp_sources_used,
        mcp_servers_offline,
    )
    return ContextBuildResult(
        prompt=prompt,
        sources=sources,
        warnings=warnings,
        mcp_sources_used=mcp_sources_used,
        mcp_chars_used=mcp_chars_used,
        mcp_servers_offline=mcp_servers_offline,
        mcp_tools_available=mcp_tools_available,
        mcp_resources_available=mcp_resources_available,
    )


def build_chat_session_prompt(project_root: Path) -> PromptSession[str]:
    style = Style.from_dict(
        {
            "prompt": "ansibrightcyan bold",
            "symbol": "ansibrightgreen bold",
            "hint": "ansibrightblack",
            "badge": "ansiblue bold",
        }
    )

    return PromptSession(
        completer=ChatCompleter(project_root),
        complete_while_typing=True,
        style=style,
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        reserve_space_for_menu=8,
        bottom_toolbar=HTML(
            "<badge>[TAB]</badge><hint> autocomplete de </hint>"
            "<b>/roles</b><hint> e </hint><b>#contexto</b>"
            "<hint>  |  </hint><badge>[CTRL+C]</badge><hint> sair</hint>"
            "<hint>  |  comandos: </hint><b>/clear</b><hint>, </hint>"
            "<b>/reset</b><hint>, </hint><b>/help</b>"
        ),
    )


def render_chat_header(thread_id: str) -> None:
    title = Text("Helo AI CLI", style="bold white")
    subtitle = Text(f"Sessao ativa: {thread_id}", style="cyan")
    body = Text.assemble(
        "Chat interativo com pipeline e roles manuais",
        "\n",
        "Digite /help para atalhos",
    )
    panel = Panel(
        Padding(Text.assemble(title, "\n", subtitle, "\n\n", body), (0, 1)),
        border_style="bright_blue",
        title="Helo AI CLI",
    )
    console.print(panel)


def render_user_message(message: str, context_sources: list[str]) -> None:
    source_text = "\n".join(f"- {source}" for source in context_sources)
    content = message if not context_sources else f"{message}\n\nContext refs:\n{source_text}"
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print()
    console.print(
        Panel(
            content,
            title=f"You  {timestamp}",
            border_style="bright_cyan",
            padding=(0, 1),
            expand=True,
        )
    )


def render_assistant_message(response: str, metadata: ChatInteractionMetadata) -> None:
    context_percent = (
        (metadata.total_tokens / metadata.context_window) * 100 if metadata.context_window else 0.0
    )
    timestamp = datetime.now().strftime("%H:%M:%S")
    response_panel = Panel(
        response.strip() or "[empty response]",
        title=f"Helo AI CLI ({metadata.role})  {timestamp}",
        border_style="green",
        padding=(0, 1),
    )

    skills = "\n".join(metadata.skills_loaded) if metadata.skills_loaded else "(none)"
    sources = "\n".join(metadata.sources) if metadata.sources else "(none)"
    mcp_sources = "\n".join(metadata.mcp_sources_used) if metadata.mcp_sources_used else "(none)"
    mcp_offline = (
        "\n".join(metadata.mcp_servers_offline) if metadata.mcp_servers_offline else "(none)"
    )
    meta_panel = Panel(
        Text.from_markup(
            "[bold]Session[/]\n"
            f"{metadata.thread_id}\n\n"
            "[bold]Skills loaded[/]\n"
            f"{skills}\n\n"
            "[bold]Sources used[/]\n"
            f"{sources}\n\n"
            "[bold]MCP[/]\n"
            f"sources={mcp_sources}\n"
            f"offline={mcp_offline}\n"
            f"tools={metadata.mcp_tools_available} resources={metadata.mcp_resources_available}\n"
            f"chars={metadata.mcp_chars_used}\n\n"
            "[bold]Tokens[/]\n"
            f"last={metadata.last_tokens} total={metadata.total_tokens}\n"
            "[bold]Context window[/]\n"
            f"{context_percent:.1f}% of {metadata.context_window}"
        ),
        title="Metadata",
        border_style="magenta",
        padding=(0, 1),
        width=42,
    )

    console.print()
    console.print(Columns([response_panel, meta_panel], expand=True, equal=False))


def is_chat_command(message: str, command: str) -> bool:
    return message.strip().lower() == command


def parse_help_text() -> str:
    return (
        "Comandos:\n"
        "- /planner <pedido>\n"
        "- /generator <pedido>\n"
        "- /implementer <pedido>\n"
        "- /clear (limpa a tela)\n"
        "- /reset (nova sessao limpa)\n"
        "- /exit\n\n"
        "Contexto:\n"
        "- Use #arquivo ou #diretorio para anexar contexto no prompt.\n"
        "- Digite / e pressione TAB para autocomplete de roles/comandos.\n"
        "- Digite # e pressione TAB para autocomplete de arquivos/pastas."
    )


def _skills_for_metadata(runtime: AgentRuntime, selected_role: str) -> list[str]:
    selected: list[str] = []
    role_map = {
        "planner": ["builtin:planner_skill.md"],
        "generator": ["builtin:generator_skill.md"],
        "implementer": ["builtin:implementer_skill.md"],
        "pipeline": [
            "builtin:planner_skill.md",
            "builtin:generator_skill.md",
            "builtin:implementer_skill.md",
        ],
        "chat": ["builtin:chat-system"],
    }
    selected.extend(role_map.get(selected_role, []))
    skills = getattr(runtime, "skills", [])
    if isinstance(skills, list):
        selected.extend(skills)
    return selected


class ModelRole(StrEnum):
    PLANNER = "planner"
    GENERATOR = "generator"
    IMPLEMENTER = "implementer"


class Provider(StrEnum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LITELLM = "litellm"


class RoleModelConfig(BaseModel):
    provider: Provider = Provider.OPENROUTER
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None

    @model_validator(mode="after")
    def validate_model_name(self) -> RoleModelConfig:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        return self


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        extra="ignore",
    )

    planner: RoleModelConfig = Field(
        default_factory=lambda: RoleModelConfig(
            provider=Provider.OPENROUTER,
            model="stepfun/step-3.5-flash:free",
            temperature=0.0,
        )
    )
    generator: RoleModelConfig = Field(
        default_factory=lambda: RoleModelConfig(
            provider=Provider.OPENROUTER,
            model="stepfun/step-3.5-flash:free",
            temperature=0.1,
        )
    )
    implementer: RoleModelConfig = Field(
        default_factory=lambda: RoleModelConfig(
            provider=Provider.OPENROUTER,
            model="stepfun/step-3.5-flash:free",
            temperature=0.0,
        )
    )

    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    enable_langsmith: bool = False
    langsmith_project: str | None = None

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    thread_id_default: str = "helo-ai-cli-session"
    skills_required: bool = False
    skills_autoload: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    @property
    def openrouter_effective_api_key(self) -> str | None:
        # OpenRouter accepts OpenAI-compatible keys; keep backwards compatibility.
        return self.openrouter_api_key or self.openai_api_key

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> AppSettings:
        for role in (self.planner, self.generator, self.implementer):
            if role.provider is Provider.OPENROUTER and not self.openrouter_effective_api_key:
                raise ValueError(
                    "openrouter_api_key or openai_api_key is required "
                    "when any role provider is openrouter"
                )
        if self.enable_langsmith and not os.getenv("LANGSMITH_API_KEY"):
            raise ValueError("LANGSMITH_API_KEY is required when enable_langsmith=true")
        file_mcp_servers = load_mcp_servers_from_workspace(self.project_root)
        self.mcp_servers = {**file_mcp_servers, **self.mcp_servers}
        return self


@lru_cache(maxsize=1)
def resolve_env_file() -> Path | None:
    candidates: list[Path] = [Path.cwd() / ".env"]
    project_root = os.getenv("PROJECT_ROOT")
    if project_root:
        candidates.append(Path(project_root).expanduser() / ".env")

    # When frozen with PyInstaller, look next to the executable as well.
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / ".env")

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists() and normalized.is_file():
            return normalized
    return None


def parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip().lstrip("\ufeff")
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()

    return key, value


def load_dotenv_into_environ(env_file: Path | None) -> int:
    if env_file is None:
        return 0

    loaded = 0
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0

    for line in lines:
        parsed = parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        existing = os.environ.get(key)
        if existing is not None and existing.strip() != "":
            continue
        os.environ[key] = value
        loaded += 1

    return loaded


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    env_file = resolve_env_file()
    load_dotenv_into_environ(env_file)
    settings_kwargs: dict[str, Any] = {}
    if env_file is not None:
        settings_kwargs["_env_file"] = env_file
    return AppSettings(**settings_kwargs)


def _parse_skill_metadata(skill_path: Path) -> tuple[str, str]:
    """Extract skill name and description from a SKILL.md file.

    The most reliable source is YAML frontmatter (---) with `name` and
    `description` fields. When frontmatter is absent, fall back to:
    1) first markdown heading (# ...)
    2) first non-empty line.
    """

    try:
        text = (skill_path / "SKILL.md").read_text(encoding="utf-8")
    except OSError:
        return (skill_path.name, "")

    # Remove a leading BOM if present so parsing works consistently
    text = text.lstrip("\ufeff")

    name = skill_path.name
    description = ""

    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        # parse frontmatter
        fm_lines: list[str] = []
        for line in lines[1:]:
            if line.strip() == "---":
                break
            fm_lines.append(line)
        for line in fm_lines:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip().strip("\"'")
            if key == "name" and val:
                name = val
            elif key == "description" and val:
                description = val
        if description:
            return name, description

    # fallback: first heading
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            # take first heading line and strip leading #s
            heading = stripped.lstrip("#").strip()
            if heading:
                name = heading
                break

    # fallback: first non-heading non-empty line
    if not description:
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            description = stripped
            break
    return name, description


def list_skills(project_root: Path) -> list[dict[str, str]]:
    """List all valid skills found under the .agents directory."""

    skills_dir = project_root / ".agents"
    if not skills_dir.exists() or not skills_dir.is_dir():
        return []

    skills: list[dict[str, str]] = []
    for item in sorted(skills_dir.iterdir(), key=lambda p: p.name):
        if not item.is_dir():
            continue
        skill_md = item / "SKILL.md"
        if not skill_md.exists():
            continue
        name, description = _parse_skill_metadata(item)
        skills.append(
            {
                "id": item.name,
                "path": f"/.agents/{item.name}/",
                "name": name,
                "description": description,
            }
        )
    return skills


def discover_skills_source(project_root: Path, required: bool) -> list[str]:
    skills_dir = project_root / ".agents"
    if not skills_dir.exists():
        if required:
            raise FileNotFoundError(".agents directory not found")
        return []

    if not skills_dir.is_dir():
        raise NotADirectoryError(".agents exists but is not a directory")

    invalid_skills: list[str] = []
    for item in skills_dir.iterdir():
        if item.is_dir() and not (item / "SKILL.md").exists():
            invalid_skills.append(item.name)

    if invalid_skills:
        names = ", ".join(sorted(invalid_skills))
        raise ValueError(f"Invalid skill directories missing SKILL.md: {names}")

    return ["/.agents/"]


ChatModelLike = Any


class ModelFactory:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    def _config_for_role(self, role: ModelRole) -> RoleModelConfig:
        if role is ModelRole.PLANNER:
            return self._settings.planner
        if role is ModelRole.GENERATOR:
            return self._settings.generator
        return self._settings.implementer

    def create(self, role: ModelRole) -> ChatModelLike:
        config = self._config_for_role(role)

        if config.provider is Provider.OPENROUTER:
            from langchain_openrouter import ChatOpenRouter
            from pydantic import SecretStr

            return ChatOpenRouter(
                model=config.model,
                api_key=(
                    SecretStr(self._settings.openrouter_effective_api_key)
                    if self._settings.openrouter_effective_api_key
                    else None
                ),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        if config.provider is Provider.OLLAMA:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=config.model,
                temperature=config.temperature,
                base_url=self._settings.ollama_base_url,
                num_predict=config.max_tokens,
            )

        from langchain_litellm import ChatLiteLLM

        return ChatLiteLLM(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )


@dataclass(frozen=True)
class OrchestrationPolicy:
    planner: ModelRole = ModelRole.PLANNER
    generator: ModelRole = ModelRole.GENERATOR
    implementer: ModelRole = ModelRole.IMPLEMENTER


class AgentRuntime:
    def __init__(
        self,
        settings: AppSettings,
        policy: OrchestrationPolicy | None = None,
        mcp_debug: bool = False,
    ) -> None:
        self.settings = settings
        self.policy = policy or OrchestrationPolicy()
        self.model_factory = ModelFactory(settings)
        self.backend = LocalShellBackend(root_dir=str(settings.project_root), virtual_mode=False)
        self.skills = discover_skills_source(
            settings.project_root, required=settings.skills_required
        )
        self._role_agents: dict[tuple[ModelRole, bool], Any] = {}
        self._chat_agent: Any | None = None
        self.mcp_manager = MCPManager(settings, debug=mcp_debug)

    def _builtin_skill_name(self, role: ModelRole) -> str:
        skill_name = {
            ModelRole.PLANNER: "planner_skill.md",
            ModelRole.GENERATOR: "generator_skill.md",
            ModelRole.IMPLEMENTER: "implementer_skill.md",
        }[role]
        return skill_name

    def _builtin_skill_paths(self, role: ModelRole) -> list[Path]:
        skill_name = self._builtin_skill_name(role)
        candidates: list[Path] = [self.settings.project_root / "skills_builtin" / skill_name]

        module_dir = Path(__file__).resolve().parent
        candidates.append(module_dir / "skills_builtin" / skill_name)

        if getattr(sys, "frozen", False):
            meipass = getattr(sys, "_MEIPASS", None)
            if isinstance(meipass, str) and meipass.strip():
                candidates.append(Path(meipass) / "skills_builtin" / skill_name)
            candidates.append(Path(sys.executable).resolve().parent / "skills_builtin" / skill_name)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                normalized = candidate.resolve()
            except OSError:
                normalized = candidate
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _fallback_builtin_skill_text(role: ModelRole) -> str:
        return FALLBACK_BUILTIN_SKILLS.get(role.value, "")

    def _load_builtin_skill_text(self, role: ModelRole) -> str:
        for skill_file in self._builtin_skill_paths(role):
            try:
                text = skill_file.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if text:
                return text
        return self._fallback_builtin_skill_text(role)

    @staticmethod
    def _pipeline_contract() -> str:
        return (
            "Pipeline contract:\n"
            "- Always respect the 3 phases in order: planner -> generator -> implementer.\n"
            "- Preserve context from previous phases and never rewrite intent.\n"
            "- Keep outputs deterministic, explicit, and directly actionable.\n"
            "- Use concise status updates and clear completion criteria."
        )

    def _role_preamble(self, role: ModelRole) -> str:
        if role is ModelRole.PLANNER:
            return (
                "You are the planner role. Your only job is to produce a concrete, testable, and "
                "ordered implementation plan for this repository. Do not write production code. "
                "Include validation checkpoints so downstream phases can execute without ambiguity. "
                "You must create or update plans/{feature-name}/plan.md and avoid source-code edits "
                "in this phase."
            )
        if role is ModelRole.GENERATOR:
            return (
                "You are the generator role. Convert planner output into a complete implementation "
                "guide with explicit steps, concrete file-level actions, "
                "and verification criteria. "
                "Avoid ambiguity and preserve the planner scope. "
                "You must create or update plans/{feature-name}/implementation.md and avoid source-"
                "code edits in this phase."
            )
        return (
            "You are the implementer role. Execute the implementation guide step-by-step and "
            "report what changed and what was validated. Never run git commands; ask the user to "
            "commit and push manually when checkpoints are reached."
        )

    def _implementer_control_rules(self, auto: bool) -> str:
        if auto:
            return (
                "AUTO MODE: execute all remaining implementation steps continuously "
                "without waiting "
                "for user intervention. Do not emit manual checkpoint pauses. At final completion, "
                f"include the exact token {IMPLEMENTER_DONE_TOKEN}."
            )
        return (
            "MANUAL CHECKPOINT MODE: after each STOP & COMMIT checkpoint, "
            "stop execution and include "
            f"the exact token {IMPLEMENTER_STOP_TOKEN} in the response. "
            "Then ask the user to review, "
            "commit, and push manually. Resume only after receiving 'continue'. "
            "At final completion, "
            f"include the exact token {IMPLEMENTER_DONE_TOKEN}."
        )

    def _system_prompt_for_role(self, role: ModelRole, auto: bool) -> str:
        parts = [self._pipeline_contract(), self._role_preamble(role)]
        if role is ModelRole.GENERATOR:
            parts.append(
                "GENERATOR CHECKPOINT CONTRACT: implementation.md must include a STOP & COMMIT "
                "instruction after every implementation step. Use this exact sentence each time: "
                f"{STOP_AND_COMMIT_SENTENCE}"
            )
        if role is ModelRole.IMPLEMENTER:
            parts.append(self._implementer_control_rules(auto))

        skill_text = self._load_builtin_skill_text(role)
        if skill_text:
            parts.append(
                "CRITICAL: Builtin skill instructions below are mandatory and must be followed "
                "strictly."
            )
            parts.append("Builtin skill instructions (must be followed):")
            parts.append(skill_text)
        return "\n\n".join(parts)

    @staticmethod
    def _chunk_to_text(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            return AgentRuntime._extract_assistant_text_from_chunk(chunk)
        return ""

    @staticmethod
    def _extract_assistant_text_from_chunk(chunk: dict[str, Any]) -> str:
        texts: list[str] = []

        def extract_messages(container: Any) -> None:
            if not isinstance(container, list):
                return
            for message in container:
                text = AgentRuntime._assistant_message_to_text(message)
                if text:
                    texts.append(text)

        if "messages" in chunk:
            extract_messages(chunk.get("messages"))

        for key in ("model", "agent", "output"):
            value = chunk.get(key)
            if isinstance(value, dict):
                extract_messages(value.get("messages"))

        return "".join(texts)

    @staticmethod
    def _assistant_message_to_text(message: Any) -> str:
        message_type = getattr(message, "type", None)
        class_name = type(message).__name__.lower()
        is_assistant = message_type == "ai" or "aimessage" in class_name

        if isinstance(message, dict):
            role = str(message.get("role", "")).lower()
            is_assistant = role in {"assistant", "ai"}
            content = message.get("content")
            if not is_assistant:
                return ""
            content_text = AgentRuntime._content_to_text(content)
            clarification = AgentRuntime._extract_clarification_text(message)
            if content_text and clarification:
                return f"{content_text}\n\n{clarification}"
            return content_text or clarification

        if not is_assistant:
            return ""

        content_text = AgentRuntime._content_to_text(getattr(message, "content", None))
        clarification = AgentRuntime._extract_clarification_text(message)
        if content_text and clarification:
            return f"{content_text}\n\n{clarification}"
        return content_text or clarification

    @staticmethod
    def _extract_clarification_text(message: Any) -> str:
        tool_calls: list[Any] = []
        if isinstance(message, dict):
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                tool_calls.extend(raw_tool_calls)
            additional_kwargs = message.get("additional_kwargs")
            if isinstance(additional_kwargs, dict):
                ak_tool_calls = additional_kwargs.get("tool_calls")
                if isinstance(ak_tool_calls, list):
                    tool_calls.extend(ak_tool_calls)
        else:
            raw_tool_calls = getattr(message, "tool_calls", None)
            if isinstance(raw_tool_calls, list):
                tool_calls.extend(raw_tool_calls)
            additional_kwargs = getattr(message, "additional_kwargs", None)
            if isinstance(additional_kwargs, dict):
                ak_tool_calls = additional_kwargs.get("tool_calls")
                if isinstance(ak_tool_calls, list):
                    tool_calls.extend(ak_tool_calls)

        parsed_questions: list[dict[str, Any]] = []
        for call in tool_calls:
            if isinstance(call, dict):
                name = str(call.get("name") or "").strip()
                args_payload = call.get("args")
                if args_payload is None and isinstance(call.get("function"), dict):
                    fn = cast(dict[str, Any], call.get("function"))
                    name = str(fn.get("name") or name).strip()
                    args_payload = fn.get("arguments")
            else:
                name = str(getattr(call, "name", "")).strip()
                args_payload = getattr(call, "args", None)

            if name not in {"vscode_askQuestions", "askQuestions"}:
                continue

            parsed = AgentRuntime._parse_tool_call_args(args_payload)
            questions = parsed.get("questions")
            if isinstance(questions, list):
                for question in questions:
                    if isinstance(question, dict):
                        parsed_questions.append(question)

        if not parsed_questions:
            return ""

        lines = ["Perguntas de Clarificacao:"]
        for idx, question in enumerate(parsed_questions, start=1):
            question_text = str(question.get("question") or question.get("header") or "").strip()
            if not question_text:
                question_text = "Pergunta sem texto"
            lines.append(f"{idx}. {question_text}")

            options = question.get("options")
            if isinstance(options, list):
                labels = [
                    str(opt.get("label", "")).strip() for opt in options if isinstance(opt, dict)
                ]
                labels = [label for label in labels if label]
                if labels:
                    lines.append(f"   Opcoes: {', '.join(labels)}")

        return "\n".join(lines)

    @staticmethod
    def _parse_tool_call_args(args_payload: Any) -> dict[str, Any]:
        if isinstance(args_payload, dict):
            return args_payload
        if isinstance(args_payload, str):
            try:
                parsed = json.loads(args_payload)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                parts.append(AgentRuntime._content_to_text(item))
            return "".join(parts)
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text
        return ""

    @staticmethod
    def _chunk_to_debug_dump(chunk: Any) -> str:
        try:
            rendered = repr(chunk)
        except Exception as exc:  # pragma: no cover - defensive fallback
            rendered = f"<unrepresentable chunk: {exc}>"

        if len(rendered) > MAX_VERBOSE_DUMP_CHARS:
            return f"{rendered[:MAX_VERBOSE_DUMP_CHARS]} ...(truncated)"
        return rendered

    @staticmethod
    def _slugify(text: str) -> str:
        """Create a filesystem-safe slug from a string."""
        import re

        if not text:
            return "feature"
        slug = text.lower().strip()
        slug = slug.splitlines()[0]
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        slug = slug.strip("-")
        max_len = 50
        if len(slug) > max_len:
            slug = slug[:max_len].rstrip("-")
        return slug or "feature"

    def _extract_paths_from_output(self, output: str) -> list[Path]:
        """Extract candidate artifact paths mentioned by role output.

        We intentionally do not write files from app code. Roles/deep agent must
        create files through their own tool execution. This parser only helps us
        discover artifacts that were already created.
        """

        import re

        candidates: list[str] = []
        for line in output.splitlines():
            m = re.search(r"(?:\*\*File:\*\*|File:)\s*`?([^`\s]+)`?", line, re.IGNORECASE)
            if m:
                candidates.append(m.group(1))

        candidates.extend(re.findall(r"(plans/[A-Za-z0-9_\-/.]+\.md)", output))

        deduped: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            path = Path(candidate)
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped

    def _collect_existing_output_artifacts(self, output: str, project_root: Path) -> list[Path]:
        """Return only artifact files that already exist and are inside the project."""

        project_root_resolved = project_root.resolve()
        existing: list[Path] = []
        seen: set[Path] = set()
        for candidate in self._extract_paths_from_output(output):
            target = candidate if candidate.is_absolute() else (project_root_resolved / candidate)
            try:
                resolved = target.resolve()
            except Exception:
                continue
            if project_root_resolved not in resolved.parents and resolved != project_root_resolved:
                continue
            if not resolved.exists() or not resolved.is_file() or resolved in seen:
                continue
            seen.add(resolved)
            existing.append(resolved)
        return existing

    def _snapshot_plan_artifacts(self, project_root: Path) -> dict[Path, tuple[int, int]]:
        """Capture plans/**/*.md file metadata for change detection across phases."""

        root = project_root.resolve()
        plans_dir = root / "plans"
        if not plans_dir.exists() or not plans_dir.is_dir():
            return {}

        snapshot: dict[Path, tuple[int, int]] = {}
        for path in plans_dir.rglob("*.md"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path.resolve()] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    def _collect_changed_plan_artifacts(
        self,
        before: dict[Path, tuple[int, int]],
        after: dict[Path, tuple[int, int]],
    ) -> list[Path]:
        """Return plan artifacts created/updated between two snapshots."""

        changed: list[Path] = []
        for path in sorted(after):
            prev = before.get(path)
            curr = after[path]
            if prev is None or prev != curr:
                changed.append(path)
        return changed

    def create_role_agent(self, role: ModelRole, auto: bool = False) -> Any:
        cache_key = (role, auto)
        cached = self._role_agents.get(cache_key)
        if cached is not None:
            return cached

        role_model = self.model_factory.create(role)
        kwargs: dict[str, Any] = {
            "model": role_model,
            "backend": self.backend,
            "system_prompt": self._system_prompt_for_role(role, auto=auto),
        }
        if self.skills:
            kwargs["skills"] = self.skills
        agent = create_deep_agent(**kwargs)
        self._role_agents[cache_key] = agent
        return agent

    def create_planner_agent(self) -> Any:
        return self.create_role_agent(ModelRole.PLANNER)

    def create_chat_agent(self) -> Any:
        if self._chat_agent is not None:
            return self._chat_agent

        chat_model = self.model_factory.create(self.policy.planner)
        kwargs: dict[str, Any] = {
            "model": chat_model,
            "backend": self.backend,
            "system_prompt": (
                "You are a helpful coding assistant. Reply naturally for casual chat. "
                "Never create, edit, or delete files and never run shell commands in chat mode. "
                "For engineering/build tasks, instruct the user to use the pipeline/role commands."
            ),
        }
        if self.skills:
            kwargs["skills"] = self.skills
        self._chat_agent = create_deep_agent(**kwargs)
        return self._chat_agent

    def stream_role(
        self, role: ModelRole, prompt: str, thread_id: str, auto: bool = False
    ) -> Iterable[Any]:
        agent = self.create_role_agent(role, auto=auto)
        return cast(
            Iterable[Any],
            agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": thread_id}},
            ),
        )

    def stream(self, prompt: str, thread_id: str) -> Iterable[Any]:
        agent = self.create_chat_agent()
        return cast(
            Iterable[Any],
            agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": thread_id}},
            ),
        )

    def run_role(
        self,
        role: ModelRole,
        prompt: str,
        thread_id: str,
        *,
        auto: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        chunks = self.stream_role(role=role, prompt=prompt, thread_id=thread_id, auto=auto)
        output_parts: list[str] = []
        for chunk in chunks:
            if on_chunk:
                on_chunk(f"{DUMP_EVENT_PREFIX}{self._chunk_to_debug_dump(chunk)}")
            text = self._chunk_to_text(chunk)
            output_parts.append(text)
            if on_chunk:
                on_chunk(text)
        return "".join(output_parts)

    @staticmethod
    def _contains_token(output: str, token: str) -> bool:
        return token in output

    @staticmethod
    def _count_implementation_steps(text: str) -> int:
        return len(re.findall(r"(?im)^\s{0,3}#{2,6}\s*step\b", text))

    @staticmethod
    def _count_stop_markers(text: str) -> int:
        return len(re.findall(re.escape(STOP_AND_COMMIT_SENTENCE), text))

    def _implementation_has_required_checkpoints(self, implementation_text: str) -> bool:
        stop_count = self._count_stop_markers(implementation_text)
        if stop_count == 0:
            return False
        step_count = self._count_implementation_steps(implementation_text)
        if step_count == 0:
            return stop_count >= 1
        return stop_count >= step_count

    @staticmethod
    def _generator_repair_prompt(
        *,
        user_prompt: str,
        planner_output: str,
        invalid_implementation: str,
    ) -> str:
        return (
            "Regenerate plans/{feature-name}/implementation.md. HARD REQUIREMENT: include a STOP "
            "& COMMIT block after EVERY step. Use this exact sentence in each checkpoint:\n"
            f"{STOP_AND_COMMIT_SENTENCE}\n\n"
            "Do not implement source code in this phase. Only create/update implementation.md. "
            "Preserve the original plan scope and make the guide deterministic.\n\n"
            f"User request:\n{user_prompt}\n\n"
            f"Planner output (fallback):\n{planner_output}\n\n"
            "Previous invalid implementation.md content:\n"
            f"{invalid_implementation}"
        )

    @staticmethod
    def _emit_status(on_chunk: Callable[[str], None] | None, message: str) -> None:
        if on_chunk is not None:
            on_chunk(f"{STATUS_EVENT_PREFIX}{message}")

    def run_pipeline(
        self,
        prompt: str,
        thread_id: str,
        *,
        auto: bool,
        request_continue: Callable[[str], bool] | None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> list[Path]:
        outputs: list[Path] = []

        def _find_artifact(name: str) -> Path | None:
            for item in outputs:
                if item.name == name:
                    return item
            return None

        self._emit_status(on_chunk, "\n=== Planner ===")
        self._emit_status(on_chunk, "Starting planner phase")
        before_planner = self._snapshot_plan_artifacts(self.settings.project_root)
        planner_output = self.run_role(
            role=ModelRole.PLANNER,
            prompt=prompt,
            thread_id=thread_id,
            auto=auto,
            on_chunk=on_chunk,
        )
        self._emit_status(on_chunk, "Planner phase completed")

        after_planner = self._snapshot_plan_artifacts(self.settings.project_root)
        outputs.extend(self._collect_changed_plan_artifacts(before_planner, after_planner))

        plan_file = _find_artifact("plan.md")
        if plan_file is None:
            self._emit_status(
                on_chunk,
                "Planner did not create plans/{feature-name}/plan.md. "
                "Stopping pipeline before generator.",
            )
            return outputs

        # In non-auto mode, pause after planner so user can review plan.md
        if not auto:
            self._emit_status(
                on_chunk,
                "Pipeline paused after planner. Review plans/{feature-name}/plan.md",
            )
            self._emit_status(
                on_chunk,
                "Type 'continue' to proceed or 'stop' to exit.",
            )
            if request_continue is None or not request_continue("planner"):
                return outputs

        self._emit_status(on_chunk, "\n=== Generator ===")
        self._emit_status(on_chunk, "Starting generator phase")
        generator_prompt = (
            "Generate a complete implementation guide with explicit verification checkpoints. "
            "After each step, include a STOP & COMMIT section with this exact sentence: "
            f"{STOP_AND_COMMIT_SENTENCE} "
            "First, locate and read plans/{feature-name}/plan.md created by planner "
            "using tools. "
            "If the file is unavailable, use the planner output fallback below. "
            "Create/update files under plans/{feature-name} according to "
            "your role instructions.\n\n"
            f"User request:\n{prompt}\n\nPlanner output (fallback):\n{planner_output}"
        )
        before_generator = self._snapshot_plan_artifacts(self.settings.project_root)
        generator_output = self.run_role(
            role=ModelRole.GENERATOR,
            prompt=generator_prompt,
            thread_id=thread_id,
            auto=auto,
            on_chunk=on_chunk,
        )
        self._emit_status(on_chunk, "Generator phase completed")

        after_generator = self._snapshot_plan_artifacts(self.settings.project_root)
        outputs.extend(self._collect_changed_plan_artifacts(before_generator, after_generator))

        impl_file = _find_artifact("implementation.md")
        if impl_file is None:
            self._emit_status(
                on_chunk,
                "Generator did not create plans/{feature-name}/implementation.md. "
                "Stopping pipeline before implementer.",
            )
            return outputs

        try:
            implementation_text = impl_file.read_text(encoding="utf-8")
        except OSError:
            implementation_text = generator_output

        if not self._implementation_has_required_checkpoints(implementation_text):
            self._emit_status(
                on_chunk,
                "Generator output missing required STOP & COMMIT checkpoints. Retrying generator once.",
            )
            before_repair = self._snapshot_plan_artifacts(self.settings.project_root)
            repaired_output = self.run_role(
                role=ModelRole.GENERATOR,
                prompt=self._generator_repair_prompt(
                    user_prompt=prompt,
                    planner_output=planner_output,
                    invalid_implementation=implementation_text,
                ),
                thread_id=thread_id,
                auto=auto,
                on_chunk=on_chunk,
            )
            after_repair = self._snapshot_plan_artifacts(self.settings.project_root)
            outputs.extend(self._collect_changed_plan_artifacts(before_repair, after_repair))

            impl_file = _find_artifact("implementation.md") or impl_file
            try:
                implementation_text = impl_file.read_text(encoding="utf-8")
            except OSError:
                implementation_text = repaired_output

            if not self._implementation_has_required_checkpoints(implementation_text):
                self._emit_status(
                    on_chunk,
                    "Generator could not produce required STOP & COMMIT checkpoints. "
                    "Stopping pipeline before implementer.",
                )
                return outputs

        # In non-auto mode, pause after generator so user can review implementation.md
        if not auto:
            self._emit_status(
                on_chunk,
                "Pipeline paused after generator. Review plans/{feature-name}/implementation.md",
            )
            self._emit_status(
                on_chunk,
                "Type 'continue' to proceed or 'stop' to exit.",
            )
            if request_continue is None or not request_continue("generator"):
                return outputs

        self._emit_status(on_chunk, "\n=== Implementer ===")
        self._emit_status(on_chunk, "Starting implementer phase")

        try:
            implementation_text = impl_file.read_text(encoding="utf-8")
        except OSError:
            implementation_text = generator_output

        implementer_prompt = (
            "Execute the implementation plan below in order. Respect STOP & COMMIT checkpoints and "
            "report progress clearly. Read implementation.md created by previous phases "
            "and execute "
            "accordingly.\n\n"
            f"Implementation guide:\n{implementation_text}"
        )
        step = 1
        while True:
            self._emit_status(on_chunk, f"Implementer step {step} running")
            implementer_output = self.run_role(
                role=ModelRole.IMPLEMENTER,
                prompt=implementer_prompt,
                thread_id=thread_id,
                auto=auto,
                on_chunk=on_chunk,
            )

            if auto or self._contains_token(implementer_output, IMPLEMENTER_DONE_TOKEN):
                self._emit_status(on_chunk, "Implementer phase completed")
                break

            if not self._contains_token(implementer_output, IMPLEMENTER_STOP_TOKEN):
                self._emit_status(on_chunk, "Implementer finished without STOP token")
                break

            if request_continue is None:
                self._emit_status(on_chunk, "Implementer paused: no continue handler")
                break

            if not request_continue("implementer"):
                self._emit_status(on_chunk, "Implementer paused by user")
                break

            implementer_prompt = "continue"
            step += 1

        return outputs

    def run_manual_role(
        self,
        role: ModelRole,
        prompt: str,
        thread_id: str,
        *,
        auto: bool,
        request_continue: Callable[[], bool] | None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> list[Path]:
        outputs: list[Path] = []

        if role is not ModelRole.IMPLEMENTER:
            self._emit_status(on_chunk, f"\n=== {role.value.capitalize()} ===")
            self._emit_status(on_chunk, f"Starting {role.value} phase")
            role_prompt = prompt
            if role is ModelRole.GENERATOR:
                role_prompt = (
                    "Use the existing plan.md generated by planner to produce "
                    "implementation.md in the "
                    "same plans/{feature-name} folder. "
                    "After each step, include a STOP & COMMIT section with this exact sentence: "
                    f"{STOP_AND_COMMIT_SENTENCE}\n\n"
                    f"User request:\n{prompt}"
                )
            before = self._snapshot_plan_artifacts(self.settings.project_root)
            output = self.run_role(
                role=role,
                prompt=role_prompt,
                thread_id=thread_id,
                auto=auto,
                on_chunk=on_chunk,
            )
            self._emit_status(on_chunk, f"{role.value.capitalize()} phase completed")

            after = self._snapshot_plan_artifacts(self.settings.project_root)
            outputs.extend(self._collect_changed_plan_artifacts(before, after))
            if not outputs:
                outputs.extend(
                    self._collect_existing_output_artifacts(output, self.settings.project_root)
                )

            if role is ModelRole.GENERATOR:
                implementation_file = next(
                    (path for path in outputs if path.name == "implementation.md"), None
                )
                implementation_text = output
                if implementation_file is not None:
                    try:
                        implementation_text = implementation_file.read_text(encoding="utf-8")
                    except OSError:
                        pass

                if not self._implementation_has_required_checkpoints(implementation_text):
                    self._emit_status(
                        on_chunk,
                        "Generator output missing required STOP & COMMIT checkpoints. Retrying generator once.",
                    )
                    repair_prompt = (
                        "Regenerate implementation.md with a STOP & COMMIT block after EVERY step. "
                        "Use this exact sentence each time:\n"
                        f"{STOP_AND_COMMIT_SENTENCE}\n\n"
                        f"Original user request:\n{prompt}\n\n"
                        f"Previous invalid implementation.md:\n{implementation_text}"
                    )
                    before_repair = self._snapshot_plan_artifacts(self.settings.project_root)
                    repair_output = self.run_role(
                        role=role,
                        prompt=repair_prompt,
                        thread_id=thread_id,
                        auto=auto,
                        on_chunk=on_chunk,
                    )
                    after_repair = self._snapshot_plan_artifacts(self.settings.project_root)
                    outputs.extend(
                        self._collect_changed_plan_artifacts(before_repair, after_repair)
                    )

                    implementation_file = next(
                        (path for path in outputs if path.name == "implementation.md"),
                        implementation_file,
                    )
                    repaired_text = repair_output
                    if implementation_file is not None:
                        try:
                            repaired_text = implementation_file.read_text(encoding="utf-8")
                        except OSError:
                            pass

                    if not self._implementation_has_required_checkpoints(repaired_text):
                        self._emit_status(
                            on_chunk,
                            "Generator still missing required STOP & COMMIT checkpoints.",
                        )

            return outputs

        self._emit_status(on_chunk, "\n=== Implementer ===")
        self._emit_status(on_chunk, "Starting implementer phase")
        current_prompt = (
            "Read implementation.md generated by generator under plans/{feature-name} "
            "and execute it "
            "in order.\n\n"
            f"User request:\n{prompt}"
        )
        step = 1
        while True:
            self._emit_status(on_chunk, f"Implementer step {step} running")
            output = self.run_role(
                role=ModelRole.IMPLEMENTER,
                prompt=current_prompt,
                thread_id=thread_id,
                auto=auto,
                on_chunk=on_chunk,
            )

            if auto:
                self._emit_status(on_chunk, "Implementer phase completed")
                break
            if self._contains_token(output, IMPLEMENTER_DONE_TOKEN):
                self._emit_status(on_chunk, "Implementer phase completed")
                break
            if not self._contains_token(output, IMPLEMENTER_STOP_TOKEN):
                self._emit_status(on_chunk, "Implementer finished without STOP token")
                break
            if request_continue is None or not request_continue():
                self._emit_status(on_chunk, "Implementer paused by user")
                break

            current_prompt = "continue"
            step += 1

        return outputs

    def run_chat(
        self, prompt: str, thread_id: str, on_chunk: Callable[[str], None] | None = None
    ) -> str:
        chunks = self.stream(prompt=prompt, thread_id=thread_id)
        parts: list[str] = []
        for chunk in chunks:
            if on_chunk:
                on_chunk(f"{DUMP_EVENT_PREFIX}{self._chunk_to_debug_dump(chunk)}")
            text = self._chunk_to_text(chunk)
            parts.append(text)
            if on_chunk:
                on_chunk(text)
        return "".join(parts)


def should_trigger_pipeline(prompt: str) -> bool:
    normalized = prompt.strip().lower()
    if not normalized:
        return False

    greetings = {
        "oi",
        "ola",
        "olá",
        "hello",
        "hi",
        "hey",
        "bom dia",
        "boa tarde",
        "boa noite",
    }
    if normalized in greetings:
        return False

    if len(normalized.split()) <= 3 and any(normalized.startswith(g) for g in greetings):
        return False

    keywords = (
        "feature",
        "implement",
        "implementar",
        "criar",
        "gera",
        "gerar",
        "construir",
        "build",
        "api",
        "fastapi",
        "endpoint",
        "serviço",
        "servico",
        "microserviço",
        "microservico",
        "docker",
        "dockerfile",
        "docker-compose",
        "postgres",
        "redis",
        "sqlalchemy",
        "pydantic",
        "estrutura",
        "pasta",
        "projeto",
        "refactor",
        "bug",
        "fix",
        "teste",
        "test",
        "code",
        "codigo",
        "código",
        "plano",
        "plan",
        "pipeline",
    )
    return any(word in normalized for word in keywords)


def parse_manual_role_command(message: str) -> tuple[ModelRole, str] | None:
    stripped = message.strip()
    if not stripped.startswith("/"):
        return None
    tokens = stripped.split(maxsplit=1)
    role_token = tokens[0][1:].strip().lower()
    prompt = tokens[1].strip() if len(tokens) > 1 else ""
    if not prompt:
        return None

    try:
        role = ModelRole(role_token)
    except ValueError:
        return None
    return role, prompt


def ask_continue_after_checkpoint(phase: str = "implementer") -> bool:
    if phase == "planner":
        prompt_text = (
            "[bold yellow]Planner paused. Review plans/{feature-name}/plan.md, "
            "then type 'continue' to proceed (or 'stop' to finish): [/]"
        )
    elif phase == "generator":
        prompt_text = (
            "[bold yellow]Generator paused. Review plans/{feature-name}/implementation.md, "
            "then type 'continue' to proceed (or 'stop' to finish): [/]"
        )
    else:
        prompt_text = (
            "[bold yellow]Implementer paused. Commit and push your changes, "
            "then type 'continue' to proceed (or 'stop' to finish): [/]"
        )

    while True:
        answer = console.input(prompt_text).strip().lower()
        if answer == "continue":
            return True
        if answer in {"stop", "exit", "quit"}:
            return False
        console.print("Type 'continue' or 'stop'.")


def is_clarification_text(text: str) -> bool:
    normalized = text.lower()
    hints = (
        "perguntas de clarificacao",
        "perguntas de clarificação",
        "clarification questions",
        "clarifying questions",
    )
    return any(hint in normalized for hint in hints)


def build_output_handler(
    verbose: bool, state: OutputStreamState | None = None
) -> Callable[[str], None]:
    def handler(text: str) -> None:
        if text.startswith(STATUS_EVENT_PREFIX):
            message = text[len(STATUS_EVENT_PREFIX) :]
            console.print(f"[bold cyan]{message}[/]")
            return
        if text.startswith(DUMP_EVENT_PREFIX):
            if verbose:
                dump = text[len(DUMP_EVENT_PREFIX) :]
                console.print(f"[dim]{dump}[/]")
            return
        if is_clarification_text(text):
            if state is not None:
                state.clarification_requested = True
            console.print(Panel(text, title="Clarificacao Necessaria", border_style="yellow"))
            return
        if verbose and text:
            console.print(text, end="")

    return handler


def render_mcp_status(mcp_manager: MCPManager) -> None:
    if not mcp_manager.has_servers():
        return
    statuses = mcp_manager.get_server_status()
    online = [name for name, status in statuses.items() if status.online]
    offline = [name for name, status in statuses.items() if not status.online]
    tools = sum(status.tools_count for status in statuses.values())
    resources = sum(status.resources_count for status in statuses.values())

    if online and not offline:
        console.print(
            f"[green]✓ MCP servers: {', '.join(online)} ({tools} tools, {resources} resources)[/]"
        )
        return
    if online and offline:
        online_desc = ", ".join(online)
        offline_desc = ", ".join(f"{name} (offline)" for name in offline)
        console.print(
            "[yellow]⚠ MCP servers: "
            f"{offline_desc}, {online_desc} ({tools} tools) - some features may be limited[/]"
        )
        return
    console.print("[yellow][MCP] All MCP servers offline - skipping MCP context[/]")


def render_mcp_context_update(contextual: ContextBuildResult) -> None:
    if contextual.mcp_sources_used:
        console.print(
            "[cyan][MCP] Added "
            f"{contextual.mcp_resources_available} resources from "
            f"{', '.join(contextual.mcp_sources_used)} server(s)[/]"
        )
        return
    if contextual.mcp_servers_offline:
        console.print("[yellow][MCP] All MCP servers offline - skipping MCP context[/]")


def _runtime_mcp_manager(runtime: Any) -> MCPManager | None:
    manager = getattr(runtime, "mcp_manager", None)
    return manager if isinstance(manager, MCPManager) else None


@contextmanager
def noop_context():  # type: ignore[no-untyped-def]
    yield


def tracing_enabled_context(enabled: bool, project: str | None):  # type: ignore[no-untyped-def]
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"

        @contextmanager
        def disabled() -> Iterator[None]:
            yield

        return disabled()

    os.environ["LANGSMITH_TRACING"] = "true"
    if project:
        os.environ["LANGSMITH_PROJECT"] = project

    from langsmith import tracing_context

    return tracing_context(enabled=True)


@app.command()
def chat(
    prompt: str | None = typer.Option(default=None, help="Prompt to run once."),
    thread_id: str = typer.Option(default="helo-ai-cli-session", help="Thread identifier."),
    auto: bool = typer.Option(
        default=False,
        help="When enabled, runs implementer in fully automatic mode without manual checkpoints.",
    ),
    verbose: bool = typer.Option(
        default=False,
        help="When enabled, prints intermediate streaming logs from model/tool execution.",
    ),
    debug: bool = typer.Option(
        default=False,
        help="When enabled, show full MCP error stack traces (use for debugging).",
    ),
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings, mcp_debug=debug)
    mcp_manager = _runtime_mcp_manager(runtime)
    metrics = ChatSessionMetrics()
    context_window = infer_context_window(settings)

    def execute_single_message(message: str, current_thread_id: str) -> str:
        contextual = build_contextual_prompt(
            message,
            settings.project_root,
            mcp_manager=mcp_manager,
            auto_load_skills=settings.skills_autoload,
        )
        effective_prompt = contextual.prompt
        effective_on_chunk = build_output_handler(verbose)
        selected_role = "chat"

        manual_role = parse_manual_role_command(message)
        if manual_role:
            role, role_prompt = manual_role
            selected_role = role.value
            role_prompt_with_context = build_contextual_prompt(
                role_prompt,
                settings.project_root,
                mcp_manager=mcp_manager,
                auto_load_skills=settings.skills_autoload,
            )
            contextual = role_prompt_with_context
            runtime.run_manual_role(
                role=role,
                prompt=role_prompt_with_context.prompt,
                thread_id=current_thread_id,
                auto=auto,
                request_continue=ask_continue_after_checkpoint if not auto else None,
                on_chunk=effective_on_chunk,
            )
            output = "Execucao de role concluida."
        elif should_trigger_pipeline(message):
            selected_role = "pipeline"
            runtime.run_pipeline(
                prompt=effective_prompt,
                thread_id=current_thread_id,
                auto=auto,
                request_continue=ask_continue_after_checkpoint if not auto else None,
                on_chunk=effective_on_chunk,
            )
            output = "Pipeline concluido."
        else:
            output = runtime.run_chat(
                prompt=effective_prompt,
                thread_id=current_thread_id,
                on_chunk=effective_on_chunk,
            )

        for warning in contextual.warnings:
            console.print(f"[yellow]{warning}[/]")

        consumed_tokens = estimate_token_count(message) + estimate_token_count(output)
        metrics.last_tokens = consumed_tokens
        metrics.total_tokens += consumed_tokens

        if not verbose and output.strip():
            render_assistant_message(
                output,
                ChatInteractionMetadata(
                    role=selected_role,
                    thread_id=current_thread_id,
                    skills_loaded=_skills_for_metadata(runtime, selected_role),
                    sources=contextual.sources,
                    last_tokens=metrics.last_tokens,
                    total_tokens=metrics.total_tokens,
                    context_window=context_window,
                    mcp_sources_used=contextual.mcp_sources_used,
                    mcp_chars_used=contextual.mcp_chars_used,
                    mcp_servers_offline=contextual.mcp_servers_offline,
                    mcp_tools_available=contextual.mcp_tools_available,
                    mcp_resources_available=contextual.mcp_resources_available,
                ),
            )
        return current_thread_id

    with tracing_enabled_context(settings.enable_langsmith, settings.langsmith_project):
        if prompt:
            execute_single_message(prompt, thread_id)
            console.print()
            return

        current_thread_id = thread_id
        session = build_chat_session_prompt(settings.project_root)
        render_chat_header(current_thread_id)
        while True:
            message = session.prompt(
                HTML(
                    "<prompt>╭─ </prompt><symbol>you</symbol><prompt> @ </prompt>"
                    f"<hint>{current_thread_id}</hint>\n"
                    "<prompt>╰─❯ </prompt>"
                )
            ).strip()
            if message.lower() in EXIT_COMMANDS:
                return
            if not message:
                continue

            if is_chat_command(message, "/help"):
                console.print(Panel(parse_help_text(), title="Ajuda", border_style="bright_blue"))
                console.print()
                continue

            if is_chat_command(message, "/clear"):
                console.clear()
                render_chat_header(current_thread_id)
                continue

            if is_chat_command(message, "/reset"):
                current_thread_id = f"helo-ai-cli-session-{uuid4().hex[:8]}"
                metrics.total_tokens = 0
                metrics.last_tokens = 0
                console.clear()
                render_chat_header(current_thread_id)
                console.print(f"Nova sessao iniciada: {current_thread_id}")
                console.print(
                    Panel(
                        f"Nova sessao iniciada: {current_thread_id}",
                        title="Reset",
                        border_style="yellow",
                    )
                )
                continue

            render_user_message(
                message,
                build_contextual_prompt(
                    message,
                    settings.project_root,
                    auto_load_skills=settings.skills_autoload,
                ).sources,
            )
            execute_single_message(message, current_thread_id)
            console.print()


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Single prompt execution."),
    thread_id: str = typer.Option(default="helo-ai-cli-session", help="Thread identifier."),
    auto: bool = typer.Option(
        default=False,
        help="When enabled, runs implementer in fully automatic mode without manual checkpoints.",
    ),
    verbose: bool = typer.Option(
        default=False,
        help="When enabled, prints intermediate streaming logs from model/tool execution.",
    ),
    debug: bool = typer.Option(
        default=False,
        help="Show full MCP error stack traces for debugging.",
    ),
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings, mcp_debug=debug)
    mcp_manager = _runtime_mcp_manager(runtime)
    with tracing_enabled_context(settings.enable_langsmith, settings.langsmith_project):
        on_chunk = build_output_handler(verbose)
        manual_role = parse_manual_role_command(prompt)
        contextual = build_contextual_prompt(
            prompt,
            settings.project_root,
            mcp_manager=mcp_manager,
            auto_load_skills=settings.skills_autoload,
        )
        if manual_role:
            role, role_prompt = manual_role
            runtime.run_manual_role(
                role=role,
                prompt=build_contextual_prompt(
                    role_prompt,
                    settings.project_root,
                    mcp_manager=mcp_manager,
                ).prompt,
                thread_id=thread_id,
                auto=auto,
                request_continue=ask_continue_after_checkpoint if not auto else None,
                on_chunk=on_chunk,
            )
        elif should_trigger_pipeline(prompt):
            runtime.run_pipeline(
                prompt=contextual.prompt,
                thread_id=thread_id,
                auto=auto,
                request_continue=ask_continue_after_checkpoint if not auto else None,
                on_chunk=on_chunk,
            )
        else:
            output = runtime.run_chat(
                prompt=contextual.prompt,
                thread_id=thread_id,
                on_chunk=on_chunk,
            )
            if not verbose and output.strip():
                console.print(output)
        console.print()


@app.command("role")
def role_command(
    role: ModelRole = ROLE_ARGUMENT,
    prompt: str = PROMPT_ARGUMENT,
    thread_id: str = typer.Option(default="helo-ai-cli-session", help="Thread identifier."),
    auto: bool = typer.Option(
        default=False,
        help="When enabled, runs implementer in fully automatic mode without manual checkpoints.",
    ),
    verbose: bool = typer.Option(
        default=False,
        help="When enabled, prints intermediate streaming logs from model/tool execution.",
    ),
    debug: bool = typer.Option(
        default=False,
        help="Show full MCP error stack traces for debugging.",
    ),
    interactive_followup: bool = typer.Option(
        default=True,
        help=(
            "When enabled, keeps role session open to answer clarification questions "
            "in the same thread."
        ),
    ),
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings, mcp_debug=debug)

    def run_once(prompt_text: str) -> OutputStreamState:
        stream_state = OutputStreamState()
        on_chunk = build_output_handler(verbose, stream_state)
        runtime.run_manual_role(
            role=role,
            prompt=prompt_text,
            thread_id=thread_id,
            auto=auto,
            request_continue=ask_continue_after_checkpoint
            if (role is ModelRole.IMPLEMENTER and not auto)
            else None,
            on_chunk=on_chunk,
        )
        return stream_state

    with tracing_enabled_context(settings.enable_langsmith, settings.langsmith_project):
        stream_state = run_once(prompt)

        while interactive_followup and stream_state.clarification_requested:
            answer = console.input(
                "[bold yellow]Responda a clarificacao (/exit para sair): [/]"
            ).strip()
            if answer.lower() in EXIT_COMMANDS:
                break
            stream_state = run_once(answer)

        console.print()


@app.command()
def doctor() -> None:
    settings = get_settings()
    env_file = resolve_env_file()
    checks = {
        "openrouter_api_key": bool(settings.openrouter_api_key),
        "openai_api_key": bool(settings.openai_api_key),
        "openrouter_effective_api_key": bool(settings.openrouter_effective_api_key),
        "dotenv_path": str(env_file) if env_file else None,
        "skills_source": discover_skills_source(settings.project_root, settings.skills_required),
        "langsmith_enabled": settings.enable_langsmith,
        "langsmith_env": os.getenv("LANGSMITH_TRACING", "false"),
        "mcp_servers_configured": sorted(settings.mcp_servers.keys()),
    }
    console.print_json(data=checks)


@app.command()
def models() -> None:
    settings = get_settings()
    payload = {
        "planner": settings.planner.model,
        "generator": settings.generator.model,
        "implementer": settings.implementer.model,
    }
    console.print_json(data=payload)


@app.command()
def skills(action: Annotated[str | None, typer.Argument()] = None) -> None:
    if action not in (None, "list"):
        raise typer.BadParameter("unexpected extra argument")

    settings = get_settings()
    payload = {
        "sources": discover_skills_source(settings.project_root, settings.skills_required),
        "skills_required": settings.skills_required,
        "skills": list_skills(settings.project_root),
    }
    console.print_json(data=payload)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
