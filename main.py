from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, cast
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

IMPLEMENTER_STOP_TOKEN = "STOP_FOR_COMMIT"
IMPLEMENTER_DONE_TOKEN = "IMPLEMENTATION_COMPLETE"
STATUS_EVENT_PREFIX = "__STATUS__::"
DUMP_EVENT_PREFIX = "__DUMP__::"
META_CONTEXT_WINDOW_FALLBACK = 128_000
MAX_CONTEXT_FILE_CHARS = 12_000
MAX_DIRECTORY_CONTEXT_ITEMS = 60
MAX_VERBOSE_DUMP_CHARS = 4_000
EXIT_COMMANDS = {"exit", "quit", "/exit"}
CHAT_COMMANDS = {"/planner", "/generator", "/implementer", "/clear", "/reset", "/help", "/exit"}
ROLE_COMMANDS = {"/planner", "/generator", "/implementer"}
ROLE_ARGUMENT = typer.Argument(..., help="Role to execute manually: planner|generator|implementer")
PROMPT_ARGUMENT = typer.Argument(..., help="Role prompt.")


@dataclass(frozen=True)
class ContextBuildResult:
    prompt: str
    sources: list[str]
    warnings: list[str]


@dataclass
class ChatSessionMetrics:
    total_tokens: int = 0
    last_tokens: int = 0


@dataclass(frozen=True)
class ChatInteractionMetadata:
    role: str
    thread_id: str
    skills_loaded: list[str]
    sources: list[str]
    last_tokens: int
    total_tokens: int
    context_window: int


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


def build_contextual_prompt(message: str, project_root: Path) -> ContextBuildResult:
    cleaned_message, refs = extract_context_references(message)
    if not refs:
        return ContextBuildResult(prompt=message, sources=[], warnings=[])

    warnings: list[str] = []
    sources: list[str] = []
    context_blocks: list[str] = []
    seen: set[Path] = set()

    for ref in refs:
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
            context_blocks.append(f"### {rel}\n{file_context}")
            continue
        directory_context = _summarize_directory_for_context(resolved, project_root)
        context_blocks.append(f"### {rel}/\n{directory_context}")

    if not context_blocks:
        return ContextBuildResult(
            prompt=cleaned_message or message, sources=sources, warnings=warnings
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
    return ContextBuildResult(prompt=prompt, sources=sources, warnings=warnings)


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
        "Chat interativo com pipeline e roles manuais", "\n", "Digite /help para atalhos"
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
    meta_panel = Panel(
        Text.from_markup(
            "[bold]Session[/]\n"
            f"{metadata.thread_id}\n\n"
            "[bold]Skills loaded[/]\n"
            f"{skills}\n\n"
            "[bold]Sources used[/]\n"
            f"{sources}\n\n"
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
    def __init__(self, settings: AppSettings, policy: OrchestrationPolicy | None = None) -> None:
        self.settings = settings
        self.policy = policy or OrchestrationPolicy()
        self.model_factory = ModelFactory(settings)
        self.backend = LocalShellBackend(root_dir=str(settings.project_root), virtual_mode=False)
        self.skills = discover_skills_source(
            settings.project_root, required=settings.skills_required
        )

    def _builtin_skill_path(self, role: ModelRole) -> Path:
        skill_name = {
            ModelRole.PLANNER: "planner_skill.md",
            ModelRole.GENERATOR: "generator_skill.md",
            ModelRole.IMPLEMENTER: "implementer_skill.md",
        }[role]
        return self.settings.project_root / "skills_builtin" / skill_name

    def _load_builtin_skill_text(self, role: ModelRole) -> str:
        skill_file = self._builtin_skill_path(role)
        try:
            return skill_file.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

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
                "Include validation checkpoints so downstream phases can execute without ambiguity."
            )
        if role is ModelRole.GENERATOR:
            return (
                "You are the generator role. Convert planner output into a complete implementation "
                "guide with explicit steps, concrete file-level actions, "
                "and verification criteria. "
                "Avoid ambiguity and preserve the planner scope."
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
        if role is ModelRole.IMPLEMENTER:
            parts.append(self._implementer_control_rules(auto))

        skill_text = self._load_builtin_skill_text(role)
        if skill_text:
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
            return AgentRuntime._content_to_text(content) if is_assistant else ""

        if not is_assistant:
            return ""

        return AgentRuntime._content_to_text(getattr(message, "content", None))

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

    def create_role_agent(self, role: ModelRole, auto: bool = False) -> Any:
        role_model = self.model_factory.create(role)
        kwargs: dict[str, Any] = {
            "model": role_model,
            "backend": self.backend,
            "system_prompt": self._system_prompt_for_role(role, auto=auto),
        }
        if self.skills:
            kwargs["skills"] = self.skills
        return create_deep_agent(**kwargs)

    def create_planner_agent(self) -> Any:
        return self.create_role_agent(ModelRole.PLANNER)

    def create_chat_agent(self) -> Any:
        chat_model = self.model_factory.create(self.policy.planner)
        kwargs: dict[str, Any] = {
            "model": chat_model,
            "backend": self.backend,
            "system_prompt": (
                "You are a helpful coding assistant. Reply naturally for casual chat. "
                "For engineering requests, provide practical guidance and concise steps."
            ),
        }
        if self.skills:
            kwargs["skills"] = self.skills
        return create_deep_agent(**kwargs)

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
    def _emit_status(on_chunk: Callable[[str], None] | None, message: str) -> None:
        if on_chunk is not None:
            on_chunk(f"{STATUS_EVENT_PREFIX}{message}")

    def run_pipeline(
        self,
        prompt: str,
        thread_id: str,
        *,
        auto: bool,
        request_continue: Callable[[], bool] | None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> list[Path]:
        self._emit_status(on_chunk, "\n=== Planner ===")
        self._emit_status(on_chunk, "Starting planner phase")
        planner_output = self.run_role(
            role=ModelRole.PLANNER,
            prompt=prompt,
            thread_id=thread_id,
            auto=auto,
            on_chunk=on_chunk,
        )
        self._emit_status(on_chunk, "Planner phase completed")

        self._emit_status(on_chunk, "\n=== Generator ===")
        self._emit_status(on_chunk, "Starting generator phase")
        generator_prompt = (
            "Use the planner output below to generate a complete implementation plan with explicit "
            "verification checkpoints. Create/update files under plans/{feature-name} as needed "
            "according to your role instructions.\n\n"
            f"User request:\n{prompt}\n\nPlanner output:\n{planner_output}"
        )
        generator_output = self.run_role(
            role=ModelRole.GENERATOR,
            prompt=generator_prompt,
            thread_id=thread_id,
            auto=auto,
            on_chunk=on_chunk,
        )
        self._emit_status(on_chunk, "Generator phase completed")

        self._emit_status(on_chunk, "\n=== Implementer ===")
        self._emit_status(on_chunk, "Starting implementer phase")
        implementer_prompt = (
            "Execute the implementation plan below in order. Respect STOP & COMMIT checkpoints and "
            "report progress clearly. Read implementation.md created by previous phases "
            "and execute "
            "accordingly.\n\n"
            f"Implementation guide:\n{generator_output}"
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

            if not request_continue():
                self._emit_status(on_chunk, "Implementer paused by user")
                break

            implementer_prompt = "continue"
            step += 1

        return []

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
        if role is not ModelRole.IMPLEMENTER:
            self._emit_status(on_chunk, f"\n=== {role.value.capitalize()} ===")
            self._emit_status(on_chunk, f"Starting {role.value} phase")
            role_prompt = prompt
            if role is ModelRole.GENERATOR:
                role_prompt = (
                    "Use the existing plan.md generated by planner to produce "
                    "implementation.md in the "
                    "same plans/{feature-name} folder.\n\n"
                    f"User request:\n{prompt}"
                )
            self.run_role(
                role=role,
                prompt=role_prompt,
                thread_id=thread_id,
                auto=auto,
                on_chunk=on_chunk,
            )
            self._emit_status(on_chunk, f"{role.value.capitalize()} phase completed")
            return []

        outputs: list[Path] = []
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


def ask_continue_after_checkpoint() -> bool:
    while True:
        answer = (
            console.input(
                "[bold yellow]Implementer paused. Commit and push your changes, "
                "then type 'continue' to proceed (or 'stop' to finish): [/]"
            )
            .strip()
            .lower()
        )
        if answer == "continue":
            return True
        if answer in {"stop", "exit", "quit"}:
            return False
        console.print("Type 'continue' or 'stop'.")


def build_output_handler(verbose: bool) -> Callable[[str], None]:
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
        if verbose and text:
            console.print(text, end="")

    return handler


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
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings)
    metrics = ChatSessionMetrics()
    context_window = infer_context_window(settings)

    def execute_single_message(message: str, current_thread_id: str) -> str:
        contextual = build_contextual_prompt(message, settings.project_root)
        effective_prompt = contextual.prompt
        effective_on_chunk = build_output_handler(verbose)
        selected_role = "chat"

        manual_role = parse_manual_role_command(message)
        if manual_role:
            role, role_prompt = manual_role
            selected_role = role.value
            role_prompt_with_context = build_contextual_prompt(role_prompt, settings.project_root)
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
                message, build_contextual_prompt(message, settings.project_root).sources
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
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings)
    with tracing_enabled_context(settings.enable_langsmith, settings.langsmith_project):
        on_chunk = build_output_handler(verbose)
        manual_role = parse_manual_role_command(prompt)
        if manual_role:
            role, role_prompt = manual_role
            runtime.run_manual_role(
                role=role,
                prompt=role_prompt,
                thread_id=thread_id,
                auto=auto,
                request_continue=ask_continue_after_checkpoint if not auto else None,
                on_chunk=on_chunk,
            )
        elif should_trigger_pipeline(prompt):
            runtime.run_pipeline(
                prompt=prompt,
                thread_id=thread_id,
                auto=auto,
                request_continue=ask_continue_after_checkpoint if not auto else None,
                on_chunk=on_chunk,
            )
        else:
            output = runtime.run_chat(
                prompt=prompt,
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
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings)
    with tracing_enabled_context(settings.enable_langsmith, settings.langsmith_project):
        on_chunk = build_output_handler(verbose)
        runtime.run_manual_role(
            role=role,
            prompt=prompt,
            thread_id=thread_id,
            auto=auto,
            request_continue=ask_continue_after_checkpoint
            if (role is ModelRole.IMPLEMENTER and not auto)
            else None,
            on_chunk=on_chunk,
        )
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
