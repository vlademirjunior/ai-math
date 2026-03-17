from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel

console = Console()
error_console = Console(stderr=True)
app = typer.Typer(help="Helo", no_args_is_help=True)

IMPLEMENTER_STOP_TOKEN = "STOP_FOR_COMMIT"
IMPLEMENTER_DONE_TOKEN = "IMPLEMENTATION_COMPLETE"
STATUS_EVENT_PREFIX = "__STATUS__::"


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
    thread_id_default: str = "deep-agents-session"
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
                    "openrouter_api_key or openai_api_key is required when any role provider is openrouter"
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
    return AppSettings(_env_file=env_file if env_file else None)


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
                "guide with explicit steps, concrete file-level actions, and verification criteria. "
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
                "AUTO MODE: execute all remaining implementation steps continuously without waiting "
                "for user intervention. Do not emit manual checkpoint pauses. At final completion, "
                f"include the exact token {IMPLEMENTER_DONE_TOKEN}."
            )
        return (
            "MANUAL CHECKPOINT MODE: after each STOP & COMMIT checkpoint, stop execution and include "
            f"the exact token {IMPLEMENTER_STOP_TOKEN} in the response. Then ask the user to review, "
            "commit, and push manually. Resume only after receiving 'continue'. At final completion, "
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
            "report progress clearly. Read implementation.md created by previous phases and execute "
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
                    "Use the existing plan.md generated by planner to produce implementation.md in the "
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
            "Read implementation.md generated by generator under plans/{feature-name} and execute it "
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
    thread_id: str = typer.Option(default="deep-agents-session", help="Thread identifier."),
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
        if prompt:
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
            return

        console.print(Panel("Interactive chat mode. Type 'exit' to stop.", title="Helo"))
        while True:
            message = console.input("[bold cyan]> [/]").strip()
            if message.lower() in {"exit", "quit"}:
                return
            if not message:
                continue

            manual_role = parse_manual_role_command(message)
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
            elif should_trigger_pipeline(message):
                runtime.run_pipeline(
                    prompt=message,
                    thread_id=thread_id,
                    auto=auto,
                    request_continue=ask_continue_after_checkpoint if not auto else None,
                    on_chunk=on_chunk,
                )
            else:
                output = runtime.run_chat(
                    prompt=message,
                    thread_id=thread_id,
                    on_chunk=on_chunk,
                )
                if not verbose and output.strip():
                    console.print(output)
            console.print()


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Single prompt execution."),
    thread_id: str = typer.Option(default="deep-agents-session", help="Thread identifier."),
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
    role: ModelRole = typer.Argument(
        ..., help="Role to execute manually: planner|generator|implementer"
    ),
    prompt: str = typer.Argument(..., help="Role prompt."),
    thread_id: str = typer.Option(default="deep-agents-session", help="Thread identifier."),
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
    }
    console.print_json(data=payload)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
