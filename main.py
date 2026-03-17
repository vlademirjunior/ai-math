from __future__ import annotations

import os
import re
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

    def _plans_root(self) -> Path:
        root = self.settings.project_root / "plans"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _thread_state_dir(self) -> Path:
        state_dir = self._plans_root() / ".state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9\s_-]", " ", value).strip().lower()
        slug = re.sub(r"[\s_]+", "-", cleaned)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "feature"

    def _feature_from_prompt(self, prompt: str) -> str:
        stripped = prompt.strip()
        if stripped.startswith("/"):
            pieces = stripped.split(maxsplit=1)
            stripped = pieces[1] if len(pieces) > 1 else ""
        return self._slugify(stripped)[:80]

    def _set_thread_feature(self, thread_id: str, feature: str) -> None:
        state_file = self._thread_state_dir() / f"{self._slugify(thread_id)}.txt"
        state_file.write_text(feature, encoding="utf-8")

    def _get_thread_feature(self, thread_id: str) -> str | None:
        state_file = self._thread_state_dir() / f"{self._slugify(thread_id)}.txt"
        if not state_file.exists():
            return None
        try:
            value = state_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    def _feature_dir(self, feature: str) -> Path:
        feature_dir = self._plans_root() / self._slugify(feature)
        feature_dir.mkdir(parents=True, exist_ok=True)
        return feature_dir

    def _plan_file(self, feature: str) -> Path:
        return self._feature_dir(feature) / "plan.md"

    def _implementation_file(self, feature: str) -> Path:
        return self._feature_dir(feature) / "implementation.md"

    def _latest_feature_with_file(self, filename: str) -> str | None:
        candidates: list[tuple[float, str]] = []
        for item in self._plans_root().iterdir():
            if not item.is_dir() or item.name.startswith("."):
                continue
            file_path = item / filename
            if file_path.exists() and file_path.is_file():
                candidates.append((file_path.stat().st_mtime, item.name))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates[0][1]

    def _resolve_feature(
        self, thread_id: str, prompt: str, *, require_file: str | None = None
    ) -> str:
        candidate = self._get_thread_feature(thread_id) or self._feature_from_prompt(prompt)

        if require_file is None:
            return candidate

        candidate_file = self._feature_dir(candidate) / require_file
        if candidate_file.exists():
            return candidate

        latest = self._latest_feature_with_file(require_file)
        if latest:
            self._set_thread_feature(thread_id, latest)
            return latest

        raise FileNotFoundError(f"No {require_file} found under plans/. Run planner first.")

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
            return str(chunk)
        return str(chunk)

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

    def run_pipeline(
        self,
        prompt: str,
        thread_id: str,
        *,
        auto: bool,
        request_continue: Callable[[], bool] | None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> list[Path]:
        generated_files: list[Path] = []

        feature = self._feature_from_prompt(prompt)
        self._set_thread_feature(thread_id, feature)

        planner_output = self.run_role(
            role=ModelRole.PLANNER,
            prompt=prompt,
            thread_id=thread_id,
            auto=auto,
            on_chunk=on_chunk,
        )
        plan_path = self._plan_file(feature)
        plan_path.write_text(planner_output, encoding="utf-8")
        generated_files.append(plan_path)

        generator_prompt = (
            "Use the planner output below to generate a complete implementation plan with explicit "
            "verification checkpoints.\n\n"
            f"User request:\n{prompt}\n\nPlanner output:\n{planner_output}"
        )
        generator_output = self.run_role(
            role=ModelRole.GENERATOR,
            prompt=generator_prompt,
            thread_id=thread_id,
            auto=auto,
            on_chunk=on_chunk,
        )
        implementation_path = self._implementation_file(feature)
        implementation_path.write_text(generator_output, encoding="utf-8")
        generated_files.append(implementation_path)

        implementer_prompt = (
            "Execute the implementation plan below in order. Respect STOP & COMMIT checkpoints and "
            "report progress clearly.\n\n"
            f"Implementation guide from {implementation_path}:\n{generator_output}"
        )
        while True:
            implementer_output = self.run_role(
                role=ModelRole.IMPLEMENTER,
                prompt=implementer_prompt,
                thread_id=thread_id,
                auto=auto,
                on_chunk=on_chunk,
            )

            if auto or self._contains_token(implementer_output, IMPLEMENTER_DONE_TOKEN):
                break

            if not self._contains_token(implementer_output, IMPLEMENTER_STOP_TOKEN):
                break

            if request_continue is None:
                break

            if not request_continue():
                break

            implementer_prompt = "continue"

        return generated_files

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
        feature = self._resolve_feature(
            thread_id,
            prompt,
            require_file="plan.md" if role is ModelRole.GENERATOR else None,
        )
        if role is ModelRole.PLANNER:
            output = self.run_role(
                role=role,
                prompt=prompt,
                thread_id=thread_id,
                auto=auto,
                on_chunk=on_chunk,
            )
            self._set_thread_feature(thread_id, feature)
            plan_path = self._plan_file(feature)
            plan_path.write_text(output, encoding="utf-8")
            return [plan_path]

        if role is ModelRole.GENERATOR:
            plan_path = self._plan_file(feature)
            if not plan_path.exists():
                raise FileNotFoundError("No plan.md found. Run planner first.")
            plan_text = plan_path.read_text(encoding="utf-8")
            generator_prompt = (
                "Generate implementation.md from the plan below.\n\n"
                f"User request:\n{prompt}\n\nPlan file ({plan_path}):\n{plan_text}"
            )
            output = self.run_role(
                role=role,
                prompt=generator_prompt,
                thread_id=thread_id,
                auto=auto,
                on_chunk=on_chunk,
            )
            implementation_path = self._implementation_file(feature)
            implementation_path.write_text(output, encoding="utf-8")
            self._set_thread_feature(thread_id, feature)
            return [implementation_path]

        implementation_feature = self._resolve_feature(
            thread_id, prompt, require_file="implementation.md"
        )
        implementation_path = self._implementation_file(implementation_feature)
        implementation_text = implementation_path.read_text(encoding="utf-8")

        outputs: list[Path] = [implementation_path]
        current_prompt = (
            "Execute the implementation guide below in order.\n\n"
            f"User request:\n{prompt}\n\nImplementation guide ({implementation_path}):\n"
            f"{implementation_text}"
        )
        while True:
            output = self.run_role(
                role=ModelRole.IMPLEMENTER,
                prompt=current_prompt,
                thread_id=thread_id,
                auto=auto,
                on_chunk=on_chunk,
            )

            if auto:
                break
            if self._contains_token(output, IMPLEMENTER_DONE_TOKEN):
                break
            if not self._contains_token(output, IMPLEMENTER_STOP_TOKEN):
                break
            if request_continue is None or not request_continue():
                break

            current_prompt = "continue"

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
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings)

    with tracing_enabled_context(settings.enable_langsmith, settings.langsmith_project):
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
                    on_chunk=lambda t: console.print(t, end=""),
                )
            elif should_trigger_pipeline(prompt):
                runtime.run_pipeline(
                    prompt=prompt,
                    thread_id=thread_id,
                    auto=auto,
                    request_continue=ask_continue_after_checkpoint if not auto else None,
                    on_chunk=lambda t: console.print(t, end=""),
                )
            else:
                runtime.run_chat(
                    prompt=prompt,
                    thread_id=thread_id,
                    on_chunk=lambda t: console.print(t, end=""),
                )
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
                    on_chunk=lambda t: console.print(t, end=""),
                )
            elif should_trigger_pipeline(message):
                runtime.run_pipeline(
                    prompt=message,
                    thread_id=thread_id,
                    auto=auto,
                    request_continue=ask_continue_after_checkpoint if not auto else None,
                    on_chunk=lambda t: console.print(t, end=""),
                )
            else:
                runtime.run_chat(
                    prompt=message,
                    thread_id=thread_id,
                    on_chunk=lambda t: console.print(t, end=""),
                )
            console.print()


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Single prompt execution."),
    thread_id: str = typer.Option(default="deep-agents-session", help="Thread identifier."),
    auto: bool = typer.Option(
        default=False,
        help="When enabled, runs implementer in fully automatic mode without manual checkpoints.",
    ),
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings)
    with tracing_enabled_context(settings.enable_langsmith, settings.langsmith_project):
        manual_role = parse_manual_role_command(prompt)
        if manual_role:
            role, role_prompt = manual_role
            runtime.run_manual_role(
                role=role,
                prompt=role_prompt,
                thread_id=thread_id,
                auto=auto,
                request_continue=ask_continue_after_checkpoint if not auto else None,
                on_chunk=lambda t: console.print(t, end=""),
            )
        elif should_trigger_pipeline(prompt):
            runtime.run_pipeline(
                prompt=prompt,
                thread_id=thread_id,
                auto=auto,
                request_continue=ask_continue_after_checkpoint if not auto else None,
                on_chunk=lambda t: console.print(t, end=""),
            )
        else:
            runtime.run_chat(
                prompt=prompt,
                thread_id=thread_id,
                on_chunk=lambda t: console.print(t, end=""),
            )
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
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings)
    with tracing_enabled_context(settings.enable_langsmith, settings.langsmith_project):
        runtime.run_manual_role(
            role=role,
            prompt=prompt,
            thread_id=thread_id,
            auto=auto,
            request_continue=ask_continue_after_checkpoint
            if (role is ModelRole.IMPLEMENTER and not auto)
            else None,
            on_chunk=lambda t: console.print(t, end=""),
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
