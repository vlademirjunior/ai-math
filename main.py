from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import typer
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel

console = Console()
error_console = Console(stderr=True)
app = typer.Typer(help="Deep Agents CLI", no_args_is_help=True)


class ModelRole(StrEnum):
    PLANNER = "planner"
    GENERATOR = "generator"
    IMPLEMENTER = "implementer"


class Provider(StrEnum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LITELLM = "litellm"


class RoleModelConfig(BaseModel):
    provider: Provider
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
            provider=Provider.OLLAMA,
            model="stepfun/step-3.5-flash:free",
            temperature=0.0,
        )
    )

    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    enable_langsmith: bool = False
    langsmith_project: str | None = None

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    thread_id_default: str = "deep-agents-session"
    skills_required: bool = False

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> AppSettings:
        for role in (self.planner, self.generator, self.implementer):
            if role.provider is Provider.OPENROUTER and not self.openrouter_api_key:
                raise ValueError(
                    "openrouter_api_key is required when any role provider is openrouter"
                )
        if self.enable_langsmith and not os.getenv("LANGSMITH_API_KEY"):
            raise ValueError("LANGSMITH_API_KEY is required when enable_langsmith=true")
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


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
                    SecretStr(self._settings.openrouter_api_key)
                    if self._settings.openrouter_api_key
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

    def create_planner_agent(self) -> Any:
        planner_model = self.model_factory.create(self.policy.planner)
        return create_deep_agent(
            model=planner_model,
            backend=self.backend,
            system_prompt=(
                "You are a software engineer. Plan before you execute."
                "Use available tools and be explicit about steps."
            ),
        )

    def stream(self, prompt: str, thread_id: str) -> Iterable[Any]:
        agent = self.create_planner_agent()
        return cast(
            Iterable[Any],
            agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": thread_id}},
            ),
        )


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
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings)

    with tracing_enabled_context(settings.enable_langsmith, settings.langsmith_project):
        if prompt:
            for chunk in runtime.stream(prompt=prompt, thread_id=thread_id):
                console.print(chunk, end="")
            console.print()
            return

        console.print(Panel("Interactive chat mode. Type 'exit' to stop.", title="Deep Agents CLI"))
        while True:
            message = console.input("[bold cyan]> [/]").strip()
            if message.lower() in {"exit", "quit"}:
                return
            if not message:
                continue
            for chunk in runtime.stream(prompt=message, thread_id=thread_id):
                console.print(chunk, end="")
            console.print()


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Single prompt execution."),
    thread_id: str = typer.Option(default="deep-agents-session", help="Thread identifier."),
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings)
    with tracing_enabled_context(settings.enable_langsmith, settings.langsmith_project):
        for chunk in runtime.stream(prompt=prompt, thread_id=thread_id):
            console.print(chunk, end="")
        console.print()


@app.command()
def doctor() -> None:
    settings = get_settings()
    checks = {
        "openrouter_api_key": bool(settings.openrouter_api_key),
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
def skills() -> None:
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
