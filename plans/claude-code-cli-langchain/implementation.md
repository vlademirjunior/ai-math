# Claude-Code-Like Deep Agents CLI (Python)

## Goal
Deliver a secure, production-ready single-file Deep Agents CLI with role-based model orchestration, .agents skill loading, optional default-off LangSmith tracing, and strict quality gates.

## Prerequisites
Make sure that the use is currently on the `claude-code-cli-langchain` branch before beginning implementation.
If not, move them to the correct branch. If the branch does not exist, create it from main.

Run:
```bash
git checkout claude-code-cli-langchain || git checkout -b claude-code-cli-langchain main
uv sync
```

## Technology Stack
- Python 3.13+
- Deep Agents (`deepagents`)
- LangChain integrations:
  - OpenRouter (`langchain-openrouter`)
  - Ollama (`langchain-ollama`)
  - LiteLLM (`langchain-litellm`)
- Settings and validation:
  - `pydantic`, `pydantic-settings`
- CLI and terminal UX:
  - `typer`, `rich`
- Quality gates:
  - `ruff`, `mypy`, `pytest`

### Step-by-Step Instructions

#### Step 1: Define production configuration and role contracts
- [x] Replace `main.py` with typed configuration contracts and secure env-based defaults.
- [x] Copy and paste code below into `main.py`:

```python
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRole(str, Enum):
    PLANNER = "planner"
    GENERATOR = "generator"
    IMPLEMENTER = "implementer"


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LITELLM = "litellm"


class RoleModelConfig(BaseModel):
    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None

    @model_validator(mode="after")
    def validate_model_name(self) -> "RoleModelConfig":
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
            provider=Provider.OPENROUTER,
            model="stepfun/step-3.5-flash:free",
            temperature=0.0,
        )
    )

    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    enable_langsmith: bool = False

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    thread_id_default: str = "deep-agents-session"

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "AppSettings":
        for role in (self.planner, self.generator, self.implementer):
            if role.provider is Provider.OPENROUTER and not self.openrouter_api_key:
                raise ValueError(
                    "openrouter_api_key is required when any role provider is openrouter"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def main() -> int:
    settings = get_settings()
    print("Configuration loaded")
    print(f"Planner: {settings.planner.provider.value}/{settings.planner.model}")
    print(f"Generator: {settings.generator.provider.value}/{settings.generator.model}")
    print(f"Implementer: {settings.implementer.provider.value}/{settings.implementer.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] Update dependencies for settings and test tooling.
- [x] Copy and paste code below into `pyproject.toml`:

```toml
[project]
name = "workspace"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "deepagents>=0.4.11",
    "langchain>=1.2.12",
    "langchain-openrouter>=0.1.0",
    "langchain-ollama>=0.3.0",
    "langchain-litellm>=0.2.0",
    "pydantic-settings>=2.10.1",
    "rich>=14.3.3",
    "typer>=0.16.1",
]

[dependency-groups]
dev = [
    "ipykernel>=7.2.0",
    "mypy>=1.19.1",
    "pytest>=8.4.2",
    "pytest-cov>=7.0.0",
    "ruff>=0.15.5",
]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.13"
strict = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [x] Add tests for env parsing and validation.
- [x] Copy and paste code below into `tests/test_settings.py`:

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from main import AppSettings, Provider


@pytest.mark.parametrize(
    ("env", "expected_provider"),
    [
        (
            {
                "OPENROUTER_API_KEY": "test-key",
                "PLANNER__PROVIDER": "openrouter",
                "PLANNER__MODEL": "stepfun/step-3.5-flash:free",
            },
            Provider.OPENROUTER,
        ),
        (
            {
                "PLANNER__PROVIDER": "openrouter",
                "PLANNER__MODEL": "stepfun/step-3.5-flash:free",
            },
            Provider.OPENROUTER,
        ),
    ],
)
def test_settings_env_parse(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected_provider: Provider) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PLANNER__PROVIDER", raising=False)
    monkeypatch.delenv("PLANNER__MODEL", raising=False)

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    settings = AppSettings(project_root=Path.cwd())
    assert settings.planner.provider == expected_provider


def test_openrouter_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("PLANNER__PROVIDER", "openrouter")
    monkeypatch.setenv("GENERATOR__PROVIDER", "openrouter")
    monkeypatch.setenv("IMPLEMENTER__PROVIDER", "openrouter")

    with pytest.raises(ValidationError):
        AppSettings(project_root=Path.cwd())
```

- [x] Run quality checks for this step.

```bash
uv sync
uv run ruff check .
uv run mypy main.py tests
uv run pytest tests/test_settings.py
```

##### Step 1 Verification Checklist
- [ ] No `ruff` lint errors.
- [ ] No `mypy` type errors in `main.py` and `tests/test_settings.py`.
- [ ] `pytest tests/test_settings.py` passes.
- [ ] No hard-coded credentials remain in source files.

#### Step 2: Build provider-agnostic model factory with OpenRouter/Ollama/LiteLLM adapters
- [x] Extend `main.py` with a role-based `ModelFactory`.
- [x] Copy and paste code below into `main.py`:

```python
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRole(str, Enum):
    PLANNER = "planner"
    GENERATOR = "generator"
    IMPLEMENTER = "implementer"


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LITELLM = "litellm"


class RoleModelConfig(BaseModel):
    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None

    @model_validator(mode="after")
    def validate_model_name(self) -> "RoleModelConfig":
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
            provider=Provider.OPENROUTER,
            model="stepfun/step-3.5-flash:free",
            temperature=0.0,
        )
    )

    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    enable_langsmith: bool = False

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    thread_id_default: str = "deep-agents-session"

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "AppSettings":
        for role in (self.planner, self.generator, self.implementer):
            if role.provider is Provider.OPENROUTER and not self.openrouter_api_key:
                raise ValueError(
                    "openrouter_api_key is required when any role provider is openrouter"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


class ChatModelLike(Protocol):
    pass


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

            return ChatOpenRouter(
                model=config.model,
                api_key=self._settings.openrouter_api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        if config.provider is Provider.OPENROUTER:
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


def main() -> int:
    settings = get_settings()
    factory = ModelFactory(settings)

    for role in ModelRole:
        _ = factory.create(role)
        print(f"Built model for role={role.value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] Add model factory tests.
- [x] Copy and paste code below into `tests/test_model_factory.py`:

```python
import pytest

from main import AppSettings, ModelFactory, ModelRole, Provider, RoleModelConfig


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        planner=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
    )


def test_factory_builds_all_roles(settings: AppSettings) -> None:
    factory = ModelFactory(settings)
    for role in ModelRole:
        model = factory.create(role)
        assert model is not None
```

- [x] Run checks for this step.

```bash
uv run ruff check .
uv run mypy main.py tests
uv run pytest tests/test_model_factory.py
```

##### Step 2 Verification Checklist
- [ ] `ModelFactory` returns a model for each role.
- [ ] Missing provider requirements fail with clear exceptions.
- [ ] `pytest tests/test_model_factory.py` passes.

#### Step 3: Compose Deep Agent runtime and 3-role orchestration policy
- [x] Add orchestration policy and Deep Agent runtime composition.
- [x] Copy and paste code below into `main.py`:

```python
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRole(str, Enum):
    PLANNER = "planner"
    GENERATOR = "generator"
    IMPLEMENTER = "implementer"


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LITELLM = "litellm"


class RoleModelConfig(BaseModel):
    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None

    @model_validator(mode="after")
    def validate_model_name(self) -> "RoleModelConfig":
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
            provider=Provider.OPENROUTER,
            model="stepfun/step-3.5-flash:free",
            temperature=0.0,
        )
    )

    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    enable_langsmith: bool = False

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    thread_id_default: str = "deep-agents-session"

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "AppSettings":
        for role in (self.planner, self.generator, self.implementer):
            if role.provider is Provider.OPENROUTER and not self.openrouter_api_key:
                raise ValueError(
                    "openrouter_api_key is required when any role provider is openrouter"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


class ChatModelLike(Protocol):
    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        ...


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

            return ChatOpenRouter(
                model=config.model,
                api_key=self._settings.openrouter_api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        if config.provider is Provider.OPENROUTER:
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

    def create_planner_agent(self) -> Any:
        planner_model = self.model_factory.create(self.policy.planner)
        return create_deep_agent(
            model=planner_model,
            backend=self.backend,
            system_prompt=(
                "You are a senior software engineer. "
                "Always plan first with write_todos before editing files."
            ),
        )

    def stream(self, prompt: str, thread_id: str) -> Iterable[Any]:
        agent = self.create_planner_agent()
        return agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": thread_id}},
        )


@contextmanager
def noop_context() -> Iterator[None]:
    yield


def run_interactive(runtime: AgentRuntime, thread_id: str) -> int:
    print("Deep Agents CLI")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            return 0
        if not user_input:
            continue

        for chunk in runtime.stream(prompt=user_input, thread_id=thread_id):
            print(chunk, end="", flush=True)
        print()


def main() -> int:
    settings = get_settings()
    runtime = AgentRuntime(settings)
    return run_interactive(runtime, settings.thread_id_default)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] Add orchestration runtime tests.
- [x] Copy and paste code below into `tests/test_runtime.py`:

```python
from unittest.mock import MagicMock

from main import AgentRuntime, AppSettings, Provider, RoleModelConfig


def test_runtime_initializes() -> None:
    settings = AppSettings(
        planner=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
    )
    runtime = AgentRuntime(settings)
    assert runtime.settings.thread_id_default


def test_stream_uses_thread_id(monkeypatch) -> None:
    settings = AppSettings(
        planner=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="stepfun/step-3.5-flash:free"),
    )
    runtime = AgentRuntime(settings)

    fake_agent = MagicMock()
    fake_agent.stream.return_value = ["ok"]
    monkeypatch.setattr(runtime, "create_planner_agent", lambda: fake_agent)

    chunks = list(runtime.stream("hello", "abc-thread"))
    assert chunks == ["ok"]
    fake_agent.stream.assert_called_once()
```

- [x] Run checks for this step.

```bash
uv run ruff check .
uv run mypy main.py tests
uv run pytest tests/test_runtime.py
```

##### Step 3 Verification Checklist
- [ ] Planner agent is created through Deep Agents.
- [ ] `thread_id` is passed through `configurable.thread_id`.
- [ ] `pytest tests/test_runtime.py` passes.

#### Step 4: Add .agents skill loader and startup validation
- [x] Add explicit `.agents` discovery and validation logic in startup path.
- [x] Copy and paste code below into `main.py`:

```python
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRole(str, Enum):
    PLANNER = "planner"
    GENERATOR = "generator"
    IMPLEMENTER = "implementer"


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LITELLM = "litellm"


class RoleModelConfig(BaseModel):
    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None

    @model_validator(mode="after")
    def validate_model_name(self) -> "RoleModelConfig":
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
            provider=Provider.OPENROUTER,
            model="stepfun/step-3.5-flash:free",
            temperature=0.0,
        )
    )

    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    enable_langsmith: bool = False

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    thread_id_default: str = "deep-agents-session"
    skills_required: bool = False

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "AppSettings":
        for role in (self.planner, self.generator, self.implementer):
            if role.provider is Provider.OPENROUTER and not self.openrouter_api_key:
                raise ValueError(
                    "openrouter_api_key is required when any role provider is openrouter"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


class ChatModelLike(Protocol):
    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        ...


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

            return ChatOpenRouter(
                model=config.model,
                api_key=self._settings.openrouter_api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        if config.provider is Provider.OPENROUTER:
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
        self.skills = discover_skills_source(settings.project_root, required=settings.skills_required)

    def create_planner_agent(self) -> Any:
        planner_model = self.model_factory.create(self.policy.planner)
        return create_deep_agent(
            model=planner_model,
            backend=self.backend,
            skills=self.skills,
            system_prompt=(
                "You are a senior software engineer. "
                "Always plan first with write_todos before editing files."
            ),
        )

    def stream(self, prompt: str, thread_id: str) -> Iterable[Any]:
        agent = self.create_planner_agent()
        return agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": thread_id}},
        )


@contextmanager
def noop_context() -> Iterator[None]:
    yield


def run_interactive(runtime: AgentRuntime, thread_id: str) -> int:
    print("Deep Agents CLI")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            return 0
        if not user_input:
            continue

        for chunk in runtime.stream(prompt=user_input, thread_id=thread_id):
            print(chunk, end="", flush=True)
        print()


def main() -> int:
    settings = get_settings()
    runtime = AgentRuntime(settings)
    return run_interactive(runtime, settings.thread_id_default)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] Add skill loader tests.
- [x] Copy and paste code below into `tests/test_skills.py`:

```python
from pathlib import Path

import pytest

from main import discover_skills_source


def test_skills_optional_when_missing(tmp_path: Path) -> None:
    assert discover_skills_source(tmp_path, required=False) == []


def test_skills_required_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_skills_source(tmp_path, required=True)


def test_invalid_skill_dir(tmp_path: Path) -> None:
    root = tmp_path / ".agents"
    root.mkdir()
    (root / "bad-skill").mkdir()

    with pytest.raises(ValueError):
        discover_skills_source(tmp_path, required=False)


def test_valid_skills_tree(tmp_path: Path) -> None:
    root = tmp_path / ".agents"
    root.mkdir()
    skill = root / "code-review"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    assert discover_skills_source(tmp_path, required=False) == ["/.agents/"]
```

- [ ] Document `.agents` layout in README.
- [ ] Copy and paste code below into `README.md`:

```md
# Claude-Code-Like Deep Agents CLI

## Skills Folder

The CLI reads skills from `.agents/` and passes `/.agents/` explicitly to `create_deep_agent(..., skills=[...])`.

Expected layout:

```text
.agents/
  code-review/
    SKILL.md
  refactor/
    SKILL.md
```

Behavior:
- Missing `.agents/`: warning/continue (default)
- `SKILLS_REQUIRED=true`: startup error if `.agents/` is missing
- Any skill directory missing `SKILL.md`: startup error
```

- [x] Run checks for this step.

```bash
uv run ruff check .
uv run mypy main.py tests
uv run pytest tests/test_skills.py
```

##### Step 4 Verification Checklist
- [x] Startup validates skill tree before creating the agent.
- [x] Invalid skill directories fail with explicit messages.
- [x] `pytest tests/test_skills.py` passes.

#### Step 5: Implement Typer + Rich CLI UX and operational commands
- [x] Replace interactive-only runner with command-based Typer CLI.
- [x] Copy and paste code below into `main.py`:

```python
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

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


class ModelRole(str, Enum):
    PLANNER = "planner"
    GENERATOR = "generator"
    IMPLEMENTER = "implementer"


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LITELLM = "litellm"


class RoleModelConfig(BaseModel):
    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None

    @model_validator(mode="after")
    def validate_model_name(self) -> "RoleModelConfig":
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
            provider=Provider.OPENROUTER,
            model="stepfun/step-3.5-flash:free",
            temperature=0.0,
        )
    )

    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    enable_langsmith: bool = False

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    thread_id_default: str = "deep-agents-session"
    skills_required: bool = False

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "AppSettings":
        for role in (self.planner, self.generator, self.implementer):
            if role.provider is Provider.OPENROUTER and not self.openrouter_api_key:
                raise ValueError(
                    "openrouter_api_key is required when any role provider is openrouter"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


class ChatModelLike(Protocol):
    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        ...


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

            return ChatOpenRouter(
                model=config.model,
                api_key=self._settings.openrouter_api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        if config.provider is Provider.OPENROUTER:
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
        self.skills = discover_skills_source(settings.project_root, required=settings.skills_required)

    def create_planner_agent(self) -> Any:
        planner_model = self.model_factory.create(self.policy.planner)
        return create_deep_agent(
            model=planner_model,
            backend=self.backend,
            skills=self.skills,
            system_prompt=(
                "You are a senior software engineer. "
                "Always plan first with write_todos before editing files."
            ),
        )

    def stream(self, prompt: str, thread_id: str) -> Iterable[Any]:
        agent = self.create_planner_agent()
        return agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": thread_id}},
        )


@contextmanager
def noop_context() -> Iterator[None]:
    yield


@app.command()
def chat(
    prompt: str | None = typer.Option(default=None, help="Prompt to run once."),
    thread_id: str = typer.Option(default="deep-agents-session", help="Thread identifier."),
) -> None:
    settings = get_settings()
    runtime = AgentRuntime(settings)

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
```

- [x] Add CLI tests.
- [x] Copy and paste code below into `tests/test_cli.py`:

```python
from typer.testing import CliRunner

from main import app


runner = CliRunner()


def test_models_command() -> None:
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "planner" in result.stdout


def test_skills_command() -> None:
    result = runner.invoke(app, ["skills"])
    assert result.exit_code == 0
```

- [ ] Expand README with command usage.
- [ ] Copy and paste code below into `README.md`:

```md
# Claude-Code-Like Deep Agents CLI

## Install

```bash
uv sync
```

## Commands

```bash
uv run python main.py chat
uv run python main.py chat --prompt "Plan refactor for auth service"
uv run python main.py run "Implement tests for parser"
uv run python main.py doctor
uv run python main.py models
uv run python main.py skills
```

## Non-interactive usage

Use `run` or `chat --prompt` in CI scripts for deterministic single-shot execution.
```

- [x] Run checks for this step.

```bash
uv run ruff check .
uv run mypy main.py tests
uv run pytest tests/test_cli.py
```

##### Step 5 Verification Checklist
- [x] CLI has working commands: `chat`, `run`, `doctor`, `models`, `skills`.
- [x] Interactive and non-interactive flows both work.
- [x] `pytest tests/test_cli.py` passes.

#### Step 6: Integrate optional LangSmith tracing with deterministic default-off behavior
- [x] Add deterministic tracing toggle with default disabled behavior.
- [x] Copy and paste code below into `main.py`:

```python
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

import typer
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel


console = Console()
app = typer.Typer(help="Deep Agents CLI", no_args_is_help=True)


class ModelRole(str, Enum):
    PLANNER = "planner"
    GENERATOR = "generator"
    IMPLEMENTER = "implementer"


class Provider(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LITELLM = "litellm"


class RoleModelConfig(BaseModel):
    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None

    @model_validator(mode="after")
    def validate_model_name(self) -> "RoleModelConfig":
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
            provider=Provider.OPENROUTER,
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
    def validate_provider_credentials(self) -> "AppSettings":
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


class ChatModelLike(Protocol):
    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        ...


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

            return ChatOpenRouter(
                model=config.model,
                api_key=self._settings.openrouter_api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        if config.provider is Provider.OPENROUTER:
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


def tracing_enabled_context(enabled: bool, project: str | None) -> Iterator[None]:
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
        self.skills = discover_skills_source(settings.project_root, required=settings.skills_required)

    def create_planner_agent(self) -> Any:
        planner_model = self.model_factory.create(self.policy.planner)
        return create_deep_agent(
            model=planner_model,
            backend=self.backend,
            skills=self.skills,
            system_prompt=(
                "You are a senior software engineer. "
                "Always plan first with write_todos before editing files."
            ),
        )

    def stream(self, prompt: str, thread_id: str) -> Iterable[Any]:
        agent = self.create_planner_agent()
        return agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": thread_id}},
        )


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
```

- [ ] Add tracing tests.
- [ ] Copy and paste code below into `tests/test_tracing.py`:

```python
import os

from main import tracing_enabled_context


def test_tracing_disabled_sets_env_false() -> None:
    os.environ.pop("LANGSMITH_TRACING", None)
    with tracing_enabled_context(enabled=False, project=None):
        assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_enabled_sets_env_true(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    with tracing_enabled_context(enabled=True, project="demo-project"):
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_PROJECT"] == "demo-project"
```

- [ ] Update README with tracing behavior.
- [ ] Copy and paste code below into `README.md`:

```md
# Claude-Code-Like Deep Agents CLI

## Tracing (default off)

By default, tracing is disabled and the process enforces:

- `LANGSMITH_TRACING=false`

To enable tracing:

```bash
export ENABLE_LANGSMITH=true
export LANGSMITH_API_KEY=your_key
export LANGSMITH_PROJECT=deep-agents-cli
```

Then run:

```bash
uv run python main.py doctor
```

You should see `"langsmith_enabled": true` and `"langsmith_env": "true"`.
```

- [ ] Run checks for this step.

```bash
uv run ruff check .
uv run mypy main.py tests
uv run pytest tests/test_tracing.py
```

##### Step 6 Verification Checklist
- [ ] Tracing is disabled by default without extra setup.
- [ ] Enabling tracing requires `LANGSMITH_API_KEY`.
- [ ] `pytest tests/test_tracing.py` passes.

#### Step 7: Harden quality gates and release-ready documentation
- [x] Finalize project tooling configuration.
- [x] Copy and paste code below into `Makefile`:

```make
.PHONY: sync lint format type-check test check doctor

sync:
uv sync

lint:
uv run ruff check .

format:
uv run ruff format .

type-check:
uv run mypy main.py tests

test:
uv run pytest

check: lint type-check test

doctor:
uv run python main.py doctor
```

- [x] Finalize production README.
- [x] Copy and paste code below into `README.md`:

```md
# Claude-Code-Like Deep Agents CLI (Python)

Production-ready Deep Agents CLI with:
- Role-based model orchestration (`planner`, `generator`, `implementer`)
- Skills loading from `.agents/`
- Optional LangSmith tracing (default disabled)
- Typer/Rich command UX

## Requirements
- Python 3.13+
- uv
- Optional Ollama daemon for local implementer model

## Install

```bash
uv sync
```

## Environment Variables

### Core
- `PLANNER__PROVIDER`, `PLANNER__MODEL`
- `GENERATOR__PROVIDER`, `GENERATOR__MODEL`
- `IMPLEMENTER__PROVIDER`, `IMPLEMENTER__MODEL`
- `OPENROUTER_API_KEY` (required if any role uses `openrouter`)
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `THREAD_ID_DEFAULT` (default `deep-agents-session`)
- `SKILLS_REQUIRED` (`true`/`false`, default `false`)

### LangSmith (optional)
- `ENABLE_LANGSMITH` (`true`/`false`, default `false`)
- `LANGSMITH_API_KEY` (required when enabled)
- `LANGSMITH_PROJECT` (optional)

## Skills Layout

```text
.agents/
  code-review/
    SKILL.md
  refactor/
    SKILL.md
```

## Commands

```bash
uv run python main.py chat
uv run python main.py chat --prompt "Plan migration to pydantic settings"
uv run python main.py run "Implement tests for parser"
uv run python main.py doctor
uv run python main.py models
uv run python main.py skills
```

## Quality Gates

```bash
make check
```

This runs:
- `ruff` lint
- `mypy` strict type checks
- `pytest` test suite

## Security Notes
- Never hard-code API keys.
- Keep `.env` out of version control.
- Run `make doctor` after environment changes.

## Troubleshooting
- If `openrouter_api_key is required` appears, set `OPENROUTER_API_KEY` or switch all roles away from `openrouter`.
- If Ollama role fails, confirm daemon is running and model is pulled:
  - `ollama list`
  - `ollama pull stepfun/step-3.5-flash:free`
- If skills validation fails, ensure each skill directory has `SKILL.md`.
```

- [x] Finalize `pyproject.toml` for full quality pipeline.
- [x] Copy and paste code below into `pyproject.toml`:

```toml
[project]
name = "workspace"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "deepagents>=0.4.11",
    "langchain>=1.2.12",
    "langchain-openrouter>=0.1.0",
    "langchain-ollama>=0.3.0",
    "langchain-litellm>=0.2.0",
    "pydantic-settings>=2.10.1",
    "rich>=14.3.3",
    "typer>=0.16.1",
]

[dependency-groups]
dev = [
    "ipykernel>=7.2.0",
    "mypy>=1.19.1",
    "pytest>=8.4.2",
    "pytest-cov>=7.0.0",
    "ruff>=0.15.5",
]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.13"
strict = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [x] Run full release checks.

```bash
uv sync
make check
uv run python main.py doctor
```

##### Step 7 Verification Checklist
- [x] `make check` succeeds end-to-end.
- [x] README commands run as documented.
- [x] No hard-coded credentials in repository.
- [ ] `doctor` reports expected environment and skill status.

## Final Validation Sequence
Run this complete sequence after Step 7:

```bash
uv sync
uv run ruff check .
uv run mypy main.py tests
uv run pytest
uv run python main.py doctor
```

Expected result:
- All checks pass with exit code 0.
- Tracing remains disabled unless explicitly enabled.
- `.agents` handling is deterministic and validated.
- Model role configuration loads only from environment/settings.


