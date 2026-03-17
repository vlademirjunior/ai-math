# Claude-Code-Like Helo (Python)
# Use Docs by LangChain MCP Tool when necessary
**Description:** Build a production-ready Python CLI inspired by Claude Code, using LangChain Deep Agents with skills from .agents/, three model roles (planner/generator/implementer), optional LangSmith tracing (default off), and a quality-first toolchain (uv, ruff, mypy, pytest).

## Research Report
### 1) Deep Agents Overview (core architecture)
- Primary API: `create_deep_agent(...)` from `deepagents`.
- Core capabilities to leverage: built-in planning via `write_todos`, filesystem tools (`ls`, `read_file`, `write_file`, `edit_file`), subagent task delegation via `task`, and long-term memory support.
- Recommended for this use case because the harness is designed for multi-step coding workflows and uses LangGraph runtime features (durable execution, streaming, HITL patterns).
- Agent invocation shape: `agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": ...}})` and stream support for interactive CLI output.

### 2) Deep Agents Skills (.agents)
- Skills format: directory-based; each skill directory includes `SKILL.md` (+ optional assets/scripts/docs/templates).
- Progressive disclosure: the agent reads skill frontmatter first, then lazily reads full skill only when relevant.
- Loading behavior: SDK only loads paths explicitly passed in `skills=[...]`; it does not auto-scan user directories.
- Path constraints: use forward slashes; paths are backend-root-relative; later skill sources override earlier ones (last wins).
- Operational constraints from spec/docs:
  - `description` in frontmatter is truncated after 1024 chars.
  - `SKILL.md` over 10 MB is skipped.

### 3) ChatLiteLLM Integration
- Package and classes:
  - `langchain-litellm`
  - `ChatLiteLLM` for direct model calls
  - `ChatLiteLLMRouter` for routed/fallback model lists
- Value for this CLI: a single abstraction to support multiple providers and model-role switching without hard-coding provider-specific logic.
- Supports async/streaming and normal LangChain chat-model invocation semantics.

### 4) ChatOpenRouter Integration
- Package/class: `langchain-openrouter`, `ChatOpenRouter`.
- Env/auth: `OPENROUTER_API_KEY`.
- Strong fit for cloud-default role(s): broad model availability, provider routing controls (`openrouter_provider` ordering/sort/fallback/data policies), standard LangChain tool/structured output patterns.
- Compatibility note: LangChain guidance prefers dedicated `ChatOpenRouter` over generic OpenAI-compatible `ChatOpenAI(base_url=...)` when provider-specific features matter.

### 5) ChatOllama Integration
- Package/class: `langchain-ollama`, `ChatOllama`.
- Runtime prerequisite: local Ollama daemon + pulled models (`ollama pull ...`).
- Fit for cheapest implementation role: local/offline execution and controllable cost profile.
- Risk: model/tool-calling behavior depends on selected Ollama model; ensure chosen model supports tools when required.

### 6) LangSmith Tracing for Deep Agents
- Native tracing is supported by Deep Agents/LangGraph.
- Env-based toggle works out-of-the-box:
  - `LANGSMITH_TRACING=true|false`
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_PROJECT` (optional)
- Programmatic selective tracing is available via `langsmith.tracing_context(...)` for scoped tracing.
- Requirement alignment: default to tracing disabled; enable only when explicitly configured.

### Constraints, Risks, Compatibility Notes
- The current repo `main.py` contains a hard-coded API key and must be replaced with env-based secure configuration.
- Three-role model orchestration is not an automatic Deep Agents primitive; it should be implemented as an explicit model-routing policy at the application layer.
- Provider package compatibility must be pinned/tested (`deepagents`, `langchain`, `langchain-openrouter`, `langchain-ollama`, `langchain-litellm`) to avoid API drift.
- Skills path handling must be deterministic (normalize `.agents/` to a backend-relative forward-slash path and verify existence).
- Ollama availability should degrade gracefully (startup health checks and clear UX guidance).

### Suggested 3-Model Orchestration Architecture
- `ModelRole` enum: `PLANNER`, `GENERATOR`, `IMPLEMENTER`.
- `ProviderConfig` and `ModelConfig` (Pydantic) per role loaded from env.
- `ModelFactory` returns a LangChain `BaseChatModel` instance per role/provider:
  - OpenRouter -> `ChatOpenRouter`
  - Ollama -> `ChatOllama`
  - LiteLLM -> `ChatLiteLLM`/`ChatLiteLLMRouter`
- `OrchestrationPolicy`:
  - Planner role: Deep Agent core planning and todo control.
  - Generator role: isolated generation phase/tool call wrapper for code synthesis prompts.
  - Implementer role: cheapest model for apply/patch-style execution decisions.
- `AgentRuntime` composes Deep Agent + orchestration hooks so each phase uses the correct role model.

### How to Load `.agents/` Skills
- Use `FilesystemBackend(root_dir=<project_root>)`.
- Pass explicit skill source path(s) in `create_deep_agent(..., skills=["/.agents/"])` (normalized forward-slash paths relative to backend root).
- Validate on startup:
  - `.agents/` exists.
  - each skill has `SKILL.md`.
  - optional warning on oversize files or malformed frontmatter.

### How to Toggle LangSmith (default false)
- App config default: `ENABLE_LANGSMITH=false`.
- If enabled:
  - set/require `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`.
  - optionally set `LANGSMITH_PROJECT`.
- If disabled:
  - enforce `LANGSMITH_TRACING=false` in process environment for deterministic behavior.

### Testing and Quality Strategy
- `ruff`: lint + formatting gate.
- `mypy`: strict-enough typing for config/model-factory/orchestration boundaries.
- `pytest`: unit tests for env parsing, provider/model construction, role routing, skill discovery, and tracing toggle logic.
- Smoke tests: CLI invocation with mocked models/backends and one real optional integration path behind env flags.

## Goal
Deliver a secure, production-ready single-file Python CLI that uses Deep Agents as the core harness, loads skills from `.agents/`, orchestrates three LLM roles via environment-driven provider/model configuration (OpenRouter/Ollama first-class, LiteLLM-supported), optionally traces to LangSmith (default disabled), and ships with robust developer quality gates.

## Implementation Steps
### Step 1: Define production configuration and role contracts
**Files:** `main.py`, `pyproject.toml`
**What:** Introduce Pydantic-based settings schema for three model roles, provider selection, API key env vars, Ollama host/model settings, and tracing toggle; define role enum + validation and secure defaults (no secrets in code).
**Testing:** Pytest unit tests for env parsing/validation; mypy checks for typed config contracts.

### Step 2: Build provider-agnostic model factory with OpenRouter/Ollama/LiteLLM adapters
**Files:** `main.py`, `pyproject.toml`
**What:** Implement a `ModelFactory` that returns the correct LangChain chat model object per role/provider. Prioritize official integrations (`ChatOpenRouter`, `ChatOllama`) and support `ChatLiteLLM` as flexible fallback/provider bridge.
**Testing:** Pytest parametrized tests for each provider-role combination; verify clear failure messages for missing envs/dependencies.

### Step 3: Compose Deep Agent runtime and 3-role orchestration policy
**Files:** `main.py`
**What:** Create orchestration service that maps phases (plan/todos, generate, implement) to role-specific models while keeping Deep Agents as central runtime. Ensure planning uses Deep Agents todo primitives and maintain session/thread IDs.
**Testing:** Unit tests with mock chat models validating role dispatch and phase transitions; smoke test for invoke/stream paths.

### Step 4: Add `.agents/` skill loader and startup validation
**Files:** `main.py`, `README.md`
**What:** Implement `.agents/` discovery and normalized skills source injection into `create_deep_agent`; add preflight checks and warnings for missing `SKILL.md` or invalid skill layout.
**Testing:** Pytest filesystem fixture tests for valid/invalid skill trees and source precedence behavior.

### Step 5: Implement Typer + Rich CLI UX and operational commands
**Files:** `main.py`, `README.md`
**What:** Build command-driven UX (`chat`, `run`, `doctor`, `models`, `skills`) with Rich panels/status/progress and production-grade error handling. Include non-interactive mode support for CI/scripted usage.
**Testing:** CLI tests via Typer test runner; snapshot-style output checks for core flows and failure scenarios.

### Step 6: Integrate optional LangSmith tracing with deterministic default-off behavior
**Files:** `main.py`, `README.md`
**What:** Wire environment-based tracing toggle and optional scoped tracing path. Ensure disabled-by-default operation without extra setup and explicit enable path with required variables.
**Testing:** Unit tests for env toggles; integration smoke test ensuring trace env is applied only when enabled.

### Step 7: Harden project quality gates and release-ready documentation
**Files:** `pyproject.toml`, `Makefile`, `README.md`
**What:** Finalize dependencies and tool config for uv/pip, ruff, mypy, pytest; document setup, env matrix, provider recipes, security notes, and troubleshooting.
**Testing:** End-to-end local quality gate run (`ruff`, `mypy`, `pytest`) and README command audit.

## Surprise-me with the subjects:
1. Should the three-role orchestration be strictly sequential (planner -> generator -> implementer) for every request, or should the planner dynamically skip phases when unnecessary?
2. For the implementation role, can Ollama be the default cheapest provider when available, with automatic fallback to OpenRouter/LiteLLM when local models are missing?
3. Do you want `.agents/` to be mandatory (hard error if missing) or optional (warn and continue without skills)?
4. Should we support one shared model across multiple roles when env vars are partially configured, or require all three roles explicitly?
5. For LangSmith, do you want only env-based toggling, or also a CLI flag that overrides env at runtime?
6. Should the CLI include an approval mode for shell/file-destructive actions similar to Helo interrupt controls by default?
7. Do you want a strict Python version target (e.g., 3.11+ per Deep Agents docs) or keep current project target (3.13+)?