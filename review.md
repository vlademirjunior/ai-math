# Deep Code Review — `main.py` & `tests/`

> **Scope:** SOLID, Clean Code, DRY analysis  
> **Date:** 2026-03-19  
> **Status:** Read-only review — no edits made  

---

## Executive Summary

`main.py` is a **2 978-line monolith** containing settings, models, MCP integration, context building, agent runtime, pipeline orchestration, CLI UI rendering, and Typer commands — all in one file.  
The tests (~65 KB across 14 files) cover many flows but inherit the codebase's coupling issues, resulting in heavy duplication and fragile mocks.

| Principle | Score | Key Problem |
|-----------|-------|-------------|
| **S** — Single Responsibility | 🔴 2/10 | One file owns ≥10 distinct responsibilities |
| **O** — Open/Closed | 🟡 5/10 | `MCPManagerLike` Protocol is good; rest is rigid |
| **L** — Liskov Substitution | 🟢 7/10 | Generally respected |
| **I** — Interface Segregation | 🟡 5/10 | `AgentRuntime` is a fat class (30+ methods) |
| **D** — Dependency Inversion | 🟡 5/10 | Concrete imports inside methods; `MCPManagerLike` is a good start |
| **Clean Code** | 🟡 5/10 | Good typing, but long functions and God Object pattern |
| **DRY** | 🔴 3/10 | Settings factory duplicated 9× in tests; repair logic duplicated in 2 methods |

---

## 1. SOLID Violations

### 1.1 Single Responsibility Principle (SRP) — 🔴 Critical

**Problem:** `main.py` holds at least **10 distinct domains** in a single file:

| Domain | Line Range (approx) | Suggested Module |
|--------|---------------------|------------------|
| Constants & tokens | 44–69 | `constants.py` |
| Data classes (ContextBuildResult, FileSnapshot…) | 93–141 | `models.py` |
| Chat auto-complete (ChatCompleter) | 143–206 | `cli/completer.py` |
| Context building & references | 209–957 | `context.py` |
| MCP config, manager, client | 292–723 | `mcp/manager.py` |
| Settings & dotenv | 1105–1271 | `settings.py` |
| Skills discovery & parsing | 1274–1381 | `skills.py` |
| Model factory | 1387–1433 | `models/factory.py` |
| AgentRuntime (orchestration) | 1442–2388 | `runtime.py` |
| CLI commands & UI rendering | 2390–2978 | `cli/commands.py` |

**Action Plan:**
1. Create a package structure (e.g., `helo/`) with one module per domain
2. Move code into respective modules
3. Keep `main.py` as a thin entry-point that imports and wires everything

---

### 1.2 Open/Closed Principle (OCP) — 🟡

**Good:** `MCPManagerLike` Protocol (L392) allows swapping MCP implementations.

**Problems:**
- `ModelFactory.create()` (L1398–1432) uses if/elif chains for each provider — adding a new provider requires modifying the method.
- `should_trigger_pipeline()` (L2390–2450) hard-codes keyword lists — new intents require editing the function.
- `_config_for_role()` (L1391–1396) uses if/elif instead of a mapping.

**Action Plan:**
1. **`ModelFactory`**: Use a registry dict `{Provider: factory_callable}` so new providers are registered, not coded inline
2. **`should_trigger_pipeline`**: Extract keyword lists to config or a data file
3. **`_config_for_role`**: Replace with `{ModelRole.PLANNER: self._settings.planner, ...}` mapping

---

### 1.3 Interface Segregation Principle (ISP) — 🟡

**Problem:** `AgentRuntime` is a God Object with **30+ methods** mixing:
- Agent creation (`create_role_agent`, `create_chat_agent`)
- Stream/run execution (`stream_role`, `stream`, `run_role`, `run_chat`)
- Pipeline orchestration (`run_pipeline`, `run_manual_role`)
- Text parsing (`_chunk_to_text`, `_extract_assistant_text_from_chunk`, `_content_to_text`, `_extract_clarification_text`, `_parse_tool_call_args`)
- File system operations (`_snapshot_plan_artifacts`, `_snapshot_project_files`, `_collect_changed_plan_artifacts`, `_collect_changed_project_files`)
- Path extraction (`_extract_paths_from_output`, `_collect_existing_output_artifacts`)
- System prompt building (`_system_prompt_for_role`, `_pipeline_contract`, `_role_preamble`, `_implementer_control_rules`)

**Action Plan:**
1. Extract **`PromptBuilder`** class for system prompt construction
2. Extract **`ArtifactCollector`** for file snapshot & diff logic
3. Extract **`ChunkParser`** for output text parsing
4. Keep `AgentRuntime` focused on orchestration only

---

### 1.4 Dependency Inversion Principle (DIP) — 🟡

**Problems:**
- `ModelFactory.create()` imports concrete LLM classes inside methods (L1402, L1416, L1426) — tightly coupled.
- `MCPManager._ensure_client()` (L497–522) imports `langchain_mcp_adapters` inline.
- `tracing_enabled_context()` (L2616) imports `langsmith` inline.

**Good:** The `client_factory` parameter on `MCPManager.__init__` is a proper DIP example.

**Action Plan:**
1. Accept provider factory callables via dependency injection in `ModelFactory`
2. Keep lazy imports for optional dependencies but abstract them behind an interface

---

## 2. Clean Code Issues

### 2.1 God Function: `execute_single_message()` (L2653–2748)

**96 lines** handling routing, context building, pipeline execution, clarification tracking, metric updates, and rendering — all in a single closure.

**Action Plan:** Extract into a `MessageRouter` class with discrete methods for each concern.

---

### 2.2 God Function: `run_pipeline()` (L2020–2238)

**218 lines** in a single method. Contains three distinct phases (planner, generator, implementer), each with its own snapshot, validation, and repair logic.

**Action Plan:** Extract each phase into its own method:
- `_run_planner_phase()`
- `_run_generator_phase()`
- `_run_implementer_phase()`

---

### 2.3 God Function: `run_manual_role()` (L2240–2373)

**133 lines** with deep nesting. The generator repair logic (L2282–2334) is almost identical to the repair logic in `run_pipeline()` (L2145–2178).

**Action Plan:** Extract shared `_repair_generator_output()` method. See DRY §3.1.

---

### 2.4 Magic Strings

String-based event protocols rely on prefix matching (L50–51):
```python
STATUS_EVENT_PREFIX = "__STATUS__::"
DUMP_EVENT_PREFIX = "__DUMP__::"
```

**Action Plan:** Replace with a proper event enum or typed event objects:
```python
@dataclass
class StatusEvent:
    message: str

@dataclass
class DumpEvent:
    content: str
```

---

### 2.5 Re-importing `re` Inside Methods

`re` is imported at module level (L7) but also imported inside `_slugify()` (L1749) and `_extract_paths_from_output()` (L1771).

**Action Plan:** Remove redundant local `import re` statements.

---

### 2.6 Mixed Language in User-Facing Strings

Portuguese and English are mixed throughout:
- Portuguese: `"Contexto adicional fornecido pelo usuario via #"` (L932)
- English: `"Pipeline paused after planner"` (L2096)

**Action Plan:** Choose one language for UI strings, or implement i18n from the start.

---

### 2.7 `_background_refresh()` Infinite Loop with Bare `except` (L451–459)

```python
def _background_refresh(self) -> None:
    while True:
        try:
            asyncio.run(self.refresh())
        except Exception:
            pass
        time.sleep(30)
```

Swallows all exceptions silently; `asyncio.run()` in a thread creates a new event loop each iteration.

**Action Plan:**
1. Log exceptions instead of silently swallowing
2. Use a dedicated event loop or `asyncio.run_coroutine_threadsafe()`
3. Add a stop mechanism (e.g., `threading.Event`)

---

### 2.8 Overly Broad Exception Handling (L629)

```python
except (ConnectionError, TimeoutError, ValueError, Exception) as error:
```

Catching `Exception` makes the preceding specific catches redundant.

**Action Plan:** Remove `Exception` from the tuple; let unexpected errors propagate or log them separately.

---

## 3. DRY Violations

### 3.1 Generator Repair Logic — Duplicated

The "check + retry generator for STOP & COMMIT" logic appears **twice**:
- `run_pipeline()` L2145–2178
- `run_manual_role()` L2282–2334

Both follow the same pattern: check → emit status → re-invoke → re-check → emit final status.

**Action Plan:** Extract a shared `_ensure_generator_checkpoints(impl_file, prompt, planner_output, thread_id, auto, on_chunk)` method.

---

### 3.2 `_settings()` Factory — Duplicated 9× Across Tests

The same factory pattern appears in **9 different test files**:

| File | Lines |
|------|-------|
| `test_pipeline_flow.py` | 15–21 |
| `test_cli_output.py` | 19–25 |
| `test_implementer_hitl.py` | 14–20 |
| `test_manual_roles.py` | 14–20 |
| `test_system_prompts.py` | 15–21 |
| `test_mcp_integration.py` | 24–33 |
| `test_chat_ui.py` | 38–44 |
| `test_cli.py` | 41–46 |
| `test_model_factory.py` | 6–12 |

**Action Plan:**
1. Move to `conftest.py` as a shared fixture:
```python
@pytest.fixture
def litellm_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        project_root=tmp_path,
        planner=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        generator=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
        implementer=RoleModelConfig(provider=Provider.LITELLM, model="openai/gpt-4o-mini"),
    )
```

---

### 3.3 `StubRuntime` — Duplicated 5× in `test_cli.py` & `test_chat_ui.py`

Each test defines its own `StubRuntime` inner class with nearly identical signatures.

**Action Plan:** Create a reusable `StubRuntime` in `conftest.py` that accepts optional callback hooks:
```python
class StubRuntime:
    def __init__(self, settings, *, on_pipeline=None, on_chat=None, on_manual_role=None):
        ...
```

---

### 3.4 `fake_stream_role` — Duplicated 8× in Pipeline & CLI Output Tests

Every pipeline test defines a local `fake_stream_role` with the same boilerplate: create `plans_dir`, write `plan.md`, write `implementation.md`, return token strings.

**Action Plan:** Create a configurable `FakeStreamRole` builder in `conftest.py` or a test utilities module.

---

### 3.5 `fake_print` Pattern — Duplicated 6×

```python
def fake_print(*args, **kwargs) -> None:
    if args:
        printed.append(str(args[0]))
```

This exact function appears in 6 tests.

**Action Plan:** Move to `conftest.py` as a fixture:
```python
@pytest.fixture
def capture_console(monkeypatch):
    printed = []
    monkeypatch.setattr(main.console, "print", lambda *a, **k: printed.append(str(a[0])) if a else None)
    return printed
```

---

## 4. Implementation Action Plan (Ordered)

> Steps ordered by priority: foundational refactors first, then progressive improvements.

### Phase 1: Test Foundation (Low Risk)
- [ ] **Step 1.1:** Create shared `litellm_settings` fixture in `conftest.py`
- [ ] **Step 1.2:** Create shared `StubRuntime` in `conftest.py`
- [ ] **Step 1.3:** Create shared `fake_stream_role` builder in `conftest.py`
- [ ] **Step 1.4:** Create shared `capture_console` fixture in `conftest.py`
- [ ] **Step 1.5:** Refactor all 14 test files to use shared fixtures
- [ ] **Step 1.6:** Run tests to confirm green

### Phase 2: Module Extraction (Medium Risk)
- [ ] **Step 2.1:** Create `helo/` package with `__init__.py`
- [ ] **Step 2.2:** Extract `constants.py` (tokens, prefixes, limits)
- [ ] **Step 2.3:** Extract `models.py` (data classes, enums, Pydantic models)
- [ ] **Step 2.4:** Extract `settings.py` (AppSettings, dotenv, env resolution)
- [ ] **Step 2.5:** Extract `context.py` (context building, references, trimming)
- [ ] **Step 2.6:** Extract `mcp/` package (config, manager, client)
- [ ] **Step 2.7:** Extract `skills.py` (discovery, parsing, listing)
- [ ] **Step 2.8:** Extract `factory.py` (ModelFactory with registry pattern)
- [ ] **Step 2.9:** Extract `runtime.py` (AgentRuntime + sub-classes)
- [ ] **Step 2.10:** Extract `cli/` package (commands, rendering, completer)
- [ ] **Step 2.11:** Keep `main.py` as a thin entry-point
- [ ] **Step 2.12:** Update all imports in test files
- [ ] **Step 2.13:** Run tests to confirm green

### Phase 3: Clean Code Fixes (Low Risk)
- [ ] **Step 3.1:** Remove duplicate `import re` in `_slugify()` and `_extract_paths_from_output()`
- [ ] **Step 3.2:** Replace magic string event prefixes with typed event objects
- [ ] **Step 3.3:** Fix `_background_refresh()` — log errors, add stop mechanism
- [ ] **Step 3.4:** Fix overly broad exception handling (remove redundant `Exception`)
- [ ] **Step 3.5:** Split `run_pipeline()` into `_run_planner_phase()`, `_run_generator_phase()`, `_run_implementer_phase()`
- [ ] **Step 3.6:** Extract `_ensure_generator_checkpoints()` shared method (DRY fix)
- [ ] **Step 3.7:** Split `run_manual_role()` to use the new phase methods
- [ ] **Step 3.8:** Run tests to confirm green

### Phase 4: Architecture Improvements (Medium Risk)
- [ ] **Step 4.1:** Refactor `ModelFactory` to use a registry dict instead of if/elif chains
- [ ] **Step 4.2:** Extract `PromptBuilder` class from `AgentRuntime`
- [ ] **Step 4.3:** Extract `ArtifactCollector` class from `AgentRuntime`
- [ ] **Step 4.4:** Extract `ChunkParser` class from `AgentRuntime`
- [ ] **Step 4.5:** Standardize language in UI strings (choose PT-BR or EN)
- [ ] **Step 4.6:** Run full test suite + manual smoke test

---

## 5. Test Quality Observations

### Strengths ✅
- Good coverage of pipeline edge cases (missing plan, missing implementation, clarification flow)
- Proper use of `monkeypatch` for isolation
- `tmp_path` used correctly for filesystem tests
- Tests are well-named and intention-revealing

### Weaknesses ❌
- **No negative test for `MCPServerConfig` with invalid transport types** beyond `stdio`
- **No test for `_background_refresh()` thread behavior or shutdown**
- **No test for `_slugify()` edge cases** (empty string, unicode, very long input)
- **No test for `list_skills()` metadata parsing** (frontmatter, fallback, BOM)
- **No integration-level test** for the full `chat` command loop with MCP + pipeline + clarification
- **StubRuntime classes don't verify call signatures** — tests pass even if the real interface changes

### Recommended New Tests
- [ ] `test_slugify_edge_cases` — empty, unicode, max length, special chars
- [ ] `test_parse_skill_metadata_frontmatter` — valid YAML, missing fields, BOM
- [ ] `test_background_refresh_logs_errors` — verify exceptions are not silently swallowed
- [ ] `test_model_factory_registry` — after registry refactor, test adding a new provider
- [ ] `test_estimate_token_count_edge_cases` — empty string, whitespace only

---

> **Next step:** Pick a phase and begin implementation. Phase 1 (test foundation) is the safest starting point.
