# MCP Integration with Priority Context System

**Branch:** main
**Description:** Implementa leitura de configurações MCP do arquivo .vscode e integração com servidores disponíveis

## Goal
Implementar um sistema que lê o arquivo .vscode/mcp.json e/ou .agents/mcp.json do repositorio atual e configura automaticamente os MCPs (Model Context Protocol) disponíveis, permitindo que o ResizeMe utilize serviços externos como GitHub Copilot e LangChain Docs de forma integrada.

## Implementation Steps

### Step 1: Add Dependencies and Configuration
**Files:** uv add, main.py
**What:**
- Add `langchain-mcp-adapters>=0.2.2` using uv add command
- Add new Pydantic config field: `mcp_servers: Dict[str, MCPServerConfig]` with structure:
  ```python
  class MCPServerConfig(BaseModel):
      transport: Literal["stdio", "streamable_http", "sse", "websocket"]
      command: Optional[str] = None  # for stdio
      args: List[str] = Field(default_factory=list)  # for stdio
      url: Optional[str] = None  # for http/websocket
      headers: Dict[str, str] = Field(default_factory=dict)
      timeout: int = 30  # seconds
  ```
- Add new constant: `MAX_MCP_CONTEXT_CHARS = 6000` (separate from `MAX_CONTEXT_FILE_CHARS=12000`)
- Add environment variable support: `MCP_SERVERS__SERVER_NAME__TRANSPORT=streamable_http`, etc.
**Testing:**
- Verify config loads from environment variables
- Verify default values and validation errors
- Test that MCP config is optional (app works without it)

### Step 2: Create MCP Client Wrapper with Offline Handling
**Files:** main.py
**What:**
- Implementar lógica para ler e parsear o arquivo .vscode/mcp.json e/ou .agents/mcp.json, extraindo informações de servidores e entradas
- Create `MCPManager` class that:
  - Accepts `AppSettings` instance
  - Maintains `MultiServerMCPClient` instance
  - Implements connection health tracking per server (online/offline status)
  - Provides async methods: `get_all_tools()`, `get_resources()`, `get_server_status()`
  - Handles connection errors gracefully, marks server as offline, logs friendly message
  - Implements retry logic with exponential backoff (optional but recommended)
  - Provides `is_available()` method to check if any MCP servers are online
- Use try/except blocks around all MCP operations to catch:
  - `ConnectionError`, `TimeoutError`, `ValueError`, `Exception`
  - When caught, set server status to offline and return empty results
- Add logging: `logger.warning(f"MCP server '{name}' is offline: {error}. Continuing without MCP context.")`
**Testing:**
- Unit test: Mock `MultiServerMCPClient` to simulate offline server, verify graceful handling
- Unit test: Verify server status tracking (online → offline transitions)
- Unit test: Verify empty results returned when server offline
- Unit test: Verify multiple servers, one offline doesn't affect others
- Testar com o arquivo .vscode/mcp.json e/ou .agents/mcp.json existente e validar parsing correto

### Step 3: Integrate MCP into Context Building Pipeline
**Files:** main.py
**What:**
- Modify context building logic to include MCP resources as a source
- Implement priority order:
  1. Local files from `#context` references (respect `MAX_CONTEXT_FILE_CHARS`)
  2. `.agents/**` skills documentation (respect same file limit)
  3. MCP resources (respect `MAX_MCP_CONTEXT_CHARS` separately)
  4. Other sources (if any)
- Add MCP resource fetching:
  - Call `mcp_manager.get_resources()` async
  - Format each blob as: `[MCP: server_name] resource_uri\n\n{content}`
  - Track source as `ContextSource.MCP` in metadata
  - Count chars separately against `MAX_MCP_CONTEXT_CHARS`
  - If MCP manager unavailable or all servers offline, skip with logged info message
- Update `ContextBuildResult` metadata to include:
  - `mcp_sources_used: List[str]` (server names)
  - `mcp_chars_used: int`
  - `mcp_servers_offline: List[str]`
**Testing:**
- Integration test: Mock MCP manager with sample resources, verify they appear in context
- Integration test: Verify MCP respects separate limit (stops at 6000 chars even if file limit not reached)
- Integration test: Verify priority order (MCP only added after files/.agents quota filled)
- Integration test: Verify MCP offline doesn't break context building

### Step 4: Add UI Notifications for MCP Status
**Files:** main.py (UI/display sections)
**What:**
- At startup/initialization, display MCP server status:
  - If any servers online: `✓ MCP servers: math, weather (3 tools, 5 resources)`
  - If any offline: `⚠ MCP servers: math (offline), weather (2 tools) - some features may be limited`
- Use Rich's `Status` or console print with appropriate styling (green for online, yellow for warnings)
- During context building, if MCP contributes: show `[MCP] Added 2 resources from weather server`
- If all MCP servers offline: show `[MCP] All MCP servers offline - skipping MCP context`
- Ensure messages are user-friendly and non-blocking (don't stop execution)
**Testing:**
- Visual/CLI test: Run with mock MCP servers, verify status messages appear
- Visual/CLI test: Run with offline server, verify warning message
- Test: Verify UI doesn't block execution even with offline servers

### Step 5: Extend Metadata and Logging
**Files:** main.py or wherever `ChatInteractionMetadata`/`ContextBuildResult` defined
**What:**
- Add fields to `ContextBuildResult`:
  ```python
  mcp_sources_used: List[str] = Field(default_factory=list)
  mcp_chars_used: int = 0
  mcp_servers_offline: List[str] = Field(default_factory=list)
  mcp_tools_available: int = 0
  mcp_resources_available: int = 0
  ```
- Update logging in context builder to include MCP stats:
  - `logger.info(f"Context built: files={file_chars}, agents={agent_chars}, mcp={mcp_chars}, total={total}")`
  - `logger.debug(f"MCP servers: online={online}, offline={offline}")`
- Ensure MCP appears in any summary displays (e.g., end-of-interaction stats)
**Testing:**
- Unit test: Verify metadata fields populated correctly
- Unit test: Verify logging output includes MCP metrics

### Step 6: Implement Comprehensive Test Suite
**Files:** tests/test_mcp_integration.py (new file)
**What:**
- **Mock HTTP MCP servers** using `pytest-httpserver` or `respx`:
  - Create fixtures that simulate MCP servers over streamable_http
  - Implement endpoints: `/mcp` that respond with valid MCP protocol messages
  - Simulate offline by not starting server or returning 503
- **Test MCPManager:**
  - `test_mcp_manager_initialization()`: Verify client creation with config
  - `test_get_tools_success()`: Mock successful tool discovery
  - `test_get_tools_offline()`: Mock connection error, verify graceful handling
  - `test_get_resources_success()`: Mock resource blobs returned
  - `test_mixed_server_status()`: One online, one offline, verify isolation
  - `test_server_status_tracking()`: Verify status transitions
- **Test Context Building:**
  - `test_mcp_resources_included()`: Verify MCP resources added to context
  - `test_mcp_priority_order()`: Verify MCP only after files/.agents quota
  - `test_mcp_separate_limit()`: Verify MAX_MCP_CONTEXT_CHARS enforced independently
  - `test_mcp_offline_no_crash()`: Verify context builds without MCP
  - `test_mcp_metadata_populated()`: Verify metadata fields
- **Test UI Notifications:**
  - `test_startup_status_display()`: Capture Rich console output, verify status messages
  - `test_offline_warning_displayed()`: Verify warning appears when server offline
- **Test Configuration:**
  - `test_mcp_config_from_env()`: Verify environment variable parsing
  - `test_invalid_config_validation()`: Verify missing required fields error
- **Test End-to-End:**
  - `test_full_flow_with_mock_mcp()`: Run planner→generator→implementer with mock MCP, verify context includes MCP
  - `test_full_flow_offline_mcp()`: Run with offline MCP, verify flow continues
**Testing:**
- All tests should pass with mocked MCP servers (no real network calls)
- Use `pytest-asyncio` for async tests
- Achieve >80% coverage on MCP-related code

### Step 7: Documentation and Final Validation
**Files:** README.md or docs/mcp.md (new), plans/main/plan.md (this file)
**What:**
- Create documentation section: "MCP Integration"
  - How to configure MCP servers in config file or environment
  - Supported transports and required fields
  - Example configurations for stdio and streamable_http
  - Explanation of priority system and context limits
  - Troubleshooting: "MCP server offline" messages and what to do
- Add inline code comments explaining MCP integration points
- Update any existing architecture diagrams or notes
**Validation Checkpoints:**
1. ✅ Dependencies installed (`langchain-mcp-adapters`)
2. ✅ Configuration loads without errors
3. ✅ MCPManager initializes and connects to servers
4. ✅ Offline servers handled gracefully (no crashes)
5. ✅ MCP resources appear in context after files/.agents quota
6. ✅ Separate MCP context limit enforced
7. ✅ Metadata includes MCP stats
8. ✅ UI shows server status at startup
9. ✅ All tests pass with mocked HTTP servers
10. ✅ Documentation complete

## Notes
- **Async requirement**: All MCP operations are async. The main pipeline may need async adjustments. Use `asyncio.run()` or convert main loop to async if not already.
- **Priority system**: Implement as sequential filling: first collect local files, .agents/**.md up to their limits, then add MCP resources up to `MAX_MCP_CONTEXT_CHARS`. (context priority)
- **Offline handling**: Never raise exceptions to user. Log warnings and continue. UI should inform but not block.
- **Testing focus**: Mock at the `MultiServerMCPClient` level for unit tests, and use `pytest-httpserver` for integration tests to verify HTTP transport.
- **Performance**: MCP resource fetching may be slow. Consider caching or parallel fetching if needed (future optimization).
