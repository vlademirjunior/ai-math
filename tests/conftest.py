import sys
from pathlib import Path
from typing import Any, cast

# Ensure the repository root is on sys.path so tests can import main.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Workaround: `mcp.client.streamable_http.streamablehttp_client` is deprecated and
# emits a DeprecationWarning when called. Some downstream libraries still use it.
# Patch it to point to the supported `streamable_http_client` implementation so
# tests run cleanly without warnings.
try:
    import mcp.client.streamable_http as _streamable_http

    _streamable_http.streamablehttp_client = cast(Any, _streamable_http.streamable_http_client)
except Exception:  # pragma: no cover
    # If MCP isn't installed in this environment, tests that depend on it will fail
    # anyway; we just avoid raising during import-time patching.
    pass
