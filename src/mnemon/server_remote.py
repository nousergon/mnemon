"""Remote HTTP server — Streamable HTTP transport for MCP.

Exposes the same MCP tools as stdio mode, accessible from Claude.ai
web and iOS via Streamable HTTP. Bearer token auth.

Usage:
    mnemon serve-remote                          # port 8502, no auth
    MNEMON_TOKEN=secret mnemon serve-remote      # with bearer token auth
    PORT=9000 mnemon serve-remote                # custom port
"""

from __future__ import annotations

import os
import sys

PORT = int(os.environ.get("PORT", "8502"))
AUTH_TOKEN = os.environ.get("MNEMON_TOKEN", "")


def run_remote() -> None:
    """Start the remote HTTP server using FastMCP's built-in Streamable HTTP transport."""
    from .server import mcp

    # Set host/port for the Streamable HTTP transport
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = PORT

    print(f"mnemon remote server starting on http://0.0.0.0:{PORT}/mcp", file=sys.stderr)
    print(f"Auth: {'enabled (Bearer token)' if AUTH_TOKEN else 'disabled'}", file=sys.stderr)

    mcp.run(transport="streamable-http")
