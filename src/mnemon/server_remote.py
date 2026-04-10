"""Remote HTTP server — Streamable HTTP transport for MCP.

Exposes the same MCP tools as stdio mode, accessible from Claude.ai
web and iOS via Streamable HTTP. Bearer token auth when MNEMON_TOKEN is set.

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
    """Start the remote HTTP server.

    When MNEMON_TOKEN is set, wraps with bearer auth middleware and runs
    via uvicorn. Otherwise uses FastMCP's native Streamable HTTP transport.
    """
    from .server import mcp

    print(f"mnemon remote server starting on http://0.0.0.0:{PORT}/mcp", file=sys.stderr)
    print(f"Auth: {'enabled (Bearer token)' if AUTH_TOKEN else 'disabled'}", file=sys.stderr)

    if AUTH_TOKEN:
        _run_with_auth(mcp)
    else:
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = PORT
        mcp.run(transport="streamable-http")


def _run_with_auth(mcp) -> None:
    """Run with bearer token auth middleware via uvicorn."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Mount, Route

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            auth_header = request.headers.get("authorization", "")
            if auth_header != f"Bearer {AUTH_TOKEN}":
                return Response("Unauthorized", status_code=401)
            return await call_next(request)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    mcp_app = mcp.streamable_http_app()

    app = Starlette(
        routes=[
            Route("/health", health),
            Mount("/", app=mcp_app),
        ],
        middleware=[Middleware(BearerAuthMiddleware)],
    )

    print(f"Health: http://0.0.0.0:{PORT}/health", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
