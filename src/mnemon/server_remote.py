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

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

PORT = int(os.environ.get("PORT", "8502"))
AUTH_TOKEN = os.environ.get("MNEMON_TOKEN", "")


# ── Auth Middleware ─────────────────────────────────────────────────────────

from starlette.middleware.base import BaseHTTPMiddleware


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid Bearer token (when MNEMON_TOKEN is set)."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health check
        if request.url.path == "/health":
            return await call_next(request)

        if AUTH_TOKEN:
            auth_header = request.headers.get("authorization", "")
            if auth_header != f"Bearer {AUTH_TOKEN}":
                return Response("Unauthorized", status_code=401)

        return await call_next(request)


# ── Health Endpoint ─────────────────────────────────────────────────────────

async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "version": "0.1.0"})


# ── Build App ──────────────────────────────────────────────────────────────

def create_app() -> Starlette:
    """Create the Starlette app with MCP at /mcp and health at /health.

    FastMCP's streamable_http_app() is a Starlette app with an internal
    route at /mcp, so we mount it at root (not at /mcp) to avoid
    double-prefixing (/mcp/mcp).
    """
    from .server import mcp

    mcp_app = mcp.streamable_http_app()

    middleware = []
    if AUTH_TOKEN:
        middleware.append(Middleware(BearerAuthMiddleware))

    app = Starlette(
        routes=[
            Route("/health", health),
            Mount("/", app=mcp_app),
        ],
        middleware=middleware,
    )

    return app


def run_remote() -> None:
    """Start the remote HTTP server."""
    import uvicorn

    print(f"mnemon remote server starting on http://0.0.0.0:{PORT}/mcp", file=sys.stderr)
    print(f"Auth: {'enabled (Bearer token)' if AUTH_TOKEN else 'disabled'}", file=sys.stderr)
    print(f"Health: http://0.0.0.0:{PORT}/health", file=sys.stderr)

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
