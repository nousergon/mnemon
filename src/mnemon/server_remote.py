"""Remote HTTP server — Streamable HTTP transport for MCP with OAuth 2.1.

Exposes the same MCP tools as stdio mode, accessible over public HTTPS to
MCP clients that speak the Streamable HTTP transport (claude.ai web, Claude
Desktop, Claude mobile apps via custom connectors, Claude Code via
``claude mcp add --transport http``, Cursor via mcp.json, etc.).

Authentication
--------------
When ``MNEMON_OAUTH_ISSUER``, ``MNEMON_OAUTH_JWKS_URL``, and
``MNEMON_OAUTH_AUDIENCE`` are set, the server operates as an OAuth 2.1
resource server per the MCP authorization spec (2025-06-18): it validates
JWT bearer tokens from an external authorization server and serves RFC 9728
Protected Resource Metadata for client discovery.

``MNEMON_LOCAL_TOKEN`` enables a secondary static-bearer auth path for
headless clients (Claude Code hooks, Cursor, scripts) that cannot complete
a browser OAuth flow. The value must match exactly — the middleware does
a constant-time comparison and skips JWT/userinfo validation on match.
Can be combined with OAuth or used alone.

When none of these env vars are set, the server runs without auth (local
development only — do NOT expose an unauthenticated server to the public
internet).

Usage
-----
Local, no auth::

    mnemon serve-remote

Cloud, with external AS (e.g., Auth0)::

    export MNEMON_OAUTH_ISSUER=https://your-tenant.us.auth0.com/
    export MNEMON_OAUTH_JWKS_URL=https://your-tenant.us.auth0.com/.well-known/jwks.json
    export MNEMON_OAUTH_AUDIENCE=https://your-mnemon.fly.dev/mcp
    export MNEMON_PUBLIC_URL=https://your-mnemon.fly.dev
    mnemon serve-remote
"""

from __future__ import annotations

import os
import sys

from .auth import OAuthConfig, OAuthMiddleware

PORT = int(os.environ.get("PORT", "8502"))


def run_remote() -> None:
    """Start the remote HTTP server wrapped in the OAuth middleware.

    Eagerly initializes the embedding model before uvicorn binds the
    port. This shifts the FastEmbed model load (~3 seconds with the
    Docker-baked cache, ~10+ without) from the user's first
    ``memory_search`` call to server startup, so the first hook
    invocation after a Fly cold start succeeds within Claude Code's
    8-second hook timeout. The server doesn't accept connections until
    the embedder is ready, which means clients see a brief connection
    delay during cold start instead of an in-flight tool-call timeout.
    """
    from .server import mcp

    # Eager embedder init — non-fatal if it fails (lazy load will retry
    # on first actual search call).
    try:
        from .embedder import _get_model

        print("Pre-loading embedding model...", file=sys.stderr)
        _get_model()
        print("Embedding model ready.", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(
            f"WARN: failed to pre-load embedding model "
            f"({type(e).__name__}: {e}); first memory_search will pay "
            "the load cost lazily",
            file=sys.stderr,
        )

    config = OAuthConfig.from_env()

    # Self-hosted Authorization Server config (Phase 2 scaffolding). When
    # MNEMON_AS_ENABLED=true, the well-known AS metadata + JWKS endpoints
    # are served. The token-issuing endpoints themselves are not yet
    # implemented — coming in PR #37.
    from .oauth_as import AuthorizationServerConfig

    as_config = AuthorizationServerConfig.from_env()
    as_problems = as_config.validate()
    if as_problems:
        print(
            "ERROR: self-hosted AS enabled but misconfigured:\n  - "
            + "\n  - ".join(as_problems),
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"mnemon remote server starting on http://0.0.0.0:{PORT}/mcp",
        file=sys.stderr,
    )
    if config.enabled:
        print(
            f"Auth: OAuth 2.1 resource server (issuer={config.issuer}, "
            f"audience={config.audience})",
            file=sys.stderr,
        )
    if as_config.enabled:
        print(
            f"Auth: self-hosted Authorization Server enabled "
            f"(issuer={as_config.issuer})",
            file=sys.stderr,
        )
    if config.local_token:
        print(
            "Auth: local static bearer token enabled (MNEMON_LOCAL_TOKEN set)",
            file=sys.stderr,
        )
    if not config.enabled and not config.local_token:
        print(
            "Auth: DISABLED — do not expose this server to the public internet. "
            "Set MNEMON_OAUTH_ISSUER, MNEMON_OAUTH_JWKS_URL, and "
            "MNEMON_OAUTH_AUDIENCE to enable OAuth, or MNEMON_LOCAL_TOKEN "
            "to enable local static bearer auth.",
            file=sys.stderr,
        )

    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: uvicorn not installed. Install with `pip install "
            "mnemon-memory[server]`.",
            file=sys.stderr,
        )
        sys.exit(1)

    mcp_app = mcp.streamable_http_app()
    wrapped = OAuthMiddleware(mcp_app, config, as_config=as_config)
    uvicorn.run(wrapped, host="0.0.0.0", port=PORT, log_level="info")
