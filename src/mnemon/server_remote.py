"""Remote HTTP server — Streamable HTTP transport for MCP with OAuth 2.1.

Exposes the same MCP tools as stdio mode, accessible over public HTTPS to
MCP clients that speak the Streamable HTTP transport (claude.ai web, Claude
Desktop, Claude mobile apps via custom connectors, Claude Code via
``claude mcp add --transport http``, Cursor via mcp.json, etc.).

Authentication
--------------
When ``MNEMON_AS_ENABLED=true`` (with ``MNEMON_AS_PASSPHRASE`` and
``MNEMON_PUBLIC_URL``), the server runs a self-hosted OAuth 2.1
Authorization Server (see ``oauth_as.py``) alongside the Resource Server
and verifies bearer JWTs against the local keypair. No external auth
vendor required.

``MNEMON_LOCAL_TOKEN`` enables a secondary static-bearer auth path for
headless clients (Claude Code hooks, Cursor, scripts) that cannot
complete a browser OAuth flow. Constant-time compared, no network hop.
Can be combined with the self-hosted AS or used alone.

When neither is set, the server runs without auth (local development
only — do NOT expose an unauthenticated server to the public internet).

Usage
-----
Local, no auth::

    mnemon serve-remote

Production (self-hosted AS)::

    export MNEMON_AS_ENABLED=true
    export MNEMON_AS_PASSPHRASE=<your-passphrase>
    export MNEMON_PUBLIC_URL=https://your-mnemon.fly.dev
    export MNEMON_LOCAL_TOKEN=<random-bearer-for-hooks>
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

    # Self-hosted Authorization Server. When MNEMON_AS_ENABLED=true, the
    # well-known AS metadata, JWKS, /authorize, /token, and /register
    # endpoints are served. Browser MCP clients (claude.ai, Claude
    # Desktop) authenticate via DCR + PKCE against these endpoints;
    # headless clients (hooks, Cursor) use MNEMON_LOCAL_TOKEN instead.
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
    if not as_config.enabled and not config.local_token:
        print(
            "Auth: DISABLED — do not expose this server to the public internet. "
            "Set MNEMON_AS_ENABLED=true (with MNEMON_AS_PASSPHRASE + "
            "MNEMON_PUBLIC_URL) to enable the self-hosted Authorization "
            "Server, or MNEMON_LOCAL_TOKEN for headless bearer auth.",
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

    # Wire the persistent session manager BEFORE streamable_http_app() is
    # called — FastMCP lazy-initializes the manager on first call and
    # caches it. By assigning our subclass first, the lazy path is
    # skipped and our manager handles all requests. This is what lets
    # MCP sessions survive Fly auto_stop_machines: the in-memory dict
    # gets replaced with a SQLite-persisted one, and unknown-but-issued
    # session IDs are transparently resumed instead of 404'd.
    from .config import vault_dir
    from .persistent_sessions import PersistentSessionManager, SessionStore

    sessions_db = vault_dir() / "mcp_sessions.sqlite"
    session_store = SessionStore(sessions_db)
    expired = session_store.expire_old()
    if expired:
        print(
            f"Pruned {expired} expired MCP session(s) from {sessions_db}",
            file=sys.stderr,
        )
    # json_response=True flips StreamableHTTP into discrete request/
    # response mode (one POST → one JSON body, no long-lived SSE stream
    # per session). Required for mnemon: upstream's session-creation
    # lock is held for the full duration of `handle_request`, and in
    # SSE mode `handle_request` keeps the stream open until the client
    # disconnects — so once one session is open, every fresh-session
    # POST queues behind it indefinitely. mnemon's tools are all
    # single-shot RPCs (no streaming, no server-initiated messages),
    # so the SSE channel buys nothing and only exposes this hang.
    # Symptom this fixes: `mnemon doctor` and any `streamablehttp_client`
    # consumer timing out at session.initialize() while concurrent
    # requests sit in the lock queue.
    # Periodic confidence-decay sweep over the memory vault. Opens a
    # thread-local Store each tick because the sweep is dispatched via
    # anyio.to_thread.run_sync (sqlite3 connections default to
    # check_same_thread=True, so reusing the foreground singleton from
    # server.py would raise across the thread boundary). Decay is non-
    # destructive — it only adjusts the confidence column on aged
    # documents — so a transient failure here is safe to swallow.
    def _decay_sweep() -> int:
        from .contradiction import apply_confidence_decay
        from .store import Store
        store = Store()
        try:
            return apply_confidence_decay(store)
        finally:
            store.close()

    mcp._session_manager = PersistentSessionManager(
        app=mcp._mcp_server,
        session_store=session_store,
        event_store=mcp._event_store,
        retry_interval=mcp._retry_interval,
        json_response=True,
        stateless=mcp.settings.stateless_http,
        security_settings=mcp.settings.transport_security,
        decay_fn=_decay_sweep,
    )
    print(
        f"MCP sessions persisted to {sessions_db} "
        f"(survives cold-stops, TTL {session_store.ttl_seconds}s, "
        f"periodic prune every {mcp._session_manager._expire_interval_seconds}s, "
        f"periodic memory decay every {mcp._session_manager._decay_interval_seconds}s)",
        file=sys.stderr,
    )

    mcp_app = mcp.streamable_http_app()
    wrapped = OAuthMiddleware(
        mcp_app,
        config,
        as_config=as_config,
        metrics_provider=mcp._session_manager.metrics,
    )
    uvicorn.run(wrapped, host="0.0.0.0", port=PORT, log_level="info")
