"""OAuth 2.1 resource server middleware for mnemon remote MCP server.

Implements the MCP authorization specification (2025-06-18) resource-server
side: serves RFC 9728 Protected Resource Metadata, returns RFC 6750 compliant
401 responses with WWW-Authenticate header on missing/invalid tokens, and
validates JWT bearer tokens against mnemon's self-hosted Authorization
Server (see ``oauth_as.py``).

Two auth paths:

1. **Self-hosted AS JWTs** — for browser-capable MCP clients (claude.ai
   web/mobile, Claude Desktop) that complete a DCR + PKCE flow against
   our own /oauth/authorize + /oauth/token endpoints. Tokens are
   RS256-signed by our local keypair and verified with no network hop.

2. **Local static bearer** (``MNEMON_LOCAL_TOKEN``) — for headless
   clients (Claude Code hooks, Cursor, scripts) that cannot complete a
   browser OAuth flow. Compared constant-time against the env secret.

The middleware is pure ASGI (not Starlette BaseHTTPMiddleware) because
BaseHTTPMiddleware has known compatibility issues with mounted ASGI sub-apps
and streaming responses.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from .oauth_as import AuthorizationServerConfig

logger = logging.getLogger(__name__)

ASGIScope = dict[str, Any]
ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]


class OAuthConfig:
    """Resource-server auth configuration loaded from environment.

    Post–Phase 2 this class only carries the local bearer token used by
    headless clients. Browser-based OAuth is handled entirely by the
    self-hosted Authorization Server in ``oauth_as.py`` (wired as a
    separate ``AuthorizationServerConfig`` passed to the middleware).

    Retained as a distinct config type rather than merged into
    ``AuthorizationServerConfig`` because the local-token path is
    orthogonal to OAuth — a deployment can enable one without the other.
    """

    def __init__(self, local_token: str | None = None) -> None:
        # local_token is a static bearer accepted for headless clients that
        # cannot complete a browser OAuth flow. When set, the middleware
        # accepts requests whose bearer matches this value exactly (using a
        # constant-time comparison) without any network roundtrip. This value
        # must never be logged — only the ``X-Mnemon-Client`` header label is.
        self.local_token = local_token

    @classmethod
    def from_env(cls) -> OAuthConfig:
        return cls(local_token=os.environ.get("MNEMON_LOCAL_TOKEN") or None)


class OAuthMiddleware:
    """Pure-ASGI OAuth 2.1 resource-server middleware.

    Wraps a downstream ASGI application (the FastMCP streamable-http app).
    Intercepts four categories of requests:

    1. ``/health`` — served unauthenticated with ``{"status": "ok"}``.
    2. Well-known metadata endpoints (``/.well-known/oauth-protected-
       resource``, ``/.well-known/oauth-authorization-server``,
       ``/.well-known/jwks.json``) — served unauthenticated.
    3. OAuth AS endpoints (``/oauth/authorize``, ``/oauth/token``,
       ``/oauth/register``) — delegated to ``oauth_as.py`` handlers,
       served unauthenticated (they bootstrap auth).
    4. Everything else — requires a valid bearer token. Either the
       static ``MNEMON_LOCAL_TOKEN`` or a JWT issued by the self-hosted
       AS; on failure returns 401 with WWW-Authenticate header.

    Pass-through mode: if neither ``MNEMON_LOCAL_TOKEN`` nor
    ``MNEMON_AS_ENABLED=true`` is set, all non-health requests go
    through unauthenticated. Intended for local development only —
    never expose an unauthenticated server to the public internet.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: OAuthConfig,
        as_config: "AuthorizationServerConfig | None" = None,
        metrics_provider: "Callable[[], dict[str, int]] | None" = None,
    ) -> None:
        self.app = app
        self.config = config
        # Optional self-hosted Authorization Server config. When set and
        # enabled, the middleware serves the AS endpoints and validates
        # bearer JWTs against the local keypair.
        self.as_config = as_config
        # Optional callable returning a counters dict — surfaced via
        # /health for cold-stop diagnostics. Failures are swallowed so
        # /health stays a reliable Fly health check.
        self._metrics_provider = metrics_provider

    async def __call__(
        self, scope: ASGIScope, receive: ASGIReceive, send: ASGISend
    ) -> None:
        if scope["type"] != "http":
            # ASGI lifespan, websockets etc. — pass through unchanged.
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # Unauthenticated endpoints.
        if path == "/health":
            payload: dict[str, object] = {"status": "ok"}
            if self._metrics_provider is not None:
                try:
                    payload["metrics"] = self._metrics_provider()
                except Exception:  # noqa: BLE001
                    # Never let a metrics-collection bug 500 the health
                    # check — Fly relies on /health for liveness.
                    logger.exception("metrics_provider failed")
            await _send_json(send, 200, payload)
            return

        if path == "/.well-known/oauth-protected-resource":
            if not self._prm_enabled():
                await _send_json(
                    send, 404, {"error": "oauth not configured on this server"}
                )
                return
            await _send_json(send, 200, self._protected_resource_metadata())
            return

        # Self-hosted Authorization Server well-known endpoints (Phase 2).
        # These are served only when an AS config is wired AND enabled;
        # otherwise they 404 so clients can tell the difference between
        # "not hosting an AS" and "misconfigured."
        if path == "/.well-known/oauth-authorization-server":
            from .oauth_as import serve_as_metadata

            if self.as_config is None:
                await _send_json(
                    send, 404, {"error": "authorization server not enabled"}
                )
                return
            await serve_as_metadata(self.as_config, send)
            return

        if path == "/.well-known/jwks.json":
            from .oauth_as import serve_jwks

            if self.as_config is None:
                await _send_json(
                    send, 404, {"error": "authorization server not enabled"}
                )
                return
            await serve_jwks(self.as_config, send)
            return

        # Self-hosted AS token-issuance endpoints. Served unauthenticated
        # (the whole point is to bootstrap a token) — request validation
        # happens inside the handlers themselves (PKCE, passphrase, etc).
        if path == "/oauth/authorize":
            from .oauth_as import serve_authorize

            if self.as_config is None:
                await _send_json(
                    send, 404, {"error": "authorization server not enabled"}
                )
                return
            await serve_authorize(self.as_config, scope, receive, send)
            return

        if path == "/oauth/token":
            from .oauth_as import serve_token

            if self.as_config is None:
                await _send_json(
                    send, 404, {"error": "authorization server not enabled"}
                )
                return
            await serve_token(self.as_config, scope, receive, send)
            return

        if path == "/oauth/register":
            from .oauth_as import serve_register

            if self.as_config is None:
                await _send_json(
                    send, 404, {"error": "authorization server not enabled"}
                )
                return
            await serve_register(self.as_config, scope, receive, send)
            return

        # Pass-through mode: no auth configured at all. Health still
        # works (handled above), but everything else goes straight to
        # the downstream app. Local-development convenience only.
        as_enabled = self.as_config is not None and self.as_config.enabled
        if not self.config.local_token and not as_enabled:
            await self.app(scope, receive, send)
            return

        # Extract Authorization header (case-insensitive).
        auth_header = _get_header(scope, b"authorization")
        if not auth_header or not auth_header.lower().startswith(b"bearer "):
            await self._send_401(send, error="missing_token")
            return

        token = auth_header[7:].decode("ascii", errors="replace").strip()

        # Local-token fast path: static bearer validated directly against
        # MNEMON_LOCAL_TOKEN with a constant-time comparison. Used by
        # headless clients (Claude Code hooks, Cursor, scripts) that
        # cannot complete a browser OAuth flow. Checked first — no
        # key-loading cost for the common case.
        if self.config.local_token and hmac.compare_digest(
            token, self.config.local_token
        ):
            client_header = _get_header(scope, b"x-mnemon-client")
            client_label = (
                client_header.decode("ascii", errors="replace")
                if client_header
                else "unknown"
            )
            logger.info("Accepted local token from client=%s", client_label)
            await self.app(scope, receive, send)
            return

        # Self-hosted AS path: validate the JWT against the local JWKS.
        # No network hop — keys loaded from the same Fly volume that
        # signs them.
        if as_enabled:
            from .oauth_as import verify_self_hosted_token

            try:
                verify_self_hosted_token(self.as_config, token)
            except ValueError as e:
                logger.info("Self-hosted token validation failed: %s", e)
                await self._send_401(
                    send, error="invalid_token", description=str(e),
                )
                return
            await self.app(scope, receive, send)
            return

        # Only local_token was configured and the bearer didn't match —
        # reject. Token validation against an AS is not possible.
        await self._send_401(
            send,
            error="invalid_token",
            description="bearer token did not match local token",
        )

    def _prm_enabled(self) -> bool:
        """Whether to serve Protected Resource Metadata at all.

        Served only when the self-hosted AS is enabled. Not served when
        only ``MNEMON_LOCAL_TOKEN`` is configured — headless clients
        don't need PRM because there's no browser flow to bootstrap.
        """
        return self.as_config is not None and self.as_config.enabled

    def _protected_resource_metadata(self) -> dict[str, Any]:
        """Build the RFC 9728 Protected Resource Metadata JSON body.

        ``_prm_enabled()`` must be True before calling this — otherwise
        ``as_config`` may be None or disabled and the assertion fails.
        """
        assert self.as_config is not None and self.as_config.enabled
        issuer = self.as_config.issuer
        return {
            "resource": f"{issuer}/mcp",
            "authorization_servers": [issuer],
            "bearer_methods_supported": ["header"],
            "resource_documentation": "https://github.com/cipher813/mnemon",
        }

    def _resource_metadata_url(self) -> str:
        """Build the PRM URL for 401 WWW-Authenticate headers.

        Uses the AS public_url when available. Falls back to an empty
        string when no AS is configured (local_token-only deployments
        don't publish a PRM — the header points nowhere useful, but
        that's fine because headless clients don't read it anyway).
        """
        if self.as_config is not None and self.as_config.public_url:
            base = self.as_config.public_url.rstrip("/")
            return f"{base}/.well-known/oauth-protected-resource"
        return ""

    async def _send_401(
        self,
        send: ASGISend,
        *,
        error: str = "invalid_token",
        description: str | None = None,
    ) -> None:
        """Return 401 with RFC 6750 + RFC 9728 WWW-Authenticate header."""
        www_auth_parts = [
            'Bearer realm="mnemon"',
            f'resource_metadata="{self._resource_metadata_url()}"',
            f'error="{error}"',
        ]
        if description:
            # Escape quotes conservatively.
            safe_desc = description.replace('"', "'")
            www_auth_parts.append(f'error_description="{safe_desc}"')
        www_authenticate = ", ".join(www_auth_parts)

        body_obj: dict[str, Any] = {"error": error}
        if description:
            body_obj["error_description"] = description
        body = json.dumps(body_obj).encode("utf-8")

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"www-authenticate", www_authenticate.encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})


def _get_header(scope: ASGIScope, name: bytes) -> bytes | None:
    """Case-insensitive header lookup from ASGI scope."""
    name_lower = name.lower()
    for header_name, header_value in scope.get("headers", []):
        if header_name.lower() == name_lower:
            return header_value
    return None


async def _send_json(send: ASGISend, status: int, body: dict[str, Any]) -> None:
    """Minimal JSON response helper."""
    body_bytes = json.dumps(body).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body_bytes, "more_body": False})
