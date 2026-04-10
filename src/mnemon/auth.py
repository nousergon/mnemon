"""OAuth 2.1 resource server middleware for mnemon remote MCP server.

Implements the MCP authorization specification (2025-06-18) resource-server
side: serves RFC 9728 Protected Resource Metadata, returns RFC 6750 compliant
401 responses with WWW-Authenticate header on missing/invalid tokens, and
validates JWT bearer tokens against an external OAuth 2.1 authorization server.

This is Phase 1 of the mnemon OAuth build — the AS (authorization server) is
external (e.g., Auth0, Logto). Phase 2 will implement a self-hosted AS inside
mnemon. See private/mnemon-plan-optimized-260410.md for context.

The middleware is pure ASGI (not Starlette BaseHTTPMiddleware) because
BaseHTTPMiddleware has known compatibility issues with mounted ASGI sub-apps
and streaming responses — the original c5828c2 commit's bearer middleware
had a bug where it did not actually enforce auth for requests reaching the
mounted FastMCP app.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ASGIScope = dict[str, Any]
ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]


class OAuthConfig:
    """OAuth resource-server configuration loaded from environment.

    All fields are optional. If ``issuer``, ``jwks_url``, and ``audience`` are
    all set, the server operates in authenticated mode. Otherwise the server
    runs without auth (intended for local development only).
    """

    def __init__(
        self,
        issuer: str | None = None,
        jwks_url: str | None = None,
        audience: str | None = None,
        public_url: str | None = None,
    ) -> None:
        self.issuer = issuer
        self.jwks_url = jwks_url
        self.audience = audience
        # public_url is the externally-reachable base URL, used to build the
        # resource_metadata URL in WWW-Authenticate headers. Falls back to
        # audience's scheme+host if unset.
        self.public_url = public_url

    @classmethod
    def from_env(cls) -> OAuthConfig:
        return cls(
            issuer=os.environ.get("MNEMON_OAUTH_ISSUER") or None,
            jwks_url=os.environ.get("MNEMON_OAUTH_JWKS_URL") or None,
            audience=os.environ.get("MNEMON_OAUTH_AUDIENCE") or None,
            public_url=os.environ.get("MNEMON_PUBLIC_URL") or None,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.issuer and self.jwks_url and self.audience)

    @property
    def resource_metadata_url(self) -> str:
        """URL of the Protected Resource Metadata endpoint."""
        base = self.public_url or self._derive_base_from_audience()
        return f"{base.rstrip('/')}/.well-known/oauth-protected-resource"

    def _derive_base_from_audience(self) -> str:
        """Fallback: derive scheme+host from audience if public_url unset."""
        if not self.audience:
            return ""
        from urllib.parse import urlparse

        parsed = urlparse(self.audience)
        return f"{parsed.scheme}://{parsed.netloc}"


class OAuthMiddleware:
    """Pure-ASGI OAuth 2.1 resource-server middleware.

    Wraps a downstream ASGI application (the FastMCP streamable-http app).
    Intercepts three categories of requests:

    1. ``/health`` — served unauthenticated with ``{"status": "ok"}``.
    2. ``/.well-known/oauth-protected-resource`` — served unauthenticated
       with RFC 9728 Protected Resource Metadata JSON pointing at the
       configured authorization server.
    3. Everything else — requires a valid JWT bearer token. The token is
       validated against the configured JWKS; on failure, returns 401 with
       an RFC 6750 / RFC 9728 compliant ``WWW-Authenticate`` header.

    If the OAuth config is not enabled (any of issuer/jwks_url/audience
    unset), the middleware operates in pass-through mode: all requests go
    through unauthenticated, /health still works, and the resource metadata
    endpoint returns 404. This mirrors the pre-c5828c2 behavior of
    ``mnemon serve-remote`` without env vars.
    """

    def __init__(self, app: ASGIApp, config: OAuthConfig) -> None:
        self.app = app
        self.config = config
        self._jwks_client: Any = None  # Lazy-init PyJWKClient

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
            await _send_json(send, 200, {"status": "ok"})
            return

        if path == "/.well-known/oauth-protected-resource":
            if not self.config.enabled:
                await _send_json(
                    send, 404, {"error": "oauth not configured on this server"}
                )
                return
            await _send_json(send, 200, self._protected_resource_metadata())
            return

        # Unauthenticated mode — pass through.
        if not self.config.enabled:
            await self.app(scope, receive, send)
            return

        # Extract Authorization header (case-insensitive).
        auth_header = _get_header(scope, b"authorization")
        if not auth_header or not auth_header.lower().startswith(b"bearer "):
            await self._send_401(send, error="missing_token")
            return

        token = auth_header[7:].decode("ascii", errors="replace").strip()
        try:
            self._validate_token(token)
        except _OAuthError as e:
            logger.info("JWT validation failed: %s", e)
            await self._send_401(send, error=e.code, description=e.description)
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected error during JWT validation")
            await self._send_401(
                send, error="invalid_token", description=f"validation error: {e}"
            )
            return

        # Token valid — forward to downstream app.
        await self.app(scope, receive, send)

    def _protected_resource_metadata(self) -> dict[str, Any]:
        """Build the RFC 9728 Protected Resource Metadata JSON body."""
        assert self.config.audience and self.config.issuer
        return {
            "resource": self.config.audience,
            "authorization_servers": [self.config.issuer],
            "bearer_methods_supported": ["header"],
            "resource_documentation": "https://github.com/cipher813/mnemon",
        }

    def _validate_token(self, token: str) -> dict[str, Any]:
        """Validate JWT signature, issuer, audience, and expiry.

        Raises :class:`_OAuthError` on validation failure.
        """
        try:
            import jwt
            from jwt import PyJWKClient
        except ImportError as e:
            raise _OAuthError(
                "server_error",
                "pyjwt not installed — install mnemon-memory[server]",
            ) from e

        if self._jwks_client is None:
            assert self.config.jwks_url
            self._jwks_client = PyJWKClient(
                self.config.jwks_url, cache_keys=True, lifespan=3600
            )

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
        except jwt.PyJWKClientError as e:
            raise _OAuthError(
                "invalid_token", f"could not fetch signing key: {e}"
            ) from e
        except jwt.DecodeError as e:
            raise _OAuthError("invalid_token", f"malformed token: {e}") from e

        try:
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={"require": ["exp", "iat", "iss", "aud"]},
            )
        except jwt.ExpiredSignatureError as e:
            raise _OAuthError("invalid_token", "token expired") from e
        except jwt.InvalidAudienceError as e:
            raise _OAuthError(
                "invalid_token", f"audience mismatch (expected {self.config.audience})"
            ) from e
        except jwt.InvalidIssuerError as e:
            raise _OAuthError(
                "invalid_token", f"issuer mismatch (expected {self.config.issuer})"
            ) from e
        except jwt.InvalidTokenError as e:
            raise _OAuthError("invalid_token", str(e)) from e

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
            f'resource_metadata="{self.config.resource_metadata_url}"',
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


class _OAuthError(Exception):
    """Internal OAuth validation error — carries an RFC 6750 error code."""

    def __init__(self, code: str, description: str) -> None:
        super().__init__(f"{code}: {description}")
        self.code = code
        self.description = description


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
