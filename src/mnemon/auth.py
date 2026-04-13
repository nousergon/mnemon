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

    All fields are optional. Two auth paths are supported and can be enabled
    independently or together:

    1. **OAuth 2.1 JWT / userinfo** — when ``issuer``, ``jwks_url``, and
       ``audience`` are set, the server accepts JWT bearer tokens from the
       configured authorization server. Intended for browser-capable MCP
       clients (claude.ai web/mobile, Claude Desktop) that can complete a
       DCR + PKCE flow.
    2. **Local static bearer** — when ``local_token`` is set, the server
       accepts a specific bearer token value directly, without any network
       roundtrip to the authorization server. Intended for headless clients
       that cannot complete a browser OAuth flow (Claude Code hooks, Cursor,
       scripts).

    If both are configured, the local token is checked first. If neither is
    configured, the server runs without auth (intended for local development
    only — do not expose an unauthenticated server to the public internet).
    """

    def __init__(
        self,
        issuer: str | None = None,
        jwks_url: str | None = None,
        audience: str | None = None,
        public_url: str | None = None,
        userinfo_url: str | None = None,
        local_token: str | None = None,
    ) -> None:
        self.issuer = issuer
        self.jwks_url = jwks_url
        self.audience = audience
        # public_url is the externally-reachable base URL, used to build the
        # resource_metadata URL in WWW-Authenticate headers. Falls back to
        # audience's scheme+host if unset.
        self.public_url = public_url
        # userinfo_url is an optional fallback for token introspection when
        # JWT validation fails. Needed because some OAuth providers issue
        # opaque tokens for OIDC-only flows instead of JWTs bound to the
        # requested audience. If unset, no fallback is used.
        self.userinfo_url = userinfo_url
        # local_token is a static bearer accepted for headless clients that
        # cannot complete a browser OAuth flow. When set, the middleware
        # accepts requests whose bearer matches this value exactly (using a
        # constant-time comparison) without any network roundtrip. This value
        # must never be logged — only the ``X-Mnemon-Client`` header label is.
        self.local_token = local_token

    @classmethod
    def from_env(cls) -> OAuthConfig:
        return cls(
            issuer=os.environ.get("MNEMON_OAUTH_ISSUER") or None,
            jwks_url=os.environ.get("MNEMON_OAUTH_JWKS_URL") or None,
            audience=os.environ.get("MNEMON_OAUTH_AUDIENCE") or None,
            public_url=os.environ.get("MNEMON_PUBLIC_URL") or None,
            userinfo_url=os.environ.get("MNEMON_OAUTH_USERINFO_URL") or None,
            local_token=os.environ.get("MNEMON_LOCAL_TOKEN") or None,
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

    def __init__(
        self,
        app: ASGIApp,
        config: OAuthConfig,
        as_config: "AuthorizationServerConfig | None" = None,
    ) -> None:
        self.app = app
        self.config = config
        # Optional self-hosted Authorization Server config. When set and
        # enabled, the middleware serves the AS well-known documents and
        # (in future PRs) the AS endpoints themselves. Kept as a separate
        # parameter from ``config`` so the resource-server concerns stay
        # decoupled from the AS concerns — callers can wire one without
        # the other.
        self.as_config = as_config
        self._jwks_client: Any = None  # Lazy-init PyJWKClient
        self._userinfo_cache: dict[str, float] = {}  # token hash -> expiry ts

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

        # Unauthenticated mode — pass through only when NEITHER auth method
        # is configured. If local_token is set but OAuth is not, we still
        # enforce auth below.
        if not self.config.enabled and not self.config.local_token:
            await self.app(scope, receive, send)
            return

        # Extract Authorization header (case-insensitive).
        auth_header = _get_header(scope, b"authorization")
        if not auth_header or not auth_header.lower().startswith(b"bearer "):
            await self._send_401(send, error="missing_token")
            return

        token = auth_header[7:].decode("ascii", errors="replace").strip()

        # Local-token fast path: static bearer validated directly against
        # MNEMON_LOCAL_TOKEN with a constant-time comparison. Used by clients
        # that cannot complete a browser OAuth flow (Claude Code hooks,
        # Cursor, headless scripts). Skips JWT and userinfo entirely — no
        # network hop, no external AS dependency. Checked first so local
        # clients don't pay the network cost of a failed JWT validation.
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

        # If only local_token is configured (no OAuth), any non-matching
        # token is invalid — do not fall through to JWT/userinfo because
        # neither is configured and _validate_token would raise on missing
        # jwks_url. This path also covers the self-hosted / Phase 2 world
        # where the OAuth block may be absent entirely.
        if not self.config.enabled:
            await self._send_401(
                send,
                error="invalid_token",
                description="bearer token did not match local token",
            )
            return

        # Try JWT validation first (spec-compliant path).
        jwt_error: _OAuthError | None = None
        try:
            self._validate_token(token)
        except _OAuthError as e:
            jwt_error = e
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected error during JWT validation")
            jwt_error = _OAuthError(
                "invalid_token", f"validation error: {e}"
            )

        if jwt_error is not None:
            # JWT validation failed. If a userinfo URL is configured, try
            # opaque-token introspection as a fallback. This works around
            # OAuth providers (notably Auth0) that issue opaque /userinfo
            # tokens instead of audience-bound JWTs for OIDC-only flows.
            if self.config.userinfo_url:
                try:
                    await self._validate_via_userinfo(token)
                except _OAuthError as ue:
                    logger.info(
                        "Both JWT and userinfo validation failed: jwt=%s userinfo=%s",
                        jwt_error,
                        ue,
                    )
                    await self._send_401(
                        send,
                        error=ue.code,
                        description=f"jwt: {jwt_error.description}; userinfo: {ue.description}",
                    )
                    return
                # Userinfo fallback succeeded — fall through.
                logger.info("Token validated via userinfo fallback (JWT failed)")
            else:
                logger.info("JWT validation failed: %s", jwt_error)
                await self._send_401(
                    send, error=jwt_error.code, description=jwt_error.description
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

    async def _validate_via_userinfo(self, token: str) -> dict[str, Any]:
        """Fallback: validate an opaque token by calling the OAuth provider's
        userinfo endpoint (OIDC Core section 5.3).

        Used when JWT decode fails — typically because the provider issued
        an opaque access token scoped to /userinfo rather than an audience-
        bound JWT. If the userinfo endpoint returns 200 with a user profile,
        the token is valid and we extract identity claims.

        Uses a small in-memory cache keyed by token hash to avoid repeat
        network calls within a short window. Returns the decoded userinfo
        payload on success. Raises :class:`_OAuthError` on failure.

        This is less strict than JWT audience validation — any token that
        the provider's /userinfo endpoint accepts will be allowed through,
        regardless of whether it was specifically issued for mnemon. That's
        acceptable for Phase 1 / single-user deployments. Phase 2's self-
        hosted AS will enforce proper audience binding.
        """
        import hashlib
        import time

        assert self.config.userinfo_url

        # Simple TTL cache: token_hash -> expiry_ts (accept until).
        now = time.time()
        token_hash = hashlib.sha256(token.encode("ascii", errors="replace")).hexdigest()
        cached_until = self._userinfo_cache.get(token_hash, 0)
        if cached_until > now:
            return {"sub": "cached"}

        try:
            import httpx
        except ImportError as e:
            raise _OAuthError(
                "server_error", "httpx not installed (required for userinfo fallback)"
            ) from e

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    self.config.userinfo_url,
                    headers={"Authorization": f"Bearer {token}"},
                )
        except httpx.TimeoutException as e:
            raise _OAuthError(
                "server_error", f"userinfo endpoint timeout: {e}"
            ) from e
        except httpx.HTTPError as e:
            raise _OAuthError(
                "server_error", f"userinfo endpoint error: {e}"
            ) from e

        if resp.status_code == 401 or resp.status_code == 403:
            raise _OAuthError(
                "invalid_token", "userinfo rejected token"
            )
        if resp.status_code != 200:
            raise _OAuthError(
                "invalid_token",
                f"userinfo returned {resp.status_code}",
            )

        try:
            profile = resp.json()
        except ValueError as e:
            raise _OAuthError(
                "invalid_token", f"userinfo returned non-JSON: {e}"
            ) from e

        if not isinstance(profile, dict) or "sub" not in profile:
            raise _OAuthError(
                "invalid_token", "userinfo response missing 'sub' claim"
            )

        # Cache for 5 minutes to avoid hammering the provider.
        self._userinfo_cache[token_hash] = now + 300
        # Opportunistic cache eviction to prevent unbounded growth.
        if len(self._userinfo_cache) > 256:
            expired = [k for k, v in self._userinfo_cache.items() if v <= now]
            for k in expired:
                self._userinfo_cache.pop(k, None)

        return profile

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
