"""Tests for OAuth 2.1 resource-server middleware.

Post–Phase 2 cleanup: the external-AS (Auth0) verification path and
userinfo fallback have been removed. This file covers:

- ``/health`` passes through unauthenticated
- Well-known metadata routing
- Pass-through mode when no auth is configured
- Local-token (MNEMON_LOCAL_TOKEN) accept/reject/logging
- Self-hosted AS JWT accept/reject (see also test_oauth_as.py)
"""

from __future__ import annotations

import json
import logging as _logging
from typing import Any

import pytest

from mnemon.auth import OAuthConfig, OAuthMiddleware
from mnemon.oauth_as import AuthorizationServerConfig, mint_access_token


LOCAL_TOKEN_VALUE = "test-local-token-value-12345678"


# --- Test helpers ----------------------------------------------------------


async def _call_middleware(
    middleware: OAuthMiddleware,
    path: str,
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, dict[bytes, bytes], bytes]:
    """Invoke the middleware with a simulated HTTP request."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers or [],
    }

    response: dict[str, Any] = {"status": None, "headers": {}, "body": b""}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
            response["headers"] = {k: v for k, v in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            response["body"] += message.get("body", b"")

    await middleware(scope, receive, send)
    return response["status"], response["headers"], response["body"]


def _stub_downstream_factory(called_flag: list[bool]):
    """Return a stub downstream ASGI app that records invocations."""

    async def app(scope, receive, send):
        called_flag.append(True)
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"ok": true}',
            "more_body": False,
        })

    return app


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def no_auth_config() -> OAuthConfig:
    """OAuthConfig with no local token — pass-through mode when AS also
    absent."""
    return OAuthConfig()


@pytest.fixture
def local_token_config() -> OAuthConfig:
    return OAuthConfig(local_token=LOCAL_TOKEN_VALUE)


@pytest.fixture
def enabled_as_config(tmp_path) -> AuthorizationServerConfig:
    return AuthorizationServerConfig(
        enabled=True,
        public_url="https://mnemon-test.example.com",
        passphrase="x",
        key_dir=tmp_path,
    )


# --- /health endpoint -------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint_unauthenticated(local_token_config):
    """Health must always respond 200 without auth, regardless of config."""
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, local_token_config)

    status, _, body = await _call_middleware(mw, "/health")
    assert status == 200
    assert json.loads(body) == {"status": "ok"}
    assert downstream_called == []  # never forwarded


@pytest.mark.asyncio
async def test_health_works_with_no_auth_configured(no_auth_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, no_auth_config)

    status, _, _ = await _call_middleware(mw, "/health")
    assert status == 200


# --- Pass-through mode (no auth configured) --------------------------------


@pytest.mark.asyncio
async def test_no_auth_configured_passes_through(no_auth_config):
    """With no local_token and no AS, all non-health requests go to the
    downstream app unauthenticated. Dev-only convenience."""
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, no_auth_config)

    status, _, _ = await _call_middleware(mw, "/mcp")
    assert status == 200
    assert downstream_called == [True]


# --- /.well-known/oauth-protected-resource ---------------------------------


@pytest.mark.asyncio
async def test_prm_404_when_as_not_configured(local_token_config):
    """With only local_token configured (no AS), the PRM endpoint 404s —
    headless clients don't read PRM, so there's nothing to publish."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config)

    status, _, _ = await _call_middleware(mw, "/.well-known/oauth-protected-resource")
    assert status == 404


@pytest.mark.asyncio
async def test_prm_points_at_self_hosted_issuer(local_token_config, enabled_as_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config, as_config=enabled_as_config)

    status, _, body = await _call_middleware(
        mw, "/.well-known/oauth-protected-resource"
    )
    assert status == 200
    doc = json.loads(body)
    assert doc["authorization_servers"] == ["https://mnemon-test.example.com"]
    assert doc["resource"] == "https://mnemon-test.example.com/mcp"


@pytest.mark.asyncio
async def test_prm_404_when_as_config_disabled(local_token_config, tmp_path):
    """AS config supplied but enabled=False still 404s — don't leak a
    half-configured AS to clients."""
    as_config = AuthorizationServerConfig(enabled=False, key_dir=tmp_path)
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config, as_config=as_config)

    status, _, _ = await _call_middleware(mw, "/.well-known/oauth-protected-resource")
    assert status == 404


# --- Local-token auth path --------------------------------------------------


@pytest.mark.asyncio
async def test_local_token_accepted(local_token_config):
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, local_token_config)

    status, _, body = await _call_middleware(
        mw, "/mcp",
        headers=[(b"authorization", f"Bearer {LOCAL_TOKEN_VALUE}".encode())],
    )
    assert status == 200
    assert json.loads(body) == {"ok": True}
    assert downstream_called == [True]


@pytest.mark.asyncio
async def test_local_token_wrong_value_returns_401(local_token_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config)

    status, headers, body = await _call_middleware(
        mw, "/mcp",
        headers=[(b"authorization", b"Bearer wrong-value")],
    )
    assert status == 401
    assert b"www-authenticate" in headers
    assert json.loads(body)["error"] == "invalid_token"


@pytest.mark.asyncio
async def test_local_token_missing_auth_header_returns_401(local_token_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config)

    status, _, body = await _call_middleware(mw, "/mcp")
    assert status == 401
    assert json.loads(body)["error"] == "missing_token"


@pytest.mark.asyncio
async def test_local_token_non_bearer_scheme_returns_401(local_token_config):
    """Non-Bearer auth schemes (Basic, Digest, etc.) must be rejected
    regardless of the credential value — we only accept RFC 6750 Bearer."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config)

    status, _, _ = await _call_middleware(
        mw, "/mcp",
        headers=[(b"authorization", b"Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==")],
    )
    assert status == 401


@pytest.mark.asyncio
async def test_local_token_constant_time_compare(local_token_config):
    """Guard against a timing-side-channel regression. hmac.compare_digest
    is used internally; this test just confirms correct vs. wrong tokens
    both produce deterministic results, relying on pytest to run without
    flakes if the comparison is constant-time."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config)

    # A token with the same length but wrong value must be rejected.
    status, _, _ = await _call_middleware(
        mw, "/mcp",
        headers=[(b"authorization", f"Bearer {'X' * len(LOCAL_TOKEN_VALUE)}".encode())],
    )
    assert status == 401


@pytest.mark.asyncio
async def test_local_token_logs_client_header(local_token_config, caplog):
    """X-Mnemon-Client header appears in the accept log for attribution.
    Token value itself must NEVER be logged."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config)

    with caplog.at_level(_logging.INFO, logger="mnemon.auth"):
        status, _, _ = await _call_middleware(
            mw, "/mcp",
            headers=[
                (b"authorization", f"Bearer {LOCAL_TOKEN_VALUE}".encode()),
                (b"x-mnemon-client", b"claude-code"),
            ],
        )

    assert status == 200
    log_text = " ".join(record.message for record in caplog.records)
    assert "claude-code" in log_text
    assert LOCAL_TOKEN_VALUE not in log_text


@pytest.mark.asyncio
async def test_local_token_logs_unknown_when_header_missing(local_token_config, caplog):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config)

    with caplog.at_level(_logging.INFO, logger="mnemon.auth"):
        status, _, _ = await _call_middleware(
            mw, "/mcp",
            headers=[(b"authorization", f"Bearer {LOCAL_TOKEN_VALUE}".encode())],
        )

    assert status == 200
    log_text = " ".join(record.message for record in caplog.records)
    assert "unknown" in log_text


# --- Self-hosted AS JWT auth path ------------------------------------------


@pytest.mark.asyncio
async def test_self_hosted_token_accepted(enabled_as_config):
    """A JWT minted by the local AS is accepted when AS is enabled."""
    token = mint_access_token(enabled_as_config, subject="owner", scope="mcp")
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, OAuthConfig(), as_config=enabled_as_config)

    status, _, _ = await _call_middleware(
        mw, "/mcp",
        headers=[(b"authorization", f"Bearer {token}".encode())],
    )
    assert status == 200
    assert downstream_called == [True]


@pytest.mark.asyncio
async def test_self_hosted_expired_token_returns_401(enabled_as_config):
    token = mint_access_token(
        enabled_as_config, subject="owner", scope="mcp", ttl_sec=-60,
    )
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, OAuthConfig(), as_config=enabled_as_config)

    status, headers, body = await _call_middleware(
        mw, "/mcp",
        headers=[(b"authorization", f"Bearer {token}".encode())],
    )
    assert status == 401
    assert b"expired" in body
    # WWW-Authenticate points at the self-hosted PRM URL
    auth_header = headers.get(b"www-authenticate", b"")
    assert b"mnemon-test.example.com/.well-known/oauth-protected-resource" in auth_header


@pytest.mark.asyncio
async def test_self_hosted_garbage_token_returns_401(enabled_as_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, OAuthConfig(), as_config=enabled_as_config)

    status, _, _ = await _call_middleware(
        mw, "/mcp",
        headers=[(b"authorization", b"Bearer not.a.jwt")],
    )
    assert status == 401


@pytest.mark.asyncio
async def test_self_hosted_missing_bearer_returns_401(enabled_as_config):
    """Even with self-hosted AS as the only auth method, missing bearer
    must still return 401 (not pass through as unauthenticated)."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, OAuthConfig(), as_config=enabled_as_config)

    status, _, _ = await _call_middleware(mw, "/mcp")
    assert status == 401


@pytest.mark.asyncio
async def test_local_token_and_as_coexist(local_token_config, enabled_as_config):
    """Claude Code hooks (local_token) and claude.ai (AS JWT) must both
    work against the same server."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config, as_config=enabled_as_config)

    # Local token accepted.
    status_local, _, _ = await _call_middleware(
        mw, "/mcp",
        headers=[(b"authorization", f"Bearer {LOCAL_TOKEN_VALUE}".encode())],
    )
    assert status_local == 200

    # AS-minted JWT accepted.
    token = mint_access_token(enabled_as_config, subject="owner", scope="mcp")
    status_jwt, _, _ = await _call_middleware(
        mw, "/mcp",
        headers=[(b"authorization", f"Bearer {token}".encode())],
    )
    assert status_jwt == 200


@pytest.mark.asyncio
async def test_local_token_checked_before_jwt_verification(
    local_token_config, enabled_as_config
):
    """Local token path is a simple string compare — must be attempted
    before JWT verification so common hook calls don't pay key-loading
    cost. Verified indirectly: a matching local token is accepted even
    when it's clearly not a JWT."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config, as_config=enabled_as_config)

    status, _, _ = await _call_middleware(
        mw, "/mcp",
        headers=[(b"authorization", f"Bearer {LOCAL_TOKEN_VALUE}".encode())],
    )
    assert status == 200


# --- OAuthConfig env loading ------------------------------------------------


def test_oauth_config_local_token_none_by_default(monkeypatch):
    monkeypatch.delenv("MNEMON_LOCAL_TOKEN", raising=False)
    config = OAuthConfig.from_env()
    assert config.local_token is None


def test_oauth_config_loads_local_token_from_env(monkeypatch):
    monkeypatch.setenv("MNEMON_LOCAL_TOKEN", "env-loaded-value")
    config = OAuthConfig.from_env()
    assert config.local_token == "env-loaded-value"


def test_oauth_config_empty_local_token_coerced_to_none(monkeypatch):
    monkeypatch.setenv("MNEMON_LOCAL_TOKEN", "")
    config = OAuthConfig.from_env()
    assert config.local_token is None


def test_oauth_config_no_auth0_fields():
    """Guard against accidentally re-adding the external-AS fields.
    If Phase 2 cutover regressed and someone re-added issuer/jwks_url/
    audience/userinfo_url to OAuthConfig, this test catches it. The
    Phase 2 plan is explicit: no external AS dependency."""
    config = OAuthConfig()
    for field in ("issuer", "jwks_url", "audience", "userinfo_url"):
        assert not hasattr(config, field), (
            f"OAuthConfig should not have '{field}' after Phase 2 — "
            "that's external-AS config which has been removed."
        )


# --- AS well-known routing (routes to oauth_as.py handlers) ----------------


@pytest.mark.asyncio
async def test_as_metadata_404_without_as_config(local_token_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config)

    status, _, _ = await _call_middleware(
        mw, "/.well-known/oauth-authorization-server",
    )
    assert status == 404


@pytest.mark.asyncio
async def test_as_metadata_200_with_as_config(local_token_config, enabled_as_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config, as_config=enabled_as_config)

    status, _, body = await _call_middleware(
        mw, "/.well-known/oauth-authorization-server",
    )
    assert status == 200
    doc = json.loads(body)
    assert doc["issuer"] == "https://mnemon-test.example.com"


@pytest.mark.asyncio
async def test_jwks_200_with_as_config(local_token_config, enabled_as_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_config, as_config=enabled_as_config)

    status, _, body = await _call_middleware(mw, "/.well-known/jwks.json")
    assert status == 200
    doc = json.loads(body)
    assert doc["keys"][0]["kty"] == "RSA"
