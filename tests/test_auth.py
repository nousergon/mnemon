"""Tests for OAuth 2.1 resource-server middleware.

Exercises the OAuthMiddleware class with locally-minted RSA-signed JWTs,
mocking the JWKS fetch so the tests are offline and deterministic.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import jwt

from mnemon.auth import OAuthConfig, OAuthMiddleware


# --- Test fixtures ---------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair() -> dict[str, Any]:
    """Generate an RSA keypair once per module for signing test JWTs."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return {
        "private_key": private_key,
        "public_key": public_key,
        "private_pem": private_pem,
        "public_pem": public_pem,
    }


@pytest.fixture
def oauth_config() -> OAuthConfig:
    return OAuthConfig(
        issuer="https://test-issuer.example.com/",
        jwks_url="https://test-issuer.example.com/.well-known/jwks.json",
        audience="https://mnemon-test.example.com/mcp",
        public_url="https://mnemon-test.example.com",
    )


@pytest.fixture
def mock_signing_key(rsa_keypair):
    """Patch PyJWKClient.get_signing_key_from_jwt to return our test key."""

    class _FakeSigningKey:
        def __init__(self, key):
            self.key = key

    def _get_key(self, _token):
        return _FakeSigningKey(rsa_keypair["public_pem"])

    with patch(
        "jwt.PyJWKClient.get_signing_key_from_jwt", _get_key
    ):
        yield


def _mint_token(
    private_key: Any,
    *,
    issuer: str = "https://test-issuer.example.com/",
    audience: str = "https://mnemon-test.example.com/mcp",
    subject: str = "user-123",
    expires_in: int = 300,
    issued_at_offset: int = 0,
) -> str:
    now = int(time.time()) + issued_at_offset
    payload = {
        "iss": issuer,
        "aud": audience,
        "sub": subject,
        "iat": now,
        "exp": now + expires_in,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


async def _call_middleware(
    middleware: OAuthMiddleware,
    path: str,
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    downstream_called: list[bool] | None = None,
) -> tuple[int, dict[bytes, bytes], bytes]:
    """Invoke the middleware synchronously and collect the response.

    Returns ``(status, headers_dict, body)``.
    """
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
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {"type": "http.response.body", "body": b'{"ok": true}', "more_body": False}
        )

    return app


# --- Health endpoint -------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint_unauthenticated(oauth_config):
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, oauth_config)

    status, headers, body = await _call_middleware(mw, "/health")

    assert status == 200
    assert json.loads(body) == {"status": "ok"}
    assert downstream_called == []  # Did not forward to downstream


@pytest.mark.asyncio
async def test_health_endpoint_works_when_oauth_disabled():
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, OAuthConfig())

    status, _, body = await _call_middleware(mw, "/health")

    assert status == 200
    assert json.loads(body) == {"status": "ok"}
    assert downstream_called == []


# --- Protected Resource Metadata -------------------------------------------


@pytest.mark.asyncio
async def test_protected_resource_metadata_returns_rfc9728_json(oauth_config):
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, oauth_config)

    status, headers, body = await _call_middleware(
        mw, "/.well-known/oauth-protected-resource"
    )

    assert status == 200
    assert headers[b"content-type"] == b"application/json"
    data = json.loads(body)
    assert data["resource"] == oauth_config.audience
    assert data["authorization_servers"] == [oauth_config.issuer]
    assert data["bearer_methods_supported"] == ["header"]
    assert downstream_called == []


@pytest.mark.asyncio
async def test_protected_resource_metadata_404_when_oauth_disabled():
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, OAuthConfig())

    status, _, body = await _call_middleware(
        mw, "/.well-known/oauth-protected-resource"
    )

    assert status == 404
    assert "error" in json.loads(body)


# --- Authenticated paths — unauth mode passes through ---------------------


@pytest.mark.asyncio
async def test_oauth_disabled_passes_through_all_requests():
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, OAuthConfig())

    status, _, body = await _call_middleware(mw, "/mcp")

    assert status == 200
    assert json.loads(body) == {"ok": True}
    assert downstream_called == [True]


# --- Missing / malformed Authorization header -----------------------------


@pytest.mark.asyncio
async def test_missing_authorization_header_returns_401(oauth_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config)

    status, headers, body = await _call_middleware(mw, "/mcp")

    assert status == 401
    www_auth = headers[b"www-authenticate"].decode()
    assert "Bearer" in www_auth
    assert "resource_metadata=" in www_auth
    assert oauth_config.resource_metadata_url in www_auth
    assert json.loads(body)["error"] == "missing_token"


@pytest.mark.asyncio
async def test_non_bearer_authorization_returns_401(oauth_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config)

    status, _, _ = await _call_middleware(
        mw, "/mcp", headers=[(b"authorization", b"Basic dXNlcjpwYXNz")]
    )

    assert status == 401


# --- Valid JWT flow -------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_jwt_passes_through(oauth_config, rsa_keypair, mock_signing_key):
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, oauth_config)

    token = _mint_token(rsa_keypair["private_pem"])
    status, _, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", f"Bearer {token}".encode())],
    )

    assert status == 200
    assert json.loads(body) == {"ok": True}
    assert downstream_called == [True]


# --- Invalid JWT variants -------------------------------------------------


@pytest.mark.asyncio
async def test_expired_jwt_returns_401(oauth_config, rsa_keypair, mock_signing_key):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config)

    # issued 10 minutes ago, expires in 1 minute → expired 9 minutes ago
    token = _mint_token(
        rsa_keypair["private_pem"], expires_in=60, issued_at_offset=-600
    )
    status, headers, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", f"Bearer {token}".encode())],
    )

    assert status == 401
    assert b"error=" in headers[b"www-authenticate"]
    assert "expired" in json.loads(body).get("error_description", "")


@pytest.mark.asyncio
async def test_wrong_audience_returns_401(oauth_config, rsa_keypair, mock_signing_key):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config)

    token = _mint_token(
        rsa_keypair["private_pem"], audience="https://some-other-server.example.com/mcp"
    )
    status, _, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", f"Bearer {token}".encode())],
    )

    assert status == 401
    assert "audience" in json.loads(body).get("error_description", "").lower()


@pytest.mark.asyncio
async def test_wrong_issuer_returns_401(oauth_config, rsa_keypair, mock_signing_key):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config)

    token = _mint_token(
        rsa_keypair["private_pem"], issuer="https://evil-issuer.example.com/"
    )
    status, _, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", f"Bearer {token}".encode())],
    )

    assert status == 401
    assert "issuer" in json.loads(body).get("error_description", "").lower()


@pytest.mark.asyncio
async def test_malformed_jwt_returns_401(oauth_config):
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config)

    status, _, _ = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", b"Bearer not.a.valid.jwt.token")],
    )

    assert status == 401


# --- Config loading -------------------------------------------------------


def test_oauth_config_disabled_by_default(monkeypatch):
    for var in (
        "MNEMON_OAUTH_ISSUER",
        "MNEMON_OAUTH_JWKS_URL",
        "MNEMON_OAUTH_AUDIENCE",
        "MNEMON_PUBLIC_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    config = OAuthConfig.from_env()
    assert not config.enabled


def test_oauth_config_enabled_from_env(monkeypatch):
    monkeypatch.setenv("MNEMON_OAUTH_ISSUER", "https://issuer.example.com/")
    monkeypatch.setenv(
        "MNEMON_OAUTH_JWKS_URL", "https://issuer.example.com/.well-known/jwks.json"
    )
    monkeypatch.setenv("MNEMON_OAUTH_AUDIENCE", "https://mnemon.example.com/mcp")
    monkeypatch.setenv("MNEMON_PUBLIC_URL", "https://mnemon.example.com")
    config = OAuthConfig.from_env()
    assert config.enabled
    assert config.issuer == "https://issuer.example.com/"
    assert (
        config.resource_metadata_url
        == "https://mnemon.example.com/.well-known/oauth-protected-resource"
    )


def test_oauth_config_public_url_derived_from_audience():
    config = OAuthConfig(
        issuer="https://issuer.example.com/",
        jwks_url="https://issuer.example.com/jwks",
        audience="https://mnemon.example.com/mcp",
        public_url=None,
    )
    assert (
        config.resource_metadata_url
        == "https://mnemon.example.com/.well-known/oauth-protected-resource"
    )


# --- Userinfo fallback --------------------------------------------------


@pytest.fixture
def oauth_config_with_userinfo() -> OAuthConfig:
    return OAuthConfig(
        issuer="https://test-issuer.example.com/",
        jwks_url="https://test-issuer.example.com/.well-known/jwks.json",
        audience="https://mnemon-test.example.com/mcp",
        public_url="https://mnemon-test.example.com",
        userinfo_url="https://test-issuer.example.com/userinfo",
    )


@pytest.mark.asyncio
async def test_userinfo_fallback_accepts_opaque_token(
    oauth_config_with_userinfo, monkeypatch
):
    """When JWT decode fails and userinfo returns 200, token is accepted."""
    import httpx

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"sub": "auth0|user123", "email": "test@example.com"}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None):
            assert "Authorization" in headers
            assert headers["Authorization"].startswith("Bearer ")
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, oauth_config_with_userinfo)

    status, _, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", b"Bearer opaque-token-not-a-jwt")],
    )

    assert status == 200
    assert json.loads(body) == {"ok": True}
    assert downstream_called == [True]


@pytest.mark.asyncio
async def test_userinfo_fallback_rejects_when_userinfo_returns_401(
    oauth_config_with_userinfo, monkeypatch
):
    import httpx

    class _FakeResponse:
        status_code = 401

        def json(self):
            return {"error": "invalid_token"}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config_with_userinfo)

    status, _, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", b"Bearer opaque-bad-token")],
    )

    assert status == 401
    assert "userinfo rejected" in json.loads(body).get("error_description", "")


@pytest.mark.asyncio
async def test_no_userinfo_fallback_when_not_configured(oauth_config):
    """When userinfo_url is unset, JWT failures produce immediate 401."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config)  # no userinfo_url

    status, _, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", b"Bearer opaque-token")],
    )

    assert status == 401
    # Description should only mention JWT failure, not userinfo
    desc = json.loads(body).get("error_description", "")
    assert "userinfo" not in desc


# --- Local static bearer token path ---------------------------------------
#
# These tests cover the MNEMON_LOCAL_TOKEN auth path used by headless
# clients (Claude Code hooks, Cursor, scripts) that cannot complete a
# browser OAuth flow. The local token is checked before JWT/userinfo and
# bypasses both on match.


LOCAL_TOKEN_VALUE = "test-local-token-abcdef123456"


@pytest.fixture
def oauth_config_with_local_token(oauth_config) -> OAuthConfig:
    """OAuth config with BOTH OAuth and local_token set."""
    return OAuthConfig(
        issuer=oauth_config.issuer,
        jwks_url=oauth_config.jwks_url,
        audience=oauth_config.audience,
        public_url=oauth_config.public_url,
        local_token=LOCAL_TOKEN_VALUE,
    )


@pytest.fixture
def local_token_only_config() -> OAuthConfig:
    """Config with ONLY local_token set — OAuth disabled entirely.

    This previews the Phase 2 world where Auth0 is gone. If any middleware
    code path silently requires OAuth env vars, tests using this fixture
    will fail loudly.
    """
    return OAuthConfig(local_token=LOCAL_TOKEN_VALUE)


@pytest.mark.asyncio
async def test_local_token_accepted(oauth_config_with_local_token):
    """Correct local token bypasses JWT validation and reaches downstream."""
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, oauth_config_with_local_token)

    status, _, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", f"Bearer {LOCAL_TOKEN_VALUE}".encode())],
    )

    assert status == 200
    assert json.loads(body) == {"ok": True}
    assert downstream_called == [True]
    # JWT client must NOT have been initialized — the local-token path
    # should have short-circuited before _validate_token ran.
    assert mw._jwks_client is None


@pytest.mark.asyncio
async def test_local_token_wrong_value_returns_401_when_oauth_enabled(
    oauth_config_with_local_token, rsa_keypair, mock_signing_key
):
    """When OAuth is also configured, a wrong token falls through to JWT
    validation (which also fails here because the wrong value isn't a JWT).
    The downstream response must be a 401 with a proper WWW-Authenticate
    header — never a 200, never a silent pass-through."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config_with_local_token)

    status, headers, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", b"Bearer definitely-not-the-local-token")],
    )

    assert status == 401
    assert b"www-authenticate" in headers
    assert b"Bearer" in headers[b"www-authenticate"]
    assert json.loads(body)["error"] == "invalid_token"


@pytest.mark.asyncio
async def test_local_token_precedence_over_oauth(
    oauth_config_with_local_token,
):
    """When both auth methods are enabled and the token matches the local
    token, the JWT path must not run at all. Verified by inspecting that
    ``_jwks_client`` was never lazily initialized.

    This test deliberately does NOT use the ``mock_signing_key`` fixture —
    if the middleware incorrectly falls into the JWT path, PyJWKClient
    would attempt a real network fetch against the fake JWKS URL and the
    test would fail loudly rather than silently pass.
    """
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, oauth_config_with_local_token)

    status, _, _ = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", f"Bearer {LOCAL_TOKEN_VALUE}".encode())],
    )

    assert status == 200
    assert downstream_called == [True]
    assert mw._jwks_client is None


@pytest.mark.asyncio
async def test_local_token_works_without_oauth_config(local_token_only_config):
    """Local token auth must work with OAuth entirely unconfigured.

    This is a guardrail test for the Phase 2 / self-hosted-AS future: when
    ``MNEMON_OAUTH_*`` env vars are all unset, the middleware should still
    enforce auth via the local token path. Any regression here means a
    hidden Auth0/OAuth dependency has crept in.
    """
    downstream_called: list[bool] = []
    app = _stub_downstream_factory(downstream_called)
    mw = OAuthMiddleware(app, local_token_only_config)

    # Correct token → accepted, downstream called.
    status, _, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", f"Bearer {LOCAL_TOKEN_VALUE}".encode())],
    )
    assert status == 200
    assert json.loads(body) == {"ok": True}
    assert downstream_called == [True]
    assert mw._jwks_client is None


@pytest.mark.asyncio
async def test_local_token_only_rejects_wrong_token(local_token_only_config):
    """With only local token configured, wrong tokens must return 401 —
    not fall through to a JWT path that would raise on missing jwks_url."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_only_config)

    status, headers, body = await _call_middleware(
        mw,
        "/mcp",
        headers=[(b"authorization", b"Bearer wrong-token-value")],
    )

    assert status == 401
    assert b"www-authenticate" in headers
    assert json.loads(body)["error"] == "invalid_token"
    # JWT path must NOT have been entered — no jwks_url to fetch.
    assert mw._jwks_client is None


@pytest.mark.asyncio
async def test_local_token_only_missing_auth_header_returns_401(
    local_token_only_config,
):
    """Missing Authorization header returns 401 even when only local token
    is configured — auth is still required, just via a different path."""
    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, local_token_only_config)

    status, _, body = await _call_middleware(mw, "/mcp")

    assert status == 401
    assert json.loads(body)["error"] == "missing_token"


@pytest.mark.asyncio
async def test_local_token_logs_client_header(
    oauth_config_with_local_token, caplog
):
    """The X-Mnemon-Client header value should appear in the accept log
    line for attribution. The token value itself must NEVER be logged."""
    import logging as _logging

    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config_with_local_token)

    with caplog.at_level(_logging.INFO, logger="mnemon.auth"):
        status, _, _ = await _call_middleware(
            mw,
            "/mcp",
            headers=[
                (b"authorization", f"Bearer {LOCAL_TOKEN_VALUE}".encode()),
                (b"x-mnemon-client", b"claude-code"),
            ],
        )

    assert status == 200
    # Client label surfaced in logs.
    assert any("claude-code" in record.message for record in caplog.records)
    # Token value must not appear anywhere in log output.
    for record in caplog.records:
        assert LOCAL_TOKEN_VALUE not in record.message


@pytest.mark.asyncio
async def test_local_token_logs_unknown_when_header_missing(
    oauth_config_with_local_token, caplog
):
    """When no X-Mnemon-Client header is sent, the accept log records
    the client as 'unknown' rather than crashing or omitting the field."""
    import logging as _logging

    app = _stub_downstream_factory([])
    mw = OAuthMiddleware(app, oauth_config_with_local_token)

    with caplog.at_level(_logging.INFO, logger="mnemon.auth"):
        status, _, _ = await _call_middleware(
            mw,
            "/mcp",
            headers=[(b"authorization", f"Bearer {LOCAL_TOKEN_VALUE}".encode())],
        )

    assert status == 200
    assert any("unknown" in record.message for record in caplog.records)


# --- Config loading — local token path -----------------------------------


def test_oauth_config_local_token_none_by_default(monkeypatch):
    monkeypatch.delenv("MNEMON_LOCAL_TOKEN", raising=False)
    config = OAuthConfig.from_env()
    assert config.local_token is None


def test_oauth_config_loads_local_token_from_env(monkeypatch):
    monkeypatch.setenv("MNEMON_LOCAL_TOKEN", "env-loaded-token-value")
    config = OAuthConfig.from_env()
    assert config.local_token == "env-loaded-token-value"


def test_oauth_config_empty_local_token_coerced_to_none(monkeypatch):
    """Empty string env var is treated as unset (matches other env fields)."""
    monkeypatch.setenv("MNEMON_LOCAL_TOKEN", "")
    config = OAuthConfig.from_env()
    assert config.local_token is None
