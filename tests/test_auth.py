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
