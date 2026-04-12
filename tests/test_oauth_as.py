"""Tests for the self-hosted OAuth Authorization Server scaffolding.

Phase 2, PR #36 scope: key management, well-known metadata documents,
ASGI handlers. The endpoints themselves (/authorize, /token, /register)
are not yet implemented — tests for those land with PR #37/#38.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mnemon.oauth_as import (
    AuthorizationServerConfig,
    authorization_server_metadata,
    ensure_keypair,
    jwks_document,
    public_key_jwk,
)


# ── Config loading ──────────────────────────────────────────────────────────


class TestAuthorizationServerConfig:
    def test_defaults_to_disabled(self, monkeypatch):
        """A fresh environment with no AS vars set must produce a
        disabled config — guards against accidentally enabling the AS
        in production before the endpoints are implemented."""
        for var in ("MNEMON_AS_ENABLED", "MNEMON_AS_PASSPHRASE",
                    "MNEMON_AS_KEY_DIR", "MNEMON_PUBLIC_URL"):
            monkeypatch.delenv(var, raising=False)
        config = AuthorizationServerConfig.from_env()
        assert config.enabled is False

    def test_enabled_requires_public_url_and_passphrase(self, monkeypatch):
        monkeypatch.setenv("MNEMON_AS_ENABLED", "true")
        monkeypatch.delenv("MNEMON_PUBLIC_URL", raising=False)
        monkeypatch.delenv("MNEMON_AS_PASSPHRASE", raising=False)
        config = AuthorizationServerConfig.from_env()
        problems = config.validate()
        assert any("MNEMON_PUBLIC_URL" in p for p in problems)
        assert any("MNEMON_AS_PASSPHRASE" in p for p in problems)

    def test_enabled_with_full_config_validates(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MNEMON_AS_ENABLED", "true")
        monkeypatch.setenv("MNEMON_PUBLIC_URL", "https://example.fly.dev")
        monkeypatch.setenv("MNEMON_AS_PASSPHRASE", "secret")
        monkeypatch.setenv("MNEMON_AS_KEY_DIR", str(tmp_path))
        config = AuthorizationServerConfig.from_env()
        assert config.enabled is True
        assert config.issuer == "https://example.fly.dev"
        assert config.passphrase == "secret"
        assert config.key_dir == tmp_path
        assert config.validate() == []

    def test_disabled_passes_validation_regardless(self):
        """When disabled, the AS has nothing to validate — don't block
        server startup because optional AS vars happen to be empty."""
        config = AuthorizationServerConfig(enabled=False)
        assert config.validate() == []

    def test_issuer_strips_trailing_slash(self):
        """Issuer claim must be exact — RFC 8414 requires no trailing
        slash in the issuer value even if the public URL has one."""
        config = AuthorizationServerConfig(
            enabled=True,
            public_url="https://example.fly.dev/",
            passphrase="x",
        )
        assert config.issuer == "https://example.fly.dev"

    def test_boolean_parsing_accepts_common_truthy_values(self, monkeypatch):
        for val in ("1", "true", "TRUE", "yes"):
            monkeypatch.setenv("MNEMON_AS_ENABLED", val)
            config = AuthorizationServerConfig.from_env()
            assert config.enabled is True, f"expected {val!r} to enable AS"

    def test_boolean_parsing_rejects_falsy_values(self, monkeypatch):
        for val in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("MNEMON_AS_ENABLED", val)
            config = AuthorizationServerConfig.from_env()
            assert config.enabled is False, f"expected {val!r} to disable AS"


# ── Key management ──────────────────────────────────────────────────────────


class TestEnsureKeypair:
    def test_generates_new_keypair_on_first_call(self, tmp_path):
        private_pem, public_pem = ensure_keypair(tmp_path)
        assert private_pem.startswith(b"-----BEGIN PRIVATE KEY-----")
        assert public_pem.startswith(b"-----BEGIN PUBLIC KEY-----")
        assert (tmp_path / "private.pem").exists()

    def test_persists_private_key_with_0600_perms(self, tmp_path):
        ensure_keypair(tmp_path)
        mode = oct((tmp_path / "private.pem").stat().st_mode)[-3:]
        assert mode == "600", (
            "private key must be readable only by owner — leaking it "
            "lets anyone sign tokens"
        )

    def test_subsequent_calls_load_existing_key(self, tmp_path):
        """Repeated calls must be idempotent — an AS restart must not
        invalidate existing tokens by generating a fresh key."""
        first_priv, first_pub = ensure_keypair(tmp_path)
        second_priv, second_pub = ensure_keypair(tmp_path)
        assert first_priv == second_priv
        assert first_pub == second_pub

    def test_creates_missing_parent_directory(self, tmp_path):
        key_dir = tmp_path / "nonexistent" / "oauth_keys"
        ensure_keypair(key_dir)
        assert key_dir.exists()


# ── JWK / JWKS documents ────────────────────────────────────────────────────


class TestPublicKeyJwk:
    def test_includes_required_fields(self, tmp_path):
        jwk = public_key_jwk(tmp_path)
        required = {"kty", "alg", "use", "kid", "n", "e"}
        assert required.issubset(jwk.keys()), f"missing fields: {required - jwk.keys()}"

    def test_advertises_rs256(self, tmp_path):
        jwk = public_key_jwk(tmp_path)
        assert jwk["alg"] == "RS256"
        assert jwk["kty"] == "RSA"
        assert jwk["use"] == "sig"

    def test_kid_is_stable(self, tmp_path):
        """JWKS kid lets clients know which key signed a given JWT. It
        must not change across calls for the same keypair."""
        jwk_a = public_key_jwk(tmp_path)
        jwk_b = public_key_jwk(tmp_path)
        assert jwk_a["kid"] == jwk_b["kid"]


class TestJwksDocument:
    def test_wraps_public_key_in_keys_array(self, tmp_path):
        doc = jwks_document(tmp_path)
        assert "keys" in doc
        assert isinstance(doc["keys"], list)
        assert len(doc["keys"]) == 1
        assert doc["keys"][0]["kty"] == "RSA"


# ── Authorization Server metadata (RFC 8414) ────────────────────────────────


class TestAuthorizationServerMetadata:
    def test_metadata_shape(self, tmp_path):
        config = AuthorizationServerConfig(
            enabled=True,
            public_url="https://example.fly.dev",
            passphrase="x",
            key_dir=tmp_path,
        )
        meta = authorization_server_metadata(config)
        assert meta["issuer"] == "https://example.fly.dev"
        assert meta["authorization_endpoint"] == "https://example.fly.dev/oauth/authorize"
        assert meta["token_endpoint"] == "https://example.fly.dev/oauth/token"
        assert meta["registration_endpoint"] == "https://example.fly.dev/oauth/register"
        assert meta["jwks_uri"] == "https://example.fly.dev/.well-known/jwks.json"

    def test_only_supports_authorization_code_and_refresh(self, tmp_path):
        """Password grant, client_credentials, and implicit are deprecated
        and must not be advertised — OAuth 2.1 compliance."""
        config = AuthorizationServerConfig(
            enabled=True,
            public_url="https://x",
            passphrase="x",
            key_dir=tmp_path,
        )
        meta = authorization_server_metadata(config)
        assert meta["grant_types_supported"] == ["authorization_code", "refresh_token"]

    def test_only_supports_pkce_s256(self, tmp_path):
        config = AuthorizationServerConfig(
            enabled=True,
            public_url="https://x",
            passphrase="x",
            key_dir=tmp_path,
        )
        meta = authorization_server_metadata(config)
        assert meta["code_challenge_methods_supported"] == ["S256"]

    def test_public_client_only_no_client_secrets(self, tmp_path):
        """All MCP clients are public (no secure secret storage). Must
        advertise token_endpoint_auth_methods=["none"] so DCR doesn't
        hand out secrets it can't actually keep secret."""
        config = AuthorizationServerConfig(
            enabled=True,
            public_url="https://x",
            passphrase="x",
            key_dir=tmp_path,
        )
        meta = authorization_server_metadata(config)
        assert meta["token_endpoint_auth_methods_supported"] == ["none"]


# ── ASGI handlers ────────────────────────────────────────────────────────────


async def _capture_response(handler, config):
    """Invoke an ASGI send-only handler and return (status, body_json)."""
    sent = []

    async def send(msg):
        sent.append(msg)

    await handler(config, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    body_msg = next(m for m in sent if m["type"] == "http.response.body")
    body_json = json.loads(body_msg["body"].decode("utf-8"))
    return start["status"], body_json


class TestServeJwks:
    @pytest.mark.asyncio
    async def test_returns_404_when_as_disabled(self):
        from mnemon.oauth_as import serve_jwks

        config = AuthorizationServerConfig(enabled=False)
        status, body = await _capture_response(serve_jwks, config)
        assert status == 404
        assert "not enabled" in body["error"]

    @pytest.mark.asyncio
    async def test_returns_jwks_when_enabled(self, tmp_path):
        from mnemon.oauth_as import serve_jwks

        config = AuthorizationServerConfig(
            enabled=True,
            public_url="https://x",
            passphrase="x",
            key_dir=tmp_path,
        )
        status, body = await _capture_response(serve_jwks, config)
        assert status == 200
        assert "keys" in body
        assert body["keys"][0]["kty"] == "RSA"


class TestServeAsMetadata:
    @pytest.mark.asyncio
    async def test_returns_404_when_as_disabled(self):
        from mnemon.oauth_as import serve_as_metadata

        config = AuthorizationServerConfig(enabled=False)
        status, body = await _capture_response(serve_as_metadata, config)
        assert status == 404

    @pytest.mark.asyncio
    async def test_returns_metadata_when_enabled(self, tmp_path):
        from mnemon.oauth_as import serve_as_metadata

        config = AuthorizationServerConfig(
            enabled=True,
            public_url="https://example.fly.dev",
            passphrase="x",
            key_dir=tmp_path,
        )
        status, body = await _capture_response(serve_as_metadata, config)
        assert status == 200
        assert body["issuer"] == "https://example.fly.dev"


# ── /oauth/authorize ────────────────────────────────────────────────────────


import base64
import hashlib
from urllib.parse import parse_qs, urlparse


@pytest.fixture
def as_config(tmp_path):
    return AuthorizationServerConfig(
        enabled=True,
        public_url="https://example.fly.dev",
        passphrase="correct-horse-battery",
        key_dir=tmp_path,
    )


@pytest.fixture(autouse=True)
def reset_oauth_state():
    from mnemon.oauth_as import _reset_state_for_tests
    _reset_state_for_tests()
    yield
    _reset_state_for_tests()


def _pkce_pair():
    """Generate a PKCE (code_verifier, code_challenge) pair matching RFC 7636."""
    import secrets
    verifier = secrets.token_urlsafe(64)[:64]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


async def _run_asgi(handler, config, method="GET", query=b"", body=b""):
    """Invoke an AS ASGI handler and return (status, headers_dict, body_bytes)."""
    scope = {"type": "http", "method": method, "path": "/oauth/test",
             "query_string": query, "headers": []}
    sent = []

    async def send(msg):
        sent.append(msg)

    body_sent = {"done": False}

    async def receive():
        if body_sent["done"]:
            return {"type": "http.disconnect"}
        body_sent["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    # Handlers take different arg shapes depending on whether they need receive.
    import inspect
    sig = inspect.signature(handler)
    if "receive" in sig.parameters:
        await handler(config, scope, receive, send)
    else:
        await handler(config, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    body_msgs = [m for m in sent if m["type"] == "http.response.body"]
    body_bytes = b"".join(m.get("body", b"") for m in body_msgs)
    headers = {k.decode(): v.decode() for k, v in start.get("headers", [])}
    return start["status"], headers, body_bytes


class TestServeAuthorizeGet:
    @pytest.mark.asyncio
    async def test_404_when_as_disabled(self):
        from mnemon.oauth_as import serve_authorize

        config = AuthorizationServerConfig(enabled=False)
        status, _, _ = await _run_asgi(serve_authorize, config, method="GET")
        assert status == 404

    @pytest.mark.asyncio
    async def test_rejects_missing_params(self, as_config):
        from mnemon.oauth_as import serve_authorize

        status, _, body = await _run_asgi(
            serve_authorize, as_config, method="GET",
            query=b"client_id=x",  # everything else missing
        )
        assert status == 400
        assert b"missing required parameter" in body

    @pytest.mark.asyncio
    async def test_rejects_non_s256_pkce(self, as_config):
        from mnemon.oauth_as import serve_authorize

        query = (
            b"client_id=c&redirect_uri=https://x/cb&response_type=code&"
            b"code_challenge=abc&code_challenge_method=plain"
        )
        status, _, body = await _run_asgi(
            serve_authorize, as_config, method="GET", query=query
        )
        assert status == 400
        assert b"S256" in body

    @pytest.mark.asyncio
    async def test_rejects_non_code_response_type(self, as_config):
        """OAuth 2.1 removes implicit grant — only response_type=code."""
        from mnemon.oauth_as import serve_authorize

        query = (
            b"client_id=c&redirect_uri=https://x/cb&response_type=token&"
            b"code_challenge=abc&code_challenge_method=S256"
        )
        status, _, _ = await _run_asgi(
            serve_authorize, as_config, method="GET", query=query
        )
        assert status == 400

    @pytest.mark.asyncio
    async def test_renders_login_form_with_valid_params(self, as_config):
        from mnemon.oauth_as import serve_authorize

        _, challenge = _pkce_pair()
        query = (
            f"client_id=test-client&redirect_uri=https://client.example/cb&"
            f"response_type=code&code_challenge={challenge}&"
            f"code_challenge_method=S256&state=abc123"
        ).encode()
        status, headers, body = await _run_asgi(
            serve_authorize, as_config, method="GET", query=query
        )
        assert status == 200
        assert "text/html" in headers.get("content-type", "")
        # Hidden fields round-trip the authorize params back on POST
        assert b'name="client_id"' in body
        assert b'value="test-client"' in body
        assert b'name="state"' in body
        assert b'value="abc123"' in body
        # Never echo the passphrase field as a value
        assert b'name="passphrase"' in body

    @pytest.mark.asyncio
    async def test_escapes_html_in_params(self, as_config):
        """state and redirect_uri are untrusted — must be HTML-escaped to
        avoid XSS via the login form."""
        from mnemon.oauth_as import serve_authorize

        _, challenge = _pkce_pair()
        query = (
            f"client_id=<script>&redirect_uri=https://x/cb&response_type=code&"
            f"code_challenge={challenge}&code_challenge_method=S256&"
            f"state=<img src=x>"
        ).encode()
        _, _, body = await _run_asgi(
            serve_authorize, as_config, method="GET", query=query
        )
        assert b"<script>" not in body
        assert b"<img src=x>" not in body


class TestServeAuthorizePost:
    @pytest.mark.asyncio
    async def test_rejects_wrong_passphrase(self, as_config):
        from mnemon.oauth_as import serve_authorize

        _, challenge = _pkce_pair()
        form = (
            f"client_id=c&redirect_uri=https://x/cb&response_type=code&"
            f"code_challenge={challenge}&code_challenge_method=S256&"
            f"passphrase=wrong"
        ).encode()
        status, headers, body = await _run_asgi(
            serve_authorize, as_config, method="POST", body=form
        )
        assert status == 401
        assert "text/html" in headers.get("content-type", "")
        # Must NOT redirect back to client with a code on failure —
        # keeps the client from logging invalid-attempt URLs.
        assert "location" not in headers
        assert b"Invalid passphrase" in body

    @pytest.mark.asyncio
    async def test_empty_configured_passphrase_never_matches(self, tmp_path):
        """If AS is somehow enabled without a passphrase set, an empty
        submission must NOT match — guards the misconfig case."""
        from mnemon.oauth_as import serve_authorize

        # Directly construct config without validation to simulate boot
        # skipping the validate() call.
        config = AuthorizationServerConfig(
            enabled=True,
            public_url="https://example.fly.dev",
            passphrase="",  # misconfigured
            key_dir=tmp_path,
        )
        _, challenge = _pkce_pair()
        form = (
            f"client_id=c&redirect_uri=https://x/cb&response_type=code&"
            f"code_challenge={challenge}&code_challenge_method=S256&"
            f"passphrase="
        ).encode()
        status, _, _ = await _run_asgi(
            serve_authorize, config, method="POST", body=form
        )
        assert status == 401

    @pytest.mark.asyncio
    async def test_correct_passphrase_redirects_with_code(self, as_config):
        from mnemon.oauth_as import _auth_codes, serve_authorize

        _, challenge = _pkce_pair()
        form = (
            f"client_id=test-client&redirect_uri=https://client/cb&"
            f"response_type=code&code_challenge={challenge}&"
            f"code_challenge_method=S256&state=xyz&passphrase=correct-horse-battery"
        ).encode()
        status, headers, _ = await _run_asgi(
            serve_authorize, as_config, method="POST", body=form
        )
        assert status == 302
        location = headers["location"]
        parsed = urlparse(location)
        assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://client/cb"
        qs = parse_qs(parsed.query)
        assert "code" in qs
        assert qs["state"] == ["xyz"]
        # Code stored server-side for later exchange
        assert qs["code"][0] in _auth_codes


# ── /oauth/token ────────────────────────────────────────────────────────────


async def _issue_code(as_config, **overrides):
    """Helper: run an authorize POST with correct passphrase, return (code,
    code_verifier, params-used-for-issuance)."""
    from mnemon.oauth_as import serve_authorize

    verifier, challenge = _pkce_pair()
    params = {
        "client_id": "test-client",
        "redirect_uri": "https://client/cb",
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "mcp",
        "passphrase": "correct-horse-battery",
    }
    params.update(overrides)
    from urllib.parse import urlencode
    form = urlencode(params).encode()
    _, headers, _ = await _run_asgi(
        serve_authorize, as_config, method="POST", body=form
    )
    location = headers["location"]
    parsed = urlparse(location)
    code = parse_qs(parsed.query)["code"][0]
    return code, verifier, params


class TestServeTokenAuthorizationCode:
    @pytest.mark.asyncio
    async def test_404_when_as_disabled(self):
        from mnemon.oauth_as import serve_token

        config = AuthorizationServerConfig(enabled=False)
        status, _, _ = await _run_asgi(serve_token, config, method="POST")
        assert status == 404

    @pytest.mark.asyncio
    async def test_rejects_non_post(self, as_config):
        from mnemon.oauth_as import serve_token

        status, _, _ = await _run_asgi(serve_token, as_config, method="GET")
        assert status == 405

    @pytest.mark.asyncio
    async def test_rejects_unsupported_grant_type(self, as_config):
        from mnemon.oauth_as import serve_token

        form = b"grant_type=password"
        status, _, body = await _run_asgi(
            serve_token, as_config, method="POST", body=form
        )
        assert status == 400
        assert b"unsupported_grant_type" in body

    @pytest.mark.asyncio
    async def test_happy_path_returns_tokens(self, as_config):
        from mnemon.oauth_as import serve_token

        code, verifier, params = await _issue_code(as_config)
        from urllib.parse import urlencode
        form = urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": params["redirect_uri"],
            "client_id": params["client_id"],
            "code_verifier": verifier,
        }).encode()
        status, _, body_bytes = await _run_asgi(
            serve_token, as_config, method="POST", body=form
        )
        assert status == 200
        doc = json.loads(body_bytes)
        assert doc["token_type"] == "Bearer"
        assert doc["access_token"]
        assert doc["refresh_token"]
        assert doc["expires_in"] == 3600
        assert doc["scope"] == "mcp"

    @pytest.mark.asyncio
    async def test_access_token_verifies_against_jwks(self, as_config):
        """Minted access token must be verifiable with the public key
        published at /.well-known/jwks.json — the contract the future
        resource-server swap (PR #39) relies on."""
        import jwt
        from mnemon.oauth_as import jwks_document, serve_token

        code, verifier, params = await _issue_code(as_config)
        from urllib.parse import urlencode
        form = urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": params["redirect_uri"],
            "client_id": params["client_id"],
            "code_verifier": verifier,
        }).encode()
        _, _, body_bytes = await _run_asgi(
            serve_token, as_config, method="POST", body=form
        )
        token = json.loads(body_bytes)["access_token"]

        jwks = jwks_document(as_config.key_dir)
        signing_key = jwt.PyJWK(jwks["keys"][0])
        payload = jwt.decode(
            token, signing_key.key, algorithms=["RS256"],
            audience=f"{as_config.issuer}/mcp",
            issuer=as_config.issuer,
        )
        assert payload["sub"] == "owner"
        assert payload["scope"] == "mcp"

    @pytest.mark.asyncio
    async def test_code_is_single_use(self, as_config):
        from mnemon.oauth_as import serve_token

        code, verifier, params = await _issue_code(as_config)
        from urllib.parse import urlencode
        form = urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": params["redirect_uri"],
            "client_id": params["client_id"],
            "code_verifier": verifier,
        }).encode()

        # First exchange succeeds...
        status1, _, _ = await _run_asgi(serve_token, as_config, method="POST", body=form)
        assert status1 == 200
        # ...second with the same code fails.
        status2, _, body2 = await _run_asgi(serve_token, as_config, method="POST", body=form)
        assert status2 == 400
        assert b"invalid_grant" in body2

    @pytest.mark.asyncio
    async def test_wrong_code_verifier_rejected(self, as_config):
        from mnemon.oauth_as import serve_token

        code, _, params = await _issue_code(as_config)
        from urllib.parse import urlencode
        form = urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": params["redirect_uri"],
            "client_id": params["client_id"],
            "code_verifier": "a-different-verifier-than-the-one-used",
        }).encode()
        status, _, body = await _run_asgi(
            serve_token, as_config, method="POST", body=form
        )
        assert status == 400
        assert b"PKCE verification failed" in body

    @pytest.mark.asyncio
    async def test_wrong_redirect_uri_rejected(self, as_config):
        from mnemon.oauth_as import serve_token

        code, verifier, params = await _issue_code(as_config)
        from urllib.parse import urlencode
        form = urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": "https://attacker.example/cb",  # different
            "client_id": params["client_id"],
            "code_verifier": verifier,
        }).encode()
        status, _, body = await _run_asgi(
            serve_token, as_config, method="POST", body=form
        )
        assert status == 400
        assert b"redirect_uri" in body

    @pytest.mark.asyncio
    async def test_wrong_client_id_rejected(self, as_config):
        from mnemon.oauth_as import serve_token

        code, verifier, params = await _issue_code(as_config)
        from urllib.parse import urlencode
        form = urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": params["redirect_uri"],
            "client_id": "different-client",
            "code_verifier": verifier,
        }).encode()
        status, _, body = await _run_asgi(
            serve_token, as_config, method="POST", body=form
        )
        assert status == 400
        assert b"client_id" in body

    @pytest.mark.asyncio
    async def test_unknown_code_rejected(self, as_config):
        from mnemon.oauth_as import serve_token

        form = (
            b"grant_type=authorization_code&code=nope&redirect_uri=https://x/cb&"
            b"client_id=c&code_verifier=v"
        )
        status, _, body = await _run_asgi(
            serve_token, as_config, method="POST", body=form
        )
        assert status == 400
        assert b"invalid_grant" in body


class TestServeTokenRefreshGrant:
    @pytest.mark.asyncio
    async def test_refresh_token_rotates(self, as_config):
        """Rotating refresh tokens means each refresh consumes the old
        one and issues a new one. A leaked refresh token only works once."""
        from mnemon.oauth_as import serve_token

        # Get an initial pair.
        code, verifier, params = await _issue_code(as_config)
        from urllib.parse import urlencode
        code_form = urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": params["redirect_uri"],
            "client_id": params["client_id"],
            "code_verifier": verifier,
        }).encode()
        _, _, body1 = await _run_asgi(serve_token, as_config, method="POST", body=code_form)
        first = json.loads(body1)

        # Refresh → new pair.
        refresh_form = urlencode({
            "grant_type": "refresh_token",
            "refresh_token": first["refresh_token"],
        }).encode()
        status, _, body2 = await _run_asgi(
            serve_token, as_config, method="POST", body=refresh_form
        )
        assert status == 200
        second = json.loads(body2)
        assert second["access_token"] != first["access_token"]
        assert second["refresh_token"] != first["refresh_token"]

        # Old refresh token is now invalid.
        status3, _, body3 = await _run_asgi(
            serve_token, as_config, method="POST", body=refresh_form
        )
        assert status3 == 400
        assert b"invalid_grant" in body3

    @pytest.mark.asyncio
    async def test_unknown_refresh_token_rejected(self, as_config):
        from mnemon.oauth_as import serve_token

        form = b"grant_type=refresh_token&refresh_token=nope"
        status, _, body = await _run_asgi(
            serve_token, as_config, method="POST", body=form
        )
        assert status == 400
        assert b"invalid_grant" in body

    @pytest.mark.asyncio
    async def test_missing_refresh_token_rejected(self, as_config):
        from mnemon.oauth_as import serve_token

        form = b"grant_type=refresh_token"
        status, _, body = await _run_asgi(
            serve_token, as_config, method="POST", body=form
        )
        assert status == 400
        assert b"invalid_request" in body


class TestPKCEVerification:
    def test_correct_verifier_validates(self):
        from mnemon.oauth_as import _verify_pkce_s256
        verifier, challenge = _pkce_pair()
        assert _verify_pkce_s256(verifier, challenge) is True

    def test_wrong_verifier_fails(self):
        from mnemon.oauth_as import _verify_pkce_s256
        _, challenge = _pkce_pair()
        assert _verify_pkce_s256("not-the-verifier", challenge) is False


class TestMintAccessToken:
    def test_token_decodes_to_expected_claims(self, as_config):
        import jwt
        from mnemon.oauth_as import mint_access_token, public_key_jwk

        token = mint_access_token(
            as_config, subject="owner", scope="mcp", ttl_sec=60
        )
        jwk = public_key_jwk(as_config.key_dir)
        signing_key = jwt.PyJWK(jwk)
        payload = jwt.decode(
            token, signing_key.key, algorithms=["RS256"],
            audience=f"{as_config.issuer}/mcp",
            issuer=as_config.issuer,
        )
        assert payload["sub"] == "owner"
        assert payload["scope"] == "mcp"
        assert payload["iss"] == as_config.issuer
        assert payload["aud"] == f"{as_config.issuer}/mcp"

    def test_token_header_includes_kid(self, as_config):
        import jwt
        from mnemon.oauth_as import mint_access_token

        token = mint_access_token(as_config, subject="owner", scope="mcp")
        header = jwt.get_unverified_header(token)
        assert header["kid"] == "mnemon-as-1"
        assert header["alg"] == "RS256"
