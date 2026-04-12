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
