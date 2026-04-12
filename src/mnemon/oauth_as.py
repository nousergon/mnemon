"""Self-hosted OAuth 2.1 Authorization Server — Phase 2 scaffolding.

Removes mnemon's external-AS (Auth0) dependency so self-hosted deployments
can run as a single process with no third-party auth vendor. This module
is the AS-side counterpart to ``auth.py`` (which is the Resource Server).

Current scope (PR #36 — scaffolding only)
-----------------------------------------
This PR lays the groundwork for the AS endpoints to follow in subsequent
PRs. What ships here:

- RSA keypair management: generated on first run, persisted to the Fly
  volume, loaded on subsequent starts. Private key never leaves the
  server; public key is published at ``/.well-known/jwks.json``.
- ``/.well-known/oauth-authorization-server`` metadata (RFC 8414). Some
  endpoint URLs point at routes that don't exist yet — clients will get
  404s if they try to use them, which is fine while this is scaffolding.
- ``/.well-known/jwks.json`` — the public key in JWKS format for any
  future resource server (mnemon itself, in Phase 2) to verify tokens.
- Feature flag ``MNEMON_AS_ENABLED``. Off by default so this PR is a
  no-op on the live deployment; turn it on in a later PR when the
  endpoints are implemented.

Out of scope (future PRs)
-------------------------
- ``/authorize`` + ``/token`` endpoints + PKCE — PR #37
- ``/register`` (DCR, RFC 7591) + refresh rotation — PR #38
- Switch resource-server middleware to verify self-hosted tokens — PR #39
- Remove Auth0 env vars from production config — PR #40

Design notes
------------
- **Single-user only.** This AS does not implement a user database. The
  server owner is the sole authorized principal, authenticated by a
  setup-time passphrase (``MNEMON_AS_PASSPHRASE``). No signup UI, no
  password reset, no account recovery — intentional scope reduction for
  personal self-host. Multi-user is explicitly out of scope indefinitely.
- **RS256 signing.** Standard for OIDC/OAuth 2.1; matches what Auth0 was
  doing, so Phase 2 cutover doesn't break existing client assumptions.
- **Key storage on Fly volume.** ``/data/oauth_keys/private.pem`` persists
  across deploys because the volume does. If the volume is destroyed,
  all issued tokens become invalid — acceptable for personal use; in
  production you'd back up the key or accept forced re-registration.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Environment variable names — keep these provider-agnostic so Phase 2
# swap from Auth0 to self-hosted is env-var-only, no code changes.
ENV_AS_ENABLED = "MNEMON_AS_ENABLED"
ENV_AS_PASSPHRASE = "MNEMON_AS_PASSPHRASE"
ENV_PUBLIC_URL = "MNEMON_PUBLIC_URL"
ENV_KEY_DIR = "MNEMON_AS_KEY_DIR"

# RS256 is the default for OIDC/OAuth 2.1 JWT signing. Don't add alg
# negotiation — pick one and stick with it.
JWT_ALG = "RS256"
KEY_ID = "mnemon-as-1"  # bump if rotating keys in a future PR


class AuthorizationServerConfig:
    """Configuration for the self-hosted AS loaded from environment.

    Attributes:
        enabled: whether the AS is active. When False, all AS endpoints
            (well-known metadata, JWKS, /authorize, /token, /register)
            return 404 and the resource server stays on Auth0.
        public_url: externally-reachable base URL (e.g.,
            ``https://mnemon-memory.fly.dev``). Used to build issuer
            claim and endpoint URLs in metadata documents.
        passphrase: single-user login credential. Required when enabled;
            server refuses to start without it.
        key_dir: directory holding the RSA keypair. Defaults to
            ``/data/oauth_keys`` on Fly, ``~/.mnemon/oauth_keys``
            locally.
    """

    def __init__(
        self,
        enabled: bool = False,
        public_url: str | None = None,
        passphrase: str | None = None,
        key_dir: Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.public_url = public_url
        self.passphrase = passphrase
        self.key_dir = key_dir or _default_key_dir()

    @classmethod
    def from_env(cls) -> AuthorizationServerConfig:
        enabled = os.environ.get(ENV_AS_ENABLED, "").lower() in ("1", "true", "yes")
        public_url = os.environ.get(ENV_PUBLIC_URL) or None
        passphrase = os.environ.get(ENV_AS_PASSPHRASE) or None
        key_dir_env = os.environ.get(ENV_KEY_DIR)
        key_dir = Path(key_dir_env) if key_dir_env else _default_key_dir()
        return cls(enabled=enabled, public_url=public_url,
                   passphrase=passphrase, key_dir=key_dir)

    @property
    def issuer(self) -> str:
        """The ``iss`` claim value for tokens the AS issues.

        Per OAuth 2.1 Authorization Server Metadata (RFC 8414), this is
        also the base from which well-known URLs are built (appending
        ``/.well-known/oauth-authorization-server``).
        """
        return (self.public_url or "").rstrip("/")

    def validate(self) -> list[str]:
        """Return a list of config problems, empty if OK to start.

        Callers should refuse to enable the AS when this returns a
        non-empty list — better to fail fast at boot than serve broken
        metadata.
        """
        problems: list[str] = []
        if not self.enabled:
            return problems
        if not self.public_url:
            problems.append(f"{ENV_PUBLIC_URL} must be set when AS is enabled")
        if not self.passphrase:
            problems.append(
                f"{ENV_AS_PASSPHRASE} must be set when AS is enabled "
                "(single-user login credential)"
            )
        return problems


def _default_key_dir() -> Path:
    """Return the default key directory.

    Uses ``MNEMON_VAULT_DIR/oauth_keys`` so keys live alongside the
    vault data on the Fly volume. Falls back to ``~/.mnemon/oauth_keys``
    for local development.
    """
    vault_dir = os.environ.get("MNEMON_VAULT_DIR")
    if vault_dir:
        return Path(vault_dir) / "oauth_keys"
    return Path.home() / ".mnemon" / "oauth_keys"


# ── Key management ───────────────────────────────────────────────────────────


def ensure_keypair(key_dir: Path) -> tuple[bytes, bytes]:
    """Load or generate the AS signing keypair.

    On first call, generates a fresh 2048-bit RSA keypair and persists
    the private key to ``key_dir/private.pem`` with 0600 permissions.
    Subsequent calls load the existing key. The public key is derived
    from the private key on every call (cheap) rather than stored
    separately to avoid the private/public drift that comes from two
    separate files.

    Returns:
        A ``(private_pem, public_pem)`` tuple. Both are PEM-encoded bytes.

    Raises:
        ImportError: if cryptography is not installed (required by authlib).
        OSError: if the key directory cannot be created.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key_dir.mkdir(parents=True, exist_ok=True)
    private_path = key_dir / "private.pem"

    if private_path.exists():
        private_pem = private_path.read_bytes()
        private_key = serialization.load_pem_private_key(private_pem, password=None)
    else:
        logger.info(
            "oauth_as: generating new RSA keypair at %s (first run)",
            private_path,
        )
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048
        )
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        private_path.write_bytes(private_pem)
        private_path.chmod(0o600)

    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def public_key_jwk(key_dir: Path) -> dict[str, Any]:
    """Return the AS public key in JWK format (RFC 7517).

    Used to serve ``/.well-known/jwks.json``. The returned dict always
    contains ``kty``, ``alg``, ``use``, ``kid``, ``n``, ``e``.
    """
    from authlib.jose import JsonWebKey

    _, public_pem = ensure_keypair(key_dir)
    jwk = JsonWebKey.import_key(public_pem, {"kty": "RSA"}).as_dict()
    jwk.update({
        "alg": JWT_ALG,
        "use": "sig",
        "kid": KEY_ID,
    })
    return jwk


def jwks_document(key_dir: Path) -> dict[str, Any]:
    """Return the full JWKS document (RFC 7517) for ``/.well-known/jwks.json``."""
    return {"keys": [public_key_jwk(key_dir)]}


# ── Metadata ────────────────────────────────────────────────────────────────


def authorization_server_metadata(config: AuthorizationServerConfig) -> dict[str, Any]:
    """Return RFC 8414 Authorization Server Metadata.

    Served at ``/.well-known/oauth-authorization-server``. Some of the
    endpoint URLs point at routes that are not yet implemented (see the
    module docstring). That's expected for PR #36; clients will fail
    loudly if they try to use them before PRs #37/#38 land, which is
    better than silently half-working.
    """
    issuer = config.issuer
    return {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/oauth/authorize",
        "token_endpoint": f"{issuer}/oauth/token",
        "registration_endpoint": f"{issuer}/oauth/register",
        "jwks_uri": f"{issuer}/.well-known/jwks.json",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],  # PKCE-only, no client secrets
        "scopes_supported": ["mcp"],
        "service_documentation": "https://github.com/cipher813/mnemon",
    }


# ── ASGI handlers ────────────────────────────────────────────────────────────


async def serve_jwks(config: AuthorizationServerConfig, send) -> None:
    """ASGI handler: serve ``/.well-known/jwks.json``.

    Returns 404 when the AS is disabled so clients can tell the
    difference between "this server doesn't host its own AS" and "AS is
    misconfigured."
    """
    from .auth import _send_json  # reuse existing JSON helper

    if not config.enabled:
        await _send_json(send, 404, {"error": "authorization server not enabled"})
        return
    try:
        doc = jwks_document(config.key_dir)
    except Exception as e:  # noqa: BLE001
        logger.exception("oauth_as: failed to build JWKS: %s", e)
        await _send_json(send, 500, {"error": "jwks unavailable"})
        return
    await _send_json(send, 200, doc)


async def serve_as_metadata(config: AuthorizationServerConfig, send) -> None:
    """ASGI handler: serve ``/.well-known/oauth-authorization-server``."""
    from .auth import _send_json

    if not config.enabled:
        await _send_json(send, 404, {"error": "authorization server not enabled"})
        return
    await _send_json(send, 200, authorization_server_metadata(config))
