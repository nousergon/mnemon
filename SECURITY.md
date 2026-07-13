# Security Policy

## Reporting a vulnerability

If you find a security vulnerability in mnemon, please report it privately:

- **Preferred:** open a [GitHub Security Advisory](https://github.com/nousergon/mnemon/security/advisories/new). This keeps the discussion private until a fix ships.
- **Alternative:** email `security@nousergon.ai` with a description and reproduction steps.

Please **do not** open a public issue for security reports. I aim to acknowledge within 72 hours and ship a fix or mitigation within 14 days for high-severity issues.

## Scope

mnemon is a self-hosted single-user memory server. The following are in scope:

- **Authentication bypass**: any path that lets an unauthenticated request reach an MCP tool or the SQLite vault.
- **OAuth flow flaws**: PKCE downgrade, code/token leakage, JWT signature verification gaps, DCR abuse, replay attacks.
- **Injection or escalation**: SQL injection, path traversal, command injection in any of the CLI tools or hooks.
- **Cryptographic mistakes**: weak key generation, insecure storage of `private.pem`, refresh token predictability, timing attacks on passphrase comparison.
- **Vault data exposure**: any path that lets one self-host user read another's vault, or that leaks vault contents through error messages, logs, or unauthenticated endpoints.

The following are **out of scope**:

- DoS via traffic volume (single-user infrastructure; rate-limit it yourself if you expose to the public internet).
- Self-XSS in the passphrase login form (it's a single-user form; no other user to attack).
- Issues that require local filesystem or process access (the SQLite vault and `private.pem` are protected by file-system permissions; if your machine is compromised, mnemon's threat model has already failed).
- Vulnerabilities in upstream dependencies that have not been disclosed publicly — please report those to the upstream project first.

## Threat model assumptions

mnemon is designed for the **single-user self-host** case. Important assumptions:

- The server owner is the sole authorized principal. There is no multi-user model and no plan to add one.
- The `MNEMON_AS_PASSPHRASE` is the only credential gating browser-client access. A `secrets.token_urlsafe(32)`-class value is required (16-char minimum enforced at boot); shorter passphrases break the security model.
- The `MNEMON_LOCAL_TOKEN` is a static bearer for headless clients (Claude Code hooks, Cursor) and bypasses OAuth entirely. Treat it as equivalent to a vault-master key.
- The Fly volume holding `private.pem`, `clients.json`, `refresh_tokens.json`, and the SQLite vault is assumed to be encrypted at rest by Fly. Local deploys (or other hosts) inherit whatever the underlying disk encryption provides.
- Network transport is HTTPS-only in production (`force_https = true` in `fly.toml`). HTTP is dev-only.

## Hardening recommendations for self-hosters

- Generate `MNEMON_LOCAL_TOKEN` and `MNEMON_AS_PASSPHRASE` with `secrets.token_urlsafe(32)` and store them in a password manager. Never reuse credentials across deployments.
- Set `MNEMON_ALLOWED_HOSTS` to your exact deployment hostname (DNS-rebinding protection).
- Run `mnemon doctor` after every deploy and verify all 7 checks pass.
- Back up the Fly volume (`/data/oauth_keys/` + the SQLite vault) regularly, or enable S3 vault sync.
- If `private.pem` is ever exposed, delete it from `/data/oauth_keys/`, restart the server (a new key auto-generates on boot), and reconnect every browser client. All previously-issued JWTs become unverifiable.
