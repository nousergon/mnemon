# Changelog

## [0.4.0] - 2026-04-14

### Added
- **Self-hosted OAuth 2.1 Authorization Server** (Phase 2). MCP-compliant `/oauth/authorize`, `/oauth/token`, `/oauth/register` endpoints with PKCE, RS256 JWTs, and RFC 7591 Dynamic Client Registration. No third-party auth vendor required — `fly deploy` gives you a working OAuth-protected MCP endpoint (PRs #36–#39).
- **Volume-backed persistence for OAuth state.** Auth codes and refresh tokens now persist to `{key_dir}/auth_codes.json` and `{key_dir}/refresh_tokens.json` (mode 0600, atomic write-through) so Fly autostop wake doesn't force every client to re-enter the passphrase (PR #44).
- **`fly.toml.example` template + self-host runbook in README.** Placeholder-driven config plus an end-to-end walkthrough (`fly launch` → `fly volume create` → secrets → deploy → `mnemon doctor` → client connection) so a new user can stand up a vault in ~10 minutes with zero Fly prior knowledge (PR #46).
- **16-char minimum on `MNEMON_AS_PASSPHRASE`.** Validated at server boot via `validate()`; error message points at `secrets.token_urlsafe(32)` (PR #47).
- Auth code + refresh token rotation: one-time-use auth codes, rotation on every refresh.

### Changed
- **Removed Auth0 / external-AS code path** (PR #40). `OAuthConfig` now only carries `local_token`; `MNEMON_OAUTH_ISSUER` / `JWKS_URL` / `AUDIENCE` / `USERINFO_URL` env vars are gone. The self-hosted AS is the only browser-auth path.
- `fly.toml` is now gitignored; `fly.toml.example` is the canonical template (PR #46).
- Test count: 253 → 460.

### Removed
- Auth0 JWKS validation, userinfo fallback, and all `MNEMON_OAUTH_*` env var parsing.
- `MNEMON_OAUTH_AUDIENCE` from `fly.toml` (dead after PR #40; PR #45).

## [0.3.0] - 2026-04-13 (not published to PyPI)

### Added
- **Remote-first architecture unification** (Phase 3). Claude Code hooks (context_surfacing, session_extractor, handoff_generator) rewritten to call the remote Fly vault via Streamable HTTP instead of the local SQLite vault (PRs #25–#30).
- `MNEMON_LOCAL_TOKEN` static bearer path for headless clients (hooks, Cursor) that can't complete a browser OAuth flow (PR #24).
- `mnemon setup` CLI: auto-configures Claude Code / Cursor / Gemini with a remote URL and bearer token (PR #28).
- `mnemon doctor` CLI: six end-to-end diagnostics (remote URL, local token, file perms, health endpoint, auth + MCP tool call, save/search/forget round-trip) (PR #41).
- Claude Code hook integration tests (PR #34).

### Changed
- Hooks read `MNEMON_REMOTE_URL` + `MNEMON_LOCAL_TOKEN` (or `~/.mnemon/remote_url` + `~/.mnemon/local_token`) instead of opening a local SQLite file.
- Dedup in hooks switched from text-parsing to structured `memory_search_structured` MCP tool (PR #33).

## [0.2.0] - 2026-04-09

### Added
- Full Python rewrite (was TypeScript/Bun in v0.1.x)
- 13 MCP tools: search, get, timeline, save, pin, forget, status, sweep, related, rebuild, check_contradictions, profile_get, profile_update
- Hybrid BM25 + vector search with Reciprocal Rank Fusion and MMR diversity filtering
- Composite scoring: 0.5 * relevance + 0.25 * recency + 0.25 * confidence
- FastEmbed embeddings (bge-small-en-v1.5, 384d ONNX — no PyTorch needed)
- Memory lifecycle: content-type-based half-life decay, pinning, archival via sweep
- Contradiction detection with confidence decay (vector similarity + optional LLM classification)
- Claude Code hooks: context surfacing (UserPromptSubmit), session extraction (Stop), handoff generation (Stop)
- Auto-configure command: `mnemon setup claude-code|cursor|gemini|hooks`
- Remote Streamable HTTP server via FastMCP native transport
- S3 vault sync (push/pull via AWS CLI)
- CLI: serve, serve-remote, status, search, save, forget, sync, setup
- Optional local LLM (QMD-1.7B via llama-cpp-python) for query expansion, extraction, contradiction detection
- 253 tests, 90% coverage

### Changed
- Rewritten from TypeScript (Bun) to Python (>=3.10)
- Storage: SQLite + FTS5 + numpy vector store (was SQLite + FTS5 + TypeScript vector store)
- Embedding: FastEmbed bge-small-en-v1.5 (was EmbeddingGemma-300M)
- Build: hatchling (was Bun bundler)
- Package distribution: PyPI (was npm)

## [0.1.x] - 2026-04-08

Initial TypeScript implementation (deprecated, replaced by Python rewrite in v0.2.0).
