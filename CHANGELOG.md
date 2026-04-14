# Changelog

## [0.5.0] - 2026-04-14

### Breaking

- **MCP tools now return JSON instead of pre-formatted prose.** The
  ``_structured`` paired variant (``memory_search_structured``) is gone
  — ``memory_search`` itself returns the JSON array directly. Same
  treatment for ``memory_get``, ``memory_timeline``, ``memory_status``,
  ``memory_sweep``, ``memory_related``, ``profile_get``. The motivation:
  a single clean-format contract beats paired tools with the same
  concern but different shapes, and modern LLM clients (claude.ai,
  Claude Code, Cursor, Claude Desktop) all parse JSON cleanly.
  Mutation/side-effect tools (``memory_save``, ``memory_pin``,
  ``memory_forget``, ``memory_rebuild``, ``memory_check_contradictions``,
  ``profile_update``) still return short confirmation strings.
- **Breaking for direct MCP consumers** that regex the old prose
  output. The ``context_surfacing`` hook was the main such consumer
  in-tree and has been migrated to JSON parsing + client-side
  formatting. Migration for your own consumers: call the same tool
  name, ``json.loads()`` the result, format or filter as you need.
  Empty result is now ``"[]"`` (not a prose sentinel).

### Added

- ``memory_export_vectors`` — new tool exposing the full embedding
  matrix joined to document metadata, for remote-aware clients that
  want to run UMAP / visualization / similarity work client-side. JSON
  only; capped at 5000 vectors per response with a ``truncated`` flag
  for vaults past that size.
- ``VecStore.export_all()`` — internal helper returning a snapshot
  copy of (ids, vectors) so ``memory_export_vectors`` callers can
  mutate the returned array safely.

### Changed

- Total tool count: 14 (was 14 in 0.4.3 — ``memory_search_structured``
  was removed and ``memory_export_vectors`` added; net zero).
- ``context_surfacing`` hook now parses JSON from ``memory_search`` and
  formats the markdown context block client-side. Output in the
  ``<mnemon-context>`` block is shape-identical to 0.4.x — Claude sees
  the same injected context, just formatted by the hook rather than
  the server.

## [0.4.3] - 2026-04-14

Rolls up all 0.4.2 changes (which was built but never uploaded to PyPI) plus a final ruff-clean pass.

### Fixed
- **`mnemon doctor` round-trip now actually forgets the probe memory.** Previously called `memory_forget` with wrong kwarg (`document_id` vs `id`); FastMCP returned a tool-error payload rather than raising, so the save/search/forget check always reported success while leaking probe memories into the vault on every run (PR #54).

### Changed
- `MMR_DEMOTION_FACTOR`, `QUERY_EXPANSION_MAX_TOKENS`, `CONTRADICTION_OVERLAP_THRESHOLD` (renamed from `OVERLAP_THRESHOLD`), and `CONTRADICTION_CONTEXT_MAX_CHARS` moved from inline literals to `config.py`. Tunable surface now discoverable in one file (PR #54).
- Hook error logging consolidated via `framework.log_hook_error(hook_name, context, exc)`. All three hooks (`context_surfacing`, `session_extractor`, `handoff_generator`) now emit a single greppable format: `mnemon {hook} {context}: {Type}: {message}`. Nine call sites replaced (PR #55).
- `Store.save()` dropped the unused `path` kwarg. Path was auto-generated in every call; the parameter had no callers. **Minor API break** — removed rather than deferring to a major version since 0.4.1 was published for only a few hours with no known external adoption (PR #55).
- `DEFAULT_CONFIDENCE` lookups in `contradiction.py` and `store.py` switched from `.get(ct, 0.5)` to strict `[ct]`. The enum has an explicit mapping for every value; the fallback was dead code that would mask future enum additions (PR #54).
- `llm.try_generate` now logs a WARNING on LLM failure instead of silently returning `None`. LLM is optional infra but hidden crashes (OOM, llama-cpp version mismatch) previously had no operator signal (PR #54).

### Removed (dead code / lint cleanup)
- `contradiction.SearchResult` import, `server.CONTENT_TYPE_VALUES` import, `store.dataclasses.field` import, and a dead `local_token = _ensure_local_token(...)` assignment in `setup.py` — all caught by `ruff check --select F`.
- Renamed ambiguous `l` loop variables to `label` / `line` in `charts.py` and `handoff_generator.py` (E741).
- Added `[tool.ruff.lint.per-file-ignores]` suppressing E402 under `src/mnemon/dashboard/` — Streamlit pages require `st.set_page_config(...)` before other imports, legit convention.
- Test-count claims switched from fixed "460 tests" to drift-tolerant "450+ tests" in `README.md` and `CLAUDE.md`.

## [0.4.1] - 2026-04-14

### Added
- **`mnemon doctor` exercises the OAuth AS metadata endpoint** (PR #49). Fetches `/.well-known/oauth-authorization-server`, validates required RFC 8414 fields, asserts `issuer` matches deployment base URL. Catches silent browser-client breakage from `MNEMON_PUBLIC_URL` typos that the local-token path doesn't surface. Warns (not fails) when the AS isn't enabled — legitimate for local-token-only deploys.
- **Per-IP rate limit on failed `/oauth/authorize` attempts** (PR #50). 10 failures per 5-minute sliding window returns HTTP 429 with `Retry-After`. Correct passphrase clears the counter. Client IP resolved from `Fly-Client-IP` → `X-Forwarded-For` → `scope["client"][0]`.

### Changed
- `CHANGELOG.md` backfilled with 0.3.0 and 0.4.0 entries (PR #48).
- `fly volume create` command in README's self-host runbook now includes `--yes` to skip the single-region-volume confirmation prompt (PR #48).

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
