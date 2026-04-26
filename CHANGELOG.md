# Changelog

## [0.6.0rc6] - 2026-04-26

### Added

- **`--mnemon-version <ver>` flag on `mnemon upgrade web`.** Pins the
  version in the deployed Dockerfile explicitly instead of always
  inheriting from the locally-installed `__version__`. Closes the trap
  where a user publishes a new RC to PyPI then forgets to `pip install
  -U` locally before redeploying, silently shipping the prior version.
  The internal redeploy path already supported the parameter; this
  exposes it on the CLI.
- **`examples/quickstart.py`** demonstrating the public Python API
  (`mnemon.store.Store` + `mnemon.search.search`) without an MCP
  transport. Three queries cover lexical, semantic, and diagnostic
  cases.
- **`bench/search_stress.py`** + a JSON baseline at 1k memories.
  Hybrid BM25 + vector search measures p50 / p95 / p99 over a
  deterministic synthetic corpus. Initial numbers on consumer hardware:
  sub-4 ms p99 at 1k memories, sub-8 ms p99 at 5k.
- **`CONTRIBUTING.md`** with dev setup, style, PR conventions, and a
  pointer to `SECURITY.md`.
- **Hourly health-monitor GitHub Actions workflow.** Hits
  `https://mnemon-memory.fly.dev/health`, asserts on the metrics shape,
  and opens / comments on / closes a single tracker GH issue
  automatically. Resets to green when the next run passes.
- **Fresh-install CI workflow.** Builds the wheel from PR source,
  installs it in clean `python:X.Y-slim` containers (3.10 / 3.12 /
  3.13), verifies CLI + import. Weekly schedule also smoke-tests the
  latest published version on PyPI.
- **Daily `pip-audit --strict .` CI** scanning the declared dependency
  graph for CVEs.
- **`/health` session-routing metrics counters.** New `metrics` key on
  the `/health` payload exposes `in_memory_hits`, `resume_hits`,
  `fresh_inits`, `stale_session_misses`, `persisted_sessions_total`,
  and `in_memory_sessions_current` — for cold-start diagnostics and
  regression detection. Backward-compatible: clients that only check
  `status == "ok"` are unaffected.

### Changed

- **`mnemon setup claude-code` and `mnemon setup claude-desktop` no
  longer frame the claude.ai-synced + stdio dual config as a "shadow"
  problem.** New copy reads as state-of-the-world: `ℹ Both local + web
  mnemon configured. <Client> will use the web version (claude.ai-
  synced) by default.` Behavior is unchanged — both registrations
  still written, web wins by default, stdio activates automatically if
  the user later removes the claude.ai entry. The previous "⚠ shadow"
  alarm framing implied the user did something wrong; they didn't.
- **`mnemon setup claude-desktop` now surfaces the dual-config notice
  too.** Claude Desktop syncs MCP from claude.ai (same Anthropic
  account) the same way Claude Code does, so the same coexistence
  applies.
- **`mnemon uninstall` no longer says "no mnemon registration found
  (or CLI errored silently)"** when the claude CLI returns non-zero.
  The non-zero case is the common one (registration was already gone);
  reworded to "no user-scope mnemon registration to remove" and
  surfaces stderr if it doesn't look like a "not found"-family error.
- **README PyPI badge is now dynamic** via `shields.io/pypi/v/...`.
  Previously hardcoded to a specific RC and went stale across releases.

## [0.6.0rc5] - 2026-04-26

### Added

- **`/health` exposes session-routing metrics.** Same shape and feature
  as listed in 0.6.0rc6 — listed here for traceability since rc5 was
  the first version on PyPI to ship the counters. (rc5 was published
  without a CHANGELOG entry; this back-fills it.)

## [0.6.0rc4] - 2026-04-25

### Added

- **Persistent MCP Streamable HTTP sessions across process restarts.**
  Issued `Mcp-Session-Id` values are now stored in SQLite (default at
  `<vault_dir>/mcp_sessions.sqlite`, 7-day TTL). When a request bearing
  a previously-issued session ID arrives at a fresh process — for
  example after a Fly cold-stop, a redeploy, or any other restart — the
  server transparently resumes the session instead of returning 404,
  by spawning a new transport keyed to the same ID and running the MCP
  app `stateless=True` so the server-side session is born already-
  initialized. The client sees no break.

  Pairs with the `/health` warm-keeper hook from the previous release:
  the hook minimizes the *frequency* of cold-stops during active use,
  this change handles the *consequence* when one happens anyway. With
  both shipped, mnemon survives Fly auto-stop without the "MCP UI says
  connected, every tool call returns 404" failure mode.

  Caveat: resumed sessions are stateless on the server side. Mnemon
  uses no client-capability-gated features (sampling, elicitation,
  roots), so this is a no-op for our toolset, but third-party forks
  adding those features will need to re-handshake explicitly.

### Fixed

- **Mid-session disconnects on Fly when `auto_stop_machines = "stop"`.**
  `mnemon setup` (any target with `--remote-url`) now installs a
  lightweight `/health` warm-keeper as the first `UserPromptSubmit`
  hook. It pings the Fly app on every prompt, resetting Fly's idle
  timer so the machine stays warm during active Claude Code sessions
  and wakes on the first prompt after idle. Independent of MCP session
  state, so it works even after a cold-stop has invalidated the
  `Mcp-Session-Id` (which the existing context-surfacing hook's MCP
  call cannot recover from). `|| true` ensures a slow Fly wake never
  blocks the user's prompt.

  Previously: machine autostops after ~5 min idle, the open MCP
  Streamable HTTP session goes stale, and Claude Code's MCP UI keeps
  showing "connected" while every subsequent tool call returns 404.

## [0.6.0rc3] - 2026-04-22

### Fixed

- **`mnemon upgrade web` is now idempotent.** Rerunning it against an
  already-deployed Fly app (previously a hard failure / confusing
  relaunch) now detects the existing app via `flyctl status` and
  switches to a redeploy-only path: skip S3 push, volume create,
  secrets set, vault seed, local vault archive, and MCP client
  reconfigure — all of which are no-ops once the web tier is
  established. Clients keep their URL and bearer token; the new image
  is picked up on the next request. First-time deploy flow is
  unchanged.

  Upgrading to a newer mnemon version is now a two-command flow:
  `pip install -U 'mnemon-memory[server]'` then
  `mnemon upgrade web --app-name <existing-app>`.

  Redeploy only requires `flyctl`; AWS creds and an S3 bucket are no
  longer required for the version-bump case.

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
- **Dashboard is remote-aware.** The Streamlit dashboard detects
  ``MNEMON_REMOTE_URL`` / ``~/.mnemon/remote_url`` and routes every
  loader through the remote MCP server, so a browser-side user sees
  the same vault their hooks write to. Local SQLite stays as the
  development fallback. (PR #59)
- **Graph page collapses chunks to one point per memory.** Long
  documents produce multiple embedding chunks; previously each chunk
  rendered as its own UMAP point, which read as duplicates. The graph
  now mean-pools (L2-normalized) per document so point count matches
  memory count. (PR #60)
- **Dashboard remote-call timeouts raised.** The dashboard was
  inheriting the 8s hook timeout, which is sized for Claude Code's
  hook budget and could time out on Fly cold-start or on the heavier
  ``memory_export_vectors`` call. Dashboard now uses 30s general / 60s
  for vector export; hook callers unchanged. (PR #61)
- **``mnemon doctor`` works in local mode.** Auto-detects config and
  runs three parallel checks (vault reachable, embedder loadable,
  save/search/forget round-trip via ``Store``) when no remote URL is
  configured. Remote mode unchanged. No new flag — same command,
  adaptive behavior. (PR #62)
- **README onboarding rewrite.** Quick Start leads with the 60-second
  local path instead of the 10-minute Fly self-host runbook; dashboard
  visible in Install and Quick Start with an embedded screenshot;
  self-host secret handling switched from ``echo "$SECRET" > file`` to
  ``read -rs`` + ``printf %s`` so tokens don't land in shell history;
  FastEmbed cold-start warning added to the first-run path. (PR #62)

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
