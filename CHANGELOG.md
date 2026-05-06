# Changelog

## [0.6.0rc13] - 2026-05-06

### Fixed

- **Fresh-init MCP requests wedged the server-wide session-creation lock.**
  Diagnosed live during the rc12 soak: `mnemon doctor` and any
  `streamablehttp_client` consumer that opens a new MCP session
  ([no `Mcp-Session-Id` header) hung past 15s while warm-keeper
  PingRequests and tool calls on existing sessions kept working. Direct
  curl confirmed: `tools/list` with the persisted session ID returned
  in <300ms; `initialize` with no session ID hung 20s with no
  `Processing request of type` log on the server. Root cause: upstream
  `StreamableHTTPSessionManager._handle_stateful_request` holds
  `_session_creation_lock` for the **full** duration of the new-session
  branch — including the `await transport.handle_request(...)` that
  dispatches the JSON-RPC payload and waits for the response. If that
  handler wedges for any reason (we never identified the underlying
  trigger, but the wedge reproduced reliably across rc11 and rc12
  fresh-deploys), every subsequent fresh-init queues behind the lock
  and times out at the client side. PR #109 patched the SSE-mode
  variant of this same lock-held-too-long pattern; the
  `json_response=True` variant still had its own way to wedge.

  Fix: `PersistentSessionManager` now bypasses upstream's locked path
  for fresh-init via a new `_create_new_session` method that holds
  `_session_creation_lock` only for the brief `_server_instances`
  mutation, then releases before `task_group.start()` and
  `transport.handle_request()`. `_resume_session` got the same
  narrowing — the lock now covers only the race-guard read + dict
  write, not the dispatch. One wedged handler can no longer take down
  the server's ability to accept new sessions.

### Tests

- `TestSessionCreationLockNarrowing` (5 cases) in
  `tests/test_persistent_sessions.py`: lock-released-before-dispatch
  invariant for both `_create_new_session` and `_resume_session`,
  wedged-handler-doesn't-block-concurrent-fresh-init reproduction
  (this directly mirrors the live-prod symptom), 5-concurrent-fresh-
  inits-get-distinct-ids sanity, and resume-race-lost-falls-through
  coverage. 708 → 713 passing.

## [0.6.0rc12] - 2026-05-06

### Fixed

- **Post-deploy doctor probe wedged a freshly-redeployed Fly machine.**
  During the 0.6.0rc11 redeploy, `mnemon doctor` failed twice in a row
  immediately after `mnemon upgrade web` returned. The deploy itself
  succeeded, but doctor's 7 rapid-fire checks landed on the machine
  while FastEmbed pre-load was still finishing and the in-memory
  session manager was fresh — exactly the request-pile-up shape that
  exposes the latent CallToolRequest hang. `fly machine restart`
  cleared it; doctor passed clean after. Fix: `_redeploy_web` (and the
  first-time `upgrade_web` path) now sleeps for a configurable settle
  window (default 30s) between `flyctl deploy` returning and the
  inline doctor invocation. Override via
  `MNEMON_UPGRADE_SETTLE_SECONDS` (set to `0` to disable).
- **`mcp_sessions.sqlite` grew unbounded under long warm uptimes.**
  `expire_old()` previously ran only at server startup
  (`server_remote.run_remote`), which is fine under
  `auto_stop_machines = "stop"` (every wake = a prune) but lets the
  table drift up under any long-running configuration or between
  cold-stops. Pre-restart on 2026-05-06 the table held 4,171 persisted
  sessions with the 7-day TTL. Fix: `PersistentSessionManager` now
  spawns an in-process anyio task during the lifespan that ticks every
  6h (default) and calls `SessionStore.expire_old()`. Failures are
  logged and swallowed so a transient SQLite hiccup can't take active
  sessions down. Pass `expire_interval_seconds=0` to disable.

### Roadmap

- `--mnemon-version` flag on `mnemon upgrade web` and `CONTRIBUTING.md`
  marked done in `private/ROADMAP.md` — both shipped earlier; the
  rc11-deploy P0 audit caught the missing strikethroughs.

### Tests

- `TestPostDeploySettleWindow` (5 cases) in `tests/test_upgrade.py`
  covers ordering (deploy → settle → doctor), `--skip-doctor`
  short-circuit, env-var override to zero/custom/garbage, and the
  first-time upgrade path. Autouse fixture sets
  `MNEMON_UPGRADE_SETTLE_SECONDS=0` so existing tests don't pay 30s
  per doctor-invoking case.
- `TestPeriodicExpireConfig` + `TestPeriodicExpireTask` (5 cases) in
  `tests/test_persistent_sessions.py` cover the default 6h interval,
  zero-disables-prune, per-tick `expire_old()` call, log-on-prune,
  and swallow-then-retry on failure.

## [0.6.0rc11] - 2026-05-06

### Fixed

- **Per-Stop handoff noise — `handoff_generator` now gates on three
  conditions before saving.** Stop fires after every assistant turn,
  not once per session, and `handoff_generator` previously had no dedup
  — every turn produced a fresh `handoff` memory. Observed in the
  production vault as 710/1376 documents (51.6%) being `handoff`-typed,
  dominated by `claude-code-hook`-sourced rows whose first-user-line
  was a `/loop` body, `<task-notification>`, pasted test output, or a
  one-word reply ("yes", "done", "pr merged"). Three gates now run in
  `main()` cheapest-first: trivial-prompt skip (regex match against
  slash-command / notification / test-output patterns plus a 15-char
  floor), per-session debounce (`~/.mnemon/handoff_session_state.json`
  keyed by hook-payload `session_id` with 600s cooldown, degrades to
  legacy "always save" on empty session_id), and a final remote vector
  dedup mirroring `session_extractor.is_duplicate_remote` posture. PR
  #112.

### Tests

- 20 new tests across `TestHandoffGeneratorTrivialPromptSkip`,
  `TestHandoffGeneratorSessionDebounce`, and
  `TestHandoffGeneratorRemoteDedup` in `tests/test_hooks_extended.py`.
  4 existing `TestHandoffGeneratorMain` tests updated to patch
  `is_duplicate_remote` so they keep exercising the save path. 689 →
  709 passing.

## [0.6.0rc10] - 2026-05-02

### Fixed

- **MCP fresh-session deadlock — `json_response=True` for the
  StreamableHTTP session manager.** `mnemon doctor`, any
  `streamablehttp_client` consumer, and direct curl POSTs to `/mcp`
  were hanging past 60s on `session.initialize()` against a freshly-
  deployed rc9 server. Reproduces immediately on a fresh
  `fly machine restart` — not stale state. Root cause: upstream's
  `StreamableHTTPSessionManager._handle_stateful_request` holds
  `_session_creation_lock` for the full duration of `handle_request`,
  and in SSE response mode (`json_response=False`, the previous
  default) `handle_request` keeps the per-session SSE stream open
  until the client disconnects — so once one session is alive,
  every fresh-session POST queues behind the lock indefinitely.
  mnemon's tools are all single-shot RPCs, so the SSE channel buys
  nothing and only exposes this hang. Fix: pin `json_response=True`
  in `server_remote.run_remote` so each POST is a discrete request/
  response pair. Symptom user-side was Claude Desktop `memory_save`
  calls stalling indefinitely. PR #109.

### Tests

- New `TestSessionManagerConfig::test_session_manager_uses_json_response`
  in `tests/test_server_remote.py` captures
  `PersistentSessionManager` kwargs during `run_remote()` and asserts
  `json_response=True` so this can't silently flip back. 688 → 689
  passing.

## [0.6.0rc9] - 2026-05-02

### Fixed

- **OAuth refresh-token rotation grace window.** A retried `/oauth/token`
  call presenting an already-rotated refresh token within 60 seconds
  now returns the *same* new pair (idempotent retry) instead of
  `invalid_grant`. Closes a class of spontaneous claude.ai connector
  disconnects: when Fly auto-stops the machine and a wake-on-request
  cold-start delays the response, claude.ai's mcp-proxy retries the
  refresh; the original request had already rotated the RT, so the
  retry would brick the connector and force the user to manually
  reconnect. The cache (`{key_dir}/rotated_tokens.json`) is persisted
  to the same volume as `refresh_tokens.json` so it survives the very
  Fly restarts that trigger the bug. Replay after the grace window
  still returns `invalid_grant` — leaked-token defense preserved.
  PR #107.

### Tests

- 3 new tests in `TestServeTokenRefreshGrant`: replay-within-grace
  returns identical pair (covering 3 retries), replay-after-grace
  rejected, and rotation-cache persists across simulated restart
  (the Fly cold-start scenario). 1 existing test updated to reflect
  the new replay-within-grace behavior. 684 → 688 passing.

## [0.6.0rc8] - 2026-04-29

### Fixed

- **`read_transcript` supports Claude Code's nested message envelope.**
  Real Claude Code JSONL nests `{role, content}` under
  `msg.message.role` / `msg.message.content` alongside metadata fields
  (`parentUuid`, `sessionId`, `timestamp`, `cwd`, etc.). The existing
  parser only handled the flat top-level `{"role": ..., "content":
  ...}` format used by synthesized test fixtures, so it silently
  returned an empty string against every real Claude Code session.
  Result: `handoff_generator` and `session_extractor` Stop hooks both
  short-circuit on the `len(transcript) < 200` check before saving
  anything. Fix accepts both wire formats and tightens content-block
  parsing so non-text blocks (`tool_use`, `tool_result`, `image`) are
  explicitly skipped — only `{"type": "text", "text": ...}` blocks
  contribute. PR #101.

  Diagnosed against a real session JSONL: 1,151 lines, 0 with
  top-level `role`, 814 with nested `message.role`. Live verified on
  the same session post-fix: extracted 7,156 chars (was 0). Both Stop
  hooks now save reliably from real Claude Code usage; the
  productized auto-handoff infra finally works end-to-end against
  the wire format Claude Code actually emits.

### Tests

- 6 new regression cases in `TestReadTranscript` using the nested
  Claude Code wire format (envelope + metadata siblings + content as
  list-of-blocks + `file-history-snapshot` lines + mixed flat/nested
  in the same transcript). Test count 675 → 681.

## [0.6.0rc7] - 2026-04-28

### Added

- **PostToolUse auto-mirror hook.** Closes the longstanding gap where
  Claude Code (and any other MCP client with a local auto-memory
  system) writes memory files to its private store but never
  propagates them to the central mnemon vault. The hook installed by
  `mnemon setup` fires on every `Write` / `Edit` / `MultiEdit` tool
  call, no-ops unless the touched file lives under an auto-memory
  directory (`~/.claude/projects/*/memory/*.md` or
  `~/.config/mnemon/auto-memory/*.md`), and otherwise dispatches the
  file's contents to mnemon via the same `MemoryClient` abstraction
  used by the existing UserPromptSubmit / Stop hooks. Local + remote
  modes both supported. Errors are surfaced via stderr per the
  no-silent-fails posture; the hook never blocks Claude Code's
  continued operation. Motivated by the 2026-04-28 alpha-test
  incident where a session handoff written to local memory was never
  mirrored to mnemon. With auto-mirror installed, `mnemon setup` is
  the only step required: subsequent client memory writes appear in
  mnemon automatically.
- **`mnemon mirror <path> [--auto] [--timeout SEC]`** CLI subcommand
  that the hook shells out to. Reads the file, parses YAML
  frontmatter (PyYAML when available, minimal fallback parser
  otherwise), dispatches `memory_save` with `title` from the `name`
  frontmatter field, `content_type` from `type` (default `note`),
  `description` prepended to the body as an italicized line, and
  `source_client = "mnemon-mirror"`. `--auto` short-circuits when the
  path doesn't match an auto-memory pattern. Idempotent: SHA-256
  (title + content) dedup with a 600s window via
  `~/.mnemon/mirror_dedup.json` so repeat saves of the same content
  within ten minutes skip cleanly. Sync-loop guard: files with
  `mnemon_sync_source: <doc_id>` in frontmatter are skipped
  (placeholder for future `mnemon sync down` integration).
- **`mnemon.mirror`** module exposing `mirror_path()` for programmatic
  use and `run_cli()` for the subcommand entry.

### Tests

- 22 new tests in `tests/test_mirror.py` covering the auto-memory
  path filter, frontmatter parsing, the four save/skip outcomes,
  error paths, and CLI exit codes.
- 13 new tests in `tests/test_hooks_auto_mirror.py` for the hook
  handler — file_path extraction, trigger-tool whitelist
  (Write/Edit/MultiEdit), unrelated-tool no-op, malformed stdin
  absorption, expected `MirrorError` surfacing, and unexpected
  exception swallow with stderr surfacing.
- 3 new assertions in `tests/test_setup.py` locking the PostToolUse
  entry shape (matcher = `Write|Edit|MultiEdit`, command points at
  `mnemon.hooks.auto_mirror`, 12s timeout) in both local and remote
  modes, plus a `setup_claude_code` integration check that the entry
  reaches `~/.claude/settings.json`.

Full suite: 675 passed (+38 from this feature; 4 sandbox-only setup
errors in `tests/test_integration_remote.py` pre-exist on `main`,
not introduced here).

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
