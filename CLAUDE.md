# mnemon

Universal long-term memory layer for AI agents via MCP.

## Session State & Wind-down

**At the start of any mnemon session, read `private/SYSTEM_STATE.md`
first** — the living buildout snapshot (version, deploy, architecture,
security posture, in-flight/deferred, known gotchas). It supersedes the
one-time `private/mnemon-system-audit-260412.md`. Honor its "Last
verified" date: if older than ~1 week, re-verify the Current State /
Deploy blocks before trusting specifics.

**The backlog lives in GitHub Issues on `cipher813/mnemon-ops`** (board:
https://github.com/users/cipher813/projects/3), migrated from
`private/ROADMAP.md` 2026-06-15 — same pattern as alpha-engine-config and
metron-ops. `private/ROADMAP.md` is now a TOMBSTONE that retains only two
reference sections (release rituals + operational watch list) plus standing
decisions; it is no longer the work list. Query with `gh issue list --repo
cipher813/mnemon-ops` (or `gh auth token` + curl — `gh` is proxy-blocked).
Labels: `P1`–`P3`, `deferred`, `speculative`, `area:*`.

**When closing/wrapping a session,** run the wind-down before ending:

1. **Issue-hygiene sweep on `cipher813/mnemon-ops`** — for every item
   touched: work merged/shipped → CLOSE the issue (one-line comment naming
   the PR); advanced-but-open → COMMENT the delta; new follow-ups / audit
   findings → FILE a new issue (priority label + gate + re-exam trigger +
   closes-when) and add it to the board (projects/3). `ROADMAP.md` itself
   changes only when a reference section (release rituals / ops watch list)
   or a standing decision changes.
2. **`private/SYSTEM_STATE.md`** — overwrite the Current State snapshot
   in place for any changed facts; append a dated Recent Changes line (or
   to `SYSTEM_STATE_changelog.md`); bump "Last verified". Confirmed
   milestones only, no speculation.
3. **Memories** — save durable cross-session facts/decisions/feedback.

`private/` is gitignored from THIS repo but is its own nested git repo
(`private/.git`) pushed to `cipher813/mnemon-ops` (private). So the
wind-down doc edits above MUST be committed + pushed there — they are
NOT local-only: `cd private && git add -A && git commit && git push
origin main` (no PR). The mnemon code repo's git history + `CHANGELOG.md`
remain the durable audit trail for the package itself.

## Stack

- **Runtime:** Python >= 3.10
- **Storage:** SQLite + FTS5 + in-process vector store (single-file vault at `~/.mnemon/default.sqlite`)
- **Embedding:** FastEmbed with bge-small-en-v1.5 (384d, ONNX, ~13MB)
- **LLM (optional):** QMD-query-expansion-1.7B (GGUF via llama-cpp-python, for extraction + expansion)
- **MCP:** FastMCP (stdio + Streamable HTTP transports)
- **Package:** hatchling build backend, `pip install -e ".[dev]"` for development

## Key files

```
src/mnemon/
  store.py              # SQLite storage — content, documents, FTS5, relations
  vecstore.py           # In-process vector store — brute-force cosine over numpy arrays
  embedder.py           # FastEmbed bge-small-en-v1.5 wrapper
  search.py             # BM25 + vector + query expansion + RRF fusion + MMR
  llm.py                # QMD-1.7B via llama-cpp-python (optional)
  contradiction.py      # Contradiction detection + confidence decay
  config.py             # Content types, half-lives, scoring constants
  server.py             # MCP server (stdio) — 17 tools
  server_remote.py      # Remote HTTP server (Streamable HTTP)
  sync.py               # S3 vault sync (push/pull via AWS CLI)
  setup.py              # Auto-configure Claude Code, Cursor, Gemini
  cli.py                # CLI dispatcher
  __init__.py           # Version
  hooks/
    framework.py        # Hook framework — stdin/stdout, dedup, noise filtering
    context_surfacing.py  # UserPromptSubmit — search + inject context
    session_extractor.py  # Stop — extract observations (LLM or regex)
    handoff_generator.py  # Stop — session summary (LLM or template)
```

## Commands

```bash
# Development
pip install -e ".[dev]"         # Install with dev deps
pytest                          # Run tests (1000+ tests, ≥88% coverage gate)
pytest -v                       # Verbose output
pytest tests/test_store.py      # Single file

# CLI
mnemon serve                    # MCP server (stdio)
mnemon serve-remote             # HTTP server (Streamable HTTP)
mnemon status                   # Vault health
mnemon search <query>           # Search memories
mnemon save <title> <content>   # Save a memory
mnemon setup <target>           # Configure (claude-code, cursor, gemini, hooks)
mnemon sync <push|pull>         # S3 vault sync
```

## Testing

All tests must pass before committing. Tests use temporary SQLite databases and mock external dependencies (LLM, embeddings, AWS CLI).

```bash
pytest                          # Quick check
pytest -v                       # See individual test names
pytest --tb=short               # Shorter tracebacks on failure
```
