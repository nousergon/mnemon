# mnemon

Universal long-term memory layer for AI agents via MCP.

## Session State & Wind-down

**At the start of any mnemon session, read `private/SYSTEM_STATE.md`
first** — the living buildout snapshot (version, deploy, architecture,
security posture, in-flight/deferred, known gotchas). It supersedes the
one-time `private/mnemon-system-audit-260412.md`. Honor its "Last
verified" date: if older than ~1 week, re-verify the Current State /
Deploy blocks before trusting specifics. `ROADMAP.md` (same dir) holds
open work + the pre-deploy ritual.

**When closing/wrapping a session,** run the wind-down before ending:

1. **`private/ROADMAP.md`** — refresh every task state touched (mark
   shipped/merged, annotate awaiting-merge, add new items surfaced).
2. **`private/SYSTEM_STATE.md`** — overwrite the Current State snapshot
   in place for any changed facts; append a dated Recent Changes line;
   bump "Last verified". Confirmed milestones only, no speculation.
3. **Memories** — save durable cross-session facts/decisions/feedback.

`private/` is gitignored — these doc edits are local-only (no commit);
the git history of code + `CHANGELOG.md` is the durable audit trail.

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
pytest                          # Run tests (450+ tests)
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
