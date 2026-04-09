# mnemon

Universal long-term memory layer for AI agents via MCP.

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
  server.py             # MCP server (stdio) — 13 tools
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
pytest                          # Run tests (102 tests)
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
