# Mnemon

Universal long-term memory layer for AI agents via MCP.

## Stack

- **Runtime:** Bun (use `bun` not `node`, `bun test` not `jest`, `bun:sqlite` not `better-sqlite3`)
- **Storage:** SQLite + FTS5 + in-process vector store (single-file vault at `~/.mnemon/default.sqlite`)
- **Embedding:** EmbeddingGemma-300M (GGUF, 768d, auto-downloaded via node-llama-cpp)
- **LLM:** QMD-query-expansion-1.7B (GGUF, for observation extraction + query expansion)
- **MCP:** @modelcontextprotocol/sdk (stdio + Streamable HTTP transports)

## Key files

```
src/store.ts                    # SQLite storage — content, documents, FTS5, relations
src/vecstore.ts                 # In-process vector store — brute-force cosine over Float32Arrays
src/embedder.ts                 # EmbeddingGemma-300M via node-llama-cpp
src/llm.ts                     # QMD-1.7B for extraction + expansion
src/search.ts                  # BM25 + vector + RRF fusion + MMR + composite scoring
src/contradiction.ts           # Contradiction detection + confidence decay
src/mcp.ts                     # MCP server (stdio) — 13 tools
src/server.ts                  # Remote HTTP server (Streamable HTTP for Claude.ai/iOS)
src/sync.ts                    # S3 vault sync (push/pull)
src/index.ts                   # CLI dispatcher
src/hooks/framework.ts         # Hook framework — stdin/stdout, dedup, noise filtering
src/hooks/context-surfacing.ts # UserPromptSubmit — search + inject context
src/hooks/session-extractor.ts # Stop — extract observations from transcript
src/hooks/handoff-generator.ts # Stop — session summary for continuity
bin/mnemon                     # CLI entry point
```

## Commands

```bash
bun run src/index.ts serve          # MCP server (stdio)
bun run src/index.ts serve-remote   # HTTP server (Streamable HTTP)
bun run src/index.ts status         # Vault health
bun run src/index.ts search <q>     # Search memories
bun run src/index.ts save <t> <c>   # Save a memory
bun run src/index.ts setup <target> # Configure (claude-code, cursor, gemini, hooks)
bun run src/index.ts sync <push|pull>  # S3 vault sync
bun test                            # Run tests (50 tests)
```
