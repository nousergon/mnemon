# Mnemon

Universal long-term memory layer for AI agents via MCP.

## Stack

- **Runtime:** Bun (use `bun` not `node`, `bun test` not `jest`, `bun:sqlite` not `better-sqlite3`)
- **Storage:** SQLite + FTS5 + sqlite-vec (single-file vault at `~/.mnemon/default.sqlite`)
- **Embedding:** EmbeddingGemma-300M (GGUF, 768d, auto-downloaded via node-llama-cpp)
- **MCP:** @modelcontextprotocol/sdk (stdio transport)

## Key files

```
src/store.ts      # SQLite storage layer — content, documents, FTS5, vectors, relations
src/embedder.ts   # EmbeddingGemma-300M via node-llama-cpp, fragment-level embedding
src/search.ts     # BM25 + vector + RRF fusion + composite scoring
src/mcp.ts        # MCP server — 10 tools (retrieval, mutation, lifecycle)
src/index.ts      # CLI dispatcher (serve, status, search, save, setup)
bin/mnemon        # CLI entry point
```

## Commands

```bash
bun run src/index.ts serve      # Start MCP server (stdio)
bun run src/index.ts status     # Vault health
bun run src/index.ts search <q> # Search memories
bun run src/index.ts save <title> <content>  # Save a memory
bun run src/index.ts setup claude-code       # Configure Claude Code integration
bun test                        # Run tests
```

## Architecture

Hybrid local + remote. Phase 1 is local-only (stdio MCP).
- Local: GGUF models on Metal, Claude Code hooks for auto-capture, hybrid search
- Remote (Phase 4): Streamable HTTP on EC2, BM25-only, S3 vault sync
