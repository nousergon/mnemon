[![Bun](https://img.shields.io/badge/bun-1.3+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-50_passing-brightgreen.svg)]()
[![CI](https://github.com/cipher813/mnemon/actions/workflows/ci.yml/badge.svg)](https://github.com/cipher813/mnemon/actions/workflows/ci.yml)

# mnemon (μνήμων)

Universal long-term memory layer for AI agents. One memory vault, every tool.

mnemon is an MCP server that gives Claude Code, Cursor, Gemini CLI, and Claude.ai access to a shared, persistent memory store with hybrid keyword + semantic search.

## How it works

```
LOCAL (Mac — Metal GPU):
  [Claude Code] --stdio--> [mnemon MCP] <--stdio-- [Cursor]
  [Gemini CLI]  --stdio-->      |
                          [SQLite + FTS5]
                          [Vector store]
                          [GGUF models on Metal]
                                |
                          S3 vault sync
                                |
REMOTE (EC2):
  [Claude.ai web] --HTTP--> [mnemon remote]
  [Claude iOS]    --HTTP-->      |
                           [SQLite + FTS5]
                           [BM25-only search]
```

- **Storage:** SQLite with FTS5 full-text search + companion vector store for semantic search
- **Embedding:** EmbeddingGemma-300M (768d) via node-llama-cpp, runs on Apple Silicon Metal
- **Search:** BM25 + vector + Reciprocal Rank Fusion + composite scoring (relevance/recency/confidence)
- **Protocol:** MCP (Model Context Protocol) — works with any MCP-compatible client

## Quick start

```bash
# Install Bun (if needed)
curl -fsSL https://bun.sh/install | bash

# Clone and install
git clone https://github.com/cipher813/mnemon.git
cd mnemon
bun install

# Run tests
bun test

# Start the MCP server
bun run src/index.ts serve
```

## Configure your tools

```bash
# Claude Code
bun run src/index.ts setup claude-code

# Cursor
bun run src/index.ts setup cursor

# Gemini CLI
bun run src/index.ts setup gemini
```

Or manually add to your MCP config:

```json
{
  "mcpServers": {
    "mnemon": {
      "command": "bun",
      "args": ["run", "/path/to/mnemon/src/mcp.ts"]
    }
  }
}
```

## MCP tools

| Tool | Description |
|------|-------------|
| `memory_search` | Hybrid BM25 + vector search with composite scoring |
| `memory_get` | Get a specific memory by ID |
| `memory_related` | Find related memories via relationship graph |
| `memory_timeline` | Recent memories in chronological order |
| `memory_save` | Save a new memory (decision, preference, observation, etc.) |
| `memory_pin` | Pin an important memory to prevent archiving |
| `memory_forget` | Soft-delete a memory |
| `memory_status` | Vault health stats |
| `memory_sweep` | Archive stale memories past their half-life |
| `memory_rebuild` | Re-embed all documents (after model upgrade) |
| `memory_check_contradictions` | Check a memory for conflicts with existing memories |
| `profile_get` | Synthesized user profile from preferences + decisions |
| `profile_update` | Add a preference to the user profile |

## Memory types

Memories are classified by content type, each with a decay half-life:

| Type | Half-life | Description |
|------|-----------|-------------|
| decision | Never | Architectural decisions, why X over Y |
| preference | Never | User preferences, workflow habits |
| antipattern | Never | Things that failed, mistakes to avoid |
| observation | 90 days | Facts learned during work |
| research | 90 days | Investigations, analysis |
| project | 120 days | Project context, goals, status |
| handoff | 30 days | Session summaries for continuity |
| note | 60 days | General notes (default) |

Pinned memories are exempt from decay. Stale memories are soft-deleted by `memory_sweep`.

## CLI

```bash
bun run src/index.ts serve              # Start MCP server (stdio)
bun run src/index.ts serve-remote       # Start HTTP server (Streamable HTTP)
bun run src/index.ts status             # Vault health stats
bun run src/index.ts search <query>     # Search memories
bun run src/index.ts save <title> <content>  # Save a memory
bun run src/index.ts setup <target>     # Configure (claude-code, cursor, gemini, hooks)
bun run src/index.ts sync push          # Push vault to S3
bun run src/index.ts sync pull          # Pull vault from S3
```

## Remote server (Claude.ai + iOS)

```bash
# Start remote server with auth
MNEMON_TOKEN=your-secret-token PORT=8502 bun run src/index.ts serve-remote

# Environment variables
PORT=8502                    # Server port (default: 8502)
MNEMON_TOKEN=                # Bearer token for auth (empty = no auth)
MNEMON_VAULT=                # Custom vault path (default: ~/.mnemon/default.sqlite)
```

Then add as a custom connector on claude.ai: Settings > Connectors > Add Custom Connector with your server URL (`https://your-domain/mcp`).

## S3 vault sync

Sync your vault between local and remote via S3:

```bash
MNEMON_S3_BUCKET=my-bucket bun run src/index.ts sync push
MNEMON_S3_BUCKET=my-bucket bun run src/index.ts sync pull
```

## Architecture

**Phase 1:** Local MCP server via stdio. SQLite + FTS5 for keyword search, in-process brute-force cosine for vector search. EmbeddingGemma-300M for embeddings via node-llama-cpp on Metal.

**Phase 2:** Claude Code hooks for automatic memory capture — context surfacing on every prompt, session extraction on exit, handoff generation. 90% of memory happens without agent intervention.

**Phase 3:** Query expansion via local LLM, MMR diversity filtering, contradiction detection with confidence decay, user profile tools.

**Phase 4:** Remote Streamable HTTP server for Claude.ai web + iOS access. S3 vault sync between local and remote. Bearer token auth.

**Phase 5:** Test coverage (50 tests), type checking, documentation.

## Claude Code hooks

mnemon automatically captures memories via Claude Code hooks — no manual intervention needed:

| Hook | Event | What it does |
|------|-------|-------------|
| Context surfacing | UserPromptSubmit (8s) | Searches vault, injects relevant memories as XML context |
| Session extractor | Stop (30s) | Extracts observations from transcript via local 1.7B LLM |
| Handoff generator | Stop (30s) | Generates session summary for continuity |

Install with: `bun run src/index.ts setup hooks`

## Stack

- [Bun](https://bun.sh) — runtime (bun:sqlite, fast startup)
- [MCP SDK](https://github.com/modelcontextprotocol/sdk) — Model Context Protocol server
- [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) — local GGUF model inference (Metal GPU)
- [EmbeddingGemma-300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) — embedding model (314MB, 768d)

## License

MIT
