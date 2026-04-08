[![Bun](https://img.shields.io/badge/bun-1.3+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-62_passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-88%25_(testable)-green.svg)]()
[![CI](https://github.com/cipher813/mnemon/actions/workflows/ci.yml/badge.svg)](https://github.com/cipher813/mnemon/actions/workflows/ci.yml)

# mnemon (μνήμων)

Universal long-term memory layer for AI agents. One memory vault, every tool.

mnemon is an MCP server that gives Claude Code, Claude Desktop, Cursor, Gemini CLI, and any MCP-compatible client access to a shared, persistent memory store with hybrid keyword + semantic search. A remote Streamable HTTP server enables access from web and mobile clients.

## How it works

```
LOCAL (Mac — Metal GPU):
  [Claude Code]    --stdio--> [mnemon MCP] <--stdio-- [Cursor]
  [Claude Desktop] --stdio-->      |       <--stdio-- [Gemini CLI]
                             [SQLite + FTS5]
                             [Vector store]
                             [GGUF models on Metal]
                                   |
                             S3 vault sync
                                   |
REMOTE (EC2):
  [Claude.ai web]  --HTTP--> [mnemon remote]
  [Claude iOS]     --HTTP-->      |
  [Gemini mobile]  --HTTP-->      |
                            [SQLite + FTS5]
                            [BM25-only search]
```

## Client compatibility

| Client | Transport | Status |
|--------|-----------|--------|
| Claude Code | stdio + hooks | Working — auto context surfacing, session extraction, vault sync |
| Claude Desktop | stdio MCP | Working — all 13 tools available |
| Cursor | stdio MCP | Working — all 13 tools available |
| Gemini CLI | stdio MCP | Working — all 13 tools available |
| claude.ai web | Streamable HTTP | Ready — server deployed, waiting on client MCP support |
| Claude iOS | Streamable HTTP | Ready — server deployed, waiting on client MCP support |
| Gemini mobile | Streamable HTTP | Ready — server deployed, waiting on client MCP support |

**Local clients** (stdio) get full hybrid search — BM25 + vector similarity with GPU-accelerated embeddings on Apple Silicon.

**Remote clients** (HTTP) get BM25 keyword search with composite scoring — no GPU required on the server. Bearer token authentication.

## Features

- **Hybrid search:** BM25 full-text + vector similarity + Reciprocal Rank Fusion + composite scoring (relevance/recency/confidence)
- **Automatic memory capture:** Claude Code hooks surface relevant context on every prompt, extract observations at session end, and generate handoff summaries for continuity
- **Contradiction detection:** New memories are checked against existing ones; conflicting memories get confidence decay
- **Query expansion:** Local GGUF model expands search queries for better recall
- **MMR diversity:** Maximal Marginal Relevance prevents redundant search results
- **Confidence decay:** Memories have type-based half-lives; stale memories are archived automatically
- **User profiles:** Synthesized from stored preferences and decisions
- **S3 vault sync:** Push/pull vault between local and remote, with automated sync on session end
- **Bearer token auth:** Secure remote access without browser-based login flows

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

# Start the MCP server (stdio, for local clients)
bun run src/index.ts serve
```

## Configure local clients

```bash
# Automated setup
bun run src/index.ts setup claude-code   # Claude Code (~/.claude/settings.json)
bun run src/index.ts setup cursor        # Cursor (~/.cursor/mcp.json)
bun run src/index.ts setup gemini        # Gemini CLI (~/.gemini/settings.json)
bun run src/index.ts setup hooks         # Claude Code auto-memory hooks
```

Or manually add to any MCP-compatible client's config:

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

**Claude Desktop:** Add the above to `~/Library/Application Support/Claude/claude_desktop_config.json` under the `mcpServers` key.

## Configure remote clients

Deploy the remote server on any host (EC2, VPS, etc.):

```bash
MNEMON_TOKEN=your-secret-token PORT=8503 bun run src/index.ts serve-remote
```

The remote server exposes the same MCP tools over Streamable HTTP at `/mcp`, with a health check at `/health`. It runs BM25-only search (no GPU, no embedding model) for minimal resource usage.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8502 | Server port |
| `MNEMON_TOKEN` | (empty) | Bearer token for auth (empty = no auth) |
| `MNEMON_VAULT` | `~/.mnemon/default.sqlite` | Custom vault path |

**Production deployment** (systemd):

```ini
[Unit]
Description=Mnemon Remote MCP Server
After=network.target

[Service]
Type=simple
ExecStart=/path/to/bun run /path/to/mnemon/src/server.ts
Environment=PORT=8503
Environment=MNEMON_TOKEN=your-secret-token
Restart=on-failure
MemoryMax=150M

[Install]
WantedBy=multi-user.target
```

Put an HTTPS reverse proxy (nginx, Caddy) in front for TLS termination. Remote MCP clients connect to `https://your-domain/mcp` with the bearer token.

## Claude Code hooks

Three hooks automate memory capture in Claude Code:

| Hook | Event | Description |
|------|-------|-------------|
| Context surfacing | UserPromptSubmit | Searches vault for relevant memories and injects them as context |
| Session extractor | Stop | Extracts observations, decisions, and preferences from the conversation |
| Handoff generator | Stop | Creates a session summary for next-conversation continuity |

Install with `bun run src/index.ts setup hooks` or add manually to `~/.claude/settings.json`.

## S3 vault sync

Sync your vault between local and remote instances via S3:

```bash
MNEMON_S3_BUCKET=my-bucket bun run src/index.ts sync push   # Local -> S3
MNEMON_S3_BUCKET=my-bucket bun run src/index.ts sync pull   # S3 -> Local
```

For automated sync, add an S3 push to your Claude Code Stop hooks and a systemd timer on the remote server to pull periodically.

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
bun run src/index.ts serve              # MCP server (stdio)
bun run src/index.ts serve-remote       # HTTP server (Streamable HTTP)
bun run src/index.ts status             # Vault health stats
bun run src/index.ts search <query>     # Search memories
bun run src/index.ts save <title> <content>  # Save a memory
bun run src/index.ts setup <target>     # Configure integration
bun run src/index.ts sync <push|pull>   # S3 vault sync
```

## Stack

- [Bun](https://bun.sh) — runtime (bun:sqlite, fast startup)
- [MCP SDK](https://github.com/modelcontextprotocol/sdk) — Model Context Protocol (stdio + Streamable HTTP)
- [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) — local GGUF model inference (Apple Silicon Metal)
- [EmbeddingGemma-300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) — embedding model (314MB, 768d)
- [QMD-1.7B](https://huggingface.co/ggml-org) — query expansion + observation extraction

## License

MIT
