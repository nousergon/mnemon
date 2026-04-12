# mnemon

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-327_passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)]()
[![MCP](https://img.shields.io/badge/MCP-compatible-blueviolet.svg)](https://modelcontextprotocol.io)
[![PyPI](https://img.shields.io/badge/PyPI-v0.3.0-blue.svg)](https://pypi.org/project/mnemon-memory/)

> Universal long-term memory layer for AI agents via [MCP](https://modelcontextprotocol.io).

mnemon gives AI agents persistent, searchable memory that survives across sessions. It uses hybrid BM25 + vector search, automatic confidence decay, and contradiction detection via the Model Context Protocol. Deploy as a remote server on Fly.io for a unified vault across all your MCP clients (Claude Code, Claude Desktop, Cursor, claude.ai), or run locally for development.

## Table of Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [MCP Tools](#mcp-tools)
- [Memory Types](#memory-types)
- [Claude Code Hooks](#claude-code-hooks)
- [Remote Server](#remote-server)
- [S3 Vault Sync](#s3-vault-sync)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Development](#development)

---

## Install

```bash
pip install mnemon-memory
```

With optional LLM support (local 1.7B model for query expansion, contradiction detection, and smarter session extraction):

```bash
pip install "mnemon-memory[llm]"
```

From source:

```bash
git clone https://github.com/cipher813/mnemon.git
cd mnemon
pip install -e ".[dev]"
```

## Quick Start

### 1. Configure your client

```bash
# Claude Code with remote vault (recommended — single vault across all clients)
mnemon setup claude-code --remote-url https://your-app.fly.dev/mcp

# Cursor with remote vault
mnemon setup cursor --remote-url https://your-app.fly.dev/mcp

# Local-only mode (development, no remote server needed)
mnemon setup claude-code
mnemon setup cursor
```

### 2. Use it

Once configured, mnemon works automatically:

- **Context surfacing**: relevant memories are injected before each prompt
- **Session extraction**: decisions, preferences, and observations are saved at session end
- **Handoff generation**: session summaries maintain continuity across sessions

You can also interact with memories directly via MCP tools or CLI:

```bash
mnemon search "deployment architecture"
mnemon save "DB migration plan" "Migrate from PostgreSQL to DynamoDB in Q3"
mnemon forget 42
mnemon status
```

## MCP Tools

### Retrieval

| Tool | Description |
|------|-------------|
| `memory_search` | Hybrid BM25 + vector search with composite scoring (relevance + recency + confidence) |
| `memory_get` | Fetch a specific memory by ID with full content |
| `memory_timeline` | Recent memories in reverse chronological order |
| `memory_related` | Find memories related to a given memory via the relationship graph |

### Mutation

| Tool | Description |
|------|-------------|
| `memory_save` | Store a new memory with content type classification and auto-embedding |
| `memory_pin` | Pin a memory to boost confidence and prevent archival |
| `memory_forget` | Soft-delete a memory (marked as invalidated, not physically removed) |

### Lifecycle

| Tool | Description |
|------|-------------|
| `memory_status` | Vault health stats — counts by type, vectors, pinned/invalidated |
| `memory_sweep` | Archive stale memories past their half-life (dry-run by default) |
| `memory_rebuild` | Re-embed all documents (use after upgrading embedding model) |

### Intelligence

| Tool | Description |
|------|-------------|
| `memory_check_contradictions` | Check a memory for conflicts using vector similarity + LLM classification |
| `profile_get` | Synthesized user profile from stored preferences and decisions |
| `profile_update` | Manually add a fact to the user profile |

## Memory Types

Each memory has a content type that determines its default confidence and decay half-life:

| Type | Default Confidence | Half-Life | Use for |
|------|-------------------|-----------|---------|
| `decision` | 0.85 | Never | Architectural choices, design decisions |
| `preference` | 0.80 | Never | User workflow habits, style preferences |
| `antipattern` | 0.80 | Never | Things that failed, approaches to avoid |
| `observation` | 0.70 | 90 days | Learned facts, discovered behaviors |
| `research` | 0.70 | 90 days | Investigation results, findings |
| `project` | 0.65 | 120 days | Project status, goals, context |
| `handoff` | 0.60 | 30 days | Session summaries for continuity |
| `note` | 0.50 | 60 days | General notes, default type |

Memories with access activity decay slower — each access extends the effective half-life by 10%, up to 3x the base value.

## Claude Code Hooks

When configured via `mnemon setup claude-code`, three hooks are installed:

| Hook | Event | Timeout | Description |
|------|-------|---------|-------------|
| Context surfacing | `UserPromptSubmit` | 8s | Searches vault and injects relevant memories as context |
| Session extractor | `Stop` | 30s | Extracts decisions, preferences, and observations from the transcript |
| Handoff generator | `Stop` | 30s | Creates a session summary for the next session |

The extractor and handoff generator use LLM-based extraction when `mnemon[llm]` is installed, with regex/heuristic fallback otherwise.

## Remote Server

Deploy mnemon as a remote Streamable HTTP server for a single vault shared across all MCP clients. This is the recommended production setup — Claude Code hooks, Claude Desktop, Cursor, and claude.ai all read and write the same memories.

```bash
# Start remote server (local testing)
MNEMON_LOCAL_TOKEN=your-secret-token mnemon serve-remote

# Custom port
PORT=9000 mnemon serve-remote
```

For production, deploy to [Fly.io](https://fly.io) with a persistent volume (1GB minimum RAM required for FastEmbed model). See `fly.toml` and `Dockerfile` in the repo. Set `MNEMON_LOCAL_TOKEN` as a Fly secret for hook authentication. OAuth (Auth0 or self-hosted) is supported for browser-based clients like claude.ai.

## S3 Vault Sync

Sync your vault across machines via S3:

```bash
# Push local vault to S3
MNEMON_S3_BUCKET=my-bucket mnemon sync push

# Pull vault from S3
MNEMON_S3_BUCKET=my-bucket mnemon sync pull
```

| Env var | Default | Description |
|---------|---------|-------------|
| `MNEMON_S3_BUCKET` | (required) | S3 bucket name |
| `MNEMON_S3_PREFIX` | `mnemon/vaults` | S3 key prefix |
| `MNEMON_VAULT_NAME` | `default` | Vault name |

Requires the AWS CLI (`aws`) on your PATH with valid credentials.

## Architecture

**Remote (production):** All clients hit a single Fly-hosted vault via Streamable HTTP. Claude Code hooks use a static bearer token (`MNEMON_LOCAL_TOKEN`). Browser clients (claude.ai, Claude Desktop) use OAuth.

**Local (development):** SQLite vault at `~/.mnemon/default.sqlite` with a companion vector store. Useful for testing and offline work.

```
~/.mnemon/
  remote_url           Remote server URL (written by mnemon setup --remote-url)
  local_token          Bearer token for remote auth (chmod 600)
  default.sqlite       Local SQLite vault (FTS5 + WAL mode, development only)
  default.vec.npz      Companion vector store (numpy, brute-force cosine)
  models/              Local LLM weights (session extraction, query expansion)
```

- **Storage**: SQLite with FTS5 full-text search, content-addressable deduplication (SHA-256)
- **Search**: Hybrid BM25 + vector (384d, bge-small-en-v1.5 via FastEmbed) fused with Reciprocal Rank Fusion
- **Scoring**: Composite score = 0.5 * relevance + 0.25 * recency + 0.25 * confidence
- **Diversity**: MMR filtering (Jaccard bigram similarity > 0.6 demoted by 50%)
- **Intelligence** (optional): Local 1.7B LLM (QMD-query-expansion) for query expansion, contradiction detection, session extraction — zero API cost
- **Transport**: MCP stdio (local) and Streamable HTTP (remote)

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `MNEMON_REMOTE_URL` | (none) | Remote server URL (or `~/.mnemon/remote_url` file) |
| `MNEMON_LOCAL_TOKEN` | (none) | Bearer token for remote auth (or `~/.mnemon/local_token` file) |
| `MNEMON_VAULT_DIR` | `~/.mnemon` | Local vault directory |
| `MNEMON_MODEL_DIR` | `~/.mnemon/models` | Directory for LLM model files |
| `PORT` | `8502` | Remote server port |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (327 tests)
pytest

# Run tests with coverage
pytest --cov=mnemon --cov-report=term-missing

# Run a specific test file
pytest tests/test_store.py -v
```

## License

MIT
