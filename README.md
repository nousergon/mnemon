# mnemon

Universal long-term memory layer for AI agents via MCP.

## Overview

mnemon provides persistent, searchable memory for AI agents across sessions. It uses SQLite with FTS5 for full-text search, composite scoring (relevance + recency + confidence), and the Model Context Protocol (MCP) for seamless integration with Claude Code, Cursor, and other MCP-compatible clients.

## Install

```bash
pip install mnemon
```

Or from source:

```bash
git clone https://github.com/cipher813/mnemon.git
cd mnemon
pip install -e ".[dev]"
```

## Quick Start

```bash
# Start MCP server (stdio transport)
mnemon serve

# Check vault health
mnemon status

# Search memories
mnemon search "deployment architecture"

# Save a memory
mnemon save "DB migration plan" "Migrate from PostgreSQL to DynamoDB in Q3"
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_search` | BM25 full-text search with composite scoring |
| `memory_get` | Fetch a specific memory by ID |
| `memory_save` | Store a new memory with content type classification |
| `memory_pin` | Pin a memory to prevent archival and boost confidence |
| `memory_forget` | Soft-delete a memory |
| `memory_status` | Vault health stats |
| `memory_sweep` | Archive stale memories past their half-life |
| `memory_timeline` | Recent memories in reverse chronological order |

## Memory Types

Each memory has a content type that determines its default confidence and decay half-life:

| Type | Default Confidence | Half-Life |
|------|-------------------|-----------|
| `decision` | 0.85 | Never |
| `preference` | 0.80 | Never |
| `antipattern` | 0.80 | Never |
| `observation` | 0.70 | 90 days |
| `research` | 0.70 | 90 days |
| `project` | 0.65 | 120 days |
| `handoff` | 0.60 | 30 days |
| `note` | 0.50 | 60 days |

## Architecture

- **Storage:** SQLite + FTS5 with WAL mode for concurrent access
- **Search:** BM25 full-text search with composite scoring (relevance x 0.5 + recency x 0.25 + confidence x 0.25)
- **Deduplication:** Content-addressable storage via SHA-256 hashing
- **Diversity:** MMR filtering to reduce redundant results
- **Transport:** MCP stdio (local) and Streamable HTTP (remote)

## Development

```bash
# Run tests
pytest

# Run with verbose output
pytest -v
```

## License

MIT
