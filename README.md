# mnemon

[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-blueviolet.svg)](https://modelcontextprotocol.io)
[![PyPI](https://img.shields.io/pypi/v/mnemon-memory.svg)](https://pypi.org/project/mnemon-memory/)
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)]()

> One memory vault. Every MCP client. Self-hosted.
>
> **Status:** alpha — interfaces may change. [Issues](https://github.com/cipher813/mnemon/issues) and PRs welcome.

Claude Code, Cursor, claude.ai web/mobile/desktop, Claude Desktop, Gemini CLI — one vault shared across all of them.

mnemon is a [Model Context Protocol](https://modelcontextprotocol.io) server with hybrid BM25 + vector search. Your data stays on your machine or your own Fly app.

**Platforms:** Tested on macOS 14+. Linux should work. Windows untested.

## Two product lanes: local and web

| Capability | **mnemon local** | **mnemon web** |
|---|---|---|
| Claude Code (CLI) | ✅ | ✅ |
| Claude Desktop (Mac/Win app) | ✅ | ✅ |
| Cursor | ✅ | ✅ |
| Gemini CLI | ✅ | ✅ |
| Any local MCP client | ✅ | ✅ |
| **claude.ai (web)** | ❌ | ✅ |
| **Claude mobile app** | ❌ | ✅ |
| Memory shared across laptop + desktop | ❌ | ✅ |
| Durable off-machine backup | ❌ (manual file copy) | ✅ (S3) |
| External accounts required | none | Fly.io + AWS |
| Credit card required | no | yes (both free tiers) |
| First-install setup time | ~2 min | ~15 min |
| Ongoing cost | $0 | ~$0 on free tiers |
| Cold-start latency after idle | none | 2–5s |

- **mnemon local** — one machine, one or more MCP clients. Zero external accounts. `pip install mnemon-memory && mnemon setup`.
- **mnemon web** — memory across devices + claude.ai + mobile. Requires Fly.io + AWS. `mnemon upgrade web --app-name my-mnemon`.

Start local, upgrade later — your vault rides along. `mnemon downgrade local` reverts. Comfortable up to ~50k memories.

## Install

```bash
pip install mnemon-memory
```

Optional: `pip install "mnemon-memory[ui]"` for the Streamlit dashboard, or `[llm]` for the on-device 1.7B model.

**From source** (e.g. to try unreleased fixes on `main`):

```bash
git clone https://github.com/cipher813/mnemon.git
cd mnemon
pip install -e .
```

**For contributors** (adds `pytest`, `ruff`, and other test/lint tooling):

```bash
pip install -e ".[dev]"
```

## Quick Start

### Local

```bash
pip install mnemon-memory
mnemon setup
```

Auto-detects Claude Code, Claude Desktop, Cursor, Gemini CLI — configures each, then runs `mnemon doctor`.

First `memory_search` takes ~10–20s (one-time FastEmbed model download). Subsequent calls are fast.

### Web

Prereqs: `flyctl` authenticated, `aws` CLI configured, an S3 bucket.

```bash
export MNEMON_S3_BUCKET=my-mnemon-vault
mnemon upgrade web --app-name my-mnemon
```

After it finishes, add `https://my-mnemon.fly.dev/mcp` to claude.ai and the Claude mobile app manually (Settings → Connectors / Connected Apps).

### Upgrade to a newer version (already on web)

Rerun the same command after `pip install -U mnemon-memory` — `upgrade web` is idempotent. If the Fly app already exists, it skips the first-time steps (S3 push, volume create, client reconfigure) and just redeploys with the new version pinned. Clients keep their URL and token; the new image is picked up on the next request.

```bash
pip install -U 'mnemon-memory[server]'
mnemon upgrade web --app-name my-mnemon
```

### Downgrade back to local

```bash
mnemon downgrade local --destroy-fly-app
```

Pulls the Fly vault back via S3, reconfigures clients to stdio, optionally destroys the Fly app. No memories lost.

### Visualize your vault

```bash
pip install "mnemon-memory[ui]"
mnemon dashboard
```

Streamlit UI at `http://localhost:8503` — stats, search, timeline, UMAP graph view, profile. Works against local and remote vaults.

### Use it

Once configured, mnemon works automatically — memories save and surface during your sessions. You can also interact directly:

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

When configured via `mnemon setup`, the following hooks are installed on Claude Code:

| Hook | Event | Timeout | Mode | Description |
|------|-------|---------|------|-------------|
| Health warm-keeper | `UserPromptSubmit` | 40s | remote only | `curl /health` to wake/keep the Fly machine warm. Runs first so the MCP call below has a warm machine. |
| Context surfacing | `UserPromptSubmit` | 8s | both | Searches vault and injects relevant memories as context |
| Session pre-warm | `SessionStart` | 90s | remote only | Polls `/health` for up to 60s in the background so the first prompt of the session lands on a warm machine |
| Session extractor | `Stop` | 30s | both | Extracts decisions, preferences, and observations from the transcript |
| Handoff generator | `Stop` | 30s | both | Creates a session summary for the next session |

The extractor and handoff generator use LLM-based extraction when `mnemon[llm]` is installed, with regex/heuristic fallback otherwise.

### Why a warm-keeper if Fly auto-stops?

A self-hosted mnemon Fly app with `auto_stop_machines = "stop"` (the default in `fly.toml.example`) will autostop after a few minutes of idle. The warm-keeper resets Fly's idle timer on every prompt, so the machine stays warm during an active Claude Code session and only autostops once you've been idle for a while. Cost stays the same — Fly bills only running time — but you get reliable mid-session access without paying for an always-on machine. The `|| true` ensures a slow Fly cold-start never blocks your prompt.

### What if a cold-stop happens anyway?

The server persists every issued MCP session ID to `<vault_dir>/mcp_sessions.sqlite` (7-day TTL). When a request bearing a known-but-not-in-memory session ID arrives at a fresh process — typical after a cold-stop or redeploy — the session is transparently resumed: a new transport is spawned with the same ID, and the underlying `ServerSession` is born already-initialized so tool calls succeed without a re-handshake. The MCP client sees no break in continuity. This is the safety net under the warm-keeper, not a replacement for it.

## Uninstall

Remove mnemon state from this machine. Nothing user-owned in the cloud is touched.

```bash
mnemon uninstall [--yes] [--keep-vault]
```

### What mnemon uninstall removes

- `~/.mnemon/` — vault (SQLite + vectors), archive/, remote_url, local_token, models cache. With `--keep-vault`, this directory is preserved.
- Claude Code MCP registration (`claude mcp remove --scope user mnemon`).
- mnemon hook + mcpServers entries in `~/.claude/settings.json`.
- mnemon entry in `~/.cursor/mcp.json`.
- mnemon entry in Claude Desktop's config.

### What mnemon uninstall does NOT touch

- **The `mnemon-memory` Python package.** Use `pip uninstall mnemon-memory` separately.
- **Your Fly.io app.** Destroy it first with `mnemon downgrade local --destroy-fly-app` if you want the app gone — that pulls the remote vault back to local so no memories are lost.
- **Your S3 bucket contents.** mnemon has no `sync delete`.
- **claude.ai + Claude mobile MCP entries.** These live in your Anthropic account. `claude mcp list` shows them with a `claude.ai` prefix. Remove via Settings → Connectors in the claude.ai web UI. If `mnemon uninstall` detects one, it surfaces a `⚠ REQUIRED` bullet pointing you there.

### Memory retention matrix

| Command | Local `~/.mnemon/default.sqlite` | Fly volume | S3 bucket contents |
|---|---|---|---|
| `mnemon uninstall` | deleted (unless `--keep-vault`) | **untouched** | **untouched** |
| `mnemon uninstall --keep-vault` | **untouched** | **untouched** | **untouched** |
| `mnemon downgrade local` | replaced with Fly state (via S3 pull) | untouched (keeps running) | untouched |
| `mnemon downgrade local --destroy-fly-app` | replaced with Fly state | destroyed (after data was pulled to local) | untouched |
| `mnemon upgrade web` | archived to `archive/pre-web-<date>.sqlite` | newly created, seeded from S3 | written to (push) |
| `mnemon sync push` / `mnemon sync pull` | read/write local | — | read/write |

Memories are always recoverable as long as at least one of {S3 backup, Fly volume, local vault, local archive} exists.

### Common flows

**Test from scratch on one machine:**

```bash
mnemon uninstall --yes
pip install -e .           # or: pip install mnemon-memory
mnemon setup
```

**Stop using mnemon entirely:**

```bash
mnemon downgrade local --destroy-fly-app    # tears down Fly, preserves vault via S3 pull
mnemon uninstall --yes                      # removes local state
pip uninstall mnemon-memory                 # removes the package
# Then remove mnemon entries in claude.ai and the Claude mobile app manually.
# Delete your S3 bucket contents if you want no residual memory data.
```

**Move to a new machine (preserve all memories):**

```bash
# Old machine:
mnemon sync push
mnemon uninstall --yes

# New machine:
pip install mnemon-memory
mnemon setup claude-code --remote-url https://<your-app>.fly.dev/mcp
```

## Install troubleshooting

### Intel Mac + Python 3.12: `pip install "mnemon-memory[ui]"` fails building `llvmlite` / `numba`

The `[ui]` extra pulls in `umap-learn`, which requires `numba` + `llvmlite`. Starting with `numba 0.63`, those packages only ship macOS wheels for Apple Silicon (arm64). On an Intel Mac (x86_64), pip falls back to a source build and fails with `llvmlite needs CMake tools to build`.

Pin to the last versions that ship x86_64 macOS wheels:

```bash
pip install 'numba==0.62.1' 'llvmlite==0.45.1' 'mnemon-memory[ui]'
```

If pip then complains about NumPy, add `'numpy<2.3'` to the same command.

Apple Silicon, Linux, and Windows are unaffected — prebuilt wheels exist on those platforms.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
