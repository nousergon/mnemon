# mnemon

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-550%2B_passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)]()
[![MCP](https://img.shields.io/badge/MCP-compatible-blueviolet.svg)](https://modelcontextprotocol.io)
[![PyPI](https://img.shields.io/badge/PyPI-v0.5.0-blue.svg)](https://pypi.org/project/mnemon-memory/)

> One memory vault. Every MCP client. Self-hosted, no third-party auth.

Claude Code remembers your architecture decisions. Cursor remembers your API conventions. claude.ai web/mobile/desktop remembers your project context. All from a single vault you own and run.

mnemon is a [Model Context Protocol](https://modelcontextprotocol.io) server with hybrid BM25 + vector search, automatic confidence decay, and contradiction detection. Deploy as a remote server on Fly.io (~$1/mo for a personal vault) or run locally for development. Browser clients authenticate via self-hosted OAuth 2.1 + PKCE — no Auth0, Clerk, or other third-party auth vendor required.

**Privacy:** mnemon sends no telemetry. Your vault never leaves your server. Embeddings run locally via FastEmbed. The optional LLM (1.7B params) runs on-device via llama.cpp. The only outbound calls are: (a) the FastEmbed model download on first run, (b) optional S3 vault sync if you configure it, and (c) `huggingface-hub` to fetch the optional LLM weights.

**Platforms:** Tested on macOS 14+. Linux should work — Python + SQLite + FastEmbed are portable. Windows untested.

## Two product lanes: local and web

mnemon ships as one tool with two distinct deployment modes. Pick the one that matches what you need; upgrade later if that changes.

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

- **mnemon local** is the right call if you use one machine and one-or-more MCP clients on that machine. Zero external accounts, zero tokens, zero config files to edit. `pip install mnemon-memory && mnemon setup` and you're done.
- **mnemon web** is the right call if you want memory across devices, via the claude.ai web app, or on mobile. Requires a Fly.io account (≈$0/mo on the free tier with auto-stop) and an AWS account (S3 backup — also free-tier for mnemon's payload size). `mnemon upgrade web` wraps the deploy + seed + reconfigure in one command.

You can start local, save a few hundred memories, and `mnemon upgrade web --app-name my-mnemon` later — your entire vault rides along. Symmetric `mnemon downgrade local` reverts if you change your mind.

**Who this is for:** individual developers who use multiple MCP clients and want a single memory vault shared across all of them. Comfortable up to ~50k memories. Beyond that, evaluate [Mem0](https://mem0.ai), [Zep](https://www.getzep.com), or [Letta](https://docs.letta.com) — they offer managed multi-tenancy, larger embeddings, and per-user isolation that mnemon intentionally doesn't.

**What mnemon doesn't do (by design):**
- Multi-tenancy / per-user isolation for teams.
- Managed SaaS — if you choose web, you run the server.
- Automatic fact extraction from arbitrary message streams at Mem0's level of polish.

## Table of Contents

- [Two product lanes: local and web](#two-product-lanes-local-and-web)
- [Install](#install)
- [Quick Start](#quick-start)
  - [Local — one command, zero accounts](#local--one-command-zero-accounts)
  - [Web — one command once you have Fly + AWS](#web--one-command-once-you-have-fly--aws)
  - [Downgrade back to local](#downgrade-back-to-local)
  - [Uninstall](#uninstall)
  - [Visualize your vault](#visualize-your-vault)
  - [Use it](#use-it)
- [MCP Tools](#mcp-tools)
- [Memory Types](#memory-types)
- [Claude Code Hooks](#claude-code-hooks)
- [Remote Server](#remote-server)
  - [Self-host on Fly.io](#self-host-on-flyio)
  - [Troubleshooting](#troubleshooting)
- [S3 Vault Sync](#s3-vault-sync)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)
- [Development](#development)

---

## Install

```bash
pip install mnemon-memory
```

Optional extras:

```bash
pip install "mnemon-memory[ui]"    # Streamlit dashboard (mnemon dashboard)
pip install "mnemon-memory[llm]"   # local 1.7B model for query expansion + smarter extraction
pip install "mnemon-memory[ui,llm]" # both
```

From source:

```bash
git clone https://github.com/cipher813/mnemon.git
cd mnemon
pip install -e ".[dev]"
```

## Quick Start

### Local — one command, zero accounts

```bash
pip install mnemon-memory
mnemon setup
```

`mnemon setup` with no target auto-detects every MCP client you have installed (Claude Code, Claude Desktop, Cursor — plus prints a copy-paste snippet for Gemini CLI), configures each, installs in-process hooks that dispatch against the local SQLite vault at `~/.mnemon/default.sqlite`, and runs `mnemon doctor` to verify. A single explicit target — `mnemon setup claude-code` etc. — also works.

No Fly account. No AWS account. No tokens. No config files to edit. Restart the clients and you're done.

**Heads-up:** the first `memory_search` after a fresh install takes ~10–20s while FastEmbed downloads the embedding model (one-time). Subsequent calls are fast.

### Web — one command once you have Fly + AWS

```bash
# Prereqs: flyctl authenticated, aws CLI configured, S3 bucket for vault backup.
export MNEMON_S3_BUCKET=my-mnemon-vault
mnemon upgrade web --app-name my-mnemon
```

`mnemon upgrade web` deploys a Fly app running `mnemon serve-remote`, seeds its volume from S3 (your local vault rides along), archives your local `~/.mnemon/default.sqlite` to `~/.mnemon/archive/pre-web-YYYY-MM-DD.sqlite`, rewrites every detected client's MCP config to point at `https://my-mnemon.fly.dev/mcp`, installs Claude Code's SessionStart pre-warm hook, and runs `mnemon doctor --fail-on-warn` against the new remote.

After it's done, add the same URL + token manually to claude.ai and the Claude mobile app (Settings → Connected Apps). Those two live in Anthropic's UI and can't be auto-configured.

See [Self-host on Fly.io](#self-host-on-flyio) for what the command does under the hood and what to do if a step fails.

### Downgrade back to local

```bash
mnemon downgrade local --destroy-fly-app
```

Symmetric exit: pulls the current Fly vault state back to `~/.mnemon/default.sqlite` via S3, reconfigures every client back to stdio mode, optionally destroys the Fly app (with a y/N confirmation). No memories lost — whatever was on the Fly vault at teardown time becomes the new local vault.

### Uninstall

Remove mnemon state from this machine. Nothing user-owned in the cloud is touched.

```bash
mnemon uninstall [--yes] [--keep-vault]
```

#### What mnemon uninstall removes

- `~/.mnemon/` — vault (SQLite + vectors), archive/, remote_url, local_token, models cache. With `--keep-vault`, this directory is preserved.
- Claude Code MCP registration (`claude mcp remove --scope user mnemon`).
- mnemon hook + mcpServers entries in `~/.claude/settings.json`.
- mnemon entry in `~/.cursor/mcp.json`.
- mnemon entry in Claude Desktop's config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS).

#### What mnemon uninstall does NOT touch

These are deliberate scope exclusions. Memory data is preserved in full:

- **The `mnemon-memory` Python package.** Use `pip uninstall mnemon-memory` separately if you want the CLI gone too.
- **Your Fly.io app.** If you deployed web via `mnemon upgrade web`, that app keeps running (and keeps billing, if above free tier). Destroy it explicitly with `mnemon downgrade local --destroy-fly-app` BEFORE running uninstall — that command pulls the remote vault back to local first, so no memories are lost.
- **Your S3 bucket contents.** Your vault backup stays in S3. mnemon has no `sync delete`; we never issue DELETE against your bucket.
- **claude.ai + Claude mobile MCP entries.** If you added mnemon via claude.ai's web UI (Settings → Connected Apps), that registration lives in your Anthropic account, not on your machine. `claude mcp list` shows it with a `claude.ai` prefix. No local command can remove it — you have to delete it in the claude.ai web UI manually. If `mnemon uninstall` detects one, it surfaces a loud `⚠ REQUIRED` bullet pointing you there.

#### Memory retention matrix

| Command | Local `~/.mnemon/default.sqlite` | Fly volume | S3 bucket contents |
|---|---|---|---|
| `mnemon uninstall` | deleted (unless `--keep-vault`) | **untouched** | **untouched** |
| `mnemon uninstall --keep-vault` | **untouched** | **untouched** | **untouched** |
| `mnemon downgrade local` | replaced with Fly state (via S3 pull) | untouched (keeps running) | untouched |
| `mnemon downgrade local --destroy-fly-app` | replaced with Fly state | destroyed (after data was pulled to local) | untouched |
| `mnemon upgrade web` | archived to `archive/pre-web-<date>.sqlite` | newly created, seeded from S3 | written to (push) |
| `mnemon sync push` / `mnemon sync pull` | read/write local | — | read/write |

Memories are always recoverable as long as at least one of {S3 backup, Fly volume, local vault, local archive} exists.

#### Common flows

**Test from scratch on one machine** (validating setup, testing install flow):

```bash
mnemon uninstall --yes
# Your Fly + S3 + claude.ai keep running; memories in the cloud are safe.
pip install -e .           # or: pip install mnemon-memory
mnemon setup               # fresh local install, no prior config
```

If a claude.ai-synced mnemon entry is also present, uninstall will tell you — remove it in claude.ai's web UI before re-setup or Claude Code will shadow the new local registration.

**Stop using mnemon entirely** (delete all memories, destroy cloud infra):

```bash
# If you were in web mode, tear down the Fly app first (preserves the
# vault in S3 as a backup before destruction):
mnemon downgrade local --destroy-fly-app

# Then remove local state:
mnemon uninstall --yes

# Then (optionally) remove the pip package:
pip uninstall mnemon-memory

# Then (manually, if applicable):
# - Remove the mnemon entry in claude.ai → Settings → Connected Apps
# - Remove the mnemon entry in the Claude mobile app
# - Delete your S3 bucket contents if you want no residual memory data
```

**Move to a new machine** (preserve all memories):

```bash
# On the old machine:
mnemon sync push           # ensure S3 has the latest vault
mnemon uninstall --yes     # optional; Fly + S3 keep serving

# On the new machine:
pip install mnemon-memory
mnemon setup claude-code --remote-url https://<your-app>.fly.dev/mcp
# If you're local-only on the old machine:
mnemon sync pull           # pulls your vault from S3 to local
mnemon setup               # wires clients
```

### Visualize your vault

```bash
pip install "mnemon-memory[ui]"
mnemon dashboard
```

Opens a Streamlit UI at `http://localhost:8503` — Home stats, Search, Timeline, Graph (UMAP 2D projection of the vector space), Profile. Works against both local and remote vaults.

![Memory Graph dashboard](docs/images/dashboard-graph.png)

### Use it

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

### Run locally (development)

```bash
MNEMON_LOCAL_TOKEN=your-secret-token mnemon serve-remote
PORT=9000 mnemon serve-remote   # custom port
```

### Self-host on Fly.io

You'll get a bearer-authenticated MCP endpoint at `https://<your-app>.fly.dev/mcp` running on your own Fly account. Budget ~$0–$2/mo for a personal vault (auto-stop idle, 1GB volume). Free tier covers most personal use.

**Prerequisites:**
- A [Fly.io](https://fly.io) account with [`flyctl`](https://fly.io/docs/hands-on/install-flyctl/) on `$PATH`.
- An AWS account with the AWS CLI v2 configured (`aws configure`), plus an S3 bucket for vault backup.
- `mnemon setup` already run in local mode (gives `mnemon upgrade web` a vault to migrate).

**One-command path.** `mnemon upgrade web --app-name <name>` handles everything: `mnemon sync push` (local → S3), `flyctl launch` with an embedded Dockerfile that installs `mnemon-memory[server]` from PyPI, `flyctl volumes create`, `flyctl secrets set` (bearer token + AWS creds so the container can pull from S3), `flyctl deploy`, and a post-deploy `flyctl ssh console -C 'mnemon sync pull'` to seed the vault. Then it rewrites every detected client's MCP config and runs `mnemon doctor --fail-on-warn` against the new remote.

If a step fails, the orchestration aborts. For failures after `flyctl launch` you may need `flyctl apps destroy <name>` to clean up the partial Fly state — the error message will tell you.

**Guardrail — don't touch prod.** Set `MNEMON_PROD_APP_NAMES=mnemon-memory,<your-other-prod-apps>` before running `mnemon upgrade web` and the command will refuse to target any of those names. Cheap insurance when testing with a scoped `--app-name`.

**Manual runbook.** The long-form steps below still work if you want to deploy by hand or debug a step of the one-command path. The files referenced (`Dockerfile`, `fly.toml.example`) live in this repo:

**1. Pick an app name and copy the template.**

```bash
cp fly.toml.example fly.toml
# Edit fly.toml: replace REPLACE_ME_fly_app_name (3 occurrences) with your chosen app name.
# Pick something globally unique on Fly — e.g. "my-mnemon-vault".
```

The real `fly.toml` is gitignored — it holds your specific app identity. `fly.toml.example` stays in the repo as the template.

**2. Create the app and the persistent volume.**

```bash
fly launch --copy-config --no-deploy      # creates the app from your edited fly.toml; no deploy yet
fly volume create mnemon_data --size 1 --region sjc --yes   # 1GB fits thousands of memories; match primary_region; --yes skips the single-region-volume confirmation prompt
```

Without the volume step, every restart wipes your vault — the `[mounts]` block in `fly.toml` expects `mnemon_data` to exist.

**3. Generate and set secrets.**

```bash
# Generate two independent high-entropy secrets. Copy both into your password
# manager now — MNEMON_LOCAL_TOKEN is needed again on the client (step 5).
python -c "import secrets; print('MNEMON_LOCAL_TOKEN   =', secrets.token_urlsafe(32))"
python -c "import secrets; print('MNEMON_AS_PASSPHRASE =', secrets.token_urlsafe(32))"

# Paste into Fly (values land in shell history — clear it after with
# `history -d <n>` if that matters to you, or prefix the line with a
# space if HISTCONTROL=ignorespace is set):
fly secrets set MNEMON_LOCAL_TOKEN=<value-1> \
                MNEMON_AS_ENABLED=true \
                MNEMON_AS_PASSPHRASE=<value-2>
```

`MNEMON_AS_PASSPHRASE` is the single-user login for browser clients (claude.ai, Claude Desktop). There is no complexity enforcement in code — use a high-entropy value. `MNEMON_LOCAL_TOKEN` is the static bearer for headless clients (Claude Code hooks, Cursor).

**4. Deploy.**

```bash
fly deploy
```

First deploy pulls the FastEmbed model (~15–25s on first `memory_search`). Subsequent deploys reuse the cached layer.

**5. Verify.**

```bash
# Write the remote URL (public — safe to echo).
mkdir -p ~/.mnemon
echo "https://<your-app>.fly.dev/mcp" > ~/.mnemon/remote_url

# Write the token without leaking it to shell history. `read -rs` reads
# silently; `printf %s` avoids a trailing newline in the file.
read -rs -p "Paste MNEMON_LOCAL_TOKEN: " t && \
  printf %s "$t" > ~/.mnemon/local_token && \
  chmod 600 ~/.mnemon/local_token && \
  unset t

mnemon doctor
```

`mnemon doctor` runs 6 checks: remote URL configured, local token configured + 0600 perms, `/health` reachable, authenticated MCP tool call round-trips, and save + search + forget cycle. All 6 should pass green. If any fail, the error message points at the specific misconfiguration. (Without a configured remote, the same command switches to local mode and runs 3 equivalent checks against the on-disk SQLite vault.)

**6. Connect clients.**

```bash
# Claude Code hooks (uses MNEMON_LOCAL_TOKEN)
mnemon setup claude-code --remote-url https://<your-app>.fly.dev/mcp

# Cursor (uses MNEMON_LOCAL_TOKEN)
mnemon setup cursor --remote-url https://<your-app>.fly.dev/mcp
```

For **claude.ai** (web/mobile) and **Claude Desktop** — no CLI needed, these use the OAuth browser flow:

1. In the client, go to Settings → Connectors → Add custom connector.
2. Paste `https://<your-app>.fly.dev/mcp` as the connector URL.
3. Click Connect. Browser redirects to your server's login page.
4. Enter `MNEMON_AS_PASSPHRASE` from step 3 above.
5. You're in. The client now sees `memory_search`, `memory_save`, etc. alongside its built-in tools.

Browser clients self-register via Dynamic Client Registration (RFC 7591) — no manual client-id provisioning. Authentication uses PKCE + RS256 JWTs signed by the AS's own keypair (auto-generated on first boot, stored in the Fly volume at `/data/oauth_keys/`).

### Troubleshooting

If `mnemon doctor` fails, check the specific failing line:
- **Health endpoint unreachable** — app may be booting (cold start takes 15–25s for FastEmbed); retry after a moment. If persistent, check `fly logs -a <your-app>` and `fly status`.
- **Auth + MCP tool call returns 401** — `MNEMON_LOCAL_TOKEN` on your machine doesn't match the Fly secret. Re-copy from your password manager into `~/.mnemon/local_token`.
- **Round-trip fails** — `MNEMON_ALLOWED_HOSTS` in `fly.toml` doesn't include the hostname you're connecting through. It should match the host portion of `MNEMON_PUBLIC_URL`.

**First prompt feels slow, or context-surfacing returns empty.** Fly's `auto_stop_machines` pauses the VM after ~5 min of inactivity to keep personal vaults at ~$1/mo. The first request after a pause pays 2–5s to wake the machine, plus a one-time 15–25s on first-ever use while FastEmbed loads. Users with infrequent usage patterns (e.g. one session a day) hit this on every first prompt; the context-surfacing hook may time out and the prompt runs without memory context. The next prompt has it.

If this bothers you, disable auto-stop: set `auto_stop_machines = false` in `fly.toml`, then `fly scale count 1 --max-per-region 1 && fly deploy`. Expect ~$5/mo for always-on. Most personal users don't need this.

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

## Design decisions

A small set of architectural choices shape the rest of the system. Documented here so self-host users know what they're signing up for and reviewers can evaluate the trade-offs.

**Why SQLite + FTS5 (not Postgres, not a vector DB).** A single-file embedded database means no operational surface area — no connection pools, no migrations against a live DB, no standalone vector store to keep in sync. FTS5 gives production-grade BM25 without a separate Elasticsearch. A numpy-backed vector store sits alongside the SQLite file; brute-force cosine over a few thousand memories is faster than any network hop to a hosted vector DB. The single-file design also makes vault portability trivial — copy one file and you've moved your entire memory.

**Why hybrid BM25 + vector (not pure semantic).** Pure vector search misses exact-identifier lookups; pure keyword misses paraphrase. Reciprocal Rank Fusion combines both rankings, then composite scoring folds in recency and confidence. In practice this catches both "find my note about bge-small-en-v1.5" (keyword wins) and "memory about embedding models" (vector wins) without tuning.

**Why Fly.io (not AWS / GCP).** mnemon is designed to idle cheaply and wake on demand. Fly's `auto_stop_machines` + `min_machines_running=0` costs ~$0.50–0.90/mo for a personal vault; the closest AWS equivalent (ECS Fargate or App Runner) can't scale to zero and starts at ~$10/mo. Fly volumes are local-attached SSD, which matches SQLite's access pattern — AWS's equivalent (EFS) is slower and pricier. Deploy is one `fly.toml` and one command, vs. the VPC + ALB + ECS + IAM setup AWS requires — which matters for any future self-host user.

**Why self-hosted OAuth 2.1 + PKCE + DCR (not Auth0 / Clerk / Logto).** Requiring users to register an Auth0 tenant before they can try mnemon is a near-guaranteed bounce. mnemon ships with its own Authorization Server (well-known endpoints, `/oauth/authorize`, `/oauth/token` with PKCE, `/oauth/register` per RFC 7591, JWT issuance) — anyone can `fly deploy` and have a working OAuth-protected MCP endpoint with no third-party signup. The trade-off is less battle-tested auth code; the mitigation is that browser clients are the only OAuth consumers, and headless clients (Claude Code, Cursor) use a simple static bearer.

**Why MCP + a separate memory server (not Claude's native memory).** Claude's native memory is account-scoped and only reaches Anthropic products (claude.ai web/mobile/desktop). It doesn't reach Claude Code, Cursor, or any other MCP-speaking client. mnemon serves the cross-client case: a single vault that Claude Code hooks, Cursor, and claude.ai can all read and write. It's also self-hosted, exportable, and programmatically introspectable — the opposite of Anthropic's closed-box model. These systems are complementary, not competing.

## Configuration

**Client-side (hooks, CLI)**

| Env var | Default | Description |
|---------|---------|-------------|
| `MNEMON_REMOTE_URL` | (none) | Remote server URL (or `~/.mnemon/remote_url` file) |
| `MNEMON_LOCAL_TOKEN` | (none) | Bearer token for remote auth (or `~/.mnemon/local_token` file) |
| `MNEMON_VAULT_DIR` | `~/.mnemon` | Local vault directory |
| `MNEMON_MODEL_DIR` | `~/.mnemon/models` | Directory for LLM model files |

**Server-side (`mnemon serve-remote`)**

| Env var | Default | Description |
|---------|---------|-------------|
| `MNEMON_AS_ENABLED` | `false` | Enable the self-hosted OAuth Authorization Server |
| `MNEMON_AS_PASSPHRASE` | (none) | Single-user login passphrase (required when AS enabled) |
| `MNEMON_AS_KEY_DIR` | `$MNEMON_VAULT_DIR/oauth_keys` | RSA keypair storage directory |
| `MNEMON_PUBLIC_URL` | (none) | Externally-reachable base URL (required when AS enabled) |
| `MNEMON_LOCAL_TOKEN` | (none) | Static bearer for headless clients (hooks, Cursor) |
| `MNEMON_ALLOWED_HOSTS` | (none) | Comma-separated host allowlist for DNS-rebinding protection |
| `PORT` | `8502` | Remote server port |

## Known limitations

Client-side behaviors that affect mnemon users but are not bugs in mnemon itself. Upstream tracking linked where applicable.

**Claude Code: MCP session invalidated after server restart.** When the remote mnemon server restarts (via `fly deploy`, `fly secrets set`, or Fly auto-stop/auto-start), Claude Code's cached MCP session ID becomes stale. Subsequent tool calls from within an active Claude Code session return `Session not found`, and the client does not auto-reinitialize. Workaround: quit and re-launch Claude Code. Hooks are unaffected — they use the static bearer path and bypass the MCP session layer. Upstream: [anthropics/claude-code#46533](https://github.com/anthropics/claude-code/issues/46533).

**Claude Code: `/mcp authenticate` CLI hang after browser OAuth success.** When authenticating a new OAuth-protected MCP connector via `/mcp`, the browser passphrase flow succeeds and the server issues a JWT, but the CLI prompt that should confirm completion does not respond to Enter (only Escape). Workaround: press Escape, then quit and re-launch Claude Code; the connector state persists. Upstream: [anthropics/claude-code#42707](https://github.com/anthropics/claude-code/issues/42707).

**FastEmbed cold start.** The first MCP tool call after a Fly machine auto-stop takes 15–25s while the FastEmbed ONNX model loads into memory. Subsequent calls are fast. Mitigated by a polling SessionStart hook and an eager initialization step in `mnemon serve-remote`; Fly's `http_service.checks.grace_period` is set accordingly.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (450+ tests)
pytest

# Run tests with coverage
pytest --cov=mnemon --cov-report=term-missing

# Run a specific test file
pytest tests/test_store.py -v
```

## License

MIT
