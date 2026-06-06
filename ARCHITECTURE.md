# mnemon — Architecture

A map of the codebase for readers and contributors. mnemon is **two products in one codebase**: a simple **local** memory vault (zero accounts, single SQLite file, stdio MCP) and an optional **hosted web** layer (Fly-deployed remote MCP with OAuth, multi-device). The most important thing to understand:

> **You can read and understand the entire local memory engine without ever touching the hosted-web code.** The core is `store → embedder → vecstore → search → server`. Everything OAuth / session / dashboard / migration is an *optional layer* on top.

If you're here to understand how memory works, read the **Core** table. If you're deploying or hacking on the hosted server, read **Hosted layer**.

---

## The two modes

| | Local (default) | Hosted web (optional) |
|---|---|---|
| Storage | Single SQLite file `~/.mnemon/default.sqlite` + `default.vec.npz` | Same engine, on Fly + S3 vault backup |
| Transport | stdio MCP (`mnemon serve`) | Streamable HTTP MCP (`mnemon serve-remote`) |
| Accounts | None | OAuth (self-hosted AS) |
| Install | `pip install mnemon-memory` | `pip install "mnemon-memory[server]"` + deploy |
| Invariant | — | **Single source of truth at all times** — never both local and remote active; no sync daemon. In remote mode the local vault is *inaccessible* (fails loud). |

---

## Core (local engine — always present, read this first)

The memory engine. A local-only user exercises exactly this set.

| Module | Purpose |
|---|---|
| `store.py` | SQLite + FTS5 storage — `documents` / `content` / `relations`, supersession chain (`invalidated_by`), the standing **`tier`** column, capture-attention **`recurrence_count`**. The heart of the system. |
| `vecstore.py` | In-process vector store — brute-force cosine over numpy arrays (`default.vec.npz`). |
| `embedder.py` | FastEmbed `bge-small-en-v1.5` wrapper (384-d ONNX, ~13 MB, lazy singleton). |
| `search.py` | Retrieval — BM25 ⊕ vector via RRF fusion → composite score (relevance / recency / confidence) → MMR diversity. Optional LLM query expansion. |
| `config.py` | Content types, decay half-lives, scoring constants — the tunables. |
| `safety.py` | Stored-injection defense — `defang_control_markup` (recall boundary) + `contains_control_markup` (capture-boundary reject). See `SECURITY.md` / the 5-layer model. |
| `contradiction.py` + `nli.py` | Contradiction detection via NLI (`cross-encoder/nli-deberta-v3-xsmall`) + confidence decay. |
| `llm.py` | **Optional** (`[llm]` extra) — QMD-1.7B GGUF via llama-cpp-python, used by `search.py` for query expansion. Not required. |
| `server.py` | The MCP server (stdio) — registers the **17 tools**. Imports only `store`, `search`, `safety`. This is the local entry point. |
| `api.py` | In-process tool surface (same shapes as `server.py`) so local hooks / `doctor` / setup work without any HTTP endpoint. |
| `server_proxy.py` | When a remote vault is configured, `mnemon serve` forwards stdio → remote (fail-loud, never opens local). |
| `cli.py` | CLI dispatcher (`serve`, `status`, `search`, `save`, `setup`, `sync`, `standing`, …). |
| `setup.py` | Auto-configure MCP clients (Claude Code / Desktop / Cursor / Gemini) + hooks. |
| `doctor.py` | Health checks (`mnemon doctor`). |
| `mirror.py` | Mirror local memory files (upsert by slug) for the `auto_mirror` hook. |
| `sync.py` | S3 vault backup (push/pull). |

### Hooks (`hooks/` — Claude Code integration, all best-effort, never block the session, exit 0)
| Hook | Trigger | Purpose |
|---|---|---|
| `context_surfacing.py` | UserPromptSubmit | `memory_search` → inject the `<mnemon-context>` spotlight envelope |
| `session_extractor.py` | Stop | LLM-or-regex observation extraction → `memory_save` (Layer-0 control-markup reject) |
| `handoff_generator.py` | Stop | Session-summary memory |
| `auto_mirror.py` | PostToolUse | Mirror local memory files (`mirror.py`) |
| `framework.py` | (shared) | stdin/stdout, SHA-256 dedup, noise filtering |
| `_client.py` / `_remote_client.py` | (shared) | Local (via `api.py`) vs remote tool-call clients |

---

## Hosted layer (optional — only for the web product; `[server]` extra)

A **local-only user never imports any of these.** They exist for the Fly-deployed multi-device product. (Verified: `server.py` does not import any module in this table.)

| Module | Purpose |
|---|---|
| `server_remote.py` | Remote HTTP MCP (Streamable HTTP) — reuses `server.mcp`; persistent sessions + warm-keeper. |
| `oauth_as.py` | Self-hosted OAuth Authorization Server (DCR, refresh-token rotation grace). |
| `auth.py` | OAuth config + middleware for the remote server. |
| `persistent_sessions.py` | Session store + warm-keeper for the remote transport (single uvicorn worker). |
| `dashboard/` | Streamlit vault dashboard (`[ui]` extra). |

### Migration tooling (advanced, not core-path)
| Module | Purpose |
|---|---|
| `upgrade.py` | `mnemon upgrade web` — local → hosted migration + Fly deploy. |
| `downgrade.py` | `mnemon downgrade local` — hosted → local. |
| `uninstall.py` | `mnemon uninstall`. |

---

## How to extend

Common extension points and where to start:

- **Add an MCP tool** → `server.py` (register with `@mcp.tool()`) + mirror the shape in `api.py` so local hooks can call it. Add a test.
- **Add a content type / change a half-life or scoring weight** → `config.py`.
- **Change retrieval behavior** (fusion, recency decay, MMR) → `search.py`.
- **Swap the embedding model** → `embedder.py` (then `mnemon rebuild` to re-embed).
- **Add a storage backend** → `store.py` is SQLite-specific; the `Store` interface is the seam.
- **Add an MCP client to auto-setup** → `setup.py`.

Hard rules for contributions: the full `pytest` suite must stay green (coverage gate ≥ 80%); storage stays **lossless** (only the model-facing copy is defanged); schema changes are **additive** (new nullable columns + migrations, never rename/drop); the fail-loud posture (raise, don't silently swallow) holds. See `CONTRIBUTING.md`.

---

## Request flow (local)

```
Claude Code prompt
  └─ hooks/context_surfacing.py (UserPromptSubmit)
       └─ api.py / server.py: memory_search
            └─ search.py  → store.py (BM25/FTS5) ⊕ vecstore.py (cosine)
                          → RRF fuse → composite score → MMR
            └─ safety.defang_control_markup (recall boundary)
       └─ inject <mnemon-context>

Claude Code Stop
  └─ hooks/session_extractor.py → memory_save
       └─ safety.contains_control_markup (capture reject)  → store.py
```
