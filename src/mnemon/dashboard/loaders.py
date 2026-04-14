"""Data access layer for the dashboard.

Post-0.5.0 the dashboard can point at either a **remote** vault (the
default on Fly, via the ``memory_*`` MCP tools over Streamable HTTP) or
a **local** vault (the SQLite file in ``~/.mnemon/``). Mode is detected
from ``MNEMON_REMOTE_URL`` env var or the ``~/.mnemon/remote_url`` file;
local mode is the fallback when neither is set.

Streamlit ``@st.cache_data`` memoizes results per page-render cycle. A
TTL of 5–15 min balances freshness (recent saves surface quickly) with
cost (remote MCP calls are ~1s cold, ~100ms warm).
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st


# ── Mode detection ──────────────────────────────────────────────────────────


def _use_remote() -> bool:
    """True if the dashboard should route through the remote MCP server."""
    if os.environ.get("MNEMON_REMOTE_URL", "").strip():
        return True
    remote_url_file = Path.home() / ".mnemon" / "remote_url"
    if remote_url_file.exists() and remote_url_file.read_text().strip():
        return True
    return False


def _call_remote(tool: str, args: dict) -> str:
    """Call an MCP tool via the shared remote client. Raises on error —
    callers decide whether to surface via ``st.error`` or silently fall
    back."""
    from mnemon.hooks._remote_client import call_tool_sync

    raw, _elapsed = call_tool_sync(
        tool,
        args,
        client_label="mnemon-dashboard",
    )
    return raw


# ── Local-mode Store singleton ──────────────────────────────────────────────


@st.cache_resource
def get_store():
    """Singleton read-only Store connection for local mode only.

    Cross-thread safe via ``check_same_thread=False`` — Streamlit runs
    pages in different threads than the cached Store. Not used when
    ``_use_remote()`` is True.
    """
    import sqlite3
    from mnemon.store import Store
    store = Store()
    store.db.close()
    store.db = sqlite3.connect(store.db_path, check_same_thread=False)
    store.db.row_factory = sqlite3.Row
    store.db.execute("PRAGMA journal_mode = WAL")
    store.db.execute("PRAGMA busy_timeout = 15000")
    return store


# ── Informational loaders (remote MCP + local fallback) ─────────────────────


@st.cache_data(ttl=300)
def load_status() -> dict:
    """Vault health stats."""
    if _use_remote():
        return json.loads(_call_remote("memory_status", {}))
    return get_store().status()


@st.cache_data(ttl=300)
def load_timeline(limit: int = 200, content_type: str | None = None) -> list[dict]:
    """Recent memories as a list of dicts."""
    if _use_remote():
        return json.loads(_call_remote(
            "memory_timeline",
            {"limit": limit, "content_type": content_type},
        ))
    docs = get_store().timeline(limit, content_type)
    return [dataclasses.asdict(d) for d in docs]


@st.cache_data(ttl=300)
def load_search(query: str, limit: int = 20, content_type: str | None = None) -> list[dict]:
    """Hybrid search results as a list of dicts."""
    if _use_remote():
        return json.loads(_call_remote(
            "memory_search",
            {"query": query, "limit": limit, "content_type": content_type},
        ))
    from mnemon.search import search
    results = search(
        get_store(), query,
        limit=limit, content_type=content_type, use_vector=True,
    )
    return [dataclasses.asdict(r) for r in results]


@st.cache_data(ttl=300)
def load_sweep() -> dict:
    """Stale-memory sweep candidates (dry-run)."""
    if _use_remote():
        return json.loads(_call_remote("memory_sweep", {"dry_run": True}))
    result = get_store().sweep(dry_run=True)
    result["candidates"] = [dataclasses.asdict(c) for c in result["candidates"]]
    return result


@st.cache_data(ttl=300)
def load_document(doc_id: int) -> dict | None:
    """Single document by ID, or None if not found."""
    if _use_remote():
        parsed = json.loads(_call_remote("memory_get", {"id": doc_id}))
        if parsed.get("error") == "not_found":
            return None
        return parsed
    doc = get_store().get(doc_id)
    return dataclasses.asdict(doc) if doc else None


@st.cache_data(ttl=300)
def load_related(doc_id: int, limit: int = 10) -> list[dict]:
    """Related documents for a given doc_id."""
    if _use_remote():
        return json.loads(_call_remote(
            "memory_related",
            {"id": doc_id, "limit": limit},
        ))
    rels = get_store().get_related(doc_id, limit)
    return [dataclasses.asdict(r) for r in rels]


# ── Graph page: vectors + doc metadata ──────────────────────────────────────


@st.cache_data(ttl=900)
def load_vectors() -> tuple[list[str], np.ndarray, dict[str, dict]] | None:
    """Return (vec_ids, vectors, vec_id_to_doc_map) or None when empty.

    In **remote** mode, all three come from ``memory_export_vectors`` in
    a single call — the server does the vec_id → doc join.

    In **local** mode, vectors come from the ``.npz`` file and the
    doc-map is built via a SQLite query keyed on content hash.

    Returns None when the vault has no vectors yet (fresh install or
    before first embedding).
    """
    if _use_remote():
        payload = json.loads(_call_remote("memory_export_vectors", {}))
        items = payload.get("items", [])
        if not items:
            return None
        vec_ids = [it["vec_id"] for it in items]
        vectors = np.array([it["vector"] for it in items], dtype=np.float32)
        doc_map = {
            it["vec_id"]: {
                "id": it["doc_id"],
                "title": it["title"],
                "content_type": it["content_type"],
                "confidence": it["confidence"],
                "created_at": it["created_at"],
            }
            for it in items
        }
        if payload.get("truncated"):
            st.warning(
                f"Vector export was truncated to {payload.get('count')} of "
                "your vault's vectors — increase the server-side cap or "
                "paginate if you need the full set."
            )
        return vec_ids, vectors, doc_map

    # Local mode
    from mnemon.config import vault_path
    vec_path = str(vault_path()).replace(".sqlite", ".vec.npz")
    if not Path(vec_path).exists():
        return None
    data = np.load(vec_path, allow_pickle=True)
    ids = data["ids"].tolist()
    vectors = data["vectors"].astype(np.float32)
    if len(ids) == 0:
        return None
    doc_map = _build_vector_doc_map_local(ids)
    return ids, vectors, doc_map


@st.cache_data(ttl=900)
def load_vectors_collapsed() -> tuple[list[str], np.ndarray, dict[str, dict]] | None:
    """Same shape as ``load_vectors``, but one row per document.

    Multi-chunk documents get mean-pooled (then L2-normalized so the
    result stays on the unit sphere for cosine UMAP). The synthesized
    vec_id is ``"doc_{doc_id}"`` so the Graph page's click-to-detail
    keeps working through the ``customdata.doc_id`` field.
    """
    raw = load_vectors()
    if raw is None:
        return None
    vec_ids, vectors, doc_map = raw

    by_doc: dict[int, list[int]] = {}
    for idx, vid in enumerate(vec_ids):
        doc_id = doc_map[vid]["id"]
        by_doc.setdefault(doc_id, []).append(idx)

    new_ids: list[str] = []
    pooled: list[np.ndarray] = []
    new_map: dict[str, dict] = {}
    for doc_id, idxs in by_doc.items():
        chunk_vecs = vectors[idxs]
        mean_vec = chunk_vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        synthetic_id = f"doc_{doc_id}"
        new_ids.append(synthetic_id)
        pooled.append(mean_vec)
        # Use any chunk's doc info — they all point to the same doc.
        new_map[synthetic_id] = doc_map[vec_ids[idxs[0]]]

    return new_ids, np.array(pooled, dtype=np.float32), new_map


@st.cache_data(ttl=900)
def load_umap_coords(_vectors: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """UMAP 384d → 2D. Cached aggressively since reprojecting is expensive."""
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(_vectors)


def _build_vector_doc_map_local(vec_ids: list[str]) -> dict[str, dict]:
    """Local-mode doc-map: open a fresh SQLite connection and resolve
    vec_ids → doc metadata via content_hash. Remote mode gets this
    baked into ``memory_export_vectors``."""
    import sqlite3
    from mnemon.config import vault_path

    hash_to_vec_ids: dict[str, list[str]] = {}
    for vid in vec_ids:
        content_hash = vid.rsplit("_", 1)[0]
        hash_to_vec_ids.setdefault(content_hash, []).append(vid)

    result: dict[str, dict] = {}
    db = sqlite3.connect(str(vault_path()), check_same_thread=False)
    db.row_factory = sqlite3.Row
    try:
        for content_hash, vids in hash_to_vec_ids.items():
            row = db.execute(
                "SELECT d.id, d.title, d.content_type, d.confidence, d.created_at "
                "FROM documents d WHERE d.hash = ? AND d.invalidated_at IS NULL LIMIT 1",
                (content_hash,),
            ).fetchone()
            if row:
                doc_info = dict(row)
                for vid in vids:
                    result[vid] = doc_info
    finally:
        db.close()
    return result
