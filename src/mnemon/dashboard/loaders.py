"""Data access layer for the dashboard — wraps Store with Streamlit caching."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import streamlit as st


@st.cache_resource
def get_store():
    """Singleton read-only Store connection."""
    from mnemon.store import Store
    return Store()


@st.cache_data(ttl=300)
def load_status() -> dict:
    """Vault health stats."""
    return get_store().status()


@st.cache_data(ttl=300)
def load_timeline(limit: int = 200, content_type: str | None = None) -> list[dict]:
    """Recent memories as list of dicts."""
    docs = get_store().timeline(limit, content_type)
    return [dataclasses.asdict(d) for d in docs]


@st.cache_data(ttl=300)
def load_search(query: str, limit: int = 20, content_type: str | None = None) -> list[dict]:
    """Hybrid search results as list of dicts."""
    from mnemon.search import search
    results = search(get_store(), query, limit=limit, content_type=content_type, use_vector=True)
    return [dataclasses.asdict(r) for r in results]


@st.cache_data(ttl=300)
def load_sweep() -> dict:
    """Stale memory candidates (dry_run=True)."""
    result = get_store().sweep(dry_run=True)
    result["candidates"] = [dataclasses.asdict(c) for c in result["candidates"]]
    return result


@st.cache_data(ttl=300)
def load_document(doc_id: int) -> dict | None:
    """Single document by ID."""
    doc = get_store().get(doc_id)
    return dataclasses.asdict(doc) if doc else None


@st.cache_data(ttl=300)
def load_related(doc_id: int, limit: int = 10) -> list[dict]:
    """Related documents for a given doc_id."""
    rels = get_store().get_related(doc_id, limit)
    return [dataclasses.asdict(r) for r in rels]


@st.cache_data(ttl=900)
def load_vectors() -> tuple[list, np.ndarray] | None:
    """Raw vectors from .npz file. Returns (ids, vectors) or None."""
    from mnemon.config import vault_path
    vec_path = str(vault_path()).replace(".sqlite", ".vec.npz")
    if not Path(vec_path).exists():
        return None
    data = np.load(vec_path, allow_pickle=True)
    ids = data["ids"].tolist()
    vectors = data["vectors"].astype(np.float32)
    if len(ids) == 0:
        return None
    return ids, vectors


@st.cache_data(ttl=900)
def load_umap_coords(_vectors: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """UMAP 384d -> 2D. Cached aggressively."""
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(_vectors)


def build_vector_doc_map(vec_ids: list[str]) -> dict[str, dict]:
    """Map vector IDs to document metadata via content_hash.

    Opens a separate SQLite connection to avoid cross-thread errors
    (Streamlit runs pages in different threads than the cached Store).
    """
    import sqlite3
    from mnemon.config import vault_path

    hash_to_vec_ids: dict[str, list[str]] = {}
    for vid in vec_ids:
        content_hash = vid.rsplit("_", 1)[0]
        hash_to_vec_ids.setdefault(content_hash, []).append(vid)

    result = {}
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
