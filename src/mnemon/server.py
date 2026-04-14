"""MCP server — exposes mnemon memory tools via stdio transport.

Tools: memory_search, memory_get, memory_save, memory_pin, memory_forget,
       memory_status, memory_sweep, memory_timeline, memory_related, memory_rebuild,
       memory_check_contradictions, profile_get, profile_update
"""

from __future__ import annotations

import json
import logging
import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .search import search
from .store import Store

logger = logging.getLogger(__name__)


def _build_transport_security() -> TransportSecuritySettings | None:
    """Build TransportSecuritySettings from the MNEMON_ALLOWED_HOSTS env var.

    FastMCP enables DNS rebinding protection by default with an empty
    allowed_hosts list, which rejects every non-localhost request. When
    running behind a reverse proxy or cloud host (Fly, Render, etc.), set
    MNEMON_ALLOWED_HOSTS to a comma-separated list of allowed Host header
    values, e.g.::

        MNEMON_ALLOWED_HOSTS=mnemon-memory.fly.dev,*.fly.dev

    Wildcards use fnmatch-style patterns (``*`` matches any characters).
    Returns None if the env var is unset, preserving FastMCP's default
    localhost-only behavior for local development.
    """
    raw = os.environ.get("MNEMON_ALLOWED_HOSTS", "").strip()
    if not raw:
        return None
    hosts = [h.strip() for h in raw.split(",") if h.strip()]
    # Origins default to https:// versions of each host (claude.ai connectors
    # use HTTPS exclusively).
    origins = [f"https://{h}" for h in hosts]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


mcp = FastMCP("mnemon", transport_security=_build_transport_security())

# Lazy-initialized store (created on first tool call)
_store: Store | None = None


def _get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


# ── Retrieval Tools ──────────────────────────────────────────────────────────


@mcp.tool()
def memory_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
) -> str:
    """Search memories using hybrid BM25 + vector search with composite scoring.

    This is the primary entry point for finding relevant memories.
    Results are ranked by a composite of relevance, recency, and confidence.
    """
    store = _get_store()
    results = search(store, query, limit=limit, content_type=content_type)

    if not results:
        return "No memories found matching your query."

    lines = []
    for i, r in enumerate(results, 1):
        snippet = r.content[:300]
        ellipsis = "..." if len(r.content) > 300 else ""
        lines.append(
            f"{i}. [{r.content_type}] **{r.title}** "
            f"(score: {r.composite_score:.3f}, confidence: {r.confidence:.2f})\n"
            f"   {snippet}{ellipsis}\n"
            f"   _id: {r.doc_id} | created: {r.created_at}_"
        )
    return "\n\n".join(lines)


@mcp.tool()
def memory_search_structured(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
) -> str:
    """Search memories and return results as a JSON array.

    Machine-readable counterpart to ``memory_search``. Use this when a
    client needs to act on score thresholds or individual fields rather
    than present the results to a human. Returns a JSON string with one
    object per result; empty array when nothing matches.

    Each result object contains: doc_id, title, content, content_type,
    confidence, composite_score, vector_similarity, created_at. Score
    fields are floats — callers can compare against thresholds without
    text parsing. ``vector_similarity`` is the raw cosine similarity
    (0.0–1.0) from the vector store, preserved before RRF fusion; it
    is None for BM25-only matches and is the right signal for dedup
    (composite_score is a weighted rank score, not a similarity).
    """
    store = _get_store()
    results = search(store, query, limit=limit, content_type=content_type)
    payload = [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "content": r.content,
            "content_type": r.content_type,
            "confidence": r.confidence,
            "composite_score": r.composite_score,
            "vector_similarity": r.vector_similarity,
            "created_at": r.created_at,
        }
        for r in results
    ]
    return json.dumps(payload)


@mcp.tool()
def memory_get(id: int) -> str:
    """Get a specific memory by its ID. Returns the full content."""
    store = _get_store()
    doc = store.get(id)
    if not doc:
        return f"Memory #{id} not found."

    return (
        f"# {doc.title}\n\n"
        f"**Type:** {doc.content_type} | **Confidence:** {doc.confidence:.2f} | "
        f"**Created:** {doc.created_at}\n\n"
        f"{doc.content}"
    )


@mcp.tool()
def memory_timeline(
    limit: int = 20,
    content_type: str | None = None,
) -> str:
    """Get recent memories in reverse chronological order."""
    store = _get_store()
    docs = store.timeline(limit, content_type)
    if not docs:
        return "No memories found."

    lines = [
        f"- **{d.title}** [{d.content_type}] (id: {d.id}, {d.created_at})"
        for d in docs
    ]
    return "\n".join(lines)


@mcp.tool()
def memory_timeline_structured(
    limit: int = 20,
    content_type: str | None = None,
) -> str:
    """Recent memories as a JSON list of document objects.

    Machine-readable counterpart to ``memory_timeline``. Used by the
    mnemon dashboard's Timeline page. Each object carries the full
    document shape (id, title, content, content_type, memory_type,
    confidence, pinned, created_at, updated_at, access_count).
    """
    import dataclasses
    store = _get_store()
    docs = store.timeline(limit, content_type)
    return json.dumps([dataclasses.asdict(d) for d in docs])


# ── Mutation Tools ───────────────────────────────────────────────────────────


@mcp.tool()
def memory_save(
    title: str,
    content: str,
    content_type: str = "note",
    collection: str = "default",
    source_client: str | None = None,
) -> str:
    """Save a new memory.

    Use this to explicitly store important information — decisions,
    preferences, observations, project context, or session handoffs.
    """
    store = _get_store()
    doc_id = store.save(
        title=title,
        content=content,
        content_type=content_type,
        collection=collection,
        source_client=source_client,
    )

    # Embed asynchronously (non-blocking, failures are non-fatal — the
    # memory is in SQLite either way; only semantic search is affected).
    try:
        from .embedder import embed_document
        doc = store.get(doc_id)
        if doc:
            embed_document(store, doc.hash, title, content)
    except Exception as exc:
        logger.warning(
            "memory_save: embedding failed for doc_id=%d (%s: %s); "
            "memory is saved but won't surface in vector search until "
            "`mnemon rebuild` runs",
            doc_id, type(exc).__name__, exc,
        )

    return f'Saved memory #{doc_id}: "{title}" [{content_type}]'


@mcp.tool()
def memory_pin(id: int) -> str:
    """Pin an important memory to boost its confidence and prevent archival."""
    store = _get_store()
    success = store.pin(id)
    return f"Pinned memory #{id}." if success else f"Memory #{id} not found."


@mcp.tool()
def memory_forget(id: int) -> str:
    """Soft-delete a memory. Marked as invalidated but not physically removed."""
    store = _get_store()
    success = store.forget(id)
    return (
        f"Forgot memory #{id}."
        if success
        else f"Memory #{id} not found or already forgotten."
    )


# ── Lifecycle Tools ──────────────────────────────────────────────────────────


@mcp.tool()
def memory_status() -> str:
    """Get vault health stats — document counts by type, pinned/invalidated counts."""
    store = _get_store()
    stats = store.status()

    by_type = "\n".join(
        f"  {t['content_type']}: {t['count']}" for t in stats["by_type"]
    )

    return (
        f"Vault: {stats['vault_path']}\n"
        f"Total memories: {stats['total_documents']}\n"
        f"Vectors: {stats['total_vectors']}\n"
        f"Pinned: {stats['pinned']}\n"
        f"Invalidated: {stats['invalidated']}\n\n"
        f"By type:\n{by_type}"
    )


@mcp.tool()
def memory_status_structured() -> str:
    """Vault health stats as a JSON object.

    Machine-readable counterpart to ``memory_status``. Used by the
    mnemon dashboard's Home page to render counts without parsing prose.

    Returns a JSON object: ``{total_documents, total_vectors, invalidated,
    pinned, by_type, vault_path}`` where ``by_type`` is a list of
    ``{content_type, count}``.
    """
    store = _get_store()
    return json.dumps(store.status())


@mcp.tool()
def memory_sweep(dry_run: bool = True) -> str:
    """Archive stale memories that have exceeded their half-life.

    Runs in dry-run mode by default — pass dry_run=False to actually archive.
    """
    store = _get_store()
    result = store.sweep(dry_run)

    if not result["candidates"]:
        return "No stale memories to archive."

    lines = [
        f'- #{c.id} "{c.title}" [{c.content_type}] — {c.age_days} days old'
        for c in result["candidates"]
    ]

    action = "Would archive" if dry_run else "Archived"
    return f"{action} {len(result['candidates'])} memories:\n" + "\n".join(lines)


@mcp.tool()
def memory_sweep_structured(dry_run: bool = True) -> str:
    """Sweep results as a JSON object.

    Machine-readable counterpart to ``memory_sweep``. Used by the
    mnemon dashboard to render stale-candidate tables. Returns
    ``{archived: int, candidates: [{id, title, content_type, age_days}]}``.
    """
    import dataclasses
    store = _get_store()
    result = store.sweep(dry_run)
    result["candidates"] = [dataclasses.asdict(c) for c in result["candidates"]]
    return json.dumps(result)


@mcp.tool()
def memory_related(id: int, limit: int = 10) -> str:
    """Find memories related to a given memory via the relationship graph."""
    store = _get_store()
    related = store.get_related(id, limit)
    if not related:
        return f"No related memories found for #{id}."

    lines = [
        f"- [{r.relation_type}] **{r.title}** (id: {r.id}, weight: {r.weight:.2f})"
        for r in related
    ]
    return "\n".join(lines)


@mcp.tool()
def memory_related_structured(id: int, limit: int = 10) -> str:
    """Related memories as a JSON list.

    Machine-readable counterpart to ``memory_related``. Each entry is
    the full document shape extended with ``relation_type`` and
    ``weight`` fields.
    """
    import dataclasses
    store = _get_store()
    return json.dumps([dataclasses.asdict(r) for r in store.get_related(id, limit)])


# Vector-export cap: 5000 vectors × 384 floats ≈ 7-10 MB JSON at the
# float representation we emit. Enough for any personal vault; explicit
# cap so a runaway vault size doesn't OOM the server process silently.
_VECTOR_EXPORT_MAX = 5000


@mcp.tool()
def memory_export_vectors() -> str:
    """Export all stored embedding vectors joined to document metadata.

    Used by the mnemon dashboard's Graph page to pull the full embedding
    matrix over MCP and run UMAP projection client-side. Avoids exposing
    a filesystem path or a bulk-SQL interface while keeping the dashboard
    remote-aware.

    Returns a JSON object ``{count, dim, truncated, items}``:

    - ``count``: number of vectors returned (may be ≤ stored count if
      capped).
    - ``dim``: vector dimensionality (384 for bge-small-en-v1.5).
    - ``truncated``: True if the vault exceeds the server's export cap
      and results were truncated.
    - ``items``: list of ``{doc_id, vec_id, title, content_type,
      confidence, created_at, pinned, vector}``. ``vector`` is a list
      of floats of length ``dim``. Invalidated docs are excluded.

    Cap is ``_VECTOR_EXPORT_MAX`` (5000) vectors. Larger vaults need a
    paginated API — file an issue if you hit this.
    """
    store = _get_store()
    vec_ids, vectors = store.vec_store.export_all()

    if not vec_ids:
        return json.dumps({"count": 0, "dim": store.vec_store.dim,
                           "truncated": False, "items": []})

    truncated = len(vec_ids) > _VECTOR_EXPORT_MAX
    if truncated:
        vec_ids = vec_ids[:_VECTOR_EXPORT_MAX]
        vectors = vectors[:_VECTOR_EXPORT_MAX]

    # vec_id format is "{content_hash}_{seq}" — split once from the right
    # since content_hash is a hex SHA-256 (no underscores).
    hashes = [vid.rsplit("_", 1)[0] for vid in vec_ids]
    unique_hashes = list(dict.fromkeys(hashes))

    # One query for all hashes — cheaper than N queries. Take the first
    # non-invalidated doc per hash (there may be multiple rows sharing a
    # content hash after supersedes/updates).
    placeholders = ",".join("?" * len(unique_hashes))
    rows = store.db.execute(
        f"""SELECT hash, id, title, content_type, confidence, created_at, pinned
            FROM documents
            WHERE hash IN ({placeholders})
              AND invalidated_at IS NULL""",
        unique_hashes,
    ).fetchall()
    hash_to_doc = {r["hash"]: dict(r) for r in rows}

    items = []
    for vec_id, content_hash, vector in zip(vec_ids, hashes, vectors):
        doc = hash_to_doc.get(content_hash)
        if doc is None:
            # Vector exists but its source document is gone (invalidated
            # or deleted). Skip — dashboard can't render it usefully.
            continue
        items.append({
            "doc_id": doc["id"],
            "vec_id": vec_id,
            "title": doc["title"],
            "content_type": doc["content_type"],
            "confidence": doc["confidence"],
            "created_at": doc["created_at"],
            "pinned": bool(doc["pinned"]),
            "vector": vector.tolist(),
        })

    return json.dumps({
        "count": len(items),
        "dim": store.vec_store.dim,
        "truncated": truncated,
        "items": items,
    })


@mcp.tool()
def memory_rebuild() -> str:
    """Re-embed all documents. Use after upgrading the embedding model."""
    store = _get_store()
    docs = store.timeline(1000)
    embedded = 0
    failed = 0

    try:
        from .embedder import embed_document
    except ImportError:
        return "FastEmbed not installed. Run: pip install fastembed"

    for doc in docs:
        try:
            embed_document(store, doc.hash, doc.title, doc.content)
            embedded += 1
        except Exception:
            failed += 1

    return f"Rebuild complete: {embedded} documents embedded, {failed} failed."


# ── Profile Tools ───────────────────────────────────────────────────────────


@mcp.tool()
def profile_get() -> str:
    """Get a synthesized user profile from stored preferences and decisions.

    Shows what mnemon knows about the user's habits, preferences, and key decisions.
    """
    store = _get_store()
    preferences = store.timeline(50, "preference")
    decisions = store.timeline(50, "decision")

    if not preferences and not decisions:
        return (
            "No profile data yet. Preferences and decisions will be "
            "collected automatically over time."
        )

    sections: list[str] = []

    if preferences:
        lines = [f"- **{d.title}**: {d.content[:200]}" for d in preferences]
        sections.append("## Preferences\n" + "\n".join(lines))

    if decisions:
        lines = [f"- **{d.title}**: {d.content[:200]}" for d in decisions]
        sections.append("## Key Decisions\n" + "\n".join(lines))

    return "\n\n".join(sections)


@mcp.tool()
def profile_update(title: str, content: str) -> str:
    """Manually add a fact to the user profile. Saved as a preference memory."""
    store = _get_store()
    doc_id = store.save(
        title=title,
        content=content,
        content_type="preference",
        source_client="mcp-profile",
    )

    try:
        from .embedder import embed_document
        doc = store.get(doc_id)
        if doc:
            embed_document(store, doc.hash, title, content)
    except Exception as exc:
        logger.warning(
            "profile_update: embedding failed for doc_id=%d (%s: %s); "
            "preference is saved but won't surface in vector search until "
            "`mnemon rebuild` runs",
            doc_id, type(exc).__name__, exc,
        )

    return f'Profile updated — saved preference #{doc_id}: "{title}"'


# ── Contradiction Check Tool ────────────────────────────────────────────────


@mcp.tool()
def memory_check_contradictions(id: int) -> str:
    """Check a memory for contradictions against existing memories.

    Uses vector similarity + LLM classification to find conflicts.
    Automatically decays confidence of superseded or contradicting memories.
    """
    store = _get_store()
    doc = store.get(id)
    if not doc:
        return f"Memory #{id} not found."

    from .contradiction import check_contradictions
    result = check_contradictions(store, doc.title, doc.content, id)

    if not result["relationships"]:
        return f"No contradictions found for memory #{id}."

    lines = [
        f'- #{r["doc_id"]} "{r["title"]}" → **{r["relationship"]}**'
        for r in result["relationships"]
    ]

    return (
        f'Contradiction check for #{id} "{doc.title}":\n'
        + "\n".join(lines)
        + f'\n\n{result["decayed"]} memories had their confidence decayed.'
    )


def run_stdio() -> None:
    """Start the MCP server on stdio transport."""
    mcp.run(transport="stdio")
