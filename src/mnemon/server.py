"""MCP server — exposes mnemon memory tools via stdio transport.

Tools: memory_search, memory_get, memory_save, memory_pin, memory_forget,
       memory_promote, memory_demote, memory_list_standing,
       memory_status, memory_sweep, memory_timeline, memory_related, memory_rebuild,
       memory_check_contradictions, profile_get, profile_update
"""

from __future__ import annotations

import json
import logging
import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .safety import defang_doc
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

    Primary entry point for finding relevant memories. Results are ranked
    by a composite of relevance, recency, and confidence.

    Returns a JSON string — a list of result objects, empty when nothing
    matches. Each object contains: ``doc_id``, ``title``, ``content``,
    ``content_type``, ``confidence``, ``composite_score``,
    ``vector_similarity``, ``created_at``. Score fields are floats.
    ``vector_similarity`` is the raw cosine similarity (0.0–1.0) from the
    vector store, preserved before RRF fusion; it is None for BM25-only
    matches and is the right signal for dedup (``composite_score`` is a
    weighted rank score, not a raw similarity).
    """
    store = _get_store()
    results = search(store, query, limit=limit, content_type=content_type)
    return json.dumps([
        defang_doc({
            "doc_id": r.doc_id,
            "title": r.title,
            "content": r.content,
            "content_type": r.content_type,
            "confidence": r.confidence,
            "composite_score": r.composite_score,
            "vector_similarity": r.vector_similarity,
            "created_at": r.created_at,
        })
        for r in results
    ])


@mcp.tool()
def memory_get(id: int) -> str:
    """Get a specific memory by its ID.

    Returns a JSON string. On hit: the full document fields (``id``,
    ``title``, ``content``, ``content_type``, ``confidence``,
    ``created_at``, ``updated_at``, ``pinned``, ``access_count``, etc.).
    On miss: ``{"error": "not_found", "id": <id>}``.
    """
    import dataclasses
    store = _get_store()
    doc = store.get(id)
    if not doc:
        return json.dumps({"error": "not_found", "id": id})
    return json.dumps(defang_doc(dataclasses.asdict(doc)))


@mcp.tool()
def memory_timeline(
    limit: int = 20,
    content_type: str | None = None,
) -> str:
    """Recent memories in reverse chronological order as a JSON list.

    Each item carries the full document shape (``id``, ``title``,
    ``content``, ``content_type``, ``confidence``, ``pinned``,
    ``created_at``, ``updated_at``, ``access_count``). Empty list when
    no matches.
    """
    import dataclasses
    store = _get_store()
    return json.dumps([
        defang_doc(dataclasses.asdict(d))
        for d in store.timeline(limit, content_type)
    ])


# ── Mutation Tools ───────────────────────────────────────────────────────────


@mcp.tool()
def memory_save(
    title: str,
    content: str,
    content_type: str = "note",
    collection: str = "default",
    source_client: str | None = None,
    source_key: str | None = None,
) -> str:
    """Save a new memory.

    Use this to explicitly store important information — decisions,
    preferences, observations, project context, or session handoffs.

    ``source_key`` is an optional stable caller-owned identity. When
    set, re-saving the same key updates in place (invalidate-prior +
    insert) instead of accumulating near-duplicates — used by the
    auto-mirror path, keyed to the local memory file's slug, so a
    memory edited several times in one session stays a single document.
    """
    store = _get_store()
    doc_id = store.save(
        title=title,
        content=content,
        content_type=content_type,
        collection=collection,
        source_client=source_client,
        source_key=source_key,
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


@mcp.tool()
def memory_promote(id: int) -> str:
    """Promote a memory to the capped standing tier.

    Standing-tier memories are injected into every recall context
    regardless of query similarity — they condition reasoning rather
    than answering it. Use sparingly: the cap is the contract
    (default 15, hard ceiling 20). Past ~20, the tier stops being
    salient and becomes noise again.

    Hook-sourced memories (auto-mirror, session_extractor) cannot be
    promoted — operator-explicit gesture only (Layer 4 composition).
    """
    from .store import (
        StandingTierCapReached,
        StandingTierError,
        StandingTierProvenanceRejected,
    )
    store = _get_store()
    try:
        ok = store.promote_to_standing(id)
        status = store.standing_tier_status()
        return (
            f"Promoted memory #{id} to standing tier "
            f"({status['count']}/{status['cap']})."
            if ok else f"Memory #{id} could not be promoted."
        )
    except StandingTierCapReached as e:
        return f"Cap reached: {e}"
    except StandingTierProvenanceRejected as e:
        return f"Provenance rejected: {e}"
    except StandingTierError as e:
        return f"Error: {e}"


@mcp.tool()
def memory_demote(id: int) -> str:
    """Demote a standing-tier memory back to situational.

    The opposite of memory_promote. Idempotent: demoting a memory
    that isn't on the standing tier is a no-op (returns "not on
    standing tier" rather than failing).
    """
    from .store import StandingTierError
    store = _get_store()
    try:
        actually_demoted = store.demote_to_situational(id)
        status = store.standing_tier_status()
        if actually_demoted:
            return (
                f"Demoted memory #{id} to situational "
                f"({status['count']}/{status['cap']} remain standing)."
            )
        return f"Memory #{id} was not on the standing tier."
    except StandingTierError as e:
        return f"Error: {e}"


@mcp.tool()
def memory_list_standing() -> str:
    """Return all live standing-tier memories as a JSON array.

    Consumed by ``hooks/context_surfacing.py`` to render the always-on
    standing block when ``STANDING_TIER_ENABLED`` is True. The same
    list is shown by ``mnemon standing list`` on the CLI.

    Each element: ``{doc_id, title, content, content_type, confidence,
    created_at}``. Empty array when nothing has been promoted.
    """
    store = _get_store()
    docs = store.list_standing()
    return json.dumps([
        {
            "doc_id": d.id,
            "title": d.title,
            "content": d.content,
            "content_type": d.content_type,
            "confidence": d.confidence,
            "created_at": d.created_at,
        }
        for d in docs
    ])


# ── Lifecycle Tools ──────────────────────────────────────────────────────────


@mcp.tool()
def memory_status() -> str:
    """Vault health stats as a JSON object.

    Returns: ``{total_documents, total_vectors, invalidated, pinned,
    by_type, vault_path}``. ``by_type`` is a list of
    ``{content_type, count}`` ordered by count descending.
    """
    store = _get_store()
    return json.dumps(store.status())


@mcp.tool()
def memory_sweep(dry_run: bool = True) -> str:
    """Archive stale memories that have exceeded their half-life.

    Runs in dry-run mode by default — pass ``dry_run=False`` to actually
    archive. Returns JSON: ``{archived: int, candidates: [{id, title,
    content_type, age_days}]}``. ``archived`` is 0 on dry-run.
    """
    import dataclasses
    store = _get_store()
    result = store.sweep(dry_run)
    result["candidates"] = [
        defang_doc(dataclasses.asdict(c)) for c in result["candidates"]
    ]
    return json.dumps(result)


@mcp.tool()
def memory_related(id: int, limit: int = 10) -> str:
    """Related memories via the relationship graph, as a JSON list.

    Each entry is the full document shape plus ``relation_type`` and
    ``weight``. Empty list when nothing is related.
    """
    import dataclasses
    store = _get_store()
    return json.dumps([
        defang_doc(dataclasses.asdict(r)) for r in store.get_related(id, limit)
    ])


# Vector-export cap: 5000 vectors × 384 floats ≈ 7-10 MB JSON at the
# float representation we emit. Enough for any personal vault; explicit
# cap so a runaway vault size doesn't OOM the server process silently.
_VECTOR_EXPORT_MAX = 5000


@mcp.tool()
def memory_export_vectors() -> str:
    """Export all stored embedding vectors joined to document metadata.

    Used by the mnemon dashboard's Graph page to pull the full embedding
    matrix over MCP and run UMAP projection client-side. Avoids exposing
    a filesystem path or bulk-SQL interface while keeping the dashboard
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

    Cap is ``_VECTOR_EXPORT_MAX`` (5000) vectors.
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

    # One query for all hashes; take the first non-invalidated doc per hash.
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
    """Synthesized user profile from stored preferences and decisions.

    Returns JSON: ``{preferences: [doc_dict, ...], decisions: [doc_dict, ...]}``.
    Each list holds the most recent 50 items of its content type, with
    the full document shape per entry. Both lists may be empty.
    """
    import dataclasses
    store = _get_store()
    return json.dumps({
        "preferences": [
            dataclasses.asdict(d) for d in store.timeline(50, "preference")
        ],
        "decisions": [
            dataclasses.asdict(d) for d in store.timeline(50, "decision")
        ],
    })


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
