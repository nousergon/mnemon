"""In-process mnemon API — the same tool surface as :mod:`server`, callable
directly without an MCP transport.

Purpose
-------
P1a of the mnemon simplification plan (``private/mnemon-simplification-plan-260421.md``).
Hooks, ``mnemon doctor``, and the setup preflight previously required an
HTTP endpoint even for single-machine use — the hook code path in
``hooks/_remote_client.py`` is HTTP-only and has no local fallback.
Installing those hooks without a remote vault produced a
``RemoteClientConfigError`` banner on every Claude Code prompt.

This module gives the hook client (``hooks/_client.py::LocalMemoryClient``)
a target to dispatch against when the user is in local-only mode. Every
function here returns the **same JSON/string shape** that the matching
``@mcp.tool()`` in :mod:`server` returns, so hooks written against a
shared ``MemoryClient`` protocol work in either mode without branching.

Design notes
------------
- Responses are ``str``. JSON-returning tools are ``json.dumps(...)``;
  plain tools return formatted strings. Matches the MCP tool contract.
- Functions accept a ``store`` keyword for testability. When omitted, a
  default :class:`Store` is created once per process and reused.
- We intentionally duplicate the response shapes from :mod:`server`
  rather than refactoring server tools to delegate here, because this PR
  is already large and the MCP-facing code path is load-bearing for
  remote users. A later PR can consolidate.
- Embedding on save mirrors the server behavior — best-effort, warns on
  failure, never raises.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

from .store import Store

logger = logging.getLogger(__name__)


_default_store: Store | None = None


def _get_default_store() -> Store:
    """Return a process-wide default Store, creating it lazily."""
    global _default_store
    if _default_store is None:
        _default_store = Store()
    return _default_store


def _resolve_store(store: Store | None) -> Store:
    return store if store is not None else _get_default_store()


def memory_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    *,
    store: Store | None = None,
) -> str:
    """Hybrid BM25 + vector search — see :func:`server.memory_search`."""
    from .search import search as _search

    store = _resolve_store(store)
    results = _search(store, query, limit=limit, content_type=content_type)
    return json.dumps(
        [
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
    )


def memory_get(id: int, *, store: Store | None = None) -> str:
    """Fetch one memory — see :func:`server.memory_get`."""
    store = _resolve_store(store)
    doc = store.get(id)
    if not doc:
        return json.dumps({"error": "not_found", "id": id})
    return json.dumps(dataclasses.asdict(doc))


def memory_timeline(
    limit: int = 20,
    content_type: str | None = None,
    *,
    store: Store | None = None,
) -> str:
    """Recent memories — see :func:`server.memory_timeline`."""
    store = _resolve_store(store)
    return json.dumps(
        [dataclasses.asdict(d) for d in store.timeline(limit, content_type)]
    )


def memory_save(
    title: str,
    content: str,
    content_type: str = "note",
    collection: str = "default",
    source_client: str | None = None,
    *,
    store: Store | None = None,
) -> str:
    """Save a memory — see :func:`server.memory_save`.

    Matches the server behavior: embedding is best-effort, never raises.
    Returns the same ``'Saved memory #{doc_id}: "{title}" [{content_type}]'``
    shape so callers can parse it identically to the MCP path.
    """
    store = _resolve_store(store)
    doc_id = store.save(
        title=title,
        content=content,
        content_type=content_type,
        collection=collection,
        source_client=source_client,
    )
    try:
        from .embedder import embed_document

        doc = store.get(doc_id)
        if doc:
            embed_document(store, doc.hash, title, content)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "memory_save: embedding failed for doc_id=%d (%s: %s); "
            "memory is saved but won't surface in vector search until "
            "`mnemon rebuild` runs",
            doc_id,
            type(exc).__name__,
            exc,
        )
    return f'Saved memory #{doc_id}: "{title}" [{content_type}]'


def memory_forget(id: int, *, store: Store | None = None) -> str:
    """Soft-delete — see :func:`server.memory_forget`."""
    store = _resolve_store(store)
    ok = store.forget(id)
    return (
        f"Forgot memory #{id}."
        if ok
        else f"Memory #{id} not found or already forgotten."
    )


def memory_pin(id: int, *, store: Store | None = None) -> str:
    """Pin — see :func:`server.memory_pin`."""
    store = _resolve_store(store)
    ok = store.pin(id)
    return f"Pinned memory #{id}." if ok else f"Memory #{id} not found."


def memory_status(*, store: Store | None = None) -> str:
    """Vault health stats — see :func:`server.memory_status`.

    Returns the raw ``store.status()`` dict as JSON. Local mode, so
    ``vault_path`` is the on-disk SQLite path.
    """
    store = _resolve_store(store)
    return json.dumps(store.status())


# Dispatch table — maps MCP tool names to the in-process implementations
# above. Keeps the client's ``call_tool(name, args)`` one-liner-simple.
# Every handler accepts ``**kwargs`` so callers don't have to pre-filter
# argument dicts down to exactly the supported keyword names.
_HANDLERS = {
    "memory_search": memory_search,
    "memory_get": memory_get,
    "memory_timeline": memory_timeline,
    "memory_save": memory_save,
    "memory_forget": memory_forget,
    "memory_pin": memory_pin,
    "memory_status": memory_status,
}


class UnsupportedToolError(ValueError):
    """Raised when an MCP tool name has no in-process implementation.

    Hooks and doctor only use a subset of the full MCP surface; tools
    that don't appear in :data:`_HANDLERS` fall into this error so we
    never silently succeed on a typo or an unexpected caller.
    """


def dispatch(name: str, arguments: dict[str, Any], *, store: Store | None = None) -> str:
    """Route an MCP tool-name + args to the matching in-process handler.

    Parameters
    ----------
    name:
        The tool name as it would appear on the MCP wire
        (``memory_search``, ``memory_save``, etc.).
    arguments:
        The tool arguments. Only keys matching the handler's signature
        are forwarded; extras are ignored to match FastMCP's looser
        argument handling.
    store:
        Optional Store injection for tests.

    Returns:
        The handler's return value (always ``str`` to match MCP).

    Raises:
        UnsupportedToolError: if ``name`` is not in :data:`_HANDLERS`.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        raise UnsupportedToolError(
            f"Tool {name!r} has no in-process implementation. "
            f"Supported: {sorted(_HANDLERS)}"
        )
    # Pass only arguments the handler accepts. This mirrors MCP's behavior
    # of ignoring unknown keys rather than erroring on them.
    import inspect

    sig = inspect.signature(handler)
    accepted = {
        k: v for k, v in (arguments or {}).items() if k in sig.parameters
    }
    return handler(**accepted, store=store)
