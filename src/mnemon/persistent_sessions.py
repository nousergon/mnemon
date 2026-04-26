"""Persistent MCP Streamable HTTP session storage.

When a Fly.io machine running mnemon auto-stops on idle, the in-memory
session dict in ``StreamableHTTPSessionManager._server_instances`` is
lost. After the machine wakes, any client request bearing a previously-
issued ``Mcp-Session-Id`` header gets a 404 (per the MCP spec — but the
spec also says clients SHOULD reinit on 404, which Claude Code's MCP
client does not reliably do, so the symptom is "MCP UI says connected
but every tool call fails").

This module sidesteps that by persisting issued session IDs to SQLite,
so a known-but-not-in-memory session can be transparently resumed on
the new process. The resumed session is born with ``stateless=True``,
which makes the underlying ``ServerSession`` skip the init handshake
and accept tool calls immediately. mnemon does not use any client-
capability-gated features (sampling, elicitation, roots), so the
absence of negotiated client capabilities on a resumed session is a
no-op for our toolset.

The complement on the client side is the ``/health`` warm-keeper hook
installed by ``mnemon setup`` (see ``setup.py:_hooks_config``). The
hook keeps Fly warm during active sessions; this module catches the
edge case where the machine *did* cold-stop anyway (long idle, manual
stop, deploy).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import TaskStatus
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.streamable_http import (
    MCP_SESSION_ID_HEADER,
    EventStore,
    StreamableHTTPServerTransport,
)
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 7 * 24 * 3600


class SessionStore:
    """SQLite-backed registry of issued MCP Streamable HTTP session IDs.

    Stores ``(session_id, created_at, last_active_at)`` for every session
    the manager hands out. Entries older than ``ttl_seconds`` are
    considered expired and treated as unknown.

    The store deliberately holds *no* per-session capability or
    handshake state — resumed sessions are born stateless on the server
    side, so there is nothing to persist beyond the ID itself.
    """

    def __init__(
        self,
        db_path: Path | str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None puts SQLite in autocommit; each statement
        # commits on its own. Suits the single-statement ops below.
        return sqlite3.connect(str(self.db_path), isolation_level=None)

    def _init_schema(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_transport_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_active_at REAL NOT NULL
                )
                """
            )

    def register(self, session_id: str) -> None:
        """Persist a newly-issued session ID (idempotent)."""
        now = time.time()
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO mcp_transport_sessions"
                "(session_id, created_at, last_active_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )

    def is_known(self, session_id: str) -> bool:
        """True if the session ID was issued and is not expired."""
        cutoff = time.time() - self.ttl_seconds
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM mcp_transport_sessions "
                "WHERE session_id = ? AND last_active_at > ?",
                (session_id, cutoff),
            ).fetchone()
        return row is not None

    def touch(self, session_id: str) -> None:
        """Update last-active timestamp on activity."""
        with self._connect() as db:
            db.execute(
                "UPDATE mcp_transport_sessions SET last_active_at = ? "
                "WHERE session_id = ?",
                (time.time(), session_id),
            )

    def expire_old(self) -> int:
        """Delete entries past TTL. Returns count deleted."""
        cutoff = time.time() - self.ttl_seconds
        with self._connect() as db:
            cur = db.execute(
                "DELETE FROM mcp_transport_sessions WHERE last_active_at <= ?",
                (cutoff,),
            )
            return cur.rowcount

    def count(self) -> int:
        """Total persisted (non-expired) sessions. Used by tests + diagnostics."""
        cutoff = time.time() - self.ttl_seconds
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM mcp_transport_sessions "
                "WHERE last_active_at > ?",
                (cutoff,),
            ).fetchone()
        return int(row[0]) if row else 0


class _PersistingInstanceDict(dict[str, StreamableHTTPServerTransport]):
    """Dict subclass that ``register()``s every key on assignment.

    Replaces ``StreamableHTTPSessionManager._server_instances`` so that
    every newly-minted session ID is durably persisted at the moment it
    enters the in-memory registry — without us having to reach into the
    SDK's session-creation code path.
    """

    def __init__(self, store: SessionStore) -> None:
        super().__init__()
        self._store = store

    def __setitem__(self, key: str, value: StreamableHTTPServerTransport) -> None:
        super().__setitem__(key, value)
        try:
            self._store.register(key)
        except Exception:  # noqa: BLE001
            # Persistence failure should not block session use — log and
            # continue; the session will still work in-memory, just won't
            # survive a restart. Hard-failing here would lose the request.
            logger.exception("Failed to persist session %s", key)


class PersistentSessionManager(StreamableHTTPSessionManager):
    """Session manager that resumes sessions across process restarts.

    Behavior overlay on the upstream manager:

    1. ``_server_instances`` is replaced with a dict that auto-persists
       every assignment, so newly-issued session IDs land in SQLite the
       moment they're handed out.
    2. ``_handle_stateful_request`` is overridden to add a resume branch:
       a request with an unknown-in-memory but known-in-SQLite session
       ID is served by spawning a fresh transport keyed to the same ID,
       running the MCP app with ``stateless=True`` so the server-side
       session is born already-initialized and tool calls succeed
       without a handshake round-trip.

    The upstream new-session and 404 paths are otherwise untouched.
    """

    def __init__(
        self,
        app: MCPServer[Any, Any],
        *,
        session_store: SessionStore,
        event_store: EventStore | None = None,
        json_response: bool = False,
        stateless: bool = False,
        security_settings: TransportSecuritySettings | None = None,
        retry_interval: int | None = None,
        session_idle_timeout: float | None = None,
    ) -> None:
        super().__init__(
            app=app,
            event_store=event_store,
            json_response=json_response,
            stateless=stateless,
            security_settings=security_settings,
            retry_interval=retry_interval,
            session_idle_timeout=session_idle_timeout,
        )
        self._session_store = session_store
        self._server_instances = _PersistingInstanceDict(session_store)
        # Process-lifetime counters scraped via /health for cold-stop diagnosis.
        # Single-event-loop server, so plain ints are race-free without a Lock.
        # NOTE: scripts/check_health.py reads these key names directly. If you
        # rename, add, or remove a key here, update the GHA monitor in the same
        # PR — the workflow hard-fails on missing keys ("schema drift").
        self._counters: dict[str, int] = {
            "in_memory_hits": 0,
            "resume_hits": 0,
            "fresh_inits": 0,
            "stale_session_misses": 0,
        }

    def metrics(self) -> dict[str, int]:
        """Snapshot of session-routing counters for /health introspection.

        ``stale_session_misses`` counts requests bearing a session ID that
        is neither in memory nor in SQLite — the actual "Session not found"
        surface. ``resume_hits`` counts the cold-stop recovery path firing.
        Persistent across process lifetime; resets on cold-stop.
        """
        return {
            **self._counters,
            "persisted_sessions_total": self._session_store.count(),
            "in_memory_sessions_current": len(self._server_instances),
        }

    async def _handle_stateful_request(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        request = Request(scope, receive)
        session_id = request.headers.get(MCP_SESSION_ID_HEADER)

        # In-memory hit: refresh persistence + delegate to upstream.
        if session_id is not None and session_id in self._server_instances:
            self._counters["in_memory_hits"] += 1
            self._session_store.touch(session_id)
            await super()._handle_stateful_request(scope, receive, send)
            return

        # Persisted but not in memory: this is the resume case (post
        # cold-start, redeploy, or process restart).
        if session_id is not None and self._session_store.is_known(session_id):
            self._counters["resume_hits"] += 1
            logger.info("Resuming persisted MCP session %s", session_id)
            await self._resume_session(session_id, scope, receive, send)
            return

        # Either a brand-new session (no header) or an unknown session
        # ID we never issued / has expired. Upstream handles both — new
        # session creation will write through _PersistingInstanceDict.
        if session_id is None:
            self._counters["fresh_inits"] += 1
        else:
            # Stale: client sent a session ID we never issued or that
            # expired. Upstream returns 404 — this is the actual
            # "Session not found" surface seen by clients.
            self._counters["stale_session_misses"] += 1
            logger.warning(
                "Stale session_id %s — not in memory, not in SQLite. "
                "Upstream will return 404 per MCP spec.",
                session_id,
            )
        await super()._handle_stateful_request(scope, receive, send)

    async def _resume_session(
        self,
        session_id: str,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Reconstruct an in-memory transport for ``session_id`` and dispatch.

        Mirrors the new-session path in upstream ``_handle_stateful_request``
        but reuses the supplied session_id instead of minting a fresh one
        and runs the MCP app stateless so the server-side ServerSession
        is born already-initialized.
        """
        async with self._session_creation_lock:
            # Race guard: another concurrent request may have already
            # resumed this session while we were waiting for the lock.
            if session_id in self._server_instances:
                self._session_store.touch(session_id)
                await super()._handle_stateful_request(scope, receive, send)
                return

            transport = StreamableHTTPServerTransport(
                mcp_session_id=session_id,
                is_json_response_enabled=self.json_response,
                event_store=self.event_store,
                security_settings=self.security_settings,
                retry_interval=self.retry_interval,
            )
            self._server_instances[session_id] = transport
            self._session_store.touch(session_id)

            async def run_server(
                *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED
            ) -> None:
                async with transport.connect() as streams:
                    read_stream, write_stream = streams
                    task_status.started()
                    try:
                        idle_scope = anyio.CancelScope()
                        if self.session_idle_timeout is not None:
                            idle_scope.deadline = (
                                anyio.current_time() + self.session_idle_timeout
                            )
                            transport.idle_scope = idle_scope

                        with idle_scope:
                            # stateless=True is the resume trick: the
                            # ServerSession is constructed with
                            # InitializationState.Initialized so it does
                            # not block waiting for an InitializeRequest
                            # the client will never re-send.
                            await self.app.run(
                                read_stream,
                                write_stream,
                                self.app.create_initialization_options(),
                                stateless=True,
                            )

                        if idle_scope.cancelled_caught:
                            logger.info(
                                "Resumed session %s idle timeout", session_id
                            )
                            self._server_instances.pop(session_id, None)
                            await transport.terminate()
                    except Exception:
                        logger.exception("Resumed session %s crashed", session_id)
                    finally:
                        if (
                            transport.mcp_session_id
                            and transport.mcp_session_id in self._server_instances
                            and not transport.is_terminated
                        ):
                            del self._server_instances[transport.mcp_session_id]

            assert self._task_group is not None
            await self._task_group.start(run_server)
            await transport.handle_request(scope, receive, send)
