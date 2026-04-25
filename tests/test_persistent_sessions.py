"""Tests for the persistent MCP session store and manager subclass."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from mnemon.persistent_sessions import (
    DEFAULT_TTL_SECONDS,
    PersistentSessionManager,
    SessionStore,
    _PersistingInstanceDict,
)


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class TestSessionStore:
    def test_register_then_is_known(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        store.register("abc123")
        assert store.is_known("abc123") is True

    def test_unknown_session_id_returns_false(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        assert store.is_known("never-issued") is False

    def test_register_is_idempotent(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        store.register("abc123")
        store.register("abc123")
        assert store.count() == 1

    def test_persists_across_instances(self, tmp_path):
        """Issued IDs survive a fresh SessionStore (i.e. process restart)."""
        db = tmp_path / "sessions.sqlite"
        SessionStore(db).register("survives-restart")
        assert SessionStore(db).is_known("survives-restart") is True

    def test_expired_session_is_unknown(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite", ttl_seconds=1)
        store.register("ephemeral")
        assert store.is_known("ephemeral") is True
        time.sleep(1.1)
        assert store.is_known("ephemeral") is False

    def test_expire_old_deletes_only_expired(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite", ttl_seconds=1)
        store.register("old")
        time.sleep(1.1)
        store.register("new")
        deleted = store.expire_old()
        assert deleted == 1
        assert store.is_known("old") is False
        assert store.is_known("new") is True

    def test_touch_extends_lifetime(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite", ttl_seconds=2)
        store.register("active")
        time.sleep(1.5)
        store.touch("active")
        time.sleep(1.0)  # original deadline passed; touched deadline has not
        assert store.is_known("active") is True

    def test_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "does" / "not" / "exist" / "sessions.sqlite"
        SessionStore(nested).register("sid")
        assert nested.exists()

    def test_default_ttl_one_week(self):
        # Sanity: the default isn't accidentally short.
        assert DEFAULT_TTL_SECONDS == 7 * 24 * 3600


# ---------------------------------------------------------------------------
# _PersistingInstanceDict
# ---------------------------------------------------------------------------


class TestPersistingInstanceDict:
    def test_setitem_registers_with_store(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        d = _PersistingInstanceDict(store)
        d["abc"] = MagicMock(name="transport")
        assert store.is_known("abc") is True

    def test_setitem_still_stores_value_in_dict(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        d = _PersistingInstanceDict(store)
        transport = MagicMock(name="transport")
        d["abc"] = transport
        assert d["abc"] is transport
        assert "abc" in d

    def test_persistence_failure_does_not_raise(self, tmp_path, caplog):
        """A broken store must not block in-memory session creation —
        sessions failing to persist still need to work in this process."""
        store = MagicMock(spec=SessionStore)
        store.register.side_effect = RuntimeError("disk on fire")
        d = _PersistingInstanceDict(store)
        # Should not propagate; transport assignment must succeed.
        d["abc"] = MagicMock()
        assert "abc" in d
        assert any(
            "Failed to persist session" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# PersistentSessionManager construction
# ---------------------------------------------------------------------------


class TestPersistentSessionManagerInit:
    def test_replaces_server_instances_with_persisting_dict(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
        )
        assert isinstance(manager._server_instances, _PersistingInstanceDict)

    def test_session_assignment_persists(self, tmp_path):
        """Smoke-test: writing a transport to _server_instances persists."""
        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
        )
        manager._server_instances["new-session-id"] = MagicMock(name="transport")
        assert store.is_known("new-session-id") is True


# ---------------------------------------------------------------------------
# Resume flow integration — the core behavior PR B is shipping
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestResumeFlow:
    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    async def test_unknown_in_memory_but_persisted_takes_resume_path(
        self, tmp_path, monkeypatch
    ):
        """Simulate the cold-start scenario.

        First "process" registers a session via _PersistingInstanceDict,
        second "process" gets a request bearing that session ID. The
        second manager has empty _server_instances but the SessionStore
        knows the ID — so the resume branch must fire.
        """
        db = tmp_path / "sessions.sqlite"
        # Pretend a session was issued by an earlier process.
        SessionStore(db).register("survivor-id")

        # New "process": fresh manager, empty in-memory dict, same DB.
        store = SessionStore(db)
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
        )
        assert "survivor-id" not in manager._server_instances

        # Stub the resume handler so we don't need a full ASGI/transport
        # round-trip — this test is about the routing decision.
        called: dict[str, str] = {}

        async def fake_resume(session_id, scope, receive, send):
            called["session_id"] = session_id

        monkeypatch.setattr(manager, "_resume_session", fake_resume)

        scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": [
            (b"mcp-session-id", b"survivor-id"),
        ]}

        async def receive():  # pragma: no cover — never invoked
            return {"type": "http.disconnect"}

        async def send(_):  # pragma: no cover — never invoked
            pass

        await manager._handle_stateful_request(scope, receive, send)
        assert called.get("session_id") == "survivor-id"

    async def test_truly_unknown_session_falls_through_to_super(
        self, tmp_path, monkeypatch
    ):
        """A session ID that was never issued must NOT take the resume
        branch — it falls through to upstream's 404 behavior."""
        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
        )

        resume_called = False

        async def fake_resume(*args, **kwargs):
            nonlocal resume_called
            resume_called = True

        super_called = False

        async def fake_super(self, scope, receive, send):  # noqa: ANN001
            nonlocal super_called
            super_called = True

        monkeypatch.setattr(manager, "_resume_session", fake_resume)
        # Patch the unbound super method on the class so super() resolves to it.
        monkeypatch.setattr(
            "mnemon.persistent_sessions.StreamableHTTPSessionManager."
            "_handle_stateful_request",
            fake_super,
        )

        scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": [
            (b"mcp-session-id", b"phantom-id"),
        ]}

        async def receive():  # pragma: no cover
            return {"type": "http.disconnect"}

        async def send(_):  # pragma: no cover
            pass

        await manager._handle_stateful_request(scope, receive, send)
        assert resume_called is False
        assert super_called is True
