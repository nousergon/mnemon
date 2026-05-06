"""Tests for the persistent MCP session store and manager subclass."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from mnemon.persistent_sessions import (
    DEFAULT_EXPIRE_INTERVAL_SECONDS,
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


# ---------------------------------------------------------------------------
# Counters / metrics surface — feeds /health for cold-stop diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestMetricsCounters:
    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    @staticmethod
    def _make_manager(tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        return PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
        )

    @staticmethod
    def _scope(session_id: bytes | None):
        headers = []
        if session_id is not None:
            headers.append((b"mcp-session-id", session_id))
        return {"type": "http", "method": "POST", "path": "/mcp", "headers": headers}

    @staticmethod
    async def _noop_receive():  # pragma: no cover — never invoked
        return {"type": "http.disconnect"}

    @staticmethod
    async def _noop_send(_):  # pragma: no cover — never invoked
        pass

    def test_metrics_initial_state(self, tmp_path):
        manager = self._make_manager(tmp_path)
        m = manager.metrics()
        assert m["in_memory_hits"] == 0
        assert m["resume_hits"] == 0
        assert m["fresh_inits"] == 0
        assert m["stale_session_misses"] == 0
        assert m["persisted_sessions_total"] == 0
        assert m["in_memory_sessions_current"] == 0

    async def test_in_memory_hit_increments_counter(self, tmp_path, monkeypatch):
        manager = self._make_manager(tmp_path)
        manager._server_instances["live-id"] = MagicMock(name="transport")

        async def fake_super(self, scope, receive, send):  # noqa: ANN001
            pass

        monkeypatch.setattr(
            "mnemon.persistent_sessions.StreamableHTTPSessionManager."
            "_handle_stateful_request",
            fake_super,
        )
        await manager._handle_stateful_request(
            self._scope(b"live-id"), self._noop_receive, self._noop_send
        )
        assert manager.metrics()["in_memory_hits"] == 1

    async def test_resume_hit_increments_counter(self, tmp_path, monkeypatch):
        manager = self._make_manager(tmp_path)
        manager._session_store.register("survivor")

        async def fake_resume(*args, **kwargs):
            pass

        monkeypatch.setattr(manager, "_resume_session", fake_resume)
        await manager._handle_stateful_request(
            self._scope(b"survivor"), self._noop_receive, self._noop_send
        )
        assert manager.metrics()["resume_hits"] == 1

    async def test_fresh_init_increments_counter(self, tmp_path, monkeypatch):
        manager = self._make_manager(tmp_path)

        async def fake_super(self, scope, receive, send):  # noqa: ANN001
            pass

        monkeypatch.setattr(
            "mnemon.persistent_sessions.StreamableHTTPSessionManager."
            "_handle_stateful_request",
            fake_super,
        )
        await manager._handle_stateful_request(
            self._scope(None), self._noop_receive, self._noop_send
        )
        assert manager.metrics()["fresh_inits"] == 1

    async def test_stale_session_miss_increments_counter_and_warns(
        self, tmp_path, monkeypatch, caplog
    ):
        manager = self._make_manager(tmp_path)

        async def fake_super(self, scope, receive, send):  # noqa: ANN001
            pass

        monkeypatch.setattr(
            "mnemon.persistent_sessions.StreamableHTTPSessionManager."
            "_handle_stateful_request",
            fake_super,
        )
        with caplog.at_level("WARNING", logger="mnemon.persistent_sessions"):
            await manager._handle_stateful_request(
                self._scope(b"phantom"), self._noop_receive, self._noop_send
            )
        assert manager.metrics()["stale_session_misses"] == 1
        assert any(
            "Stale session_id phantom" in rec.message for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Periodic expire_old() background task — bounds mcp_sessions.sqlite under
# long warm uptimes (no cold-stop, no redeploy)
# ---------------------------------------------------------------------------


class TestPeriodicExpireConfig:
    def test_default_interval_is_six_hours(self):
        assert DEFAULT_EXPIRE_INTERVAL_SECONDS == 6 * 3600

    def test_default_interval_applied_when_unspecified(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"), session_store=store
        )
        assert manager._expire_interval_seconds == DEFAULT_EXPIRE_INTERVAL_SECONDS

    def test_zero_interval_disables_periodic_prune(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            expire_interval_seconds=0,
        )
        assert manager._expire_interval_seconds == 0


@pytest.mark.anyio
class TestPeriodicExpireTask:
    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    async def test_periodic_task_calls_expire_old_on_each_tick(
        self, tmp_path, monkeypatch
    ):
        """Drive the periodic loop directly: replace anyio.sleep with a
        cancel-after-N-ticks shim and assert expire_old fires N times."""
        import anyio
        import mnemon.persistent_sessions as mod

        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            expire_interval_seconds=1,
        )
        # Track expire_old calls without actually pruning anything.
        calls: list[None] = []
        original_expire = store.expire_old

        def _counting_expire():
            calls.append(None)
            return original_expire()

        store.expire_old = _counting_expire  # type: ignore[method-assign]

        # Replace anyio.sleep so the loop doesn't actually sleep — and
        # cancel after 3 iterations to exit the otherwise-forever loop.
        sleep_count = {"n": 0}

        async def fake_sleep(_seconds):
            sleep_count["n"] += 1
            if sleep_count["n"] >= 3:
                raise anyio.get_cancelled_exc_class()()

        monkeypatch.setattr(mod.anyio, "sleep", fake_sleep)

        with pytest.raises(BaseException):
            await manager._run_periodic_expire()
        assert len(calls) == 2  # ran on tick 1, tick 2; tick 3 cancelled before expire

    async def test_periodic_task_logs_pruned_count_when_nonzero(
        self, tmp_path, monkeypatch, caplog
    ):
        import anyio
        import mnemon.persistent_sessions as mod

        store = SessionStore(tmp_path / "sessions.sqlite", ttl_seconds=1)
        store.register("doomed")
        time.sleep(1.1)
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            expire_interval_seconds=1,
        )

        async def fake_sleep_then_cancel(_seconds):
            raise anyio.get_cancelled_exc_class()()

        # First call ticks; second call cancels.
        first_call = {"done": False}

        async def fake_sleep(_seconds):
            if not first_call["done"]:
                first_call["done"] = True
                return
            raise anyio.get_cancelled_exc_class()()

        monkeypatch.setattr(mod.anyio, "sleep", fake_sleep)

        with caplog.at_level("INFO", logger="mnemon.persistent_sessions"), \
            pytest.raises(BaseException):
            await manager._run_periodic_expire()

        assert any(
            "Periodic prune" in rec.message and "1 expired" in rec.message
            for rec in caplog.records
        )

    async def test_periodic_task_swallows_expire_failures(
        self, tmp_path, monkeypatch, caplog
    ):
        """A transient SQLite hiccup must not kill the task and take
        every active session with it."""
        import anyio
        import mnemon.persistent_sessions as mod

        store = MagicMock(spec=SessionStore)
        store.expire_old.side_effect = [
            RuntimeError("disk on fire"),  # tick 1: bang
            0,  # tick 2: recovers
        ]
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            expire_interval_seconds=1,
        )

        ticks = {"n": 0}

        async def fake_sleep(_seconds):
            ticks["n"] += 1
            if ticks["n"] >= 3:
                raise anyio.get_cancelled_exc_class()()

        monkeypatch.setattr(mod.anyio, "sleep", fake_sleep)

        with caplog.at_level("ERROR", logger="mnemon.persistent_sessions"), \
            pytest.raises(BaseException):
            await manager._run_periodic_expire()

        # Both calls happened — second one wasn't blocked by the first failure
        assert store.expire_old.call_count == 2
        assert any(
            "Periodic session prune raised" in rec.message
            for rec in caplog.records
        )
