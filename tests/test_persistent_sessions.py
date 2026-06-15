"""Tests for the persistent MCP session store and manager subclass."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from mnemon.persistent_sessions import (
    DEFAULT_DECAY_INTERVAL_SECONDS,
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

    def test_oldest_age_seconds_empty_is_zero(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        assert store.oldest_age_seconds() == 0.0

    def test_oldest_age_seconds_tracks_least_recently_active(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        store.register("old")
        time.sleep(0.5)
        store.register("new")
        age = store.oldest_age_seconds()
        # Reflects the oldest row (the first register), so ≥ the gap.
        assert age >= 0.5

    def test_oldest_age_seconds_surfaces_unpruned_overdue_rows(self, tmp_path):
        # The prune-health signal: a row past the TTL that expire_old()
        # hasn't removed must show up here even though count() hides it.
        store = SessionStore(tmp_path / "sessions.sqlite", ttl_seconds=1)
        store.register("overdue")
        time.sleep(1.1)
        assert store.count() == 0  # expired → excluded from count()
        assert store.oldest_age_seconds() > 1.0  # but still visible here


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
        assert m["oldest_session_age_seconds"] == 0

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
        """Fresh-init now flows through ``_create_new_session`` rather
        than upstream's locked path. Patch that directly so we don't
        have to spin up a real task group."""
        manager = self._make_manager(tmp_path)

        async def fake_create(scope, receive, send):
            pass

        monkeypatch.setattr(manager, "_create_new_session", fake_create)
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


# ---------------------------------------------------------------------------
# Request-path prune (_maybe_prune) — the suspend-robust companion to the
# periodic task. Keyed on wall-clock time.time() + an in-memory timestamp,
# so it fires on machine wake even when the event-loop timer froze under a
# Fly suspend (issue #215). Gated to once per expire_interval_seconds.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestRequestPathPrune:
    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    @staticmethod
    def _make_manager(tmp_path, *, expire_interval_seconds=3600):
        store = SessionStore(tmp_path / "sessions.sqlite")
        return PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            expire_interval_seconds=expire_interval_seconds,
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

    async def _drive_stale_request(self, manager, monkeypatch):
        """Send one request through _handle_stateful_request (stale-id
        branch, the cheapest) so _maybe_prune runs at the top."""

        async def fake_super(self, scope, receive, send):  # noqa: ANN001
            pass

        monkeypatch.setattr(
            "mnemon.persistent_sessions.StreamableHTTPSessionManager."
            "_handle_stateful_request",
            fake_super,
        )
        await manager._handle_stateful_request(
            self._scope(b"phantom"), self._noop_receive, self._noop_send
        )

    async def test_first_request_prunes_immediately(self, tmp_path, monkeypatch):
        """_last_prune_ts starts at 0.0, so the first request after a cold
        boot always prunes — the cold-boot-then-suspend case."""
        manager = self._make_manager(tmp_path)
        calls: list[None] = []
        original = manager._session_store.expire_old
        monkeypatch.setattr(
            manager._session_store,
            "expire_old",
            lambda: (calls.append(None), original())[1],
        )
        await self._drive_stale_request(manager, monkeypatch)
        assert len(calls) == 1

    async def test_second_request_within_interval_is_gated(
        self, tmp_path, monkeypatch
    ):
        """A 6h-interval deploy must not prune on every request — only the
        first one past the interval boundary."""
        manager = self._make_manager(tmp_path, expire_interval_seconds=3600)
        calls: list[None] = []
        original = manager._session_store.expire_old
        monkeypatch.setattr(
            manager._session_store,
            "expire_old",
            lambda: (calls.append(None), original())[1],
        )
        await self._drive_stale_request(manager, monkeypatch)  # prunes (ts was 0)
        await self._drive_stale_request(manager, monkeypatch)  # gated
        assert len(calls) == 1

    async def test_request_after_interval_prunes_again(self, tmp_path, monkeypatch):
        """Advance wall-clock past the interval → the next request prunes."""
        import mnemon.persistent_sessions as mod

        manager = self._make_manager(tmp_path, expire_interval_seconds=3600)
        calls: list[None] = []
        original = manager._session_store.expire_old
        monkeypatch.setattr(
            manager._session_store,
            "expire_old",
            lambda: (calls.append(None), original())[1],
        )
        clock = {"t": 1_000_000.0}
        monkeypatch.setattr(mod.time, "time", lambda: clock["t"])

        await self._drive_stale_request(manager, monkeypatch)  # prunes (ts was 0)
        clock["t"] += 1800  # half an interval — still gated
        await self._drive_stale_request(manager, monkeypatch)
        clock["t"] += 1801  # now past 3600s since the first prune
        await self._drive_stale_request(manager, monkeypatch)
        assert len(calls) == 2

    async def test_zero_interval_disables_request_path_prune(
        self, tmp_path, monkeypatch
    ):
        manager = self._make_manager(tmp_path, expire_interval_seconds=0)
        calls: list[None] = []
        monkeypatch.setattr(
            manager._session_store,
            "expire_old",
            lambda: calls.append(None),
        )
        await self._drive_stale_request(manager, monkeypatch)
        assert calls == []

    async def test_request_path_removes_overdue_row_without_periodic_task(
        self, tmp_path, monkeypatch
    ):
        """The issue #215 scenario end-to-end: the periodic timer never
        ran (suspended machine), a row is past the TTL, and a single
        request prunes it — bounding oldest_session_age_seconds."""
        store = SessionStore(tmp_path / "sessions.sqlite", ttl_seconds=1)
        store.register("overdue")
        time.sleep(1.1)
        assert store.oldest_age_seconds() > 1.0  # survives, prune hasn't run
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            expire_interval_seconds=3600,
        )
        await self._drive_stale_request(manager, monkeypatch)
        assert store.oldest_age_seconds() == 0.0  # row gone

    async def test_request_path_prune_swallows_errors(
        self, tmp_path, monkeypatch, caplog
    ):
        """A transient SQLite hiccup in the prune must not fail the client
        request riding on it."""
        manager = self._make_manager(tmp_path)
        monkeypatch.setattr(
            manager._session_store,
            "expire_old",
            MagicMock(side_effect=RuntimeError("disk on fire")),
        )
        with caplog.at_level("ERROR", logger="mnemon.persistent_sessions"):
            await self._drive_stale_request(manager, monkeypatch)  # must not raise
        assert manager.metrics()["stale_session_misses"] == 1  # request still served
        assert any(
            "Request-path session prune raised" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# health_snapshot() — the /health entry point. Triggers the gated
# request-path prune BEFORE snapshotting so the hourly probe bounds
# oldest_session_age_seconds even with no real MCP traffic (the false-warn
# fixed here). metrics() itself stays a pure read.
# ---------------------------------------------------------------------------


class TestHealthSnapshot:
    @staticmethod
    def _make_manager(tmp_path, *, ttl_seconds=1, expire_interval_seconds=3600):
        store = SessionStore(tmp_path / "sessions.sqlite", ttl_seconds=ttl_seconds)
        return PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            expire_interval_seconds=expire_interval_seconds,
        )

    def test_health_snapshot_prunes_overdue_row_without_real_traffic(self, tmp_path):
        """The reported failure: an idle suspend-on-idle deploy where the only
        hourly request is the /health probe. health_snapshot() must prune the
        overdue row so oldest_session_age_seconds reads bounded (0 here)."""
        manager = self._make_manager(tmp_path)
        manager._session_store.register("overdue")
        time.sleep(1.1)  # age past the 1s TTL
        assert manager._session_store.oldest_age_seconds() > 1.0  # not yet pruned
        snap = manager.health_snapshot()
        assert snap["oldest_session_age_seconds"] == 0  # pruned during the probe

    def test_metrics_stays_a_pure_read(self, tmp_path):
        """metrics() must NOT prune — only health_snapshot() does. Guards the
        separation so non-/health callers keep a side-effect-free snapshot."""
        manager = self._make_manager(tmp_path)
        manager._session_store.register("overdue")
        time.sleep(1.1)
        assert manager.metrics()["oldest_session_age_seconds"] >= 1  # untouched
        assert manager._session_store.oldest_age_seconds() > 1.0  # still present

    def test_health_snapshot_returns_same_keys_as_metrics(self, tmp_path):
        """health_snapshot is metrics-after-prune — identical schema, so
        check_health.py reads the same fields regardless of which is wired."""
        manager = self._make_manager(tmp_path)
        assert set(manager.health_snapshot()) == set(manager.metrics())


# ---------------------------------------------------------------------------
# Periodic memory-decay sweep — wired alongside the prune task in run().
# Runs apply_confidence_decay() on the vault every decay_interval_seconds
# in a worker thread. Failures swallowed; counts logged when nonzero.
# ---------------------------------------------------------------------------


class TestPeriodicDecayConfig:
    def test_default_interval_is_twenty_four_hours(self):
        assert DEFAULT_DECAY_INTERVAL_SECONDS == 24 * 3600

    def test_default_interval_applied_when_unspecified(self, tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"), session_store=store
        )
        assert manager._decay_interval_seconds == DEFAULT_DECAY_INTERVAL_SECONDS

    def test_decay_fn_defaults_to_none(self, tmp_path):
        """Without an injected decay_fn the periodic decay task is a no-op."""
        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"), session_store=store
        )
        assert manager._decay_fn is None


@pytest.mark.anyio
class TestPeriodicDecayTask:
    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    async def test_periodic_task_calls_decay_fn_on_each_tick(
        self, tmp_path, monkeypatch
    ):
        """Drive the periodic loop directly: replace anyio.sleep with a
        cancel-after-N-ticks shim and assert decay_fn fires N times."""
        import anyio
        import mnemon.persistent_sessions as mod

        calls: list[None] = []

        def _decay_fn() -> int:
            calls.append(None)
            return 0

        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            decay_fn=_decay_fn,
            decay_interval_seconds=1,
        )

        sleep_count = {"n": 0}

        async def fake_sleep(_seconds):
            sleep_count["n"] += 1
            if sleep_count["n"] >= 3:
                raise anyio.get_cancelled_exc_class()()

        monkeypatch.setattr(mod.anyio, "sleep", fake_sleep)

        with pytest.raises(BaseException):
            await manager._run_periodic_decay()
        assert len(calls) == 2  # ran on tick 1, tick 2; tick 3 cancelled before decay

    async def test_periodic_task_logs_decayed_count_when_nonzero(
        self, tmp_path, monkeypatch, caplog
    ):
        import anyio
        import mnemon.persistent_sessions as mod

        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            decay_fn=lambda: 7,
            decay_interval_seconds=1,
        )

        first_call = {"done": False}

        async def fake_sleep(_seconds):
            if not first_call["done"]:
                first_call["done"] = True
                return
            raise anyio.get_cancelled_exc_class()()

        monkeypatch.setattr(mod.anyio, "sleep", fake_sleep)

        with caplog.at_level("INFO", logger="mnemon.persistent_sessions"), \
            pytest.raises(BaseException):
            await manager._run_periodic_decay()

        assert any(
            "Periodic decay" in rec.message and "7 memory" in rec.message
            for rec in caplog.records
        )

    async def test_periodic_task_swallows_decay_failures(
        self, tmp_path, monkeypatch, caplog
    ):
        """Same contract as the prune task: a transient failure in the
        decay sweep must not crash the manager and take active sessions
        with it. The next tick must retry."""
        import anyio
        import mnemon.persistent_sessions as mod

        side_effects: list[Exception | int] = [
            RuntimeError("disk on fire"),  # tick 1: bang
            0,  # tick 2: recovers
        ]

        def _decay_fn() -> int:
            value = side_effects.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        store = SessionStore(tmp_path / "sessions.sqlite")
        manager = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            decay_fn=_decay_fn,
            decay_interval_seconds=1,
        )

        ticks = {"n": 0}

        async def fake_sleep(_seconds):
            ticks["n"] += 1
            if ticks["n"] >= 3:
                raise anyio.get_cancelled_exc_class()()

        monkeypatch.setattr(mod.anyio, "sleep", fake_sleep)

        with caplog.at_level("ERROR", logger="mnemon.persistent_sessions"), \
            pytest.raises(BaseException):
            await manager._run_periodic_decay()

        # Both ticks happened — second one wasn't blocked by the first failure
        assert side_effects == []
        assert any(
            "Periodic memory decay raised" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Request-path decay (_maybe_decay) — the suspend-robust companion to the
# periodic decay timer (issue #217). Wall-clock-gated; spawns the sweep onto
# the lifespan task group so the full-vault walk doesn't block the request.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestRequestPathDecay:
    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    @staticmethod
    def _make_manager(tmp_path, *, decay_fn=lambda: 0, decay_interval_seconds=3600):
        store = SessionStore(tmp_path / "sessions.sqlite")
        return PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=store,
            decay_fn=decay_fn,
            decay_interval_seconds=decay_interval_seconds,
        )

    def test_last_decay_ts_inits_to_construction_time_not_zero(self, tmp_path):
        """Unlike the prune (_last_prune_ts = 0.0 → prune on first request),
        decay inits to wall-clock now so it does NOT walk the full vault on
        every cold boot — only once the interval has elapsed."""
        manager = self._make_manager(tmp_path)
        assert manager._last_decay_ts > 0.0

    def test_fires_after_interval_elapsed(self, tmp_path):
        manager = self._make_manager(tmp_path, decay_interval_seconds=3600)
        manager._task_group = MagicMock(name="task_group")
        manager._last_decay_ts = 0.0  # far in the past → gate open
        manager._maybe_decay()
        manager._task_group.start_soon.assert_called_once_with(
            manager._run_request_path_decay
        )

    def test_gated_within_interval(self, tmp_path):
        """Fresh manager (_last_decay_ts ≈ now) must not fire."""
        manager = self._make_manager(tmp_path, decay_interval_seconds=3600)
        manager._task_group = MagicMock(name="task_group")
        manager._maybe_decay()
        manager._task_group.start_soon.assert_not_called()

    def test_stamps_before_spawn_so_concurrent_requests_fire_once(self, tmp_path):
        manager = self._make_manager(tmp_path, decay_interval_seconds=3600)
        manager._task_group = MagicMock(name="task_group")
        manager._last_decay_ts = 0.0
        manager._maybe_decay()  # fires + stamps _last_decay_ts ≈ now
        manager._maybe_decay()  # gated by the fresh stamp
        assert manager._task_group.start_soon.call_count == 1

    def test_disabled_when_decay_fn_none(self, tmp_path):
        manager = self._make_manager(tmp_path, decay_fn=None)
        manager._task_group = MagicMock(name="task_group")
        manager._last_decay_ts = 0.0
        manager._maybe_decay()
        manager._task_group.start_soon.assert_not_called()

    def test_disabled_when_interval_zero(self, tmp_path):
        manager = self._make_manager(tmp_path, decay_interval_seconds=0)
        manager._task_group = MagicMock(name="task_group")
        manager._last_decay_ts = 0.0
        manager._maybe_decay()
        manager._task_group.start_soon.assert_not_called()

    def test_noop_when_task_group_not_started(self, tmp_path):
        """A request racing lifespan startup must not crash."""
        manager = self._make_manager(tmp_path)
        manager._task_group = None
        manager._last_decay_ts = 0.0
        manager._maybe_decay()  # must not raise

    async def test_spawned_sweep_runs_decay_through_real_task_group(self, tmp_path):
        """End-to-end: the gate spawns onto a real anyio task group and the
        sweep actually executes (in a worker thread) before the group exits."""
        import anyio

        calls: list[int] = []
        manager = self._make_manager(
            tmp_path,
            decay_fn=lambda: (calls.append(1), 5)[1],
            decay_interval_seconds=3600,
        )
        manager._last_decay_ts = 0.0
        async with anyio.create_task_group() as tg:
            manager._task_group = tg
            manager._maybe_decay()
        assert calls == [1]

    async def test_run_request_path_decay_logs_count(self, tmp_path, caplog):
        manager = self._make_manager(tmp_path, decay_fn=lambda: 4)
        with caplog.at_level("INFO", logger="mnemon.persistent_sessions"):
            await manager._run_request_path_decay()
        assert any(
            "Request-path decay" in rec.message and "4 memory" in rec.message
            for rec in caplog.records
        )

    async def test_run_request_path_decay_swallows_errors(self, tmp_path, caplog):
        def _boom():
            raise RuntimeError("disk on fire")

        manager = self._make_manager(tmp_path, decay_fn=_boom)
        with caplog.at_level("ERROR", logger="mnemon.persistent_sessions"):
            await manager._run_request_path_decay()  # must not raise
        assert any(
            "Request-path memory decay raised" in rec.message
            for rec in caplog.records
        )

    def test_metrics_emits_decay_age_only_when_decay_wired(self, tmp_path):
        with_decay = self._make_manager(tmp_path, decay_fn=lambda: 0)
        assert "seconds_since_last_decay" in with_decay.metrics()
        without = PersistentSessionManager(
            app=MagicMock(name="mcp_server"),
            session_store=SessionStore(tmp_path / "s2.sqlite"),
            decay_fn=None,
        )
        assert "seconds_since_last_decay" not in without.metrics()


# ---------------------------------------------------------------------------
# Narrow-lock contract — _session_creation_lock must be released BEFORE
# transport.handle_request is awaited, so a wedged handler can't block
# subsequent fresh-init / resume requests from acquiring the lock
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestSessionCreationLockNarrowing:
    """Regression coverage for the 2026-05-06 lock-held wedge.

    Symptom: upstream's ``_handle_stateful_request`` held
    ``_session_creation_lock`` for the full duration of a fresh-init
    request including ``transport.handle_request``. When that handler
    wedged for any reason (observed in prod), every subsequent fresh-
    init queued behind the lock and timed out at the client side.

    Fix: narrow the lock to only the ``_server_instances`` mutation in
    both ``_create_new_session`` and ``_resume_session``. After the
    lock releases, the dispatch (``transport.handle_request``) runs
    outside it, so concurrent fresh-init / resume requests can proceed
    even if one handler is stuck.
    """

    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    @staticmethod
    def _make_manager(tmp_path):
        store = SessionStore(tmp_path / "sessions.sqlite")
        return PersistentSessionManager(
            app=MagicMock(name="mcp_server"), session_store=store
        )

    @staticmethod
    def _scope(session_id: bytes | None):
        headers = []
        if session_id is not None:
            headers.append((b"mcp-session-id", session_id))
        return {"type": "http", "method": "POST", "path": "/mcp", "headers": headers}

    @staticmethod
    async def _noop_receive():  # pragma: no cover
        return {"type": "http.disconnect"}

    @staticmethod
    async def _noop_send(_):  # pragma: no cover
        pass

    async def test_create_new_session_releases_lock_before_handle_request(
        self, tmp_path, monkeypatch
    ):
        """Probe the narrow-lock invariant: when ``transport.handle_request``
        is awaited, ``_session_creation_lock`` must already be released.

        We patch ``handle_request`` to assert the lock is non-locked at
        call time. Pre-fix this would have failed (lock still held).
        """
        import anyio

        manager = self._make_manager(tmp_path)

        # Stub task_group so _create_new_session doesn't need a real run().
        class _FakeTaskGroup:
            async def start(self, _coro):
                return None

        manager._task_group = _FakeTaskGroup()  # type: ignore[assignment]

        observed = {"locked_during_dispatch": None}

        async def fake_handle_request(self_transport, scope, receive, send):
            # If the lock is held here, the narrow-lock invariant is broken.
            observed["locked_during_dispatch"] = (
                manager._session_creation_lock.locked()
            )

        monkeypatch.setattr(
            "mnemon.persistent_sessions.StreamableHTTPServerTransport."
            "handle_request",
            fake_handle_request,
        )

        await manager._create_new_session(
            self._scope(None), self._noop_receive, self._noop_send
        )

        assert observed["locked_during_dispatch"] is False, (
            "_session_creation_lock was still held when handle_request was "
            "awaited — narrow-lock invariant broken"
        )

    async def test_resume_session_releases_lock_before_handle_request(
        self, tmp_path, monkeypatch
    ):
        manager = self._make_manager(tmp_path)
        manager._session_store.register("survivor")

        class _FakeTaskGroup:
            async def start(self, _coro):
                return None

        manager._task_group = _FakeTaskGroup()  # type: ignore[assignment]

        observed = {"locked_during_dispatch": None}

        async def fake_handle_request(self_transport, scope, receive, send):
            observed["locked_during_dispatch"] = (
                manager._session_creation_lock.locked()
            )

        monkeypatch.setattr(
            "mnemon.persistent_sessions.StreamableHTTPServerTransport."
            "handle_request",
            fake_handle_request,
        )

        await manager._resume_session(
            "survivor", self._scope(b"survivor"), self._noop_receive, self._noop_send
        )

        assert observed["locked_during_dispatch"] is False

    async def test_wedged_handler_does_not_block_concurrent_fresh_init(
        self, tmp_path, monkeypatch
    ):
        """The reproduction we lived through: one fresh-init handler
        hangs forever (simulated). A second fresh-init request must
        still be able to acquire the lock + register its session and
        reach its own ``handle_request`` call. Pre-fix this hung
        forever; post-fix it completes."""
        import anyio

        manager = self._make_manager(tmp_path)

        class _FakeTaskGroup:
            async def start(self, _coro):
                return None

        manager._task_group = _FakeTaskGroup()  # type: ignore[assignment]

        wedge_event = anyio.Event()
        second_dispatched = anyio.Event()

        async def fake_handle_request(self_transport, scope, receive, send):
            # First call hangs forever; second completes immediately.
            if not wedge_event.is_set():
                wedge_event.set()
                await anyio.sleep_forever()
            else:
                second_dispatched.set()

        monkeypatch.setattr(
            "mnemon.persistent_sessions.StreamableHTTPServerTransport."
            "handle_request",
            fake_handle_request,
        )

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                manager._create_new_session,
                self._scope(None),
                self._noop_receive,
                self._noop_send,
            )
            # Wait for the first request to be IN handle_request (lock
            # released, hang in progress).
            await wedge_event.wait()
            # Second request should sail through.
            tg.start_soon(
                manager._create_new_session,
                self._scope(None),
                self._noop_receive,
                self._noop_send,
            )
            with anyio.fail_after(2.0):
                await second_dispatched.wait()
            tg.cancel_scope.cancel()

        assert second_dispatched.is_set()

    async def test_concurrent_fresh_inits_get_distinct_session_ids(
        self, tmp_path, monkeypatch
    ):
        """Sanity: each fresh-init mints its own session_id. The narrow
        lock still serializes ``_server_instances`` writes so two
        sessions can't end up with the same key."""
        manager = self._make_manager(tmp_path)

        class _FakeTaskGroup:
            async def start(self, _coro):
                return None

        manager._task_group = _FakeTaskGroup()  # type: ignore[assignment]

        async def fake_handle_request(self_transport, scope, receive, send):
            return None

        monkeypatch.setattr(
            "mnemon.persistent_sessions.StreamableHTTPServerTransport."
            "handle_request",
            fake_handle_request,
        )

        import anyio

        async with anyio.create_task_group() as tg:
            for _ in range(5):
                tg.start_soon(
                    manager._create_new_session,
                    self._scope(None),
                    self._noop_receive,
                    self._noop_send,
                )

        assert len(manager._server_instances) == 5
        # All session_ids unique
        assert len(set(manager._server_instances.keys())) == 5

    async def test_resume_race_lost_falls_through_to_in_memory_path(
        self, tmp_path, monkeypatch
    ):
        """Race-guard: if another coroutine resumed the session while
        we were waiting for the lock, drop our minted transport and
        delegate to upstream's lock-free in-memory hit branch."""
        import anyio

        manager = self._make_manager(tmp_path)
        manager._session_store.register("survivor")
        # Pre-populate _server_instances so the race-guard fires.
        manager._server_instances["survivor"] = MagicMock(name="winning_transport")

        super_called = {"yes": False}

        async def fake_super(self, scope, receive, send):
            super_called["yes"] = True

        monkeypatch.setattr(
            "mnemon.persistent_sessions.StreamableHTTPSessionManager."
            "_handle_stateful_request",
            fake_super,
        )

        await manager._resume_session(
            "survivor",
            self._scope(b"survivor"),
            self._noop_receive,
            self._noop_send,
        )

        assert super_called["yes"] is True
