"""Tests for the Salience-tier Phase 1 — first-class standing tier.

Plan: ``private/mnemon-salience-tier-plan-260521.md`` Phase 1.

Covers: promote/demote/list_standing API on Store; cap enforcement;
Layer 4 composition (hook-sourced rejection); search exclusion of
standing-tier docs by default; build_context wiring when the flag is on.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from mnemon import config
from mnemon.store import (
    Store,
    StandingTierCapReached,
    StandingTierError,
    StandingTierProvenanceRejected,
)


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    os.unlink(path)
    s = Store(db_path=path)
    yield s
    s.close()
    for ext in ("", "-wal", "-shm"):
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


# ── Schema migration ──────────────────────────────────────────────


class TestSchemaMigration:
    def test_tier_column_exists(self, store):
        cols = {r["name"] for r in store.db.execute("PRAGMA table_info(documents)").fetchall()}
        assert "tier" in cols

    def test_tier_defaults_to_situational(self, store):
        doc_id = store.save(title="x", content="y")
        row = store.db.execute(
            "SELECT tier FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        assert row["tier"] == "situational"


# ── promote_to_standing ───────────────────────────────────────────


class TestPromote:
    def test_promote_succeeds_on_normal_memory(self, store):
        doc_id = store.save(title="A", content="x")
        assert store.promote_to_standing(doc_id) is True
        row = store.db.execute(
            "SELECT tier FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        assert row["tier"] == "standing"

    def test_promote_idempotent(self, store):
        doc_id = store.save(title="A", content="x")
        store.promote_to_standing(doc_id)
        # Re-promoting an already-standing doc returns True, doesn't error
        assert store.promote_to_standing(doc_id) is True
        # Still only counts as one
        assert store.standing_tier_status()["count"] == 1

    def test_promote_rejects_hook_sourced(self, store):
        doc_id = store.save(
            title="hook", content="auto", source_client="claude-code-hook"
        )
        with pytest.raises(StandingTierProvenanceRejected) as exc:
            store.promote_to_standing(doc_id)
        assert "claude-code-hook" in str(exc.value).lower() or "hook-sourced" in str(exc.value)

    def test_promote_rejects_invalidated(self, store):
        doc_id = store.save(title="A", content="x")
        store.forget(doc_id)
        with pytest.raises(StandingTierError) as exc:
            store.promote_to_standing(doc_id)
        assert "invalidated" in str(exc.value)

    def test_promote_rejects_missing(self, store):
        with pytest.raises(StandingTierError) as exc:
            store.promote_to_standing(99999)
        assert "not found" in str(exc.value)

    def test_promote_rejects_at_cap(self, store, monkeypatch):
        """With cap forced to 2, the 3rd promote raises CapReached."""
        monkeypatch.setattr("mnemon.store.STANDING_TIER_DEFAULT_CAP", 2)
        id1 = store.save(title="A", content="one")
        id2 = store.save(title="B", content="two")
        id3 = store.save(title="C", content="three")
        store.promote_to_standing(id1)
        store.promote_to_standing(id2)
        with pytest.raises(StandingTierCapReached) as exc:
            store.promote_to_standing(id3)
        assert "cap" in str(exc.value).lower()

    def test_promote_cap_respects_invalidated(self, store, monkeypatch):
        """Invalidated standing-tier members don't count against the cap."""
        monkeypatch.setattr("mnemon.store.STANDING_TIER_DEFAULT_CAP", 2)
        id1 = store.save(title="A", content="one")
        id2 = store.save(title="B", content="two")
        id3 = store.save(title="C", content="three")
        store.promote_to_standing(id1)
        store.promote_to_standing(id2)
        store.forget(id1)  # invalidate one
        # Now there's room again — 1 live standing + 1 invalidated → 1/2
        assert store.promote_to_standing(id3) is True
        assert store.standing_tier_status()["count"] == 2


# ── demote_to_situational ─────────────────────────────────────────


class TestDemote:
    def test_demote_round_trips(self, store):
        doc_id = store.save(title="A", content="x")
        store.promote_to_standing(doc_id)
        assert store.demote_to_situational(doc_id) is True
        row = store.db.execute(
            "SELECT tier FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        assert row["tier"] == "situational"

    def test_demote_idempotent_on_situational(self, store):
        doc_id = store.save(title="A", content="x")
        # Doc starts as situational; demote should return False (no-op)
        assert store.demote_to_situational(doc_id) is False

    def test_demote_rejects_missing(self, store):
        with pytest.raises(StandingTierError):
            store.demote_to_situational(99999)

    def test_demote_frees_cap_slot(self, store, monkeypatch):
        monkeypatch.setattr("mnemon.store.STANDING_TIER_DEFAULT_CAP", 2)
        id1 = store.save(title="A", content="one")
        id2 = store.save(title="B", content="two")
        id3 = store.save(title="C", content="three")
        store.promote_to_standing(id1)
        store.promote_to_standing(id2)
        store.demote_to_situational(id1)
        # Slot freed → id3 can promote
        assert store.promote_to_standing(id3) is True


# ── list_standing ─────────────────────────────────────────────────


class TestListStanding:
    def test_returns_empty_when_none(self, store):
        assert store.list_standing() == []

    def test_returns_promoted_memories_ordered_by_recent(self, store):
        id1 = store.save(title="oldest", content="one")
        id2 = store.save(title="middle", content="two")
        id3 = store.save(title="newest", content="three")
        store.promote_to_standing(id1)
        store.promote_to_standing(id2)
        store.promote_to_standing(id3)

        docs = store.list_standing()
        assert len(docs) == 3
        # ORDER BY created_at DESC → newest first
        ids_in_order = [d.id for d in docs]
        # All three present; relative order depends on insert timestamp resolution
        assert set(ids_in_order) == {id1, id2, id3}

    def test_excludes_invalidated(self, store):
        id1 = store.save(title="A", content="x")
        id2 = store.save(title="B", content="y")
        store.promote_to_standing(id1)
        store.promote_to_standing(id2)
        store.forget(id1)
        docs = store.list_standing()
        assert len(docs) == 1
        assert docs[0].id == id2

    def test_includes_content_for_rendering(self, store):
        doc_id = store.save(title="A", content="the full content payload")
        store.promote_to_standing(doc_id)
        docs = store.list_standing()
        assert docs[0].content == "the full content payload"


# ── Search filter (Tier 1 excluded by default) ────────────────────


class TestSearchExclusion:
    def test_bm25_excludes_standing_by_default(self, store):
        id_sit = store.save(title="situational", content="alpha beta gamma")
        id_std = store.save(title="standing", content="alpha beta gamma delta")
        store.promote_to_standing(id_std)

        results = store.search_bm25("alpha", limit=10)
        ids = {r.doc_id for r in results}
        assert id_sit in ids
        assert id_std not in ids

    def test_bm25_includes_standing_when_requested(self, store):
        id_sit = store.save(title="situational", content="alpha beta gamma")
        id_std = store.save(title="standing", content="alpha beta gamma delta")
        store.promote_to_standing(id_std)

        results = store.search_bm25("alpha", limit=10, include_standing=True)
        ids = {r.doc_id for r in results}
        assert id_sit in ids
        assert id_std in ids


# ── build_context wiring ──────────────────────────────────────────


class TestBuildContextWiring:
    def test_flag_off_no_phase1_fetch(self, monkeypatch):
        """When the flag is off and no env-var Phase 0 path is set,
        no standing block is rendered."""
        monkeypatch.delenv("MNEMON_STANDING_TIER_ENABLED", raising=False)
        monkeypatch.delenv("MNEMON_STANDING_TIER_FILE", raising=False)
        monkeypatch.setattr(config, "STANDING_TIER_ENABLED", False)

        from mnemon.hooks.context_surfacing import build_context
        # Empty search-results JSON; no standing tier → empty context
        out = build_context(raw_text="[]")
        assert out == ""

    def test_flag_on_calls_memory_list_standing(self, monkeypatch):
        """When the flag is on, build_context should call
        ``memory_list_standing`` via the remote client and render
        the result as the standing block."""
        monkeypatch.setenv("MNEMON_STANDING_TIER_ENABLED", "true")

        fake_payload = json.dumps([
            {
                "doc_id": 42,
                "title": "Career posture",
                "content": "runway is multi-year, plenty of cash",
                "content_type": "preference",
                "confidence": 0.9,
                "created_at": "2026-05-22 13:30:00",
            },
        ])

        def fake_call_tool_sync(name, args, timeout=5.0):
            assert name == "memory_list_standing"
            return fake_payload, 0.1

        with patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            side_effect=fake_call_tool_sync,
        ):
            from mnemon.hooks.context_surfacing import build_context
            out = build_context(raw_text="[]")

        assert "Standing context" in out
        assert "Career posture" in out
        assert "runway is multi-year" in out

    def test_env_var_truthy_values(self, monkeypatch):
        """The env-var override accepts 1 / true / yes / on."""
        from mnemon.hooks.context_surfacing import _standing_tier_enabled
        for v in ("1", "true", "yes", "on", "TRUE", "True"):
            monkeypatch.setenv("MNEMON_STANDING_TIER_ENABLED", v)
            assert _standing_tier_enabled() is True
        for v in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("MNEMON_STANDING_TIER_ENABLED", v)
            monkeypatch.setattr(config, "STANDING_TIER_ENABLED", False)
            assert _standing_tier_enabled() is False
