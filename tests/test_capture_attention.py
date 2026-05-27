"""Tests for capture attention Phase A — preserve+relate+boost.

Plan: ``private/mnemon-capture-attention-plan-260522.md``.

Invariant under test: EVERY trigger preserves the new memory (no
information loss). The recurrence-detected branch adds 'restates'
relations + boosts the canonical's confidence + increments its
recurrence_count — but never skips the save.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mnemon import config
from mnemon.store import (
    CaptureAttentionUnavailableError,
    Store,
    _capture_attention_enabled,
)


# ── Fixtures ──────────────────────────────────────────────────────


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


@pytest.fixture
def attention_on(monkeypatch):
    """Flip the feature flag on for the test scope.

    The helper ``_capture_attention_enabled`` re-reads ``config`` at
    every call, so a single monkeypatch on the config constant covers
    every call site (Store.save and CLI status alike).
    """
    monkeypatch.setattr(config, "CAPTURE_ATTENTION_ENABLED", True)


def _fake_embed_constant(_text: str) -> np.ndarray:
    """Embedder stub that returns a fixed unit vector — every save
    produces the same embedding so every pair has similarity ~1.0.
    Used to force the recurrence path independent of real content."""
    v = np.ones(384, dtype=np.float32)
    return v / np.linalg.norm(v)


def _fake_embed_orthogonal(text: str) -> np.ndarray:
    """Embedder stub that returns a different unit vector per text.
    Used to force the NO-recurrence path."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=384).astype(np.float32)
    return v / np.linalg.norm(v)


def _index_with(store, doc_id: int, content: str, fake_embed):
    """Index a document's content into the vec store using a stubbed
    embedder. Mirrors what embed_document() does at runtime."""
    # The Store decodes vec_ids as ``{content_hash}_{seq}``; we use
    # seq=0 to match the "full document" fragment convention.
    import hashlib
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    store.save_embedding(content_hash, 0, fake_embed(content))
    store.flush_vectors()


def _set_created_at(store, doc_id: int, days_ago: int) -> None:
    """Backdate a document's created_at for distinct-sessions tests."""
    when = (dt.datetime.now() - dt.timedelta(days=days_ago)).isoformat(sep=" ")
    store.db.execute(
        "UPDATE documents SET created_at = ? WHERE id = ?",
        (when, doc_id),
    )
    store.db.commit()


# ── Feature flag resolution (env-var override) ────────────────────


class TestFeatureFlagResolution:
    """``MNEMON_CAPTURE_ATTENTION_ENABLED`` env var must take precedence
    over the config default — mirrors ``MNEMON_STANDING_TIER_ENABLED``
    pattern so operators can flip activation on Fly via ``flyctl secrets
    set`` without a code change + redeploy.
    """

    def test_defaults_to_config_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("MNEMON_CAPTURE_ATTENTION_ENABLED", raising=False)
        monkeypatch.setattr(config, "CAPTURE_ATTENTION_ENABLED", False)
        assert _capture_attention_enabled() is False
        monkeypatch.setattr(config, "CAPTURE_ATTENTION_ENABLED", True)
        assert _capture_attention_enabled() is True

    @pytest.mark.parametrize("truthy", ["1", "true", "True", "TRUE", "yes", "on"])
    def test_env_truthy_overrides_config_false(self, monkeypatch, truthy):
        monkeypatch.setattr(config, "CAPTURE_ATTENTION_ENABLED", False)
        monkeypatch.setenv("MNEMON_CAPTURE_ATTENTION_ENABLED", truthy)
        assert _capture_attention_enabled() is True

    @pytest.mark.parametrize("falsy", ["0", "false", "False", "FALSE", "no", "off"])
    def test_env_falsy_overrides_config_true(self, monkeypatch, falsy):
        monkeypatch.setattr(config, "CAPTURE_ATTENTION_ENABLED", True)
        monkeypatch.setenv("MNEMON_CAPTURE_ATTENTION_ENABLED", falsy)
        assert _capture_attention_enabled() is False

    def test_env_unrecognized_falls_back_to_config(self, monkeypatch):
        monkeypatch.setattr(config, "CAPTURE_ATTENTION_ENABLED", True)
        monkeypatch.setenv("MNEMON_CAPTURE_ATTENTION_ENABLED", "maybe")
        assert _capture_attention_enabled() is True
        monkeypatch.setattr(config, "CAPTURE_ATTENTION_ENABLED", False)
        assert _capture_attention_enabled() is False

    def test_env_whitespace_stripped(self, monkeypatch):
        monkeypatch.setattr(config, "CAPTURE_ATTENTION_ENABLED", False)
        monkeypatch.setenv("MNEMON_CAPTURE_ATTENTION_ENABLED", "  true  ")
        assert _capture_attention_enabled() is True


# ── Schema migration ──────────────────────────────────────────────


class TestSchemaMigration:
    def test_recurrence_count_column_exists(self, store):
        cols = {r["name"] for r in store.db.execute("PRAGMA table_info(documents)").fetchall()}
        assert "recurrence_count" in cols

    def test_recurrence_count_defaults_to_zero(self, store):
        doc_id = store.save(title="x", content="y")
        row = store.db.execute(
            "SELECT recurrence_count FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        assert row["recurrence_count"] == 0

    def test_migration_idempotent_on_reopen(self, store):
        # Close and re-open; migration runs again, no error
        store.close()
        s2 = Store(db_path=str(store.db_path) if hasattr(store, "db_path") else
                   store.db.execute("PRAGMA database_list").fetchone()["file"])
        cols = {r["name"] for r in s2.db.execute("PRAGMA table_info(documents)").fetchall()}
        assert "recurrence_count" in cols
        s2.close()


# ── Feature flag respected ────────────────────────────────────────


class TestFeatureFlagDefaultOff:
    def test_no_attention_when_flag_off(self, store):
        """With the flag default-off, behavior matches the pre-PR save."""
        # Two near-identical saves — would trigger if attention were on
        with patch("mnemon.embedder.embed", _fake_embed_constant):
            id1 = store.save(title="A", content="runway is multi-year, plenty of cash")
            _index_with(store, id1, "runway is multi-year, plenty of cash", _fake_embed_constant)
            _set_created_at(store, id1, days_ago=5)

            id2 = store.save(title="B", content="cash runway extends years out")
            _index_with(store, id2, "cash runway extends years out", _fake_embed_constant)

        # Both rows present, no relations, no recurrence increments
        live = store.db.execute(
            "SELECT id, recurrence_count FROM documents WHERE invalidated_at IS NULL"
        ).fetchall()
        assert len(live) == 2
        for row in live:
            assert row["recurrence_count"] == 0

        rels = store.db.execute("SELECT * FROM relations").fetchall()
        assert len(rels) == 0


# ── Preserve-everything invariant ─────────────────────────────────


class TestPreserveEverything:
    def test_new_memory_always_saved_even_on_trigger(self, store, attention_on):
        """The fundamental SOTA invariant: every restatement lands as
        a row in documents, regardless of the attention trigger."""
        with patch("mnemon.embedder.embed", _fake_embed_constant):
            id1 = store.save(title="A", content="content one")
            _index_with(store, id1, "content one", _fake_embed_constant)
            _set_created_at(store, id1, days_ago=4)

            id2 = store.save(title="B", content="content two")
            _index_with(store, id2, "content two", _fake_embed_constant)
            _set_created_at(store, id2, days_ago=2)

            # This save triggers the recurrence path
            id3 = store.save(title="C", content="content three")

        # All three rows live in documents — no skip
        live_ids = {
            r["id"] for r in store.db.execute(
                "SELECT id FROM documents WHERE invalidated_at IS NULL"
            ).fetchall()
        }
        assert live_ids == {id1, id2, id3}


# ── Recurrence-detected: relate + boost + count ───────────────────


class TestRecurrenceDetected:
    def test_distinct_sessions_trigger_boost(self, store, attention_on):
        with patch("mnemon.embedder.embed", _fake_embed_constant):
            id1 = store.save(title="A", content="content one")
            _index_with(store, id1, "content one", _fake_embed_constant)
            _set_created_at(store, id1, days_ago=5)

            id2 = store.save(title="B", content="content two")
            _index_with(store, id2, "content two", _fake_embed_constant)
            _set_created_at(store, id2, days_ago=3)

            initial_conf = store.db.execute(
                "SELECT confidence FROM documents WHERE id = ?", (id1,)
            ).fetchone()["confidence"]

            id3 = store.save(title="C", content="content three")

        # Canonical (id1: oldest, highest conf among the two prior since
        # they tie on confidence; tie broken by created_at DESC then -id)
        # — actually since both id1 and id2 have the same default
        # confidence, the most-recent created_at wins → id2 is canonical.
        canonical_id = id2

        # Confidence bumped on canonical
        new_conf = store.db.execute(
            "SELECT confidence, recurrence_count FROM documents WHERE id = ?",
            (canonical_id,),
        ).fetchone()
        assert new_conf["recurrence_count"] == 1
        assert new_conf["confidence"] > initial_conf

        # 'restates' relations from new doc → each prior neighbor
        rels = store.db.execute(
            "SELECT target_id, relation_type FROM relations WHERE source_id = ?",
            (id3,),
        ).fetchall()
        assert len(rels) == 2
        assert all(r["relation_type"] == "restates" for r in rels)
        assert {r["target_id"] for r in rels} == {id1, id2}

    def test_same_session_no_trigger(self, store, attention_on):
        """All neighbors created same day → distinct-sessions gate
        suppresses the trigger."""
        with patch("mnemon.embedder.embed", _fake_embed_constant):
            id1 = store.save(title="A", content="content one")
            _index_with(store, id1, "content one", _fake_embed_constant)
            id2 = store.save(title="B", content="content two")
            _index_with(store, id2, "content two", _fake_embed_constant)
            id3 = store.save(title="C", content="content three")
            id4 = store.save(title="D", content="content four")

        # No relations, no recurrence increments — all same day
        rels = store.db.execute("SELECT * FROM relations").fetchall()
        assert len(rels) == 0
        counts = store.db.execute(
            "SELECT SUM(recurrence_count) AS s FROM documents"
        ).fetchone()
        assert counts["s"] == 0


# ── Threshold respected ───────────────────────────────────────────


class TestThresholdRespected:
    def test_below_threshold_no_trigger(self, store, attention_on):
        """Orthogonal embeddings (similarity ~0) never trigger."""
        with patch("mnemon.embedder.embed", _fake_embed_orthogonal):
            id1 = store.save(title="A", content="unrelated thing one")
            _index_with(store, id1, "unrelated thing one", _fake_embed_orthogonal)
            _set_created_at(store, id1, days_ago=5)

            id2 = store.save(title="B", content="totally different topic two")
            _index_with(store, id2, "totally different topic two", _fake_embed_orthogonal)
            _set_created_at(store, id2, days_ago=3)

            id3 = store.save(title="C", content="yet another distinct subject")

        rels = store.db.execute("SELECT * FROM relations").fetchall()
        assert len(rels) == 0


# ── Hook-source ceiling ───────────────────────────────────────────


class TestHookCeiling:
    def test_hook_canonical_capped_at_hook_ceiling(self, store, attention_on):
        """A canonical with source_client='claude-code-hook' cannot be
        boosted past HOOK_SOURCE_CONFIDENCE_CEILING (0.5)."""
        from mnemon.config import HOOK_SOURCE_CONFIDENCE_CEILING

        with patch("mnemon.embedder.embed", _fake_embed_constant):
            # Two hook-sourced priors at the ceiling
            id1 = store.save(
                title="A", content="content one",
                source_client="claude-code-hook",
            )
            _index_with(store, id1, "content one", _fake_embed_constant)
            _set_created_at(store, id1, days_ago=5)

            id2 = store.save(
                title="B", content="content two",
                source_client="claude-code-hook",
            )
            _index_with(store, id2, "content two", _fake_embed_constant)
            _set_created_at(store, id2, days_ago=3)

            # Many triggers shouldn't push past the ceiling
            for i in range(10):
                store.save(title=f"X{i}", content=f"content trigger {i}")

        # Canonical's confidence should be == ceiling, not over it
        for doc_id in (id1, id2):
            row = store.db.execute(
                "SELECT confidence FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            assert row["confidence"] <= HOOK_SOURCE_CONFIDENCE_CEILING + 1e-6, \
                f"doc {doc_id} confidence {row['confidence']} exceeded hook ceiling"


class TestHookSourcedSaveSkipped:
    """Regression for 2026-05-27 Phase A soak failure (boost-rate 0.714
    vs 0.25 ceiling). Hook-sourced saves are best-effort transcripts of
    chat sessions — the same provenance set that's blocked from
    standing-tier promotion. They must NOT drive capture-attention,
    otherwise session-handoff fragments inflate confidence on
    near-neighbors (e.g. "Session: pr merged, continue" patterns
    self-boosting via the session_extractor hook)."""

    def test_hook_sourced_save_does_not_trigger_capture_attention(
        self, store, attention_on
    ):
        """Two user-authored priors at distinct dates would normally
        trigger. Saving a hook-sourced new doc with the same embedding
        must NOT create 'restates' relations or increment
        recurrence_count — the gate fires upstream."""
        with patch("mnemon.embedder.embed", _fake_embed_constant):
            id1 = store.save(title="A", content="content one")
            _index_with(store, id1, "content one", _fake_embed_constant)
            _set_created_at(store, id1, days_ago=5)

            id2 = store.save(title="B", content="content two")
            _index_with(store, id2, "content two", _fake_embed_constant)
            _set_created_at(store, id2, days_ago=3)

            # Hook-sourced save — should be filtered out at the gate.
            id3 = store.save(
                title="Session: something extracted",
                content="content three",
                source_client="claude-code-hook",
            )

        # No 'restates' relations from id3 → priors.
        rels = store.db.execute(
            "SELECT * FROM relations WHERE source_id = ?", (id3,)
        ).fetchall()
        assert len(rels) == 0, "hook-sourced save must not emit restates"

        # Neither prior had its recurrence_count incremented.
        counts = store.db.execute(
            "SELECT SUM(recurrence_count) AS s FROM documents "
            "WHERE id IN (?, ?)",
            (id1, id2),
        ).fetchone()
        assert counts["s"] == 0, (
            "hook-sourced save must not boost a canonical's recurrence_count"
        )

    def test_user_save_still_triggers_against_hook_sourced_neighbors(
        self, store, attention_on
    ):
        """Defense isn't symmetric: an OPERATOR save with hook-sourced
        neighbors still fires (operator-authored signal is the intent).
        This guards against an over-aggressive future tightening that
        would also exclude hook-source neighbors and lose the
        consolidation signal entirely."""
        with patch("mnemon.embedder.embed", _fake_embed_constant):
            id1 = store.save(
                title="Session A", content="content one",
                source_client="claude-code-hook",
            )
            _index_with(store, id1, "content one", _fake_embed_constant)
            _set_created_at(store, id1, days_ago=5)

            id2 = store.save(
                title="Session B", content="content two",
                source_client="claude-code-hook",
            )
            _index_with(store, id2, "content two", _fake_embed_constant)
            _set_created_at(store, id2, days_ago=3)

            # User save → should trigger; hook-source ceiling caps the
            # boost on the canonical, but the trigger itself fires.
            id3 = store.save(title="User assertion", content="content three")

        rels = store.db.execute(
            "SELECT * FROM relations WHERE source_id = ?", (id3,)
        ).fetchall()
        assert len(rels) == 2, "user save must trigger against hook-source neighbors"


class TestUserUncapped:
    def test_user_canonical_can_exceed_hook_ceiling(self, store, attention_on):
        """User-authored canonical (source_client=None) can be boosted
        past HOOK_SOURCE_CONFIDENCE_CEILING up to 1.0."""
        from mnemon.config import HOOK_SOURCE_CONFIDENCE_CEILING

        with patch("mnemon.embedder.embed", _fake_embed_constant):
            id1 = store.save(title="A", content="content one")  # source_client=None
            _index_with(store, id1, "content one", _fake_embed_constant)
            _set_created_at(store, id1, days_ago=5)

            id2 = store.save(title="B", content="content two")
            _index_with(store, id2, "content two", _fake_embed_constant)
            _set_created_at(store, id2, days_ago=3)

            # Enough triggers to push past 0.5
            for i in range(20):
                store.save(title=f"X{i}", content=f"content trigger {i}")

        # At least one of the priors should be above hook ceiling
        max_conf = store.db.execute(
            "SELECT MAX(confidence) AS m FROM documents WHERE id IN (?, ?)",
            (id1, id2),
        ).fetchone()["m"]
        assert max_conf > HOOK_SOURCE_CONFIDENCE_CEILING


# ── Canonical selection ───────────────────────────────────────────


class TestCanonicalSelection:
    def test_pinned_beats_high_confidence_unpinned(self, store, attention_on):
        """Pinned (operator gesture) wins over higher unpinned confidence."""
        with patch("mnemon.embedder.embed", _fake_embed_constant):
            id_pinned = store.save(title="A", content="content one")
            _index_with(store, id_pinned, "content one", _fake_embed_constant)
            _set_created_at(store, id_pinned, days_ago=10)
            store.pin(id_pinned)  # boosts confidence + sets pinned=1

            id_unpinned = store.save(title="B", content="content two")
            _index_with(store, id_unpinned, "content two", _fake_embed_constant)
            _set_created_at(store, id_unpinned, days_ago=3)
            # Manually inflate unpinned confidence above the pinned one
            store.db.execute(
                "UPDATE documents SET confidence = 0.99 WHERE id = ?",
                (id_unpinned,),
            )
            store.db.commit()

            # Trigger
            store.save(title="C", content="content three")

        # Canonical should be id_pinned (recurrence_count=1) not id_unpinned
        pinned_row = store.db.execute(
            "SELECT recurrence_count FROM documents WHERE id = ?", (id_pinned,)
        ).fetchone()
        unpinned_row = store.db.execute(
            "SELECT recurrence_count FROM documents WHERE id = ?", (id_unpinned,)
        ).fetchone()
        assert pinned_row["recurrence_count"] == 1
        assert unpinned_row["recurrence_count"] == 0


# ── correction_of override ────────────────────────────────────────


class TestCorrectionOfOverride:
    def test_correction_of_skips_attention(self, store, attention_on):
        """When correction_of is set, capture attention is skipped —
        operator gesture beats automated recurrence detection."""
        with patch("mnemon.embedder.embed", _fake_embed_constant):
            id1 = store.save(title="A", content="content one")
            _index_with(store, id1, "content one", _fake_embed_constant)
            _set_created_at(store, id1, days_ago=5)

            id2 = store.save(title="B", content="content two")
            _index_with(store, id2, "content two", _fake_embed_constant)
            _set_created_at(store, id2, days_ago=3)

            # correction_of set → skip attention even though trigger
            # conditions are met
            store.save(
                title="C", content="content three",
                correction_of=id1,
            )

        rels = store.db.execute(
            "SELECT * FROM relations WHERE relation_type = 'restates'"
        ).fetchall()
        assert len(rels) == 0
        counts = store.db.execute(
            "SELECT SUM(recurrence_count) AS s FROM documents"
        ).fetchone()
        assert counts["s"] == 0


# ── Fail-loud on embedder unavailability ──────────────────────────


class TestFailLoud:
    def test_embedder_failure_raises_named_error(self, store, attention_on):
        """Embedder unavailable → apply_capture_attention raises a NAMED
        exception (per feedback_no_silent_fails). save() catches and
        WARNs (acceptable swallow: secondary observability)."""
        id1 = store.save(title="A", content="x")
        _set_created_at(store, id1, days_ago=5)

        def boom(_text):
            raise RuntimeError("embedder offline")

        with patch("mnemon.embedder.embed", boom):
            # Direct call surfaces the named error
            with pytest.raises(CaptureAttentionUnavailableError) as exc_info:
                store.apply_capture_attention(new_doc_id=id1, content="x")
            assert "embedder offline" in str(exc_info.value) or \
                   "embed" in str(exc_info.value).lower()

            # save() path catches + WARNs; the new memory is still saved
            id2 = store.save(title="B", content="y")
            assert id2 > 0
            row = store.db.execute(
                "SELECT id FROM documents WHERE id = ?", (id2,)
            ).fetchone()
            assert row is not None
