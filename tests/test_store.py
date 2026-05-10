"""Tests for the storage layer."""

import os
import tempfile

import pytest

from mnemon.store import Store


@pytest.fixture
def store():
    """Create a store with a temporary database."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    os.unlink(path)  # Store will create it
    s = Store(db_path=path)
    yield s
    s.close()
    for ext in ("", "-wal", "-shm"):
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


class TestSave:
    def test_save_returns_doc_id(self, store):
        doc_id = store.save(title="Test", content="Hello world")
        assert doc_id > 0

    def test_save_defaults_to_note(self, store):
        doc_id = store.save(title="Test", content="Hello world")
        doc = store.get(doc_id)
        assert doc.content_type == "note"

    def test_save_with_content_type(self, store):
        doc_id = store.save(title="Test", content="Hello", content_type="decision")
        doc = store.get(doc_id)
        assert doc.content_type == "decision"
        assert doc.confidence == 0.85

    def test_save_deduplicates_by_content_hash(self, store):
        id1 = store.save(title="First", content="same content")
        id2 = store.save(title="Second", content="same content")
        assert id1 == id2

    def test_save_dedup_bumps_access_count(self, store):
        doc_id = store.save(title="First", content="same content")
        store.save(title="Second", content="same content")
        doc = store.get(doc_id)
        # access_count: +1 from dedup save, +1 from get
        assert doc.access_count >= 1

    def test_save_different_content_different_ids(self, store):
        id1 = store.save(title="A", content="content one")
        id2 = store.save(title="B", content="content two")
        assert id1 != id2


class TestSaveHookConfidenceCap:
    """Hook-sourced saves are capped below the explicit-mirror band so
    fragments can't outrank deliberate mirror saves at recall time.
    Closes the 2026-05-10 fragment-confidence regression (default vault
    ids 1994/1997/1998 saved as preference@0.80 / decision@0.85)."""

    def test_hook_sourced_preference_capped(self, store):
        doc_id = store.save(
            title="t",
            content="An observation worth remembering.",
            content_type="preference",
            source_client="claude-code-hook",
        )
        doc = store.get(doc_id)
        assert doc.confidence == 0.5

    def test_hook_sourced_decision_capped(self, store):
        doc_id = store.save(
            title="t",
            content="An observation worth remembering.",
            content_type="decision",
            source_client="claude-code-hook",
        )
        doc = store.get(doc_id)
        assert doc.confidence == 0.5

    def test_explicit_mirror_uses_per_type_default(self, store):
        doc_id = store.save(
            title="t",
            content="An observation worth remembering.",
            content_type="handoff",
            source_client="mnemon-mirror",
        )
        doc = store.get(doc_id)
        assert doc.confidence == 0.6

    def test_no_source_client_uses_per_type_default(self, store):
        doc_id = store.save(
            title="t",
            content="An observation worth remembering.",
            content_type="preference",
        )
        doc = store.get(doc_id)
        assert doc.confidence == 0.8

    def test_hook_cap_does_not_raise_below_default(self, store):
        """min(default, ceiling) — a default lower than the ceiling is
        kept as-is, never inflated up to the ceiling."""
        doc_id = store.save(
            title="t",
            content="An observation worth remembering.",
            content_type="note",
            source_client="claude-code-hook",
        )
        doc = store.get(doc_id)
        assert doc.confidence == 0.5  # NOTE default == ceiling, no movement

    def test_explicit_confidence_arg_still_capped_for_hook(self, store):
        """Explicit confidence arg is still subject to the hook ceiling —
        the hook itself never passes a confidence, but if it did, we want
        defense-in-depth."""
        doc_id = store.save(
            title="t",
            content="An observation worth remembering.",
            content_type="decision",
            source_client="claude-code-hook",
            confidence=0.95,
        )
        doc = store.get(doc_id)
        assert doc.confidence == 0.5


class TestGet:
    def test_get_returns_content(self, store):
        doc_id = store.save(title="Test", content="Hello world")
        doc = store.get(doc_id)
        assert doc is not None
        assert doc.title == "Test"
        assert doc.content == "Hello world"

    def test_get_nonexistent_returns_none(self, store):
        assert store.get(9999) is None

    def test_get_forgotten_returns_none(self, store):
        doc_id = store.save(title="Test", content="Hello world")
        store.forget(doc_id)
        assert store.get(doc_id) is None


class TestPin:
    def test_pin_boosts_confidence(self, store):
        doc_id = store.save(title="Test", content="Hello", content_type="note")
        original = store.get(doc_id)
        store.pin(doc_id)
        pinned = store.get(doc_id)
        assert pinned.confidence > original.confidence
        assert pinned.pinned == 1

    def test_pin_caps_at_1(self, store):
        doc_id = store.save(title="Test", content="Hello", content_type="decision")
        store.pin(doc_id)
        doc = store.get(doc_id)
        assert doc.confidence <= 1.0

    def test_pin_nonexistent_returns_false(self, store):
        assert store.pin(9999) is False


class TestForget:
    def test_forget_soft_deletes(self, store):
        doc_id = store.save(title="Test", content="Hello")
        assert store.forget(doc_id) is True
        assert store.get(doc_id) is None

    def test_forget_removes_from_fts(self, store):
        doc_id = store.save(title="Test", content="searchable content")
        store.forget(doc_id)
        results = store.search_bm25("searchable")
        assert len(results) == 0

    def test_forget_nonexistent_returns_false(self, store):
        assert store.forget(9999) is False

    def test_forget_idempotent(self, store):
        doc_id = store.save(title="Test", content="Hello")
        assert store.forget(doc_id) is True
        assert store.forget(doc_id) is False


class TestTimeline:
    def test_timeline_returns_recent_first(self, store):
        store.save(title="Old", content="old content")
        store.save(title="New", content="new content")
        docs = store.timeline(limit=10)
        assert len(docs) == 2
        assert docs[0].title == "New"

    def test_timeline_filters_by_type(self, store):
        store.save(title="Note", content="note content", content_type="note")
        store.save(title="Decision", content="decision content", content_type="decision")
        docs = store.timeline(limit=10, content_type="decision")
        assert len(docs) == 1
        assert docs[0].title == "Decision"

    def test_timeline_excludes_forgotten(self, store):
        doc_id = store.save(title="Forgotten", content="gone")
        store.save(title="Kept", content="here")
        store.forget(doc_id)
        docs = store.timeline(limit=10)
        assert len(docs) == 1
        assert docs[0].title == "Kept"


class TestSearchBM25:
    def test_search_finds_by_content(self, store):
        store.save(title="Python", content="Python is a programming language")
        results = store.search_bm25("programming")
        assert len(results) >= 1
        assert results[0].title == "Python"

    def test_search_finds_by_title(self, store):
        store.save(title="Python language", content="Some content here")
        results = store.search_bm25("Python")
        assert len(results) >= 1

    def test_search_empty_query_returns_empty(self, store):
        store.save(title="Test", content="Hello")
        assert store.search_bm25("") == []

    def test_search_no_match_returns_empty(self, store):
        store.save(title="Test", content="Hello world")
        results = store.search_bm25("xyznonexistent")
        assert len(results) == 0

    def test_search_excludes_forgotten(self, store):
        doc_id = store.save(title="Test", content="searchable content")
        store.forget(doc_id)
        results = store.search_bm25("searchable")
        assert len(results) == 0


class TestRelations:
    def test_add_and_get_related(self, store):
        id1 = store.save(title="A", content="content A")
        id2 = store.save(title="B", content="content B")
        store.add_relation(id1, id2, "supersedes", 0.8)
        related = store.get_related(id1)
        assert len(related) == 1
        assert related[0].title == "B"
        assert related[0].relation_type == "supersedes"

    def test_relations_are_bidirectional(self, store):
        id1 = store.save(title="A", content="content A")
        id2 = store.save(title="B", content="content B")
        store.add_relation(id1, id2, "related")
        # Should find from either direction
        assert len(store.get_related(id1)) == 1
        assert len(store.get_related(id2)) == 1


class TestStatus:
    def test_status_counts(self, store):
        store.save(title="A", content="a", content_type="note")
        store.save(title="B", content="b", content_type="decision")
        stats = store.status()
        assert stats["total_documents"] == 2
        assert stats["pinned"] == 0
        assert stats["invalidated"] == 0

    def test_status_tracks_pinned(self, store):
        doc_id = store.save(title="A", content="a")
        store.pin(doc_id)
        stats = store.status()
        assert stats["pinned"] == 1


class TestSweep:
    def test_sweep_dry_run_does_not_delete(self, store):
        store.save(title="Old note", content="old stuff", content_type="note")
        result = store.sweep(dry_run=True)
        # Note half-life is 60 days, fresh doc won't be a candidate
        assert result["archived"] == 0
