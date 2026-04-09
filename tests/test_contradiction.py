"""Tests for contradiction detection and confidence decay."""

import math
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mnemon.store import Store, SearchResult
from mnemon.contradiction import (
    CONFIDENCE_FLOOR,
    CONTRADICTION_DECAY,
    UPDATE_DECAY,
    OVERLAP_THRESHOLD,
    check_contradictions,
    apply_confidence_decay,
)


@pytest.fixture
def store():
    """Create a store with a temporary database."""
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


def _mock_embedding():
    return np.zeros(384, dtype=np.float32)


class TestCheckContradictions:
    def test_returns_empty_when_no_vectors(self, store):
        doc_id = store.save(title="Test", content="Some content")
        result = check_contradictions(store, "New title", "New content", doc_id)
        assert result["decayed"] == 0
        assert result["relationships"] == []

    def test_classifies_update_and_decays(self, store):
        id1 = store.save(title="DB choice", content="We use PostgreSQL for storage")
        id2 = store.save(title="DB migration", content="Migrating to MySQL for storage")
        doc1 = store.get(id1)
        original_confidence = doc1.confidence

        mock_results = [
            SearchResult(
                doc_id=id1, title="DB choice", content="We use PostgreSQL for storage",
                content_type="note", memory_type="semantic", confidence=original_confidence,
                created_at="2026-01-01", score=0.85, source="vector",
            )
        ]

        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.llm.generate", return_value="update"):
            result = check_contradictions(store, "DB migration", "Migrating to MySQL", id2)

        assert result["decayed"] == 1
        assert result["relationships"][0]["relationship"] == "update"

        doc1_after = store.get(id1)
        assert doc1_after.confidence < original_confidence

    def test_classifies_contradiction_and_decays_more(self, store):
        id1 = store.save(title="Auth method", content="Always use JWT tokens")
        id2 = store.save(title="Auth method v2", content="Never use JWT tokens, use sessions")
        doc1 = store.get(id1)
        original_confidence = doc1.confidence

        mock_results = [
            SearchResult(
                doc_id=id1, title="Auth method", content="Always use JWT tokens",
                content_type="note", memory_type="semantic", confidence=original_confidence,
                created_at="2026-01-01", score=0.9, source="vector",
            )
        ]

        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.llm.generate", return_value="contradiction"):
            result = check_contradictions(store, "Auth method v2", "Never use JWT", id2)

        assert result["decayed"] == 1
        assert result["relationships"][0]["relationship"] == "contradiction"

    def test_same_classification_adds_relation(self, store):
        id1 = store.save(title="Deploy step", content="Deploy via Lambda")
        id2 = store.save(title="Deploy step copy", content="We deploy using Lambda")

        mock_results = [
            SearchResult(
                doc_id=id1, title="Deploy step", content="Deploy via Lambda",
                content_type="note", memory_type="semantic", confidence=0.5,
                created_at="2026-01-01", score=0.95, source="vector",
            )
        ]

        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.llm.generate", return_value="same"):
            result = check_contradictions(store, "Deploy step copy", "We deploy using Lambda", id2)

        assert result["decayed"] == 0
        assert result["relationships"][0]["relationship"] == "same"

        related = store.get_related(id2)
        assert len(related) > 0

    def test_skips_self_in_vector_results(self, store):
        """Ensure a document doesn't conflict with itself."""
        id1 = store.save(title="Test", content="Test content")

        mock_results = [
            SearchResult(
                doc_id=id1, title="Test", content="Test content",
                content_type="note", memory_type="semantic", confidence=0.5,
                created_at="2026-01-01", score=0.99, source="vector",
            )
        ]

        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()):
            result = check_contradictions(store, "Test", "Test content", id1)

        assert result["decayed"] == 0
        assert result["relationships"] == []

    def test_filters_below_overlap_threshold(self, store):
        """Memories with low vector similarity should be skipped."""
        id1 = store.save(title="A", content="Apples")
        id2 = store.save(title="B", content="Bananas")

        mock_results = [
            SearchResult(
                doc_id=id1, title="A", content="Apples",
                content_type="note", memory_type="semantic", confidence=0.5,
                created_at="2026-01-01", score=0.5, source="vector",
            )
        ]

        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()):
            result = check_contradictions(store, "B", "Bananas", id2)

        assert result["relationships"] == []

    def test_invalid_classification_skipped(self, store):
        """LLM returning garbage should be skipped."""
        id1 = store.save(title="A", content="Content A")
        id2 = store.save(title="B", content="Content B")

        mock_results = [
            SearchResult(
                doc_id=id1, title="A", content="Content A",
                content_type="note", memory_type="semantic", confidence=0.5,
                created_at="2026-01-01", score=0.85, source="vector",
            )
        ]

        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.llm.generate", return_value="i dont know"):
            result = check_contradictions(store, "B", "Content B", id2)

        assert result["decayed"] == 0
        assert result["relationships"] == []


class TestConfidenceDecay:
    def test_decays_observation_over_time(self, store):
        doc_id = store.save(title="Obs", content="Something I learned", content_type="observation")

        store.db.execute(
            "UPDATE documents SET updated_at = datetime('now', '-100 days') WHERE id = ?",
            (doc_id,),
        )
        store.db.commit()

        updated = apply_confidence_decay(store)
        assert updated > 0

        doc = store.get(doc_id)
        assert doc.confidence < 0.70

    def test_does_not_decay_pinned_memories(self, store):
        doc_id = store.save(title="Pinned", content="Important decision", content_type="observation")
        store.pin(doc_id)

        store.db.execute(
            "UPDATE documents SET updated_at = datetime('now', '-200 days') WHERE id = ?",
            (doc_id,),
        )
        store.db.commit()

        apply_confidence_decay(store)
        doc = store.get(doc_id)
        assert doc.confidence >= 0.70

    def test_does_not_decay_permanent_types(self, store):
        doc_id = store.save(title="Decision", content="Use PostgreSQL", content_type="decision")

        store.db.execute(
            "UPDATE documents SET updated_at = datetime('now', '-365 days') WHERE id = ?",
            (doc_id,),
        )
        store.db.commit()

        apply_confidence_decay(store)
        doc = store.get(doc_id)
        assert doc.confidence == 0.85

    def test_access_reinforcement_slows_decay(self, store):
        id1 = store.save(title="Low access", content="Rarely accessed observation", content_type="observation")
        id2 = store.save(title="High access", content="Frequently accessed observation", content_type="observation")

        store.db.execute("UPDATE documents SET access_count = 20 WHERE id = ?", (id2,))
        store.db.execute("UPDATE documents SET updated_at = datetime('now', '-90 days') WHERE id = ?", (id1,))
        store.db.execute("UPDATE documents SET updated_at = datetime('now', '-90 days') WHERE id = ?", (id2,))
        store.db.commit()

        apply_confidence_decay(store)

        doc1 = store.get(id1)
        doc2 = store.get(id2)
        assert doc2.confidence > doc1.confidence

    def test_confidence_floor(self, store):
        doc_id = store.save(title="Ancient", content="Very old observation", content_type="handoff")

        store.db.execute(
            "UPDATE documents SET updated_at = datetime('now', '-365 days') WHERE id = ?",
            (doc_id,),
        )
        store.db.commit()

        apply_confidence_decay(store)
        doc = store.get(doc_id)
        assert doc.confidence >= CONFIDENCE_FLOOR
