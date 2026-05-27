"""Tests for contradiction detection (NLI-based, 2026-05-22 rebuild)
and confidence decay."""

import math
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mnemon.config import CONTRADICTION_OVERLAP_THRESHOLD as OVERLAP_THRESHOLD
from mnemon.store import Store, SearchResult
from mnemon.contradiction import (
    CONFIDENCE_FLOOR,
    CONTRADICTION_DECAY,
    UPDATE_DECAY,
    check_contradictions,
    apply_confidence_decay,
    sweep_contradictions,
)
from mnemon.nli import BidirectionalResult, NLIResult, NLIUnavailableError


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


def _bidir(mnemon_label: str) -> BidirectionalResult:
    """Build a BidirectionalResult stub with the given mnemon label.
    Sub-result probabilities are placeholders — only the
    ``mnemon_label`` field is used by ``check_contradictions``."""
    placeholder = NLIResult(
        label="neutral",
        probs={"contradiction": 0.1, "entailment": 0.1, "neutral": 0.8},
    )
    return BidirectionalResult(
        mnemon_label=mnemon_label,
        a_implies_b=placeholder,
        b_implies_a=placeholder,
    )


class TestCheckContradictions:
    def test_returns_empty_when_no_vectors(self, store):
        doc_id = store.save(title="Test", content="Some content")
        result = check_contradictions(store, "New title", "New content", doc_id)
        assert result["decayed"] == 0
        assert result["relationships"] == []
        assert result["nli_unavailable"] is False
        assert result["dry_run"] is False

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
             patch("mnemon.nli.classify_pair_bidirectional", return_value=_bidir("update")):
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
             patch("mnemon.nli.classify_pair_bidirectional", return_value=_bidir("contradiction")):
            result = check_contradictions(store, "Auth method v2", "Never use JWT", id2)

        assert result["decayed"] == 1
        assert result["relationships"][0]["relationship"] == "contradiction"

        doc1_after = store.get(id1)
        # contradiction decay (0.25) > update decay (0.15) — confirm the steeper one applied
        expected = max(CONFIDENCE_FLOOR, original_confidence - CONTRADICTION_DECAY)
        assert abs(doc1_after.confidence - expected) < 1e-6

    def test_update_bumps_winner_contradiction_win_count(self, store):
        """Salience Phase 2: the new doc (winner side) gets its
        contradiction_win_count bumped on an `update` outcome."""
        id1 = store.save(title="Old framing", content="prior phrasing")
        id2 = store.save(title="New framing", content="stronger phrasing")

        mock_results = [
            SearchResult(
                doc_id=id1, title="Old framing", content="prior phrasing",
                content_type="note", memory_type="semantic", confidence=0.8,
                created_at="2026-01-01", score=0.85, source="vector",
            )
        ]
        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.nli.classify_pair_bidirectional",
                   return_value=_bidir("update")):
            check_contradictions(store, "New framing", "stronger phrasing", id2)

        row = store.db.execute(
            "SELECT contradiction_win_count FROM documents WHERE id = ?",
            (id2,),
        ).fetchone()
        assert row["contradiction_win_count"] == 1
        # Loser side stays at 0.
        row_loser = store.db.execute(
            "SELECT contradiction_win_count FROM documents WHERE id = ?",
            (id1,),
        ).fetchone()
        assert row_loser["contradiction_win_count"] == 0

    def test_contradiction_bumps_winner_contradiction_win_count(self, store):
        id1 = store.save(title="A", content="prior")
        id2 = store.save(title="B", content="contradicts prior")

        mock_results = [
            SearchResult(
                doc_id=id1, title="A", content="prior",
                content_type="note", memory_type="semantic", confidence=0.8,
                created_at="2026-01-01", score=0.9, source="vector",
            )
        ]
        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.nli.classify_pair_bidirectional",
                   return_value=_bidir("contradiction")):
            check_contradictions(store, "B", "contradicts prior", id2)

        row = store.db.execute(
            "SELECT contradiction_win_count FROM documents WHERE id = ?",
            (id2,),
        ).fetchone()
        assert row["contradiction_win_count"] == 1

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
             patch("mnemon.nli.classify_pair_bidirectional", return_value=_bidir("same")):
            result = check_contradictions(store, "Deploy step copy", "We deploy using Lambda", id2)

        assert result["decayed"] == 0
        assert result["relationships"][0]["relationship"] == "same"

        related = store.get_related(id2)
        assert len(related) > 0

    def test_skips_self_in_vector_results(self, store):
        """A document doesn't conflict with itself."""
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
        """Memories with low vector similarity should be skipped before NLI."""
        id1 = store.save(title="A", content="Apples")
        id2 = store.save(title="B", content="Bananas")

        mock_results = [
            SearchResult(
                doc_id=id1, title="A", content="Apples",
                content_type="note", memory_type="semantic", confidence=0.5,
                created_at="2026-01-01", score=0.5, source="vector",
            )
        ]

        # Even if NLI would (wrongly) say "contradiction" on every call,
        # below-threshold pairs never reach it.
        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.nli.classify_pair_bidirectional",
                   return_value=_bidir("contradiction")) as nli_mock:
            result = check_contradictions(store, "B", "Bananas", id2)

        assert result["relationships"] == []
        assert nli_mock.call_count == 0  # cosine gate intercepted

    def test_unrelated_does_not_decay(self, store):
        id1 = store.save(title="A", content="Apples")
        id2 = store.save(title="B", content="Bananas")
        doc1 = store.get(id1)
        original_confidence = doc1.confidence

        mock_results = [
            SearchResult(
                doc_id=id1, title="A", content="Apples",
                content_type="note", memory_type="semantic", confidence=original_confidence,
                created_at="2026-01-01", score=0.8, source="vector",
            )
        ]

        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.nli.classify_pair_bidirectional",
                   return_value=_bidir("unrelated")):
            result = check_contradictions(store, "B", "Bananas", id2)

        # Unrelated → relationship recorded but no decay
        assert result["decayed"] == 0
        # Confidence on the older memory unchanged
        doc1_after = store.get(id1)
        assert abs(doc1_after.confidence - original_confidence) < 1e-6

    def test_dry_run_skips_mutations(self, store):
        id1 = store.save(title="DB choice", content="We use PostgreSQL")
        id2 = store.save(title="DB migration", content="Migrating to MySQL")
        doc1 = store.get(id1)
        original_confidence = doc1.confidence

        mock_results = [
            SearchResult(
                doc_id=id1, title="DB choice", content="We use PostgreSQL",
                content_type="note", memory_type="semantic", confidence=original_confidence,
                created_at="2026-01-01", score=0.85, source="vector",
            )
        ]

        with patch.object(store, "search_vector", return_value=mock_results), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.nli.classify_pair_bidirectional", return_value=_bidir("update")):
            result = check_contradictions(store, "DB migration", "Migrating to MySQL", id2,
                                          dry_run=True)

        # Reported as a would-have-decayed for operator visibility
        assert result["decayed"] == 1
        assert result["dry_run"] is True
        # But the actual confidence is unchanged
        doc1_after = store.get(id1)
        assert abs(doc1_after.confidence - original_confidence) < 1e-6
        # And no 'supersedes' relation was inserted
        rels = store.db.execute(
            "SELECT COUNT(*) AS c FROM relations WHERE relation_type = 'supersedes'"
        ).fetchone()
        assert rels["c"] == 0

    def test_nli_unavailable_returns_clear_flag(self, store):
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
             patch("mnemon.nli.classify_pair_bidirectional",
                   side_effect=NLIUnavailableError("model load failed")):
            result = check_contradictions(store, "B", "Content B", id2)

        assert result["nli_unavailable"] is True
        assert result["decayed"] == 0
        # No mutations applied
        rels = store.db.execute("SELECT COUNT(*) AS c FROM relations").fetchone()
        assert rels["c"] == 0

    def test_invalid_classification_skipped(self, store):
        """NLI returning an out-of-taxonomy label should be skipped, not crash."""
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
             patch("mnemon.nli.classify_pair_bidirectional",
                   return_value=_bidir("garbage_label")):
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
        # High-access memory should retain more confidence than low-access at same age
        assert doc2.confidence > doc1.confidence


class TestSweepContradictions:
    """Retroactive sweep — closes the gap for pairs that bypassed
    save-time check_contradictions. Mirrors the same NLI classifier +
    decay/relation side effects but bounded by --max-pairs."""

    def test_empty_vault_returns_zero(self, store):
        result = sweep_contradictions(store)
        assert result["pairs_examined"] == 0
        assert result["pairs_classified"] == 0
        assert result["decayed"] == 0
        assert result["relations_added"] == 0

    def test_single_doc_vault_returns_zero(self, store):
        store.save(title="Only one", content="lonely memory")
        result = sweep_contradictions(store)
        assert result["pairs_classified"] == 0

    def test_classifies_overlapping_pair_and_decays_loser(self, store):
        id_a = store.save(title="Old framing", content="old phrasing")
        id_b = store.save(title="New framing", content="new phrasing")
        # Force search_vector to return id_a as a candidate for id_b
        # and vice versa, both above OVERLAP_THRESHOLD.
        results_for_a = [SearchResult(
            doc_id=id_b, title="New framing", content="new phrasing",
            content_type="note", memory_type="semantic", confidence=0.8,
            created_at="2026-01-01", score=0.95, source="vector",
        )]
        results_for_b = [SearchResult(
            doc_id=id_a, title="Old framing", content="old phrasing",
            content_type="note", memory_type="semantic", confidence=0.8,
            created_at="2026-01-01", score=0.95, source="vector",
        )]
        loser_conf_before = store.get(id_a).confidence

        def _search_side_effect(emb, k):
            # Cheap: same neighbor list both directions for the test
            return results_for_a
        with patch.object(store, "search_vector", side_effect=_search_side_effect), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.nli.classify_pair_bidirectional",
                   return_value=_bidir("update")):
            result = sweep_contradictions(store, max_pairs=10)

        assert result["pairs_classified"] >= 1
        assert result["decayed"] >= 1
        # id_b is the higher id → winner per the sweep's convention.
        winner_after = store.db.execute(
            "SELECT contradiction_win_count FROM documents WHERE id = ?",
            (id_b,),
        ).fetchone()
        assert winner_after["contradiction_win_count"] >= 1
        loser_doc = store.get(id_a)
        assert loser_doc.confidence < loser_conf_before

    def test_skips_already_classified_pairs(self, store):
        id_a = store.save(title="A", content="first")
        id_b = store.save(title="B", content="second")
        # Pre-seed a 'related' relation so the pair is already classified
        store.add_relation(id_a, id_b, "related", 1.0)
        neighbors = [SearchResult(
            doc_id=id_b, title="B", content="second",
            content_type="note", memory_type="semantic", confidence=0.8,
            created_at="2026-01-01", score=0.95, source="vector",
        )]
        with patch.object(store, "search_vector", return_value=neighbors), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.nli.classify_pair_bidirectional",
                   return_value=_bidir("contradiction")) as mock_nli:
            result = sweep_contradictions(store, max_pairs=10)

        # No classification should have been attempted — the pre-existing
        # 'related' relation means the pair has been classified.
        mock_nli.assert_not_called()
        assert result["pairs_skipped"] >= 1
        assert result["pairs_classified"] == 0

    def test_dry_run_classifies_but_does_not_mutate(self, store):
        id_a = store.save(title="A", content="first")
        id_b = store.save(title="B", content="second")
        loser_conf = store.get(id_a).confidence
        neighbors = [SearchResult(
            doc_id=id_b, title="B", content="second",
            content_type="note", memory_type="semantic", confidence=0.8,
            created_at="2026-01-01", score=0.95, source="vector",
        )]
        with patch.object(store, "search_vector", return_value=neighbors), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.nli.classify_pair_bidirectional",
                   return_value=_bidir("contradiction")):
            result = sweep_contradictions(store, max_pairs=10, dry_run=True)

        assert result["pairs_classified"] >= 1
        # No mutation — confidence unchanged, no relations added.
        assert store.get(id_a).confidence == loser_conf
        rels = store.db.execute(
            "SELECT COUNT(*) AS c FROM relations "
            "WHERE relation_type IN ('contradicts', 'supersedes', 'related')"
        ).fetchone()["c"]
        assert rels == 0

    def test_max_pairs_caps_work(self, store):
        # 4 docs → 12 directed pair-classifications possible, but we cap at 2.
        for i in range(4):
            store.save(title=f"M{i}", content=f"content {i}")
        # Each doc's "neighbors" includes the others above threshold
        all_docs = store.timeline(10)
        def _neighbors(emb, k):
            return [
                SearchResult(
                    doc_id=d.id, title=d.title, content="",
                    content_type="note", memory_type="semantic",
                    confidence=0.8, created_at="2026-01-01",
                    score=0.95, source="vector",
                ) for d in all_docs
            ]
        with patch.object(store, "search_vector", side_effect=_neighbors), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.nli.classify_pair_bidirectional",
                   return_value=_bidir("unrelated")):
            result = sweep_contradictions(store, max_pairs=2)

        assert result["pairs_classified"] <= 2

    def test_nli_unavailable_aborts_early(self, store):
        store.save(title="A", content="first")
        store.save(title="B", content="second")
        neighbors = [SearchResult(
            doc_id=2, title="B", content="second",
            content_type="note", memory_type="semantic", confidence=0.8,
            created_at="2026-01-01", score=0.95, source="vector",
        )]
        with patch.object(store, "search_vector", return_value=neighbors), \
             patch("mnemon.embedder.embed", return_value=_mock_embedding()), \
             patch("mnemon.nli.classify_pair_bidirectional",
                   side_effect=NLIUnavailableError("model missing")):
            result = sweep_contradictions(store, max_pairs=10)

        assert result["nli_unavailable"] is True
        # No decays since we aborted before applying side effects.
        assert result["decayed"] == 0
