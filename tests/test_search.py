"""Tests for the search pipeline."""

import os
import tempfile

import pytest

from mnemon.config import PROVENANCE_DEMOTION_FACTOR
from mnemon.search import (
    ScoredResult,
    _bigrams,
    _jaccard_similarity,
    composite_score,
    compute_recency,
    mmr_rerank,
    rrf_fuse,
    search,
)
from mnemon.store import SearchResult, Store


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


class TestCompositeScoring:
    def test_recent_scores_higher(self):
        recent = compute_recency("2026-04-09T00:00:00")
        old = compute_recency("2025-01-01T00:00:00")
        assert recent > old

    def test_composite_combines_signals(self):
        result = SearchResult(
            doc_id=1,
            title="Test",
            content="Hello",
            content_type="note",
            memory_type="semantic",
            confidence=0.8,
            created_at="2026-04-09T00:00:00",
            score=1.0,
        )
        scored = composite_score(result)
        assert scored.composite_score > 0
        assert scored.recency_score > 0


class TestProvenanceDemotion:
    """Layer 4 — auto-captured transcript memories must not outrank an
    equal-relevance deliberate user assertion in unprompted recall."""

    def _result(self, source_client):
        return SearchResult(
            doc_id=1,
            title="Test",
            content="Hello",
            content_type="note",
            memory_type="semantic",
            confidence=0.5,
            created_at="2026-04-09T00:00:00",
            score=1.0,
            source_client=source_client,
        )

    def test_hook_source_is_demoted_by_exactly_the_factor(self):
        user = composite_score(self._result(None))
        hook = composite_score(self._result("claude-code-hook"))
        assert hook.composite_score == pytest.approx(
            user.composite_score * PROVENANCE_DEMOTION_FACTOR
        )
        assert hook.composite_score < user.composite_score

    def test_non_hook_sources_not_demoted(self):
        # compute_recency reads datetime.now(), so two calls differ by a
        # sub-millisecond epsilon — compare with approx, not exact ==.
        baseline = composite_score(self._result(None)).composite_score
        for sc in (None, "mnemon-mirror", "claude-desktop", "cli"):
            assert composite_score(self._result(sc)).composite_score == pytest.approx(
                baseline, rel=1e-6
            )

    def test_source_client_carried_into_scored_result(self):
        assert composite_score(self._result("claude-code-hook")).source_client == (
            "claude-code-hook"
        )

    def test_hook_capture_ranks_below_equal_user_memory(self):
        # Identical relevance/recency/confidence — provenance is the only
        # differentiator; the user memory must sort first.
        user = composite_score(self._result(None))
        hook = composite_score(self._result("claude-code-hook"))
        assert sorted(
            [hook, user], key=lambda r: r.composite_score, reverse=True
        )[0] is user

    def test_provenance_survives_rrf_fusion(self):
        hook = SearchResult(
            doc_id=7, title="H", content="h", content_type="note",
            memory_type="semantic", confidence=0.5, created_at="2026-04-09",
            score=1.0, source_client="claude-code-hook",
        )
        fused = rrf_fuse([hook])
        assert fused[0].source_client == "claude-code-hook"
        # And the demotion then actually fires on the fused result.
        assert composite_score(fused[0]).composite_score == pytest.approx(
            composite_score(
                SearchResult(**{**fused[0].__dict__, "source_client": None})
            ).composite_score * PROVENANCE_DEMOTION_FACTOR
        )


class TestMMR:
    def test_bigrams(self):
        bg = _bigrams("hello world foo")
        assert "hello world" in bg
        assert "world foo" in bg
        assert len(bg) == 2

    def test_jaccard_identical(self):
        a = {"hello world", "world foo"}
        assert _jaccard_similarity(a, a) == 1.0

    def test_jaccard_disjoint(self):
        a = {"hello world"}
        b = {"foo bar"}
        assert _jaccard_similarity(a, b) == 0.0

    def test_mmr_keeps_diverse(self):
        results = [
            ScoredResult(doc_id=1, title="A", content="unique content here", content_type="note", memory_type="semantic", confidence=0.5, created_at="2026-04-09", score=1.0, source="bm25", composite_score=0.9, recency_score=0.8),
            ScoredResult(doc_id=2, title="B", content="completely different text", content_type="note", memory_type="semantic", confidence=0.5, created_at="2026-04-09", score=0.8, source="bm25", composite_score=0.7, recency_score=0.8),
        ]
        reranked = mmr_rerank(results)
        assert len(reranked) == 2
        # Both should keep their scores since they're different
        assert reranked[0].composite_score == 0.9


class TestRRF:
    def test_rrf_fuses_single_set(self):
        results = [
            SearchResult(doc_id=1, title="A", content="a", content_type="note", memory_type="semantic", confidence=0.5, created_at="2026-04-09", score=1.0),
        ]
        fused = rrf_fuse(results)
        assert len(fused) == 1

    def test_rrf_boosts_overlapping(self):
        set1 = [
            SearchResult(doc_id=1, title="A", content="a", content_type="note", memory_type="semantic", confidence=0.5, created_at="2026-04-09", score=1.0),
            SearchResult(doc_id=2, title="B", content="b", content_type="note", memory_type="semantic", confidence=0.5, created_at="2026-04-09", score=0.5),
        ]
        set2 = [
            SearchResult(doc_id=2, title="B", content="b", content_type="note", memory_type="semantic", confidence=0.5, created_at="2026-04-09", score=0.8),
            SearchResult(doc_id=3, title="C", content="c", content_type="note", memory_type="semantic", confidence=0.5, created_at="2026-04-09", score=0.3),
        ]
        fused = rrf_fuse(set1, set2)
        # Doc 2 appears in both sets, should have higher fused score
        scores = {r.doc_id: r.score for r in fused}
        assert scores[2] > scores[3]


class TestSearch:
    def test_search_returns_results(self, store):
        store.save(title="Python", content="Python is great for data engineering")
        results = search(store, "Python")
        assert len(results) >= 1
        assert results[0].composite_score > 0

    def test_search_filters_by_type(self, store):
        store.save(title="Note", content="a note about Python", content_type="note")
        store.save(title="Decision", content="decided to use Python", content_type="decision")
        results = search(store, "Python", content_type="decision")
        assert all(r.content_type == "decision" for r in results)

    def test_search_empty_returns_empty(self, store):
        results = search(store, "nonexistent term xyz")
        assert len(results) == 0

    def test_search_attaches_vector_similarity_when_vector_match(self, store):
        """Results that came back from the vector store must carry the
        raw cosine similarity on ScoredResult.vector_similarity — it's
        the dedup signal used by is_duplicate_remote.

        Patches ``mnemon.embedder.embed`` (the real import target from
        inside search()), not ``mnemon.search.embed`` — the latter
        doesn't exist at the module level since search() imports embed
        lazily, so patching it silently did nothing and the test
        depended on the real FastEmbed being loadable (flaky)."""
        from unittest.mock import patch

        import numpy as np

        from mnemon.store import SearchResult

        fake_vector_hit = SearchResult(
            doc_id=42, title="Foo", content="bar", content_type="note",
            memory_type="semantic", confidence=0.8, created_at="2026-04-12",
            score=0.87, source="vector",
        )
        with patch.object(store, "search_bm25", return_value=[fake_vector_hit]), \
             patch.object(store, "search_vector", return_value=[fake_vector_hit]), \
             patch("mnemon.embedder.embed", return_value=np.zeros(384, dtype=np.float32)):
            results = search(store, "foo")
        assert len(results) >= 1
        assert results[0].vector_similarity == 0.87

    def test_search_vector_similarity_is_none_for_bm25_only(self, store):
        store.save(title="Python", content="Python is great")
        # Disable vector search so results come only from BM25.
        results = search(store, "Python", use_vector=False)
        assert len(results) >= 1
        assert results[0].vector_similarity is None

    def test_search_increments_access_count_for_surfaced_results(self, store):
        """Capture-attention Phase B: surfaced search results bump
        access_count so the attention-report can identify the
        load-bearing fragments."""
        doc_id = store.save(title="Python", content="Python is great")
        # Reset access_count to 0 (Store.save left it implicit; we want
        # to assert only the search-side increment).
        store.db.execute(
            "UPDATE documents SET access_count = 0 WHERE id = ?", (doc_id,),
        )
        store.db.commit()
        search(store, "Python", use_vector=False)
        row = store.db.execute(
            "SELECT access_count FROM documents WHERE id = ?", (doc_id,),
        ).fetchone()
        assert row["access_count"] == 1
        # Second hit on the same query bumps it again.
        search(store, "Python", use_vector=False)
        row = store.db.execute(
            "SELECT access_count FROM documents WHERE id = ?", (doc_id,),
        ).fetchone()
        assert row["access_count"] == 2

    def test_search_no_results_does_not_error_on_access_increment(self, store):
        """Empty-result path must not attempt the IN () update — that's
        a malformed SQL statement. Regression for the branch."""
        results = search(store, "nonexistent xyz term", use_vector=False)
        assert results == []
        # Just reaching here without an OperationalError is the assertion.
