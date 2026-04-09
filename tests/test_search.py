"""Tests for the search pipeline."""

import os
import tempfile

import pytest

from mnemon.search import (
    ScoredResult,
    _bigrams,
    _jaccard_similarity,
    composite_score,
    compute_recency,
    mmr_filter,
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
        filtered = mmr_filter(results)
        assert len(filtered) == 2
        # Both should keep their scores since they're different
        assert filtered[0].composite_score == 0.9


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
