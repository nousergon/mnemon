"""Search pipeline — BM25 + vector + query expansion + RRF fusion +
composite scoring + MMR diversity filtering.

Hybrid search: BM25 full-text search fused with vector semantic search via
Reciprocal Rank Fusion (RRF). Optional LLM-based query expansion generates
alternative search terms. Falls back to BM25-only when vectors unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import COMPOSITE_WEIGHTS, MMR_THRESHOLD, RECENCY_HALF_LIFE_DAYS, RRF_K
from .store import SearchResult, Store


@dataclass
class ScoredResult:
    doc_id: int
    title: str
    content: str
    content_type: str
    memory_type: str
    confidence: float
    created_at: str
    score: float
    source: str
    composite_score: float = 0.0
    recency_score: float = 0.0


def compute_recency(created_at: str) -> float:
    """Exponential recency decay with configurable half-life."""
    try:
        created = datetime.fromisoformat(created_at).replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
    except (ValueError, TypeError):
        age_days = 365
    return math.exp(-0.693 * age_days / RECENCY_HALF_LIFE_DAYS)


def composite_score(result: SearchResult) -> ScoredResult:
    """Apply composite scoring: relevance + recency + confidence."""
    w_rel, w_rec, w_conf = COMPOSITE_WEIGHTS
    recency = compute_recency(result.created_at)
    composite = w_rel * result.score + w_rec * recency + w_conf * result.confidence

    return ScoredResult(
        doc_id=result.doc_id,
        title=result.title,
        content=result.content,
        content_type=result.content_type,
        memory_type=result.memory_type,
        confidence=result.confidence,
        created_at=result.created_at,
        score=result.score,
        source=result.source,
        composite_score=composite,
        recency_score=recency,
    )


def _bigrams(text: str) -> set[str]:
    """Extract bigrams from text for diversity filtering."""
    tokens = text.lower().split()
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def mmr_rerank(results: list[ScoredResult]) -> list[ScoredResult]:
    """MMR diversity reranking — demote (not remove) results too similar
    to already-selected ones.

    All candidate results are kept in the output; similar ones have their
    composite score cut by 50%, then the list is re-sorted. Name is
    ``rerank`` (not ``filter``) to reflect that items are never dropped —
    the 50% demotion is usually enough to push near-duplicates below more
    diverse lower-scoring results.
    """
    if len(results) <= 1:
        return results

    selected: list[ScoredResult] = [results[0]]
    selected_bigrams: list[set[str]] = [_bigrams(results[0].content)]

    for candidate in results[1:]:
        candidate_bg = _bigrams(candidate.content)
        too_similar = any(
            _jaccard_similarity(candidate_bg, bg) > MMR_THRESHOLD
            for bg in selected_bigrams
        )

        if too_similar:
            selected.append(ScoredResult(
                **{k: getattr(candidate, k) for k in candidate.__dataclass_fields__},
                # Demote by 50%
            ))
            selected[-1].composite_score = candidate.composite_score * 0.5
        else:
            selected.append(candidate)

        selected_bigrams.append(candidate_bg)

    selected.sort(key=lambda r: r.composite_score, reverse=True)
    return selected


def rrf_fuse(*result_sets: list[SearchResult]) -> list[SearchResult]:
    """Reciprocal Rank Fusion across multiple result sets."""
    scores: dict[int, dict] = {}

    for results in result_sets:
        for rank, r in enumerate(results):
            rrf_score = 1 / (RRF_K + rank + 1)
            bonus = 0.05 if rank == 0 else 0.02 if rank <= 2 else 0

            if r.doc_id in scores:
                scores[r.doc_id]["score"] += rrf_score + bonus
            else:
                scores[r.doc_id] = {
                    "score": rrf_score + bonus,
                    "result": SearchResult(
                        doc_id=r.doc_id,
                        title=r.title,
                        content=r.content,
                        content_type=r.content_type,
                        memory_type=r.memory_type,
                        confidence=r.confidence,
                        created_at=r.created_at,
                        score=0,
                        source="fused",
                    ),
                }

    fused = sorted(scores.values(), key=lambda s: s["score"], reverse=True)
    return [SearchResult(**{**s["result"].__dict__, "score": s["score"]}) for s in fused]


def expand_query(query: str) -> list[str]:
    """Expand a query into lexical/semantic variants using the local LLM.

    Returns up to 3 alternative queries. Falls back to empty list if LLM unavailable.
    """
    from .llm import try_generate

    response = try_generate(
        "Generate 3 alternative search queries for the given query. "
        "Output one per line, no numbering or bullets. "
        "Keep them short and diverse — include synonyms, related concepts, "
        "and different phrasings.",
        query,
        max_tokens=200,
    )
    if response is None:
        return []
    expansions = [
        line.strip() for line in response.split("\n")
        if 3 < len(line.strip()) < 200
    ]
    return expansions[:3]


def search(
    store: Store,
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    use_vector: bool = True,
    use_expansion: bool = False,
) -> list[ScoredResult]:
    """Main search entry point. Hybrid BM25 + vector search with composite scoring."""
    bm25_results = store.search_bm25(query, limit * 2)

    # Skip expansion if BM25 already has a strong signal
    strong_bm25 = (
        len(bm25_results) > 0
        and bm25_results[0].score >= 0.85
        and (len(bm25_results) < 2 or bm25_results[0].score - bm25_results[1].score >= 0.15)
    )

    # Query expansion (optional, skip if strong BM25 match)
    expansion_results: list[SearchResult] = []
    if use_expansion and not strong_bm25:
        for eq in expand_query(query):
            expansion_results.extend(store.search_bm25(eq, limit))

    # Vector search (optional, fails gracefully)
    vector_results: list[SearchResult] = []
    if use_vector:
        try:
            from .embedder import embed
            query_embedding = embed(f"query: {query}")
            vector_results = store.search_vector(query_embedding, limit * 2)
        except Exception:
            pass  # Fall back to BM25-only

    # Fuse all result sets
    result_sets = [bm25_results]
    if vector_results:
        result_sets.append(vector_results)
    if expansion_results:
        result_sets.append(expansion_results)

    fused = rrf_fuse(*result_sets) if len(result_sets) > 1 else bm25_results

    scored = sorted(
        (composite_score(r) for r in fused),
        key=lambda r: r.composite_score,
        reverse=True,
    )

    if content_type:
        scored = [r for r in scored if r.content_type == content_type]

    return mmr_rerank(scored[:limit])
