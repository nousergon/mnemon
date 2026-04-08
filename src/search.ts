/**
 * Search pipeline — BM25 + vector + RRF fusion + composite scoring.
 */

import type { Store, SearchResult, ContentType } from "./store.ts";
import { embed } from "./embedder.ts";

// ── Types ────────────────────────────────────────────────────────────────────

export interface SearchOptions {
  query: string;
  limit?: number;
  contentType?: ContentType;
  useVector?: boolean;
}

export interface ScoredResult extends SearchResult {
  composite_score: number;
  recency_score: number;
}

// ── Reciprocal Rank Fusion ──────────────────────────────────────────────────

const RRF_K = 60;

function rrfFuse(
  ...resultSets: SearchResult[][]
): SearchResult[] {
  const scores = new Map<number, { score: number; result: SearchResult }>();

  for (const results of resultSets) {
    for (let rank = 0; rank < results.length; rank++) {
      const r = results[rank]!;
      const rrfScore = 1 / (RRF_K + rank + 1);

      // Top-rank bonuses (from ClawMem)
      const bonus = rank === 0 ? 0.05 : rank <= 2 ? 0.02 : 0;

      const existing = scores.get(r.doc_id);
      if (existing) {
        existing.score += rrfScore + bonus;
      } else {
        scores.set(r.doc_id, {
          score: rrfScore + bonus,
          result: { ...r, source: "fused" },
        });
      }
    }
  }

  return Array.from(scores.values())
    .sort((a, b) => b.score - a.score)
    .map((s) => ({ ...s.result, score: s.score }));
}

// ── Composite Scoring ───────────────────────────────────────────────────────

function computeRecency(createdAt: string): number {
  const ageMs = Date.now() - new Date(createdAt).getTime();
  const ageDays = ageMs / (1000 * 60 * 60 * 24);

  // Exponential decay: half-life of 30 days
  return Math.exp(-0.693 * ageDays / 30);
}

function compositeScore(result: SearchResult): ScoredResult {
  const recency = computeRecency(result.created_at);
  const composite =
    0.5 * result.score +
    0.25 * recency +
    0.25 * result.confidence;

  return {
    ...result,
    recency_score: recency,
    composite_score: composite,
  };
}

// ── Main Search Function ────────────────────────────────────────────────────

/**
 * Hybrid search: BM25 + vector + RRF fusion + composite scoring.
 * Falls back to BM25-only if useVector is false (remote mode).
 */
export async function search(
  store: Store,
  opts: SearchOptions,
): Promise<ScoredResult[]> {
  const limit = opts.limit ?? 10;
  const useVector = opts.useVector ?? true;

  // BM25 search
  const bm25Results = store.searchBM25(opts.query, limit * 2);

  if (!useVector) {
    // BM25-only mode (remote server, no GPU)
    return bm25Results
      .map(compositeScore)
      .sort((a, b) => b.composite_score - a.composite_score)
      .slice(0, limit);
  }

  // Vector search
  let vectorResults: SearchResult[] = [];
  try {
    const queryEmbedding = await embed(`query: ${opts.query}`);
    vectorResults = store.searchVector(queryEmbedding, limit * 2);
  } catch (err) {
    // Vector search may fail if no embeddings exist yet
    console.error("Vector search failed, using BM25 only:", err);
  }

  // Fuse results
  const fused =
    vectorResults.length > 0
      ? rrfFuse(bm25Results, vectorResults)
      : bm25Results;

  // Apply composite scoring
  const scored = fused
    .map(compositeScore)
    .sort((a, b) => b.composite_score - a.composite_score);

  // Filter by content type if specified
  const filtered = opts.contentType
    ? scored.filter((r) => r.content_type === opts.contentType)
    : scored;

  return filtered.slice(0, limit);
}
