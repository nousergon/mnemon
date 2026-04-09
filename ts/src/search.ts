/**
 * Search pipeline — BM25 + vector + query expansion + RRF fusion +
 * composite scoring + MMR diversity filtering.
 */

import type { Store, SearchResult, ContentType } from "./store.ts";
import { embed } from "./embedder.ts";

// ── Types ────────────────────────────────────────────────────────────────────

export interface SearchOptions {
  query: string;
  limit?: number;
  contentType?: ContentType;
  useVector?: boolean;
  useExpansion?: boolean;
}

export interface ScoredResult extends SearchResult {
  composite_score: number;
  recency_score: number;
}

// ── Reciprocal Rank Fusion ──────────────────────────────────────────────────

const RRF_K = 60;

function rrfFuse(...resultSets: SearchResult[][]): SearchResult[] {
  const scores = new Map<number, { score: number; result: SearchResult }>();

  for (const results of resultSets) {
    for (let rank = 0; rank < results.length; rank++) {
      const r = results[rank]!;
      const rrfScore = 1 / (RRF_K + rank + 1);
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

// ── MMR Diversity Filtering ─────────────────────────────────────────────────

function bigrams(text: string): Set<string> {
  const tokens = text.toLowerCase().split(/\s+/);
  const bg = new Set<string>();
  for (let i = 0; i < tokens.length - 1; i++) {
    bg.add(`${tokens[i]} ${tokens[i + 1]}`);
  }
  return bg;
}

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  let intersection = 0;
  for (const item of a) {
    if (b.has(item)) intersection++;
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

const MMR_THRESHOLD = 0.6;

function mmrFilter(results: ScoredResult[]): ScoredResult[] {
  if (results.length <= 1) return results;

  const selected: ScoredResult[] = [results[0]!];
  const selectedBigrams: Set<string>[] = [bigrams(results[0]!.content)];

  for (let i = 1; i < results.length; i++) {
    const candidate = results[i]!;
    const candidateBg = bigrams(candidate.content);

    const tooSimilar = selectedBigrams.some(
      (bg) => jaccardSimilarity(candidateBg, bg) > MMR_THRESHOLD,
    );

    if (!tooSimilar) {
      selected.push(candidate);
      selectedBigrams.push(candidateBg);
    } else {
      // Demote rather than remove — reduce score by 50%
      selected.push({ ...candidate, composite_score: candidate.composite_score * 0.5 });
      selectedBigrams.push(candidateBg);
    }
  }

  return selected.sort((a, b) => b.composite_score - a.composite_score);
}

// ── Query Expansion ─────────────────────────────────────────────────────────

/**
 * Expand a query into lexical/semantic variants using the local LLM.
 * Falls back to the original query if expansion fails.
 */
async function expandQuery(query: string): Promise<string[]> {
  try {
    const { generate } = await import("./llm.ts");
    const response = await generate(
      "Generate 3 alternative search queries for the given query. Output one per line, no numbering or bullets. Keep them short and diverse — include synonyms, related concepts, and different phrasings.",
      query,
      200,
    );

    const expansions = response
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 3 && l.length < 200);

    return expansions.slice(0, 3);
  } catch {
    return [];
  }
}

// ── Main Search Function ────────────────────────────────────────────────────

export async function search(
  store: Store,
  opts: SearchOptions,
): Promise<ScoredResult[]> {
  const limit = opts.limit ?? 10;
  const useVector = opts.useVector ?? true;
  const useExpansion = opts.useExpansion ?? false;

  // BM25 search
  const bm25Results = store.searchBM25(opts.query, limit * 2);

  // Skip expansion if BM25 already has a strong signal
  const strongBm25 =
    bm25Results.length > 0 &&
    bm25Results[0]!.score >= 0.85 &&
    (bm25Results.length < 2 || bm25Results[0]!.score - bm25Results[1]!.score >= 0.15);

  // Query expansion (optional, skip if strong BM25 match)
  let expansionResults: SearchResult[] = [];
  if (useExpansion && !strongBm25) {
    const expanded = await expandQuery(opts.query);
    for (const eq of expanded) {
      const results = store.searchBM25(eq, limit);
      expansionResults.push(...results);
    }
  }

  if (!useVector) {
    const allBm25 = expansionResults.length > 0
      ? rrfFuse(bm25Results, expansionResults)
      : bm25Results;

    const scored = allBm25
      .map(compositeScore)
      .sort((a, b) => b.composite_score - a.composite_score);

    const filtered = opts.contentType
      ? scored.filter((r) => r.content_type === opts.contentType)
      : scored;

    return mmrFilter(filtered.slice(0, limit));
  }

  // Vector search
  let vectorResults: SearchResult[] = [];
  try {
    const queryEmbedding = await embed(`query: ${opts.query}`);
    vectorResults = store.searchVector(queryEmbedding, limit * 2);
  } catch (err) {
    console.error("Vector search failed, using BM25 only:", err);
  }

  // Fuse all result sets
  const resultSets = [bm25Results];
  if (vectorResults.length > 0) resultSets.push(vectorResults);
  if (expansionResults.length > 0) resultSets.push(expansionResults);

  const fused = resultSets.length > 1 ? rrfFuse(...resultSets) : bm25Results;

  // Composite scoring + MMR
  const scored = fused
    .map(compositeScore)
    .sort((a, b) => b.composite_score - a.composite_score);

  const filtered = opts.contentType
    ? scored.filter((r) => r.content_type === opts.contentType)
    : scored;

  return mmrFilter(filtered.slice(0, limit));
}

// Exported for testing
export { bigrams, jaccardSimilarity, mmrFilter, computeRecency, rrfFuse };
