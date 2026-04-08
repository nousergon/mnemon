import { test, expect, beforeEach, afterEach } from "bun:test";
import { Store } from "../src/store.ts";
import {
  bigrams,
  jaccardSimilarity,
  mmrFilter,
  computeRecency,
  rrfFuse,
  type ScoredResult,
} from "../src/search.ts";
import { applyConfidenceDecay } from "../src/contradiction.ts";
import { unlinkSync, existsSync } from "node:fs";
import { join } from "node:path";

const TEST_DB = join(import.meta.dir, "search-test.sqlite");
const TEST_VEC = join(import.meta.dir, "search-test.vec");

// ── MMR Tests ────────────────────────────────────────────────────────────────

test("bigrams: extracts word pairs", () => {
  const bg = bigrams("the quick brown fox");
  expect(bg.size).toBe(3);
  expect(bg.has("the quick")).toBe(true);
  expect(bg.has("quick brown")).toBe(true);
  expect(bg.has("brown fox")).toBe(true);
});

test("jaccardSimilarity: identical sets", () => {
  const a = new Set(["a b", "b c"]);
  expect(jaccardSimilarity(a, a)).toBeCloseTo(1.0);
});

test("jaccardSimilarity: disjoint sets", () => {
  const a = new Set(["a b"]);
  const b = new Set(["c d"]);
  expect(jaccardSimilarity(a, b)).toBe(0);
});

test("jaccardSimilarity: partial overlap", () => {
  const a = new Set(["a b", "b c", "c d"]);
  const b = new Set(["a b", "b c", "d e"]);
  expect(jaccardSimilarity(a, b)).toBeCloseTo(0.5);
});

test("mmrFilter: demotes similar results", () => {
  const results: ScoredResult[] = [
    makeScoredResult(1, "Use SQLite for database storage layer", 0.9),
    makeScoredResult(2, "Use SQLite for database storage implementation", 0.85), // very similar
    makeScoredResult(3, "Deploy to EC2 using systemd services", 0.7), // different topic
  ];

  const filtered = mmrFilter(results);
  // Result 2 should be demoted due to similarity with result 1
  const result2 = filtered.find((r) => r.doc_id === 2)!;
  expect(result2.composite_score).toBeLessThan(0.85);
  // Result 3 should keep its score (different topic)
  const result3 = filtered.find((r) => r.doc_id === 3)!;
  expect(result3.composite_score).toBe(0.7);
});

// ── RRF Tests ────────────────────────────────────────────────────────────────

test("rrfFuse: merges multiple result sets", () => {
  const set1 = [
    makeSearchResult(1, 0.9),
    makeSearchResult(2, 0.5),
  ];
  const set2 = [
    makeSearchResult(2, 0.8), // appears in both
    makeSearchResult(3, 0.6),
  ];

  const fused = rrfFuse(set1, set2);
  // Doc 2 appears in both sets, should have highest fused score
  expect(fused[0]!.doc_id).toBe(2);
  expect(fused.length).toBe(3);
});

// ── Recency Tests ────────────────────────────────────────────────────────────

test("computeRecency: recent = high score", () => {
  const recent = computeRecency(new Date().toISOString());
  expect(recent).toBeGreaterThan(0.95);
});

test("computeRecency: old = low score", () => {
  const old = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString();
  const score = computeRecency(old);
  expect(score).toBeLessThan(0.2);
});

// ── Confidence Decay Tests ──────────────────────────────────────────────────

let store: Store;

beforeEach(() => {
  for (const f of [TEST_DB, TEST_VEC, TEST_DB + "-wal", TEST_DB + "-shm"]) {
    if (existsSync(f)) unlinkSync(f);
  }
  store = new Store(TEST_DB);
});

afterEach(() => {
  store.close();
  for (const f of [TEST_DB, TEST_VEC, TEST_DB + "-wal", TEST_DB + "-shm"]) {
    if (existsSync(f)) unlinkSync(f);
  }
});

test("confidence decay: decisions never decay", () => {
  store.save({ title: "Architecture decision", content: "Use microservices.", content_type: "decision" });

  // Simulate aging the document
  store.db.run("UPDATE documents SET updated_at = datetime('now', '-365 days')");

  const decayed = applyConfidenceDecay(store);
  expect(decayed).toBe(0); // decisions don't decay
});

test("confidence decay: notes decay over time", () => {
  const id = store.save({ title: "Temporary note", content: "Some note.", content_type: "note" });

  // Simulate aging
  store.db.run("UPDATE documents SET updated_at = datetime('now', '-120 days')");

  const decayed = applyConfidenceDecay(store);
  expect(decayed).toBe(1);

  const doc = store.get(id);
  expect(doc!.confidence).toBeLessThan(0.5); // decayed from 0.5
});

test("confidence decay: pinned documents exempt", () => {
  const id = store.save({ title: "Pinned note", content: "Important.", content_type: "note" });
  store.pin(id);

  store.db.run("UPDATE documents SET updated_at = datetime('now', '-120 days') WHERE id = ?", [id]);

  const decayed = applyConfidenceDecay(store);
  expect(decayed).toBe(0); // pinned = exempt
});

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeSearchResult(docId: number, score: number): any {
  return {
    doc_id: docId,
    title: `Doc ${docId}`,
    content: `Content for document ${docId}`,
    content_type: "note",
    memory_type: "semantic",
    confidence: 0.5,
    created_at: new Date().toISOString(),
    score,
    source: "bm25",
  };
}

function makeScoredResult(docId: number, content: string, score: number): ScoredResult {
  return {
    doc_id: docId,
    title: `Doc ${docId}`,
    content,
    content_type: "note" as any,
    memory_type: "semantic" as any,
    confidence: 0.5,
    created_at: new Date().toISOString(),
    score,
    source: "fused",
    composite_score: score,
    recency_score: 1.0,
  };
}
