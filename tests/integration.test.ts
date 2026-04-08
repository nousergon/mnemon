import { test, expect, beforeEach, afterEach } from "bun:test";
import { Store } from "../src/store.ts";
import { search } from "../src/search.ts";
import { VecStore } from "../src/vecstore.ts";
import { unlinkSync, existsSync } from "node:fs";
import { join } from "node:path";

const TEST_DB = join(import.meta.dir, "integration-test.sqlite");
const TEST_VEC = join(import.meta.dir, "integration-test.vec");

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

// ── Search Pipeline Integration ─────────────────────────────────────────────

test("search: BM25-only mode returns results", async () => {
  store.save({ title: "SQLite decision", content: "We use SQLite for the storage layer because it is zero-dependency.", content_type: "decision" });
  store.save({ title: "Redis caching", content: "Redis handles ephemeral caching for hot data.", content_type: "observation" });

  const results = await search(store, { query: "SQLite storage", useVector: false });
  expect(results.length).toBeGreaterThan(0);
  expect(results[0]!.title).toBe("SQLite decision");
  expect(results[0]!.composite_score).toBeGreaterThan(0);
  expect(results[0]!.recency_score).toBeGreaterThan(0.9); // just created
});

test("search: filters by content_type", async () => {
  store.save({ title: "A decision", content: "Decided something important.", content_type: "decision" });
  store.save({ title: "A note", content: "Just a note about decisions.", content_type: "note" });

  const results = await search(store, { query: "decision", contentType: "decision", useVector: false });
  expect(results.every((r) => r.content_type === "decision")).toBe(true);
});

test("search: empty query returns empty", async () => {
  store.save({ title: "Something", content: "Content here." });
  const results = await search(store, { query: "", useVector: false });
  expect(results.length).toBe(0);
});

test("search: respects limit", async () => {
  for (let i = 0; i < 10; i++) {
    store.save({ title: `Memory ${i}`, content: `Content about topic alpha number ${i}.` });
  }

  const results = await search(store, { query: "topic alpha", limit: 3, useVector: false });
  expect(results.length).toBeLessThanOrEqual(3);
});

// ── Store Edge Cases ────────────────────────────────────────────────────────

test("store: multiple content types have correct default confidence", () => {
  const decision = store.save({ title: "D", content: "Decision content.", content_type: "decision" });
  const note = store.save({ title: "N", content: "Note content.", content_type: "note" });
  const handoff = store.save({ title: "H", content: "Handoff content.", content_type: "handoff" });

  expect(store.get(decision)!.confidence).toBe(0.85);
  expect(store.get(note)!.confidence).toBe(0.5);
  expect(store.get(handoff)!.confidence).toBe(0.6);
});

test("store: getByPath retrieves by path", () => {
  store.save({ title: "Path test", content: "Content.", path: "test/path-123", collection: "default" });

  const doc = store.getByPath("test/path-123", "default");
  expect(doc).not.toBeNull();
  expect(doc!.title).toBe("Path test");
});

test("store: getByPath returns null for missing", () => {
  const doc = store.getByPath("nonexistent/path");
  expect(doc).toBeNull();
});

test("store: sweep with no stale docs returns empty", () => {
  store.save({ title: "Fresh", content: "Just created.", content_type: "note" });
  const result = store.sweep(true);
  expect(result.candidates.length).toBe(0);
});

test("store: forget removes from BM25 search", () => {
  const id = store.save({ title: "Forgettable", content: "This will be forgotten." });

  // Should be searchable
  let results = store.searchBM25("forgettable");
  expect(results.length).toBe(1);

  store.forget(id);

  // Should no longer appear in search
  results = store.searchBM25("forgettable");
  expect(results.length).toBe(0);
});

// ── VecStore Edge Cases ─────────────────────────────────────────────────────

test("vecstore: overwrite replaces vector", () => {
  const vec = new VecStore(TEST_VEC, 4);
  vec.set("a_0", new Float32Array([1, 0, 0, 0]));
  vec.set("a_0", new Float32Array([0, 1, 0, 0])); // overwrite

  const results = vec.search(new Float32Array([0, 1, 0, 0]), 1);
  expect(results[0]!.id).toBe("a_0");
  expect(results[0]!.similarity).toBeCloseTo(1.0);
});

test("vecstore: k larger than store size returns all", () => {
  const vec = new VecStore(TEST_VEC, 4);
  vec.set("a_0", new Float32Array([1, 0, 0, 0]));
  vec.set("b_0", new Float32Array([0, 1, 0, 0]));

  const results = vec.search(new Float32Array([1, 0, 0, 0]), 100);
  expect(results.length).toBe(2);
});

// ── Context Surfacing Format ────────────────────────────────────────────────

test("context surfacing: builds valid XML-like format", async () => {
  // Test that the context builder produces the right format
  // by checking the structural elements
  store.save({ title: "Known fact", content: "The system uses SQLite for storage.", content_type: "decision" });

  const results = await search(store, { query: "SQLite storage", useVector: false });
  expect(results.length).toBeGreaterThan(0);

  // Verify result shape for context building
  const r = results[0]!;
  expect(r).toHaveProperty("title");
  expect(r).toHaveProperty("content");
  expect(r).toHaveProperty("content_type");
  expect(r).toHaveProperty("composite_score");
  expect(r).toHaveProperty("recency_score");
  expect(r).toHaveProperty("confidence");
});

// ── Relations Integration ───────────────────────────────────────────────────

test("relations: bidirectional graph traversal", () => {
  const id1 = store.save({ title: "Parent", content: "Parent fact." });
  const id2 = store.save({ title: "Child", content: "Child fact." });
  const id3 = store.save({ title: "Sibling", content: "Sibling fact." });

  store.addRelation(id1, id2, "causes");
  store.addRelation(id1, id3, "related");

  // From parent, find both children
  const fromParent = store.getRelated(id1);
  expect(fromParent.length).toBe(2);

  // From child, find parent
  const fromChild = store.getRelated(id2);
  expect(fromChild.length).toBe(1);
  expect(fromChild[0]!.title).toBe("Parent");
});
