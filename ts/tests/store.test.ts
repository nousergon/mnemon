import { test, expect, beforeEach, afterEach } from "bun:test";
import { Store } from "../src/store.ts";
import { unlinkSync, existsSync } from "node:fs";
import { join } from "node:path";

const TEST_DB = join(import.meta.dir, "test.sqlite");

let store: Store;

beforeEach(() => {
  if (existsSync(TEST_DB)) unlinkSync(TEST_DB);
  store = new Store(TEST_DB);
});

afterEach(() => {
  store.close();
  if (existsSync(TEST_DB)) unlinkSync(TEST_DB);
  // Clean up WAL/SHM files
  for (const suffix of ["-wal", "-shm"]) {
    const f = TEST_DB + suffix;
    if (existsSync(f)) unlinkSync(f);
  }
});

test("save and retrieve a memory", () => {
  const id = store.save({
    title: "Test decision",
    content: "We decided to use SQLite for storage.",
    content_type: "decision",
  });

  expect(id).toBeGreaterThan(0);

  const doc = store.get(id);
  expect(doc).not.toBeNull();
  expect(doc!.title).toBe("Test decision");
  expect(doc!.doc).toBe("We decided to use SQLite for storage.");
  expect(doc!.content_type).toBe("decision");
  expect(doc!.confidence).toBe(0.85);
});

test("content-addressable dedup", () => {
  const id1 = store.save({
    title: "First save",
    content: "Same content here.",
  });
  const id2 = store.save({
    title: "Second save",
    content: "Same content here.",
  });

  // Same content = same document
  expect(id1).toBe(id2);

  // Access count should be bumped (dedup bumps +1, get bumps +1)
  const doc = store.get(id1);
  expect(doc!.access_count).toBeGreaterThanOrEqual(1);
});

test("BM25 search", () => {
  store.save({ title: "SQLite decision", content: "We use SQLite for the storage layer." });
  store.save({ title: "Redis caching", content: "Redis is used for caching hot data." });
  store.save({ title: "Python choice", content: "Python is the primary language." });

  const results = store.searchBM25("SQLite storage");
  expect(results.length).toBeGreaterThan(0);
  expect(results[0]!.title).toBe("SQLite decision");
});

test("pin boosts confidence", () => {
  const id = store.save({
    title: "Important fact",
    content: "This should be pinned.",
    content_type: "note",
  });

  const before = store.get(id);
  expect(before!.confidence).toBe(0.5);

  store.pin(id);

  const after = store.get(id);
  expect(after!.pinned).toBe(1);
  expect(after!.confidence).toBe(0.8);
});

test("forget soft-deletes", () => {
  const id = store.save({
    title: "Temporary note",
    content: "This will be forgotten.",
  });

  expect(store.get(id)).not.toBeNull();

  store.forget(id);

  expect(store.get(id)).toBeNull();
});

test("timeline returns recent docs", () => {
  store.save({ title: "First", content: "First content." });
  store.save({ title: "Second", content: "Second content." });
  store.save({ title: "Third", content: "Third content." });

  const timeline = store.timeline(10);
  expect(timeline.length).toBe(3);
  // Most recent first
  expect(timeline[0]!.title).toBe("Third");
});

test("status returns vault health", () => {
  store.save({ title: "A", content: "A content.", content_type: "decision" });
  store.save({ title: "B", content: "B content.", content_type: "note" });

  const stats = store.status();
  expect(stats.total_documents).toBe(2);
  expect(stats.by_type).toHaveLength(2);
});

test("relations graph", () => {
  const id1 = store.save({ title: "Cause", content: "The root cause." });
  const id2 = store.save({ title: "Effect", content: "The resulting effect." });

  store.addRelation(id1, id2, "causes", 0.9);

  const related = store.getRelated(id1);
  expect(related.length).toBe(1);
  expect(related[0]!.title).toBe("Effect");
});
