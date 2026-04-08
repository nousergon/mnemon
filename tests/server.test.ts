import { test, expect, beforeEach, afterEach } from "bun:test";
import { Store } from "../src/store.ts";
import { unlinkSync, existsSync } from "node:fs";
import { join } from "node:path";

const TEST_DB = join(import.meta.dir, "server-test.sqlite");
const TEST_VEC = join(import.meta.dir, "server-test.vec");

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

// ── source_client tracking ──────────────────────────────────────────────────

test("source_client is tracked on save", () => {
  const id = store.save({
    title: "Web memory",
    content: "Saved from Claude.ai",
    source_client: "claude-web",
  });

  const doc = store.get(id);
  expect(doc!.source_client).toBe("claude-web");
});

test("source_client defaults to null", () => {
  const id = store.save({
    title: "No client",
    content: "No source specified.",
  });

  const doc = store.get(id);
  expect(doc!.source_client).toBeNull();
});

// ── Multi-vault (different db paths) ────────────────────────────────────────

test("separate vaults are independent", () => {
  const vault2Path = join(import.meta.dir, "vault2-test.sqlite");
  if (existsSync(vault2Path)) unlinkSync(vault2Path);

  const vault2 = new Store(vault2Path);

  store.save({ title: "In vault 1", content: "Vault 1 content." });
  vault2.save({ title: "In vault 2", content: "Vault 2 content." });

  expect(store.status().total_documents).toBe(1);
  expect(vault2.status().total_documents).toBe(1);

  const results1 = store.searchBM25("vault");
  const results2 = vault2.searchBM25("vault");

  expect(results1[0]!.title).toBe("In vault 1");
  expect(results2[0]!.title).toBe("In vault 2");

  vault2.close();
  for (const f of [vault2Path, vault2Path + "-wal", vault2Path + "-shm"]) {
    if (existsSync(f)) unlinkSync(f);
  }
});

// ── Sync module ─────────────────────────────────────────────────────────────

test("sync: push fails gracefully without S3_BUCKET", async () => {
  // Temporarily clear env
  const orig = process.env.MNEMON_S3_BUCKET;
  delete process.env.MNEMON_S3_BUCKET;

  const { push } = await import("../src/sync.ts");
  const result = await push();
  expect(result.errors.length).toBeGreaterThan(0);
  expect(result.errors[0]).toContain("MNEMON_S3_BUCKET");

  if (orig) process.env.MNEMON_S3_BUCKET = orig;
});

test("sync: pull fails gracefully without S3_BUCKET", async () => {
  const orig = process.env.MNEMON_S3_BUCKET;
  delete process.env.MNEMON_S3_BUCKET;

  const { pull } = await import("../src/sync.ts");
  const result = await pull();
  expect(result.errors.length).toBeGreaterThan(0);

  if (orig) process.env.MNEMON_S3_BUCKET = orig;
});
