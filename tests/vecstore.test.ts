import { test, expect, beforeEach, afterEach } from "bun:test";
import { VecStore } from "../src/vecstore.ts";
import { unlinkSync, existsSync } from "node:fs";
import { join } from "node:path";

const TEST_VEC = join(import.meta.dir, "test.vec");

let vec: VecStore;

beforeEach(() => {
  if (existsSync(TEST_VEC)) unlinkSync(TEST_VEC);
  vec = new VecStore(TEST_VEC, 4); // tiny dim for tests
});

afterEach(() => {
  if (existsSync(TEST_VEC)) unlinkSync(TEST_VEC);
});

function makeVec(...values: number[]): Float32Array {
  return new Float32Array(values);
}

test("set and search vectors", () => {
  vec.set("a_0", makeVec(1, 0, 0, 0));
  vec.set("b_0", makeVec(0, 1, 0, 0));
  vec.set("c_0", makeVec(0.9, 0.1, 0, 0)); // similar to a

  const results = vec.search(makeVec(1, 0, 0, 0), 3);
  expect(results.length).toBe(3);
  expect(results[0]!.id).toBe("a_0"); // exact match
  expect(results[1]!.id).toBe("c_0"); // most similar
  expect(results[0]!.similarity).toBeCloseTo(1.0);
});

test("persist and reload", () => {
  vec.set("a_0", makeVec(1, 0, 0, 0));
  vec.set("b_0", makeVec(0, 1, 0, 0));
  vec.save();

  // Create new instance from same file
  const vec2 = new VecStore(TEST_VEC, 4);
  expect(vec2.size()).toBe(2);

  const results = vec2.search(makeVec(1, 0, 0, 0), 1);
  expect(results[0]!.id).toBe("a_0");
  expect(results[0]!.similarity).toBeCloseTo(1.0);
});

test("delete vector", () => {
  vec.set("a_0", makeVec(1, 0, 0, 0));
  vec.set("b_0", makeVec(0, 1, 0, 0));
  expect(vec.size()).toBe(2);

  vec.delete("a_0");
  expect(vec.size()).toBe(1);
  expect(vec.has("a_0")).toBe(false);
  expect(vec.has("b_0")).toBe(true);
});

test("empty store returns no results", () => {
  const results = vec.search(makeVec(1, 0, 0, 0), 5);
  expect(results.length).toBe(0);
});
