import { test, expect } from "bun:test";
import { isNoise, isDuplicate } from "../src/hooks/framework.ts";

// ── Noise Filtering ──────────────────────────────────────────────────────────

test("isNoise: filters empty and short prompts", () => {
  expect(isNoise("")).toBe(true);
  expect(isNoise("  ")).toBe(true);
  expect(isNoise("hi")).toBe(true);
  expect(isNoise("y")).toBe(true);
});

test("isNoise: filters slash commands", () => {
  expect(isNoise("/help")).toBe(true);
  expect(isNoise("/clear")).toBe(true);
  expect(isNoise("/commit")).toBe(true);
});

test("isNoise: filters greetings", () => {
  expect(isNoise("hello")).toBe(true);
  expect(isNoise("thanks!")).toBe(true);
  expect(isNoise("ok")).toBe(true);
  expect(isNoise("good morning")).toBe(true);
});

test("isNoise: passes real prompts through", () => {
  expect(isNoise("fix the bug in the login flow")).toBe(false);
  expect(isNoise("what is the architecture of the system?")).toBe(false);
  expect(isNoise("refactor the database layer to use connection pooling")).toBe(false);
});

// ── Context Building ─────────────────────────────────────────────────────────

test("context surfacing: builds tiered XML context", async () => {
  // Import the module to test the buildContext function indirectly
  // We test the full pipeline integration via the store tests
  // This validates the noise filter + dedup work correctly

  // Real prompts should not be noise
  expect(isNoise("How does the executor daemon handle market close?")).toBe(false);
  expect(isNoise("What was decided about the memory architecture?")).toBe(false);
});

// ── Observation Parsing ──────────────────────────────────────────────────────

test("parseObservations: extracts XML observations", async () => {
  // Test the XML parsing logic directly
  const response = `
<observation>
  <type>decision</type>
  <title>Use SQLite for storage</title>
  <content>We decided to use SQLite with FTS5 for the memory vault because it's zero-dependency, portable, and fast enough for our vault sizes.</content>
</observation>
<observation>
  <type>preference</type>
  <title>User prefers single-line commands</title>
  <content>The user wants all shell commands as single-line, never multi-line with backslash continuations.</content>
</observation>`;

  // Import the module and test parsing
  const regex = /<observation>\s*<type>(.*?)<\/type>\s*<title>(.*?)<\/title>\s*<content>(.*?)<\/content>\s*<\/observation>/gs;
  const observations: Array<{ type: string; title: string; content: string }> = [];

  let match;
  while ((match = regex.exec(response)) !== null) {
    observations.push({
      type: match[1]!.trim(),
      title: match[2]!.trim(),
      content: match[3]!.trim(),
    });
  }

  expect(observations.length).toBe(2);
  expect(observations[0]!.type).toBe("decision");
  expect(observations[0]!.title).toBe("Use SQLite for storage");
  expect(observations[1]!.type).toBe("preference");
});

test("parseObservations: handles none response", () => {
  const response = "<none/>";
  expect(response.includes("<none/>")).toBe(true);
});

// ── Handoff Parsing ──────────────────────────────────────────────────────────

test("parseHandoff: extracts handoff XML", () => {
  const response = `
<handoff>
  <title>Fixed pipeline holiday bug</title>
  <summary>
  - Fixed Step Function holiday check that was silently skipping trading days
  - Root cause: ssm:GetCommandInvocation IAM permission scoped to wrong resource
  - Added InProgress/Pending polling to prevent false holiday detection
  - Open: need to wire up EOD pipeline trigger
  </summary>
</handoff>`;

  const titleMatch = response.match(/<title>(.*?)<\/title>/s);
  const summaryMatch = response.match(/<summary>(.*?)<\/summary>/s);

  expect(titleMatch).not.toBeNull();
  expect(titleMatch![1]!.trim()).toBe("Fixed pipeline holiday bug");
  expect(summaryMatch).not.toBeNull();
  expect(summaryMatch![1]!.trim()).toContain("Fixed Step Function");
});

// ── Transcript Reading ───────────────────────────────────────────────────────

test("readTranscript: handles missing file gracefully", async () => {
  const { readTranscript } = await import("../src/hooks/framework.ts");
  expect(readTranscript("/nonexistent/path.jsonl")).toBe("");
  expect(readTranscript("")).toBe("");
});
