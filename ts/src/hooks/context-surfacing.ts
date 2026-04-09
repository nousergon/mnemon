/**
 * Context surfacing hook — UserPromptSubmit.
 *
 * Searches the vault for relevant memories and injects them as
 * XML context before Claude processes the prompt.
 *
 * Pipeline:
 *   1. Skip noise (slash commands, greetings, short prompts, duplicates)
 *   2. BM25 + vector search
 *   3. Composite scoring (relevance + recency + confidence)
 *   4. Tiered injection (HOT/WARM/COLD) within 800 token budget
 */

import { Store } from "../store.ts";
import { search, type ScoredResult } from "../search.ts";
import { VECTOR_DIM } from "../embedder.ts";
import {
  readStdin,
  writeOutput,
  isDuplicate,
  isNoise,
  type HookInput,
  type HookOutput,
} from "./framework.ts";

const TOKEN_BUDGET = 800;
const CHARS_PER_TOKEN = 4; // rough approximation
const CHAR_BUDGET = TOKEN_BUDGET * CHARS_PER_TOKEN;

// Tiered thresholds
const HOT_THRESHOLD = 0.15;   // full snippet (300 chars)
const WARM_THRESHOLD = 0.10;  // summary (150 chars)
// Below WARM = COLD (title only)

const HOT_SNIPPET_LEN = 300;
const WARM_SNIPPET_LEN = 150;

function buildContext(results: ScoredResult[]): string {
  if (results.length === 0) return "";

  const lines: string[] = [];
  let charsUsed = 0;

  for (const r of results) {
    let entry: string;

    if (r.composite_score >= HOT_THRESHOLD) {
      const snippet = r.content.slice(0, HOT_SNIPPET_LEN).replace(/\n/g, " ");
      entry = `[${r.content_type}] ${r.title}: ${snippet}${r.content.length > HOT_SNIPPET_LEN ? "..." : ""}`;
    } else if (r.composite_score >= WARM_THRESHOLD) {
      const snippet = r.content.slice(0, WARM_SNIPPET_LEN).replace(/\n/g, " ");
      entry = `[${r.content_type}] ${r.title}: ${snippet}...`;
    } else {
      entry = `[${r.content_type}] ${r.title}`;
    }

    if (charsUsed + entry.length > CHAR_BUDGET) break;

    lines.push(entry);
    charsUsed += entry.length;
  }

  if (lines.length === 0) return "";

  return `<mnemon-context>\nRelevant memories from previous sessions:\n${lines.join("\n")}\n</mnemon-context>`;
}

async function handleContextSurfacing(input: HookInput): Promise<HookOutput | null> {
  const prompt = input.prompt ?? "";

  // Filter noise
  if (isNoise(prompt)) return null;

  // Dedup (same prompt within 10 min window)
  if (isDuplicate(prompt)) return null;

  // Search vault
  const store = new Store(undefined, VECTOR_DIM);
  try {
    const results = await search(store, {
      query: prompt,
      limit: 8,
      useVector: true,
    });

    if (results.length === 0) return null;

    const context = buildContext(results);
    if (!context) return null;

    return {
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: context,
      },
    };
  } finally {
    store.close();
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  try {
    const input = await readStdin();
    const output = await handleContextSurfacing(input);

    if (output) {
      writeOutput(output);
    }
  } catch (err) {
    // Non-blocking — log to stderr, exit 0
    console.error("mnemon context-surfacing error:", err);
  }
}

main();
