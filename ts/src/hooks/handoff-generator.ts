/**
 * Handoff generator hook — Stop event.
 *
 * Generates a session summary for continuity across sessions.
 * Saved as a "handoff" memory with 30-day half-life.
 *
 * Runs on session end (Stop hook). Timeout: 30s.
 */

import { Store } from "../store.ts";
import { generate } from "../llm.ts";
import { embedDocument, VECTOR_DIM } from "../embedder.ts";
import { readStdin, readTranscript, type HookInput } from "./framework.ts";

const SYSTEM_PROMPT = `You are a session summarizer. Given a conversation transcript, produce a brief handoff summary for the next session.

Format your response as:
<handoff>
  <title>Short descriptive title of the session (max 60 chars)</title>
  <summary>
  2-4 bullet points covering:
  - What was accomplished
  - Key decisions made
  - Open questions or unfinished work
  - Files or systems that were modified
  </summary>
</handoff>

Rules:
- Be concise — this is a handoff note, not a full report.
- Focus on what the NEXT session needs to know.
- If the session was trivial (just a question, no real work), output: <none/>`;

interface Handoff {
  title: string;
  summary: string;
}

function parseHandoff(response: string): Handoff | null {
  const titleMatch = response.match(/<title>(.*?)<\/title>/s);
  const summaryMatch = response.match(/<summary>(.*?)<\/summary>/s);

  if (!titleMatch || !summaryMatch) return null;

  const title = titleMatch[1]!.trim();
  const summary = summaryMatch[1]!.trim();

  if (!title || !summary) return null;
  return { title, summary };
}

async function handleHandoffGenerator(input: HookInput): Promise<void> {
  const transcript = readTranscript(input.transcript_path ?? "", 6000);
  if (!transcript || transcript.length < 200) return;

  // Generate handoff via local LLM
  let response: string;
  try {
    response = await generate(SYSTEM_PROMPT, transcript, 500);
  } catch (err) {
    console.error("mnemon: handoff generation failed:", err);
    return;
  }

  if (response.includes("<none/>")) return;

  const handoff = parseHandoff(response);
  if (!handoff) return;

  const store = new Store(undefined, VECTOR_DIM);
  try {
    const docId = store.save({
      title: `Session: ${handoff.title}`,
      content: handoff.summary,
      content_type: "handoff",
      source_client: "claude-code-hook",
    });

    // Embed
    const doc = store.get(docId);
    if (doc) {
      await embedDocument(store, doc.hash, handoff.title, handoff.summary);
    }

    console.error(`mnemon: saved handoff "${handoff.title}"`);
  } finally {
    store.close();
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  try {
    const input = await readStdin();
    await handleHandoffGenerator(input);
  } catch (err) {
    console.error("mnemon handoff-generator error:", err);
  }
}

main();
