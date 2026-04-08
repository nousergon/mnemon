/**
 * Session extractor hook — Stop event.
 *
 * Extracts observations from the conversation transcript using a local
 * 1.7B LLM model. Deduplicates against existing memories via vector
 * similarity. Saves new observations to the vault.
 *
 * Runs on session end (Stop hook). Timeout: 30s.
 */

import { Store } from "../store.ts";
import { generate } from "../llm.ts";
import { embed, embedDocument, VECTOR_DIM } from "../embedder.ts";
import { readStdin, readTranscript, type HookInput } from "./framework.ts";

const SYSTEM_PROMPT = `You are a memory extraction assistant. Your job is to extract important observations from a conversation transcript.

For each observation, output an XML block:
<observation>
  <type>decision|preference|observation|antipattern|research|project</type>
  <title>Short descriptive title (max 80 chars)</title>
  <content>2-3 sentences explaining what was learned, decided, or discovered. Include WHY, not just WHAT.</content>
</observation>

Rules:
- Extract 1-5 observations. Only extract what is genuinely worth remembering.
- Skip routine coding tasks (fix typo, run tests, read file) — only capture decisions, insights, preferences, and discoveries.
- Each observation should be self-contained — understandable without the original conversation.
- Use "decision" for architectural choices, "preference" for user workflow habits, "antipattern" for things that failed, "observation" for learned facts, "research" for investigations, "project" for project status/goals.
- If the conversation has nothing worth remembering, output: <none/>`;

interface ExtractedObservation {
  type: string;
  title: string;
  content: string;
}

function parseObservations(response: string): ExtractedObservation[] {
  const observations: ExtractedObservation[] = [];
  const regex = /<observation>\s*<type>(.*?)<\/type>\s*<title>(.*?)<\/title>\s*<content>(.*?)<\/content>\s*<\/observation>/gs;

  let match;
  while ((match = regex.exec(response)) !== null) {
    const type = match[1]!.trim();
    const title = match[2]!.trim();
    const content = match[3]!.trim();

    if (title && content) {
      observations.push({ type, title, content });
    }
  }

  return observations;
}

/**
 * Check if an observation is too similar to existing memories (> 0.92 cosine).
 */
async function isDuplicate(store: Store, title: string, content: string): Promise<boolean> {
  try {
    const queryEmb = await embed(`title: ${title} | text: ${content}`);
    const results = store.searchVector(queryEmb, 3);
    return results.some((r) => r.score > 0.92);
  } catch {
    // If embedding fails, allow the save (false negative is better than lost memory)
    return false;
  }
}

async function handleSessionExtractor(input: HookInput): Promise<void> {
  const transcript = readTranscript(input.transcript_path ?? "", 6000);
  if (!transcript || transcript.length < 100) return;

  // Extract observations via local LLM
  let response: string;
  try {
    response = await generate(SYSTEM_PROMPT, transcript, 2000);
  } catch (err) {
    console.error("mnemon: LLM extraction failed:", err);
    return;
  }

  if (response.includes("<none/>")) return;

  const observations = parseObservations(response);
  if (observations.length === 0) return;

  const store = new Store(undefined, VECTOR_DIM);
  try {
    let saved = 0;

    for (const obs of observations) {
      // Dedup check
      if (await isDuplicate(store, obs.title, obs.content)) {
        console.error(`mnemon: skipping duplicate observation: "${obs.title}"`);
        continue;
      }

      // Save to vault
      const validTypes = ["decision", "preference", "observation", "antipattern", "research", "project"];
      const contentType = validTypes.includes(obs.type) ? obs.type : "observation";

      const docId = store.save({
        title: obs.title,
        content: obs.content,
        content_type: contentType as any,
        source_client: "claude-code-hook",
      });

      // Embed
      const doc = store.get(docId);
      if (doc) {
        await embedDocument(store, doc.hash, obs.title, obs.content);
      }

      saved++;
      console.error(`mnemon: saved [${contentType}] "${obs.title}"`);
    }

    if (saved > 0) {
      console.error(`mnemon: extracted ${saved} observations from session`);
    }
  } finally {
    store.close();
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  try {
    const input = await readStdin();
    await handleSessionExtractor(input);
  } catch (err) {
    console.error("mnemon session-extractor error:", err);
  }
  // Stop hook: output nothing (allow stop)
}

main();
