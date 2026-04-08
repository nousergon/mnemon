/**
 * Contradiction detection — finds and resolves conflicting memories.
 *
 * When a new memory is saved, searches for existing memories on the same
 * topic. Uses the local LLM to classify the relationship:
 *   - same: identical fact, no action
 *   - update: new supersedes old, decay old confidence
 *   - contradiction: direct conflict, decay old confidence more aggressively
 */

import type { Store, SearchResult } from "./store.ts";
import { embed } from "./embedder.ts";
import { generate } from "./llm.ts";

const OVERLAP_THRESHOLD = 0.7; // minimum vector similarity to consider overlapping
const UPDATE_DECAY = 0.15;     // confidence reduction for superseded memories
const CONTRADICTION_DECAY = 0.25;
const CONFIDENCE_FLOOR = 0.2;

const SYSTEM_PROMPT = `You classify the relationship between two memories. Given an existing memory and a new memory, respond with exactly one word:

- same: they express the same fact or decision
- update: the new memory supersedes or refines the old one
- contradiction: they directly conflict
- unrelated: different topics

Respond with ONLY the classification word, nothing else.`;

type Relationship = "same" | "update" | "contradiction" | "unrelated";

/**
 * Check a new memory against existing memories for contradictions.
 * Returns the number of memories whose confidence was decayed.
 */
export async function checkContradictions(
  store: Store,
  newTitle: string,
  newContent: string,
  newDocId: number,
): Promise<{ decayed: number; relationships: Array<{ docId: number; title: string; relationship: Relationship }> }> {
  const relationships: Array<{ docId: number; title: string; relationship: Relationship }> = [];
  let decayed = 0;

  // Find overlapping memories via vector similarity
  let overlapping: SearchResult[] = [];
  try {
    const queryEmb = await embed(`title: ${newTitle} | text: ${newContent}`);
    overlapping = store.searchVector(queryEmb, 5);
  } catch {
    return { decayed: 0, relationships: [] };
  }

  // Filter to genuinely overlapping results (exclude self)
  const candidates = overlapping.filter(
    (r) => r.doc_id !== newDocId && r.score >= OVERLAP_THRESHOLD,
  );

  if (candidates.length === 0) {
    return { decayed: 0, relationships: [] };
  }

  // Classify each relationship via LLM
  for (const candidate of candidates) {
    try {
      const prompt = `Existing memory:\nTitle: ${candidate.title}\nContent: ${candidate.content.slice(0, 500)}\n\nNew memory:\nTitle: ${newTitle}\nContent: ${newContent.slice(0, 500)}`;

      const response = await generate(SYSTEM_PROMPT, prompt, 10);
      const classification = response.trim().toLowerCase() as Relationship;

      if (!["same", "update", "contradiction", "unrelated"].includes(classification)) {
        continue;
      }

      relationships.push({
        docId: candidate.doc_id,
        title: candidate.title,
        relationship: classification,
      });

      // Apply confidence decay
      if (classification === "update") {
        const doc = store.get(candidate.doc_id);
        if (doc) {
          const newConfidence = Math.max(CONFIDENCE_FLOOR, doc.confidence - UPDATE_DECAY);
          store.db.run(
            "UPDATE documents SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
            [newConfidence, candidate.doc_id],
          );
          store.addRelation(newDocId, candidate.doc_id, "supersedes", 0.8);
          decayed++;
        }
      } else if (classification === "contradiction") {
        const doc = store.get(candidate.doc_id);
        if (doc) {
          const newConfidence = Math.max(CONFIDENCE_FLOOR, doc.confidence - CONTRADICTION_DECAY);
          store.db.run(
            "UPDATE documents SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
            [newConfidence, candidate.doc_id],
          );
          store.addRelation(newDocId, candidate.doc_id, "contradicts", 0.9);
          decayed++;
        }
      } else if (classification === "same") {
        store.addRelation(newDocId, candidate.doc_id, "related", 1.0);
      }
    } catch {
      // LLM classification failed for this candidate — skip
      continue;
    }
  }

  return { decayed, relationships };
}

// ── Confidence Decay ────────────────────────────────────────────────────────

const HALF_LIVES: Record<string, number | null> = {
  decision: null,
  preference: null,
  antipattern: null,
  observation: 90,
  research: 90,
  project: 120,
  handoff: 30,
  note: 60,
};

/**
 * Apply time-based confidence decay to all documents.
 * Documents with access activity decay slower (access reinforcement).
 * Returns the number of documents whose confidence was updated.
 */
export function applyConfidenceDecay(store: Store): number {
  let updated = 0;

  for (const [contentType, halfLife] of Object.entries(HALF_LIVES)) {
    if (halfLife === null) continue;

    // Get documents of this type that haven't been invalidated
    const docs = store.db.query<any, [string]>(`
      SELECT id, confidence, access_count, pinned,
             CAST(julianday('now') - julianday(updated_at) AS REAL) AS age_days
      FROM documents
      WHERE content_type = ?
        AND invalidated_at IS NULL
        AND pinned = 0
    `).all(contentType);

    for (const doc of docs) {
      // Access reinforcement: each access extends effective half-life
      // More accesses = slower decay (up to 3x half-life extension)
      const accessMultiplier = Math.min(3.0, 1.0 + doc.access_count * 0.1);
      const effectiveHalfLife = halfLife * accessMultiplier;

      // Exponential decay: confidence * 2^(-age/halflife)
      const decayFactor = Math.pow(2, -doc.age_days / effectiveHalfLife);
      const baseConfidence = store.defaultConfidence(contentType as any);
      const decayedConfidence = Math.max(CONFIDENCE_FLOOR, baseConfidence * decayFactor);

      // Only update if confidence changed meaningfully
      if (Math.abs(decayedConfidence - doc.confidence) > 0.01) {
        store.db.run(
          "UPDATE documents SET confidence = ? WHERE id = ?",
          [decayedConfidence, doc.id],
        );
        updated++;
      }
    }
  }

  return updated;
}
