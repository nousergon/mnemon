"""Contradiction detection — finds and resolves conflicting memories.

When a new memory is saved, searches for existing memories on the same
topic. Uses the local LLM to classify the relationship:
  - same: identical fact, no action (adds "related" relation)
  - update: new supersedes old, decay old confidence
  - contradiction: direct conflict, decay old confidence more aggressively
  - unrelated: different topics, no action

Also provides time-based confidence decay with access reinforcement.

Phase 3: LLM-based contradiction detection + confidence decay.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import SearchResult, Store

OVERLAP_THRESHOLD = 0.7  # minimum vector similarity to consider overlapping
UPDATE_DECAY = 0.15      # confidence reduction for superseded memories
CONTRADICTION_DECAY = 0.25
CONFIDENCE_FLOOR = 0.2

CLASSIFY_SYSTEM_PROMPT = (
    "You classify the relationship between two memories. "
    "Given an existing memory and a new memory, respond with exactly one word:\n\n"
    "- same: they express the same fact or decision\n"
    "- update: the new memory supersedes or refines the old one\n"
    "- contradiction: they directly conflict\n"
    "- unrelated: different topics\n\n"
    "Respond with ONLY the classification word, nothing else."
)

VALID_CLASSIFICATIONS = {"same", "update", "contradiction", "unrelated"}


def check_contradictions(
    store: "Store",
    new_title: str,
    new_content: str,
    new_doc_id: int,
) -> dict:
    """Check a new memory against existing memories for contradictions.

    Returns {decayed: int, relationships: [{doc_id, title, relationship}]}.
    """
    relationships: list[dict] = []
    decayed = 0

    # Find overlapping memories via vector similarity
    try:
        from .embedder import embed
        query_emb = embed(f"title: {new_title} | text: {new_content}")
        overlapping = store.search_vector(query_emb, 5)
    except Exception:
        return {"decayed": 0, "relationships": []}

    # Filter to genuinely overlapping results (exclude self)
    candidates = [
        r for r in overlapping
        if r.doc_id != new_doc_id and r.score >= OVERLAP_THRESHOLD
    ]

    if not candidates:
        return {"decayed": 0, "relationships": []}

    # Classify each relationship via LLM
    try:
        from .llm import generate
    except ImportError:
        return {"decayed": 0, "relationships": []}

    for candidate in candidates:
        try:
            prompt = (
                f"Existing memory:\nTitle: {candidate.title}\n"
                f"Content: {candidate.content[:500]}\n\n"
                f"New memory:\nTitle: {new_title}\n"
                f"Content: {new_content[:500]}"
            )

            response = generate(CLASSIFY_SYSTEM_PROMPT, prompt, max_tokens=10)
            classification = response.strip().lower()

            if classification not in VALID_CLASSIFICATIONS:
                continue

            relationships.append({
                "doc_id": candidate.doc_id,
                "title": candidate.title,
                "relationship": classification,
            })

            # Apply confidence decay
            if classification == "update":
                doc = store.get(candidate.doc_id)
                if doc:
                    new_confidence = max(CONFIDENCE_FLOOR, doc.confidence - UPDATE_DECAY)
                    store.db.execute(
                        "UPDATE documents SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
                        (new_confidence, candidate.doc_id),
                    )
                    store.db.commit()
                    store.add_relation(new_doc_id, candidate.doc_id, "supersedes", 0.8)
                    decayed += 1

            elif classification == "contradiction":
                doc = store.get(candidate.doc_id)
                if doc:
                    new_confidence = max(CONFIDENCE_FLOOR, doc.confidence - CONTRADICTION_DECAY)
                    store.db.execute(
                        "UPDATE documents SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
                        (new_confidence, candidate.doc_id),
                    )
                    store.db.commit()
                    store.add_relation(new_doc_id, candidate.doc_id, "contradicts", 0.9)
                    decayed += 1

            elif classification == "same":
                store.add_relation(new_doc_id, candidate.doc_id, "related", 1.0)

        except Exception:
            continue

    return {"decayed": decayed, "relationships": relationships}


# ── Confidence Decay ────────────────────────────────────────────────────────

from .config import DEFAULT_CONFIDENCE, HALF_LIVES  # noqa: E402


def apply_confidence_decay(store: "Store") -> int:
    """Apply time-based confidence decay to all documents.

    Documents with access activity decay slower (access reinforcement).
    Each access extends the effective half-life by 10%, up to 3x.

    Returns the number of documents whose confidence was updated.
    """
    updated = 0

    for content_type, half_life in HALF_LIVES.items():
        if half_life is None:
            continue

        rows = store.db.execute(
            """SELECT id, confidence, access_count, pinned,
                      CAST(julianday('now') - julianday(updated_at) AS REAL) AS age_days
               FROM documents
               WHERE content_type = ?
                 AND invalidated_at IS NULL
                 AND pinned = 0""",
            (content_type.value,),
        ).fetchall()

        for row in rows:
            # Access reinforcement: each access extends effective half-life
            # More accesses = slower decay (up to 3x half-life extension)
            access_multiplier = min(3.0, 1.0 + row["access_count"] * 0.1)
            effective_half_life = half_life * access_multiplier

            # Exponential decay: base_confidence * 2^(-age/halflife)
            decay_factor = math.pow(2, -row["age_days"] / effective_half_life)
            base_confidence = DEFAULT_CONFIDENCE.get(content_type, 0.5)
            decayed_confidence = max(CONFIDENCE_FLOOR, base_confidence * decay_factor)

            # Only update if confidence changed meaningfully
            if abs(decayed_confidence - row["confidence"]) > 0.01:
                store.db.execute(
                    "UPDATE documents SET confidence = ? WHERE id = ?",
                    (decayed_confidence, row["id"]),
                )
                updated += 1

    if updated > 0:
        store.db.commit()

    return updated
