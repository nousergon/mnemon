"""Contradiction detection — finds and resolves conflicting memories.

When a new memory is saved, searches for existing memories on the
same topic. Uses **NLI** (Natural Language Inference) to classify
the relationship between each candidate pair:

  - ``same``         : semantic equivalence, no action (adds ``related`` relation)
  - ``update``       : new supersedes old, decay old confidence
  - ``contradiction``: direct conflict, decay old confidence more aggressively
  - ``unrelated``    : different topics, no action

Two-stage pipeline:
  1. Cosine similarity gate (``CONTRADICTION_OVERLAP_THRESHOLD``) —
     cheap filter; unrelated pairs never reach the classifier
  2. NLI cross-encoder bidirectional classification — outputs the
     mnemon taxonomy label

Replaces the prior LLM-based classifier (2026-05-22 — see
``private/mnemon-salience-tier-plan-260521.md``) per the standing
"mnemon is LLM-free by design" decision. NLI is the SOTA non-LLM
ML primitive for this exact task; the embedded cross-encoder model
(``cross-encoder/nli-deberta-v3-xsmall``, ~87 MB INT8) ships through
the same FastEmbed-style ONNX path that already powers embeddings —
zero new deps.

Also provides time-based confidence decay with access reinforcement.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from .config import (
    CONTRADICTION_CONTEXT_MAX_CHARS,
    CONTRADICTION_OVERLAP_THRESHOLD,
    DEFAULT_CONFIDENCE,
    HALF_LIVES,
)

if TYPE_CHECKING:
    from .store import Store

logger = logging.getLogger(__name__)

UPDATE_DECAY = 0.15      # confidence reduction for superseded memories
CONTRADICTION_DECAY = 0.25
CONFIDENCE_FLOOR = 0.2

VALID_CLASSIFICATIONS = {"same", "update", "contradiction", "unrelated"}


def check_contradictions(
    store: "Store",
    new_title: str,
    new_content: str,
    new_doc_id: int,
    *,
    dry_run: bool = False,
) -> dict:
    """Check a new memory against existing memories for contradictions.

    Two-stage pipeline:
      1. Vector-similarity gate filters to genuinely overlapping
         candidates (``CONTRADICTION_OVERLAP_THRESHOLD``).
      2. NLI bidirectional classify maps each candidate to mnemon's
         taxonomy (``same`` / ``update`` / ``contradiction`` /
         ``unrelated``).

    Side effects on classification (skipped when ``dry_run=True``):
      - ``update``        : decay old confidence by ``UPDATE_DECAY``,
                            insert ``'supersedes'`` relation
      - ``contradiction`` : decay old confidence by ``CONTRADICTION_DECAY``,
                            insert ``'contradicts'`` relation
      - ``same``          : insert ``'related'`` relation, no decay
      - ``unrelated``     : no action

    Returns:
        {
          "decayed": int,                  # # of confidence decays applied
          "relationships": [
            {
              "doc_id": int,
              "title": str,
              "relationship": "same" | "update" | "contradiction" | "unrelated",
              "probs": {"contradiction": float, "entailment": float, "neutral": float},  # NLI a→b
            },
            ...
          ],
          "nli_unavailable": bool,         # True iff NLI couldn't load (downgrades to cosine-only)
          "dry_run": bool,                 # echoes the input flag
        }
    """
    relationships: list[dict] = []
    decayed = 0

    # Stage 1 — vector similarity gate
    try:
        from .embedder import embed
        query_emb = embed(f"title: {new_title} | text: {new_content}")
        overlapping = store.search_vector(query_emb, 5)
    except Exception as e:
        logger.warning("contradiction: embed/search failed (%s); skipping check", e)
        return {
            "decayed": 0, "relationships": [],
            "nli_unavailable": False, "dry_run": dry_run,
        }

    candidates = [
        r for r in overlapping
        if r.doc_id != new_doc_id and r.score >= CONTRADICTION_OVERLAP_THRESHOLD
    ]

    if not candidates:
        return {
            "decayed": 0, "relationships": [],
            "nli_unavailable": False, "dry_run": dry_run,
        }

    # Stage 2 — NLI classify (bidirectional)
    try:
        from .nli import NLIUnavailableError, classify_pair_bidirectional
    except ImportError as e:
        logger.warning("contradiction: NLI module import failed: %s", e)
        return {
            "decayed": 0, "relationships": [],
            "nli_unavailable": True, "dry_run": dry_run,
        }

    for candidate in candidates:
        try:
            premise = (
                f"title: {candidate.title} | "
                f"text: {candidate.content[:CONTRADICTION_CONTEXT_MAX_CHARS]}"
            )
            hypothesis = (
                f"title: {new_title} | "
                f"text: {new_content[:CONTRADICTION_CONTEXT_MAX_CHARS]}"
            )
            result = classify_pair_bidirectional(premise, hypothesis)
            classification = result.mnemon_label
        except NLIUnavailableError as e:
            # First candidate's NLI failure → bail out entirely with the
            # named-error path; subsequent candidates would fail
            # identically (singleton model load). Surfaces a clear
            # "nli unavailable" flag for the caller to communicate.
            logger.warning("contradiction: NLI unavailable: %s", e)
            return {
                "decayed": decayed,
                "relationships": relationships,
                "nli_unavailable": True,
                "dry_run": dry_run,
            }
        except Exception as e:
            # Per-candidate failure (tokenization edge case, etc.) —
            # log + skip this candidate, continue with others.
            logger.warning(
                "contradiction: classify failed for candidate #%d: %s",
                candidate.doc_id, e,
            )
            continue

        if classification not in VALID_CLASSIFICATIONS:
            logger.warning(
                "contradiction: unexpected classification %r for #%d; skipping",
                classification, candidate.doc_id,
            )
            continue

        relationships.append({
            "doc_id": candidate.doc_id,
            "title": candidate.title,
            "relationship": classification,
            "probs": result.b_implies_a.probs,
        })

        # Side effects — skipped under dry_run
        if dry_run:
            if classification in ("update", "contradiction"):
                decayed += 1  # would-decay count
            continue

        if classification == "update":
            doc = store.get(candidate.doc_id)
            if doc:
                new_confidence = max(CONFIDENCE_FLOOR, doc.confidence - UPDATE_DECAY)
                store.db.execute(
                    "UPDATE documents SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
                    (new_confidence, candidate.doc_id),
                )
                # Salience Phase 2: the new doc "won" — bump its
                # contradiction_win_count so the promotion-signal
                # scorer can identify structurally load-bearing memories
                # (those that regularly demote others).
                store.db.execute(
                    "UPDATE documents SET contradiction_win_count = "
                    "contradiction_win_count + 1 WHERE id = ?",
                    (new_doc_id,),
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
                # Salience Phase 2: see `update` branch above.
                store.db.execute(
                    "UPDATE documents SET contradiction_win_count = "
                    "contradiction_win_count + 1 WHERE id = ?",
                    (new_doc_id,),
                )
                store.db.commit()
                store.add_relation(new_doc_id, candidate.doc_id, "contradicts", 0.9)
                decayed += 1

        elif classification == "same":
            store.add_relation(new_doc_id, candidate.doc_id, "related", 1.0)

    return {
        "decayed": decayed,
        "relationships": relationships,
        "nli_unavailable": False,
        "dry_run": dry_run,
    }


# ── Confidence Decay ────────────────────────────────────────────────────────


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
            # Strict lookup — every ContentType has an explicit mapping in
            # config.DEFAULT_CONFIDENCE, so a KeyError here means someone
            # added an enum value without updating the map. Fail loud
            # instead of silently falling back to a made-up 0.5.
            base_confidence = DEFAULT_CONFIDENCE[content_type]
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
