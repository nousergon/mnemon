"""scripts/build_standing_set.py — Phase 0 standing-tier scorer (v3).

Embedding-based exemplar-projection scoring for the salience-tier
standing set. Uses mnemon's existing FastEmbed infrastructure
(bge-small-en-v1.5, 384d, ONNX). NO external LLM, NO new dependency.

This is the SOTA pattern for few-shot text classification with
hand-defined class prototypes — anchor texts as positive/negative
exemplars, project candidates against them, score by max cosine.
See: SetFit, prototype networks, anchor-based reranking.

LLM-judge would be a more accurate alternative but is queued as a
future opt-in (private/ROADMAP.md) to keep mnemon's public-release
onboarding dependency-free.

Plan: private/mnemon-salience-tier-plan-260521.md

Scoring (all signals in [0, 1] before weighting):

    constraint_score   max cosine(memory, constraint_exemplars)
                       "does this look like a standing rule?"
    time_penalty       max cosine(memory, time_bounded_exemplars)
                       "is this a time-bounded status update?"
    correction         1.0 if content_type='feedback' AND confidence
                       ≥ 0.85 AND title/content matches correction
                       pattern. Behavioral signal — operator-corrected
                       memories are explicit standing-tier material.
    contradiction      normalized count of incoming relations where
                       memory is winning side of 'contradicts' or
                       'supersedes'. Behavioral signal — memories
                       that displace others on load-bearing facts.
    breadth            normalized count of distinct content_types
                       in top-50 FTS neighbors. Cross-domain proxy.

    combined = 2.0·constraint + 2.0·correction + 1.0·contradiction
             + 0.5·breadth − 1.5·time_penalty

Weights are operator-tunable at the top of this file. Default
auto-selects top-10 (capped at hard ceiling 20 per the plan's
"cap is the contract" invariant). Operator can override via the
companion `scripts/salience_phase0.sh select <IDS>` subcommand.

Outputs (two files, both consumed by the recall hook):

    ~/.mnemon/standing.json          {"ids": [...]}
    ~/.mnemon/standing-rendered.md   pre-fetched rendered content,
                                     so the recall hook reads from
                                     disk (microseconds) instead of
                                     per-prompt HTTP fetch (~5s for
                                     N=10)

Usage:
    .venv/bin/python scripts/build_standing_set.py --db /path/to/snapshot.sqlite
    .venv/bin/python scripts/build_standing_set.py --top 10  # default
    .venv/bin/python scripts/build_standing_set.py --print-only  # don't write standing.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# Add src/ to path so we can use mnemon's embedder + safety modules.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

# ── Tunable exemplars ──────────────────────────────────────────────
#
# Positive exemplars: anchor texts representing the "standing
# constraint" shape. A memory with high max-cosine to any of these
# is structurally like a standing rule.
#
# Negative exemplars: time-bounded status updates. A memory with
# high cosine here gets penalized — even if its content seems
# rule-like, anchoring to date/event markers means it's not durable.

CONSTRAINT_EXEMPLARS = [
    # SOTA / institutional rules
    "default to the SOTA / institutional approach, no shortcuts",
    "use the right primitive, not the smaller diff",
    "the most-correct, most-robust route is also the faster path to production",
    "only deviate from SOTA with an explicit written rationale",
    "lift to library when two or more consumers exist",
    # Verification / discipline
    "always verify before promoting to production",
    "audit before scoping, the codebase is the source of truth",
    "every audit finding becomes a ROADMAP follow-up item",
    "verify branch state before running apply.sh",
    "test reproduction before shipping the static root cause",
    # Failure / error handling
    "fail loud and fast on errors, no silent swallows",
    "any swallow must carry an inline comment naming the failure mode",
    "raise so the failure surfaces at the earliest possible callsite",
    "graceful degrade is forbidden on producer / writer paths",
    # Process / coordination
    "every PR deploy appends to the system-wide changelog automatically",
    "never argmax-route to a per-regime sub-model",
    "use canonical alpha labels with explicit clipping over raw arithmetic returns",
    # Existential constraints
    "runway is not a constraint, optimize for preference not necessity",
    "X is not Y — assert the constraint explicitly",
    "this fact conditions reasoning regardless of query similarity",
    # ── Declarative-posture exemplars (added 2026-05-22 per ROADMAP P1) ──
    # Imperative-shape exemplars above ("never," "always," "must,"
    # "default to") under-weight career / lifestyle / posture constraints
    # that the user encodes declaratively. The 2026-05-22 finding: the
    # auto-selected top-10 against the real vault was 100% engineering
    # rules; career-context memories spanning multi-year load-bearing
    # posture (runway, recruiter posture, start-date framing, search
    # mode) did not surface despite being equally durable. These
    # exemplars represent the declarative shape of the same constraint
    # class — facts stated as if they govern future advice across many
    # domains, but phrased as posture not as imperative.
    "Brian's stance is correct as-is, posture is by design",
    "current preference is to wait, not to push",
    "his stated preference: keep replies minimal, not desperate",
    "passive / selective mode is correct given runway and pipeline",
    "their silence is information; outreach signals desperation",
    "this decouples cash pressure from outreach push timing",
    "lump sum severance through August is in hand, not biweekly",
    "deliberately niche, not chasing scale or virality",
    "the constraint binding decisions is preference, not necessity",
    "soft target for start date preserves negotiating leverage",
]

TIME_BOUNDED_EXEMPLARS = [
    # Session handoffs (these consistently get caught by FTS-breadth noise)
    "Session: proceed with option A",
    "Session: pr merged",
    "Session: ok so you are telling me",
    "Session: i was wondering",
    "Session: i think it may be best",
    # Status updates with date markers
    "today's saturday SF run completed successfully",
    "yesterday's deploy of PR was merged to main",
    "shipped this week as part of the rc18 release",
    "tomorrow's market open at 6:30 AM PT",
    "the 2026-05-21 incident response writeup",
    "scheduled for Wednesday afternoon trigger",
    "merged commit abc123 to main on Friday",
    "this morning's MorningEnrich step in the weekday SF",
    "post-market reconciliation completed for May 21",
    # PR / commit specific
    "PR #143 merged at 22:17:17Z",
    "v0.6.0 tag pushed to origin",
    # Tiny single-thought memories (no constraint, no context)
    "halt the run",
    "propagate",
    "Option 1 or 3",
]

# Operator-tunable correction patterns (existing heuristic, kept).
CORRECTION_PATTERNS = [
    re.compile(r"^\s*(stop|don'?t|never|always)\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\b(correction|corrected|wrong about|i was wrong)\b", re.IGNORECASE),
]

# Weights — tune by inspection of top-N output.
W_CONSTRAINT = 2.0
W_CORRECTION = 2.0
W_CONTRADICTION = 1.0
W_BREADTH = 0.5
W_TIME_PENALTY = 1.5

DEFAULT_TOP_N = 10
HARD_CEILING = 20  # plan invariant: never exceed 20

# Minimum content length for standing-tier consideration. A 2-word
# memory like "halt the run" or "propagate" technically scores via
# breadth (FTS-matches many queries) but carries no actual constraint
# — too thin to condition reasoning. Hard-filter rather than soft-penalty
# because no amount of other signal saves a 2-word memory from being
# noise in a standing-tier context.
DEFAULT_MIN_CONTENT_LENGTH = 50


def _resolve_db(vault_override: str | None, db_override: str | None) -> Path:
    if db_override:
        return Path(db_override)
    if vault_override:
        return Path(vault_override) / "default.sqlite"
    env = os.environ.get("MNEMON_VAULT_DIR")
    if env:
        return Path(env) / "default.sqlite"
    return Path.home() / ".mnemon" / "default.sqlite"


def _resolve_vecstore(db_path: Path) -> Path:
    """The vecstore lives next to the sqlite as <vault>/default.vec.npz."""
    return db_path.with_suffix(".vec.npz")


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalize a score dict to [0, 1]. Returns 0s if all equal."""
    if not scores:
        return scores
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _correction_score(title: str, content: str, content_type: str, confidence: float) -> float:
    if content_type != "feedback" or confidence < 0.85:
        return 0.0
    text = f"{title or ''}\n{content or ''}"
    return 1.0 if any(p.search(text) for p in CORRECTION_PATTERNS) else 0.0


def _contradiction_counts(conn: sqlite3.Connection) -> dict[int, int]:
    rows = conn.execute(
        """
        SELECT source_id, COUNT(*) FROM relations
        WHERE relation_type IN ('contradicts', 'supersedes')
        GROUP BY source_id
        """
    ).fetchall()
    return {sid: n for sid, n in rows}


def _breadth_counts(conn: sqlite3.Connection, doc_ids: list[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for doc_id in doc_ids:
        title_row = conn.execute(
            "SELECT title FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not title_row or not title_row[0]:
            out[doc_id] = 0
            continue
        cleaned = re.sub(r"[\"'*\-+:^]", " ", title_row[0])
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            out[doc_id] = 0
            continue
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT d.content_type
                FROM documents_fts f
                JOIN documents d ON d.id = f.rowid
                WHERE documents_fts MATCH ?
                  AND d.invalidated_at IS NULL
                LIMIT 50
                """,
                (cleaned,),
            ).fetchall()
            out[doc_id] = len(rows)
        except sqlite3.OperationalError:
            out[doc_id] = 0
    return out


def _load_vectors_for_docs(
    vecstore_path: Path, docs: list[dict]
) -> tuple[list[int], np.ndarray]:
    """Load embeddings for the given docs. Returns (doc_ids_with_emb,
    embedding_matrix) — embeddings stored keyed to f'{hash}_0' in mnemon's
    VecStore convention.
    """
    if not vecstore_path.exists():
        print(
            f"WARN: vecstore not found at {vecstore_path} — embedding-based "
            f"signals (constraint/time_penalty) will be zero.",
            file=sys.stderr,
        )
        return [], np.zeros((0, 384), dtype=np.float32)

    data = np.load(str(vecstore_path), allow_pickle=True)
    stored_ids = list(data["ids"])
    stored_vecs = np.asarray(data["vectors"], dtype=np.float32)
    idx_by_vec_id = {vid: i for i, vid in enumerate(stored_ids)}

    doc_ids: list[int] = []
    rows: list[np.ndarray] = []
    for d in docs:
        vec_id = f"{d['hash']}_0"
        idx = idx_by_vec_id.get(vec_id)
        if idx is None:
            continue
        doc_ids.append(d["id"])
        rows.append(stored_vecs[idx])
    if not rows:
        return [], np.zeros((0, 384), dtype=np.float32)
    return doc_ids, np.vstack(rows)


def _cosine_max(memory_vecs: np.ndarray, exemplar_vecs: np.ndarray) -> np.ndarray:
    """For each memory, max cosine sim to any exemplar. Returns shape (N,)."""
    if memory_vecs.shape[0] == 0 or exemplar_vecs.shape[0] == 0:
        return np.zeros(memory_vecs.shape[0], dtype=np.float32)
    # L2-normalize both → cosine = dot product.
    m_norm = memory_vecs / (np.linalg.norm(memory_vecs, axis=1, keepdims=True) + 1e-9)
    e_norm = exemplar_vecs / (np.linalg.norm(exemplar_vecs, axis=1, keepdims=True) + 1e-9)
    sims = m_norm @ e_norm.T  # shape (N_mem, N_exemplar)
    return sims.max(axis=1)


def _render_block(memories: list[dict]) -> str:
    """Render selected memories as the markdown block the recall hook
    will inject. Format mirrors `_format_results` in
    context_surfacing.py for stable token shape."""
    from mnemon.safety import defang_control_markup

    SNIPPET_CHARS = 300
    lines = []
    for m in memories:
        title = defang_control_markup(m.get("title", "") or "")
        content = m.get("content", "") or ""
        ct = m.get("content_type", "note")
        snippet = defang_control_markup(content[:SNIPPET_CHARS])
        ellipsis = "..." if len(content) > SNIPPET_CHARS else ""
        lines.append(
            f"- [{ct}] **{title}** (id={m['id']})\n"
            f"  {snippet}{ellipsis}"
        )
    return "\n\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--db", default=None, help="direct sqlite path")
    ap.add_argument("--vault", default=None, help="vault dir (default: ~/.mnemon)")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                    help=f"how many to auto-select (default: {DEFAULT_TOP_N}, hard ceiling: {HARD_CEILING})")
    ap.add_argument("--print-only", action="store_true",
                    help="print candidates but don't write standing.json / standing-rendered.md")
    ap.add_argument("--show", type=int, default=30,
                    help="how many top-scored candidates to display (default: 30)")
    ap.add_argument("--min-content-length", type=int, default=DEFAULT_MIN_CONTENT_LENGTH,
                    help=f"drop memories whose content is shorter than N chars BEFORE scoring "
                         f"(default: {DEFAULT_MIN_CONTENT_LENGTH}; set to 0 to disable). "
                         f"Filters out tiny noise memories like 'halt the run' / 'propagate' that "
                         f"FTS-match many queries but carry no actual constraint.")
    args = ap.parse_args()

    if args.top > HARD_CEILING:
        print(f"ERROR: --top {args.top} exceeds the hard ceiling of {HARD_CEILING} "
              f"(the cap is the contract — past ~20 standing stops being salient)",
              file=sys.stderr)
        return 2

    db_path = _resolve_db(args.vault, args.db)
    if not db_path.exists():
        print(f"ERROR: sqlite not found: {db_path}", file=sys.stderr)
        return 2

    vecstore_path = _resolve_vecstore(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    all_docs = [dict(r) for r in conn.execute(
        """
        SELECT d.id, d.title, d.content_type, d.confidence, d.hash, d.source_key, c.doc AS content
        FROM documents d
        JOIN content c ON d.hash = c.hash
        WHERE d.invalidated_at IS NULL
        """
    ).fetchall()]

    if not all_docs:
        print(f"vault {db_path} has no live memories", file=sys.stderr)
        return 1

    # Length filter — drop tiny noise BEFORE scoring.
    n_before = len(all_docs)
    min_len = max(0, args.min_content_length)
    if min_len > 0:
        docs = [d for d in all_docs if len(d["content"] or "") >= min_len]
    else:
        docs = all_docs
    n_filtered = n_before - len(docs)

    # Dedup near-duplicate iterations of the same memory. Caught
    # 2026-05-21 when "System-wide deploy changelog" appeared 3x in
    # auto-selected top-10 — same title, content edited iteratively
    # across sessions, three different content_hashes.
    #
    # Dedup key priority:
    #   1. source_key (post-rc16 canonical identity from mnemon.store.save)
    #   2. title (lowercased, stripped)
    #   3. id (untouchable — no-title memories or unique titles)
    #
    # Keep the most recent (highest id) per dedup key — it has the
    # most current title / confidence / content metadata.
    by_key: dict[str, dict] = {}
    for d in docs:
        sk = (d["source_key"] or "").strip() if "source_key" in d.keys() else ""
        title = (d["title"] or "").strip().lower()
        if sk:
            key = f"sk:{sk}"
        elif title:
            key = f"title:{title}"
        else:
            # No source_key, no title — never dedup with another memory.
            key = f"id:{d['id']}"
        existing = by_key.get(key)
        if existing is None or d["id"] > existing["id"]:
            by_key[key] = d
    n_deduped = len(docs) - len(by_key)
    docs = list(by_key.values())

    print(f"# Standing-tier scoring (embedding-based, SOTA non-LLM)", file=sys.stderr)
    print(f"# Vault:    {db_path}", file=sys.stderr)
    print(f"# Vecstore: {vecstore_path}", file=sys.stderr)
    print(
        f"# Live memories: {len(docs)} "
        f"(filtered {n_filtered} below {min_len}-char, "
        f"deduped {n_deduped} content-hash duplicates)",
        file=sys.stderr,
    )

    # Embedding-based signals
    print(f"# Embedding exemplars + memories ...", file=sys.stderr)
    from mnemon.embedder import embed_batch

    constraint_emb = np.vstack(embed_batch(CONSTRAINT_EXEMPLARS))
    time_emb = np.vstack(embed_batch(TIME_BOUNDED_EXEMPLARS))

    embedded_ids, memory_vecs = _load_vectors_for_docs(vecstore_path, docs)
    constraint_raw = dict(zip(embedded_ids, _cosine_max(memory_vecs, constraint_emb).tolist()))
    time_raw = dict(zip(embedded_ids, _cosine_max(memory_vecs, time_emb).tolist()))

    # Behavioral + breadth signals (cheap, all docs)
    correction = {
        d["id"]: _correction_score(d["title"], d["content"], d["content_type"], d["confidence"])
        for d in docs
    }
    contradiction_raw = _contradiction_counts(conn)
    breadth_raw = _breadth_counts(conn, [d["id"] for d in docs])

    # Normalize all to [0, 1]
    constraint = _normalize(constraint_raw)
    time_penalty = _normalize(time_raw)
    contradiction = _normalize({k: float(v) for k, v in contradiction_raw.items()})
    breadth = _normalize({k: float(v) for k, v in breadth_raw.items()})

    combined: dict[int, dict] = {}
    for d in docs:
        doc_id = d["id"]
        c = constraint.get(doc_id, 0.0)
        cor = correction.get(doc_id, 0.0)
        x = contradiction.get(doc_id, 0.0)
        b = breadth.get(doc_id, 0.0)
        tp = time_penalty.get(doc_id, 0.0)
        score = (
            W_CONSTRAINT * c
            + W_CORRECTION * cor
            + W_CONTRADICTION * x
            + W_BREADTH * b
            - W_TIME_PENALTY * tp
        )
        combined[doc_id] = {
            "score": score, "c": c, "cor": cor, "x": x, "b": b, "tp": tp,
        }

    by_id = {d["id"]: d for d in docs}
    ranked = sorted(combined.items(), key=lambda kv: kv[1]["score"], reverse=True)

    # Display top-show
    print(f"\n# Top-{args.show} candidates (by combined score)")
    print(f"# Format: [id] score=X.XX (constraint=C, correction=R, contra=X, breadth=B, time=T)")
    print(f"#         type confidence | title | snippet\n")
    for rank, (doc_id, sc) in enumerate(ranked[: args.show], 1):
        d = by_id[doc_id]
        title = d["title"] or "(no title)"
        snippet = (d["content"] or "")[:200].replace("\n", " ")
        marker = "★" if rank <= args.top else " "
        print(
            f"{marker} [{doc_id:>5}] score={sc['score']:6.3f} "
            f"(c={sc['c']:.2f}, cor={sc['cor']:.2f}, x={sc['x']:.2f}, "
            f"b={sc['b']:.2f}, tp={sc['tp']:.2f}) "
            f"{d['content_type']} conf={d['confidence']:.2f}"
        )
        print(f"        {title}")
        print(f"        {snippet}{'...' if len(d['content'] or '') > 200 else ''}\n")

    # Auto-select top-N and write outputs
    selected = [doc_id for doc_id, _ in ranked[: args.top]]
    selected_memories = [by_id[i] for i in selected]

    if args.print_only:
        print(f"\n# --print-only set; not writing standing.json", file=sys.stderr)
        return 0

    mnemon_dir = Path.home() / ".mnemon"
    mnemon_dir.mkdir(parents=True, exist_ok=True)
    standing_json = mnemon_dir / "standing.json"
    standing_md = mnemon_dir / "standing-rendered.md"

    standing_json.write_text(json.dumps({"ids": selected}, indent=2) + "\n")
    standing_md.write_text(_render_block(selected_memories) + "\n")

    print(f"\n# Wrote:")
    print(f"#   {standing_json} ({len(selected)} IDs)")
    print(f"#   {standing_md} (pre-rendered content for fast recall)")
    print(f"# Activate: export MNEMON_STANDING_TIER_FILE={standing_json}")
    print(f"# Override: scripts/salience_phase0.sh select <id1,id2,...>")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
