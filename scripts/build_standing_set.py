"""scripts/build_standing_set.py — Phase 0 of the salience-tier plan.

Score every live memory on cheap signals available today and print the
top-N candidates for *manual* standing-tier curation. The operator
inspects the candidates and hand-writes the selected IDs to
`~/.mnemon/standing.json` (no auto-selection — the cap is the contract,
and Phase 0 is hypothesis validation, not automation).

Phase 0 plan: private/mnemon-salience-tier-plan-260521.md
ROADMAP: "Salience tier — standing-context recall (added 2026-05-21)"

Scoring:
  correction_score   — 1.0 if content_type='feedback' AND confidence ≥ 0.85
                       AND title or content matches a correction pattern.
                       Operator-tunable patterns below.
  contradiction_score — count of relations where the memory is the
                       'winning' (source) side of a 'contradicts' or
                       'supersedes' relation. Mnemon's contradiction
                       classifier (contradiction.py) emits these when a
                       new save resolves vs a prior conflicting memory.
  breadth_score      — distinct content_types among the memory's top-50
                       FTS neighbors. Crude proxy for "this fact
                       conditions reasoning across many query types."

  combined = 2.0·correction + 1.0·contradiction + 0.5·breadth
  (weights provisional per the plan — tune by inspection)

Usage:
    .venv/bin/python scripts/build_standing_set.py
    .venv/bin/python scripts/build_standing_set.py --top 30  # default
    .venv/bin/python scripts/build_standing_set.py --top 50 --vault /alt/path
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable


# Patterns that mark a memory as a correction. Operator-tunable —
# the plan flags these as "small set" and asks for tuning by inspection.
CORRECTION_PATTERNS = [
    re.compile(r"^\s*(stop|don'?t|never|always)\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\b(correction|corrected|wrong about|i was wrong)\b", re.IGNORECASE),
]


def _resolve_db(vault_override: str | None, db_override: str | None) -> Path:
    """Resolve the sqlite path to score.

    Precedence:
      --db <path>           direct file path (highest)
      --vault <dir>         vault-dir → <dir>/default.sqlite
      MNEMON_VAULT_DIR env  env-dir   → <env>/default.sqlite
      ~/.mnemon             default   → ~/.mnemon/default.sqlite
    """
    if db_override:
        return Path(db_override)
    if vault_override:
        return Path(vault_override) / "default.sqlite"
    env = os.environ.get("MNEMON_VAULT_DIR")
    if env:
        return Path(env) / "default.sqlite"
    return Path.home() / ".mnemon" / "default.sqlite"


def _correction_score(title: str, content: str, content_type: str, confidence: float) -> float:
    """Return 1.0 if the memory shape matches a correction, else 0.0."""
    if content_type != "feedback":
        return 0.0
    if confidence < 0.85:
        return 0.0
    text = f"{title or ''}\n{content or ''}"
    for pat in CORRECTION_PATTERNS:
        if pat.search(text):
            return 1.0
    return 0.0


def _contradiction_scores(conn: sqlite3.Connection) -> dict[int, int]:
    """For each memory, count incoming relations where it's the 'winning'
    side of a 'contradicts' or 'supersedes' relation.

    Mnemon's contradiction classifier emits:
      - source_id 'contradicts' target_id  → source won
      - source_id 'supersedes' target_id   → source is the newer/correct version
    (See src/mnemon/contradiction.py)
    """
    out: dict[int, int] = {}
    rows = conn.execute(
        """
        SELECT source_id, COUNT(*) AS n
        FROM relations
        WHERE relation_type IN ('contradicts', 'supersedes')
        GROUP BY source_id
        """
    ).fetchall()
    for source_id, n in rows:
        out[source_id] = n
    return out


def _breadth_scores(conn: sqlite3.Connection, doc_ids: Iterable[int]) -> dict[int, int]:
    """For each memory, count distinct content_types among its top-50
    FTS neighbors (using the memory's title as the query).

    Cheap proxy for cross-domain applicability — facts that touch many
    types are more "constraint-like" than narrow single-context facts.
    FTS (BM25 over the FTS5 index) rather than vector similarity to
    avoid an O(N²) cosine pass on the full vault.
    """
    out: dict[int, int] = {}
    for doc_id in doc_ids:
        title_row = conn.execute(
            "SELECT title FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not title_row or not title_row[0]:
            out[doc_id] = 0
            continue
        title = title_row[0]
        # Strip FTS-special characters that would error the MATCH.
        # Single quotes need doubling for SQL parameter substitution
        # but we use parametric, so that's safe; we DO need to strip
        # FTS operators (* " + - etc) and column qualifiers.
        cleaned = re.sub(r"[\"'*\-+:^]", " ", title)
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
            # FTS MATCH error (e.g. malformed query) — skip this doc's breadth.
            out[doc_id] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--top",
        type=int,
        default=30,
        help="how many top-scored candidates to print (default: 30)",
    )
    ap.add_argument(
        "--vault",
        default=None,
        help="vault dir (overrides MNEMON_VAULT_DIR / ~/.mnemon)",
    )
    ap.add_argument(
        "--db",
        default=None,
        help="direct sqlite path (overrides --vault and env). Use for "
             "scoring a snapshot of the prod Fly vault.",
    )
    args = ap.parse_args()

    db_path = _resolve_db(args.vault, args.db)
    if not db_path.exists():
        print(f"ERROR: sqlite not found: {db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Pull live memories with content.
    docs = conn.execute(
        """
        SELECT d.id, d.title, d.content_type, d.confidence, c.doc AS content
        FROM documents d
        JOIN content c ON d.hash = c.hash
        WHERE d.invalidated_at IS NULL
        """
    ).fetchall()

    if not docs:
        print(f"vault {db_path} has no live memories", file=sys.stderr)
        return 1

    print(f"# Standing-tier candidate scoring", file=sys.stderr)
    print(f"# Vault: {db_path}", file=sys.stderr)
    print(f"# Live memories: {len(docs)}", file=sys.stderr)

    correction = {
        d["id"]: _correction_score(d["title"], d["content"], d["content_type"], d["confidence"])
        for d in docs
    }
    contradiction = _contradiction_scores(conn)
    breadth = _breadth_scores(conn, [d["id"] for d in docs])

    n_correction = sum(1 for v in correction.values() if v > 0)
    n_contradiction = sum(1 for v in contradiction.values() if v > 0)
    print(f"# Correction-flagged: {n_correction}", file=sys.stderr)
    print(f"# Contradiction-winning: {n_contradiction}", file=sys.stderr)
    print(f"# Mean breadth: {sum(breadth.values()) / max(1, len(breadth)):.1f}", file=sys.stderr)
    print(file=sys.stderr)

    combined: dict[int, tuple[float, float, float, float]] = {}
    for d in docs:
        doc_id = d["id"]
        c = correction.get(doc_id, 0.0)
        x = float(contradiction.get(doc_id, 0))
        b = float(breadth.get(doc_id, 0))
        score = 2.0 * c + 1.0 * x + 0.5 * b
        combined[doc_id] = (score, c, x, b)

    by_id = {d["id"]: d for d in docs}
    top = sorted(combined.items(), key=lambda kv: kv[1][0], reverse=True)[: args.top]

    print(f"# Top-{args.top} candidates (rank by combined score)")
    print(f"# Format: [id] score=X.XX (corr=C, contra=X, breadth=B) type confidence")
    print(f"#         title")
    print(f"#         content snippet")
    print(f"#")
    print(f"# Operator: pick N≤20 IDs by hand, write to ~/.mnemon/standing.json as:")
    print(f"#     {{\"ids\": [<id1>, <id2>, ...]}}")
    print(f"# Then set MNEMON_STANDING_TIER_FILE=~/.mnemon/standing.json in your shell")
    print(f"# (or per claude-code session config).")
    print()

    for rank, (doc_id, (score, c, x, b)) in enumerate(top, 1):
        d = by_id[doc_id]
        title = d["title"] or "(no title)"
        ct = d["content_type"]
        conf = d["confidence"]
        snippet = (d["content"] or "")[:240].replace("\n", " ")
        print(
            f"[{doc_id:>5}] score={score:5.2f} "
            f"(corr={c:.1f}, contra={int(x)}, breadth={int(b)}) "
            f"{ct} conf={conf:.2f}"
        )
        print(f"        {title}")
        print(f"        {snippet}{'...' if len(d['content'] or '') > 240 else ''}")
        print()

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
