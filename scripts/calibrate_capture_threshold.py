#!/usr/bin/env python3
"""Calibrate ``CAPTURE_ATTENTION_THRESHOLD`` against the operator's vault.

Plan: ``private/mnemon-capture-attention-plan-260522.md`` §Calibration.

The default threshold of 0.85 is a conservative starting point. Real
vault content has its own embedding distribution, and the precision-
recall sweet spot moves accordingly. This script:

1. Samples N random pairs of live memories from a vault snapshot
2. Computes cosine similarity for each pair (using the in-store
   indexed vectors — no re-embedding required)
3. Prompts the operator to tag each as same-assertion / different /
   unclear
4. Persists tagged pairs to ``tests/fixtures/capture_attention_pairs.json``
   (regression-locking fixture, consumed by test_capture_attention.py)
5. Computes precision-recall at thresholds {0.70, 0.75, 0.80, 0.85,
   0.90}
6. Recommends the threshold at the precision-leaning sweet spot
   (highest precision with recall ≥ 0.70)

Usage:
    python scripts/calibrate_capture_threshold.py --db <vault.sqlite> [--n 20]

Defaults to the prod-snapshot path used by ``salience_phase0.sh``.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "capture_attention_pairs.json"
DEFAULT_DB = "/tmp/mnemon-prod-snap.sqlite"
DEFAULT_N = 20
THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90)


def _load_pairs(db_path: Path, n: int) -> list[dict]:
    """Sample N near-neighbor memory pairs from the threshold decision region.

    The naive uniform-random sample over a 2510-memory vault produces pairs
    whose cosines cluster at 0.1–0.4 (clearly-different topics) — operator
    verdicts on those pairs carry no information about whether the
    CAPTURE_ATTENTION_THRESHOLD cut should be 0.80 or 0.85. Every pair
    a calibration operator tags should sit in the decision region (cosine
    near the candidate thresholds).

    Strategy: pick a random anchor, take its top non-self neighbor via
    vector search, and accept the pair if cosine ≥ ``MIN_PAIR_COSINE``.
    Repeat until ``n`` pairs are collected or the search budget is
    exhausted. Bias toward higher-cosine pairs is the desired calibration
    behavior — the threshold lives in the high-cosine tail.
    """
    import numpy as np

    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from mnemon.vecstore import VecStore

    MIN_PAIR_COSINE = 0.55  # well below the lowest calibration threshold (0.70)
                            # — preserves edge-negatives the threshold should NOT flag
    MAX_ATTEMPTS = max(n * 20, 200)  # generous budget for the rejection loop

    vec_path = str(db_path).replace(".sqlite", ".vec")
    if not Path(vec_path + ".npz").exists():
        sys.exit(
            f"ERROR: vec store not found at {vec_path}.npz — "
            "snapshot must include vectors. Run "
            "scripts/salience_phase0.sh snapshot first."
        )

    vs = VecStore(vec_path, dim=384)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # Pull live document ids + their content_hash. We compare via
    # the indexed full-document fragment (seq=0).
    rows = db.execute(
        """SELECT id, title, hash
           FROM documents
           WHERE invalidated_at IS NULL
           ORDER BY id"""
    ).fetchall()

    # Build hash → (id, title, embedding) map (seq=0 only — full-doc fragment).
    by_hash: dict[str, dict] = {}
    for r in rows:
        vec_id = f"{r['hash']}_0"
        vec = vs.get(vec_id)
        if vec is not None:
            by_hash[r["hash"]] = {"id": r["id"], "title": r["title"], "vec": vec}

    if len(by_hash) < n + 5:
        sys.exit(
            f"ERROR: only {len(by_hash)} eligible memories in vault "
            f"(need at least {n + 5} for {n} near-neighbor pairs)"
        )

    random.seed(42)
    candidate_hashes = list(by_hash.keys())
    pairs: list[dict] = []
    seen_pair_keys: set[tuple[str, str]] = set()
    attempts = 0

    while len(pairs) < n and attempts < MAX_ATTEMPTS:
        attempts += 1
        anchor_hash = random.choice(candidate_hashes)
        anchor = by_hash[anchor_hash]
        # k=3 → first hit is the anchor itself (cosine 1.0); take the next
        # distinct hash. Occasionally a vault has duplicate-content fragments
        # so we filter by hash, not just rank.
        results = vs.search(anchor["vec"], k=3)
        neighbor = None
        for res in results:
            res_hash = res["id"].rsplit("_", 1)[0]
            if res_hash == anchor_hash or res_hash not in by_hash:
                continue
            neighbor = (res_hash, float(res["similarity"]))
            break
        if neighbor is None:
            continue
        nhash, cos = neighbor
        if cos < MIN_PAIR_COSINE:
            continue
        pair_key = tuple(sorted([anchor_hash, nhash]))
        if pair_key in seen_pair_keys:
            continue
        seen_pair_keys.add(pair_key)

        a, b = by_hash[anchor_hash], by_hash[nhash]
        ac = db.execute("SELECT doc FROM content WHERE hash = ?", (anchor_hash,)).fetchone()
        bc = db.execute("SELECT doc FROM content WHERE hash = ?", (nhash,)).fetchone()
        pairs.append({
            "id_a": a["id"], "id_b": b["id"],
            "title_a": a["title"], "title_b": b["title"],
            "snippet_a": (ac["doc"] if ac else "")[:200],
            "snippet_b": (bc["doc"] if bc else "")[:200],
            "cosine": cos,
        })

    db.close()
    if len(pairs) < n:
        print(
            f"WARNING: only {len(pairs)} pairs found above cosine "
            f"{MIN_PAIR_COSINE} in {attempts} attempts — vault may lack "
            f"semantic clusters. Proceeding with what we have."
        )
    # Sort by cosine descending so the operator tags high-confidence
    # near-dupes first (catches the calibration intuition early).
    pairs.sort(key=lambda p: -p["cosine"])
    return pairs


def _prompt_operator(pairs: list[dict]) -> list[dict]:
    """Interactive tagging loop. Operator marks each pair."""
    print(f"\nTagging {len(pairs)} pairs. For each: same / different / unclear.")
    print("Type 's' (same), 'd' (different), 'u' (unclear), or 'q' to quit.\n")

    tagged = []
    for i, p in enumerate(pairs, 1):
        print(f"━━━ Pair {i}/{len(pairs)}  (cosine={p['cosine']:.3f}) ━━━")
        print(f"  A ({p['id_a']}): {p['title_a']}")
        print(f"     {p['snippet_a']!r}")
        print(f"  B ({p['id_b']}): {p['title_b']}")
        print(f"     {p['snippet_b']!r}")
        while True:
            verdict = input("  same/different/unclear [s/d/u/q]: ").strip().lower()
            if verdict in {"s", "same"}:
                p["verdict"] = "same"
                break
            elif verdict in {"d", "different"}:
                p["verdict"] = "different"
                break
            elif verdict in {"u", "unclear"}:
                p["verdict"] = "unclear"
                break
            elif verdict in {"q", "quit"}:
                print("Quitting — saving partial results")
                return tagged
            else:
                print("  → please enter s, d, u, or q")
        tagged.append(p)
    return tagged


def _precision_recall(tagged: list[dict], threshold: float) -> tuple[float, float]:
    """Compute (precision, recall) at a given cosine threshold.

    Precision = of pairs the threshold flags as same, what fraction were
                operator-tagged 'same'?
    Recall    = of operator-tagged 'same' pairs, what fraction did the
                threshold flag?
    'unclear' pairs are excluded from both numerator and denominator.
    """
    relevant = [p for p in tagged if p["verdict"] in ("same", "different")]
    if not relevant:
        return 0.0, 0.0

    flagged = [p for p in relevant if p["cosine"] >= threshold]
    true_positives = sum(1 for p in flagged if p["verdict"] == "same")
    all_positives = sum(1 for p in relevant if p["verdict"] == "same")

    precision = (true_positives / len(flagged)) if flagged else 1.0
    recall = (true_positives / all_positives) if all_positives else 0.0
    return precision, recall


def _recommend(tagged: list[dict]) -> tuple[float, dict]:
    """Pick the precision-leaning threshold: highest precision with
    recall ≥ 0.70."""
    table = {}
    for t in THRESHOLDS:
        p, r = _precision_recall(tagged, t)
        table[t] = {"precision": p, "recall": r}

    # Precision-leaning sweet spot
    eligible = [(t, m) for t, m in table.items() if m["recall"] >= 0.70]
    if not eligible:
        # No threshold meets the recall floor — fall back to highest recall
        recommended = max(table.items(), key=lambda kv: kv[1]["recall"])[0]
    else:
        recommended = max(eligible, key=lambda kv: kv[1]["precision"])[0]
    return recommended, table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB,
                        help=f"vault snapshot path (default {DEFAULT_DB})")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"number of pairs to tag (default {DEFAULT_N})")
    parser.add_argument("--use-fixture", action="store_true",
                        help="recompute PR table from existing fixture, skip tagging")
    args = parser.parse_args()

    if args.use_fixture:
        if not FIXTURE_PATH.exists():
            sys.exit(f"no fixture at {FIXTURE_PATH} — drop --use-fixture")
        tagged = json.loads(FIXTURE_PATH.read_text())
    else:
        db_path = Path(args.db)
        if not db_path.exists():
            sys.exit(f"vault snapshot not found at {db_path}")
        pairs = _load_pairs(db_path, args.n)
        tagged = _prompt_operator(pairs)
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(json.dumps(tagged, indent=2))
        print(f"\nFixture written: {FIXTURE_PATH}")

    recommended, table = _recommend(tagged)
    print("\n━━━ Precision–Recall by threshold ━━━")
    print(f"  {'threshold':>10}  {'precision':>10}  {'recall':>8}")
    for t, m in table.items():
        marker = "  ←" if t == recommended else ""
        print(f"  {t:>10.2f}  {m['precision']:>10.3f}  {m['recall']:>8.3f}{marker}")
    print(f"\nRecommended CAPTURE_ATTENTION_THRESHOLD = {recommended}")
    print("(precision-leaning: highest precision with recall ≥ 0.70)")
    print("\nIf this differs from src/mnemon/config.py, edit and re-soak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
