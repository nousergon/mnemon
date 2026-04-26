"""Stress benchmark for hybrid BM25 + vector search at scale.

Saves N synthetic memories (default 1000) into a throwaway vault,
embeds them all, then runs M queries (default 50) against the search
pipeline. Reports p50 / p95 / p99 wall-clock latency.

Run from a clone:

    python bench/search_stress.py
    python bench/search_stress.py --memories 5000 --queries 200
    python bench/search_stress.py --json results.json

The vault uses a deterministic seed so the corpus is reproducible
across runs — re-running locally won't shift numbers from corpus
randomness, only from machine variance.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
import time
from pathlib import Path

# Throwaway vault — never touches user's real ~/.mnemon. Must be set
# before importing Store.
EXAMPLE_VAULT = Path(tempfile.mkdtemp(prefix="mnemon-bench-"))
os.environ["MNEMON_VAULT_DIR"] = str(EXAMPLE_VAULT)

from mnemon.search import search  # noqa: E402
from mnemon.store import Store, _sha256  # noqa: E402

# Synthetic-memory templates. Each template has 3-4 placeholder slots
# (e.g. {service}, {action}). Filling each slot with a random choice
# from the corresponding pool generates plausible-looking domain
# memories — good enough for vector embeddings to differentiate.
TEMPLATES = [
    "The {service} pipeline {action} on {failure_type} errors. "
    "{action_target} are escalated to {escalation}.",
    "We chose {component} over {alternative} for the {use_case} "
    "layer because {reason}.",
    "Cold-start latency for {service} on {platform}: {duration}. "
    "Acceptable for {tier} tier; documented in {doc}.",
    "The {team} team owns the {service} {component}. "
    "Pages route via {channel} between {start_time} and {end_time}.",
    "Migration {migration_id} adds a {column_type} column to "
    "{table}. Backfill strategy: {strategy}. ETA: {eta}.",
    "{service} returns {status_code} when {condition}. Retry policy: "
    "{retry_policy}. Circuit breaker opens after {threshold}.",
    "Decision: {decision}. Trade-off: {tradeoff}. Reversal trigger: "
    "{trigger}.",
    "Bug {bug_id}: {symptom} when {trigger_condition}. Root cause: "
    "{root_cause}. Fix: {fix}. Filed by {reporter}.",
]

POOLS = {
    "service": ["api", "ingest", "scheduler", "billing", "search", "auth", "dispatch", "rendering"],
    "action": ["retries", "fails fast", "queues", "logs", "alerts", "circuit-breaks"],
    "failure_type": ["transient", "permanent", "network", "timeout", "5xx", "auth"],
    "action_target": ["Permanent errors", "Network errors", "Auth failures", "Timeouts"],
    "escalation": ["the on-call", "Slack #incidents", "PagerDuty", "the #ops channel"],
    "component": ["SQLite", "Postgres", "Redis", "DuckDB", "Cassandra", "FoundationDB"],
    "alternative": ["Postgres", "Redis", "MySQL", "DynamoDB", "Mongo", "Cassandra"],
    "use_case": ["caching", "session", "vault", "queue", "feature flag", "rate limit"],
    "reason": ["zero ops", "ACID", "single file", "low latency", "predictable cost"],
    "platform": ["Fly.io", "Render", "AWS Lambda", "Cloudflare Workers", "GCP Cloud Run"],
    "duration": ["5-7s", "<1s", "30s p99", "200ms p50, 2s p99", "negligible"],
    "tier": ["hobby", "pro", "enterprise", "free", "internal"],
    "doc": ["README", "RUNBOOK", "ROADMAP", "ARCHITECTURE.md"],
    "team": ["platform", "infra", "growth", "billing", "search"],
    "channel": ["PagerDuty", "Slack", "OpsGenie", "Discord"],
    "start_time": ["09:00 UTC", "midnight UTC", "06:00 PT", "14:00 ET"],
    "end_time": ["21:00 UTC", "noon UTC", "18:00 PT", "23:00 ET"],
    "migration_id": ["0042", "0117", "0203", "0388"],
    "column_type": ["NOT NULL bigint", "indexed varchar", "JSONB", "timestamptz"],
    "table": ["users", "events", "orders", "memories", "sessions"],
    "strategy": ["lazy", "background backfill", "blue-green", "online with shadow reads"],
    "eta": ["next sprint", "Q3", "this week", "blocked on ops review"],
    "status_code": ["503", "429", "504", "502", "499"],
    "condition": ["the upstream is unavailable", "rate limit hit", "request times out"],
    "retry_policy": ["3 retries with exponential backoff", "no retry", "1 retry only"],
    "threshold": ["50% error rate over 30s", "10 consecutive failures", "5 timeouts"],
    "decision": [
        "ship the simple version first", "reject the elastic plan",
        "split into two services", "merge into a monorepo",
    ],
    "tradeoff": [
        "more ops complexity short-term", "harder to roll back",
        "loses some flexibility", "ties us to one vendor",
    ],
    "trigger": [
        "p95 latency over 1s", "monthly cost above $500",
        "team grows past 3 engineers", "second customer asks for it",
    ],
    "bug_id": ["#1042", "#2117", "#3088", "#4221"],
    "symptom": ["search returns empty", "duplicate memories", "embedding fails", "auth 401"],
    "trigger_condition": ["query has unicode", "vault is empty", "S3 returns 503"],
    "root_cause": ["missing index", "race condition", "stale cache", "off-by-one"],
    "fix": ["add bounds check", "wrap in mutex", "invalidate on write", "use COALESCE"],
    "reporter": ["alpha tester #1", "team-internal", "via Discord", "GitHub issue"],
}


def generate_corpus(n: int, seed: int = 42) -> list[tuple[str, str]]:
    """Deterministic synthetic corpus. Each entry is (title, content)."""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for i in range(n):
        template = rng.choice(TEMPLATES)
        # Fill every {placeholder} that appears in the template.
        filled = template
        for key in POOLS:
            placeholder = "{" + key + "}"
            while placeholder in filled:
                filled = filled.replace(placeholder, rng.choice(POOLS[key]), 1)
        # Keep titles short — first sentence or first 60 chars.
        title = filled.split(". ")[0][:60] + f" #{i}"
        out.append((title, filled))
    return out


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round(q * (len(s) - 1)))
    return s[k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write results as JSON (for CI baselines).",
    )
    args = parser.parse_args()

    print(f"Vault: {EXAMPLE_VAULT}")
    print(f"Corpus: {args.memories} memories, seed={args.seed}")

    corpus = generate_corpus(args.memories, seed=args.seed)
    store = Store()

    # Save phase. Time saving so we know how long building the corpus
    # took — useful for CI / regression detection.
    t0 = time.perf_counter()
    for title, content in corpus:
        store.save(title=title, content=content)
    save_elapsed = time.perf_counter() - t0
    print(f"Saved {args.memories} memories in {save_elapsed:.1f}s "
          f"({save_elapsed / args.memories * 1000:.1f}ms/save)")

    # Embed phase. The MCP server does this post-save; the bare API
    # needs to opt in. Time it separately so the search numbers below
    # reflect search-only cost.
    try:
        from mnemon.embedder import embed_document
    except ImportError:
        print("FastEmbed not installed; vector search disabled.")
        embed_document = None  # type: ignore[assignment]

    if embed_document is not None:
        t0 = time.perf_counter()
        for title, content in corpus:
            embed_document(store, _sha256(content), title, content)
        embed_elapsed = time.perf_counter() - t0
        print(f"Embedded {args.memories} memories in {embed_elapsed:.1f}s "
              f"({embed_elapsed / args.memories * 1000:.1f}ms/embed)")
    else:
        embed_elapsed = 0.0

    # Query phase. Reuse a fixed RNG so query selection is also
    # deterministic across runs.
    rng = random.Random(args.seed + 1)
    query_pool = [
        # Lexical: a few terms straight from the pools above.
        "billing pipeline retries",
        "Postgres caching layer",
        "PagerDuty on-call escalation",
        "online backfill strategy",
        # Semantic: terms NOT in the pools, conceptually adjacent.
        "production rollout error handling",
        "database choice for the lookup tier",
        "alert routing during business hours",
        "schema change without downtime",
        # Diagnostic: vague phrasing.
        "what happens when the API is slow",
        "why did we pick this approach",
    ]

    timings_ms: list[float] = []
    for _ in range(args.queries):
        q = rng.choice(query_pool)
        t0 = time.perf_counter()
        _ = search(store, q, limit=args.limit)
        timings_ms.append((time.perf_counter() - t0) * 1000)

    p50 = quantile(timings_ms, 0.50)
    p95 = quantile(timings_ms, 0.95)
    p99 = quantile(timings_ms, 0.99)
    pmax = max(timings_ms)
    pmean = statistics.mean(timings_ms)

    print()
    print(f"Search latency over {args.queries} queries (limit={args.limit}):")
    print(f"  mean : {pmean:6.1f} ms")
    print(f"  p50  : {p50:6.1f} ms")
    print(f"  p95  : {p95:6.1f} ms")
    print(f"  p99  : {p99:6.1f} ms")
    print(f"  max  : {pmax:6.1f} ms")

    store.close()

    if args.json:
        result = {
            "memories": args.memories,
            "queries": args.queries,
            "limit": args.limit,
            "seed": args.seed,
            "save_total_seconds": round(save_elapsed, 2),
            "embed_total_seconds": round(embed_elapsed, 2),
            "search_ms": {
                "mean": round(pmean, 2),
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2),
                "max": round(pmax, 2),
            },
        }
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
