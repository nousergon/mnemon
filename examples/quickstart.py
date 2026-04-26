"""Quickstart: save a few memories, then search them.

Demonstrates mnemon's public Python API directly (no MCP transport
required). Useful for embedding mnemon in a Python script, a notebook,
or an evaluation harness — the same Store and search() functions back
the MCP server, the CLI, and this example.

Run:

    pip install mnemon-memory
    python examples/quickstart.py

The first call builds an isolated vault under ./quickstart-vault/
(via MNEMON_VAULT_DIR), so this script never touches your real
~/.mnemon vault. Re-running is idempotent — `Store.save` deduplicates
by content hash.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Point Store at a throwaway vault so the example never touches the
# user's real ~/.mnemon. This must be set BEFORE importing Store.
EXAMPLE_VAULT = Path(tempfile.mkdtemp(prefix="mnemon-quickstart-"))
os.environ["MNEMON_VAULT_DIR"] = str(EXAMPLE_VAULT)

from mnemon.search import search  # noqa: E402
from mnemon.store import Store  # noqa: E402

MEMORIES = [
    (
        "Pipeline retries on transient failures",
        "The deployment pipeline retries any step that exits with a "
        "transient error (network timeout, 5xx, container pull fail). "
        "Permanent errors abort immediately.",
    ),
    (
        "Why we picked SQLite + FTS5",
        "We evaluated Postgres, DuckDB, and SQLite for the local vault. "
        "SQLite + FTS5 won because it's a single file, no daemon, "
        "ACID, and BM25 ships in the stdlib build.",
    ),
    (
        "Cold-start latency on Fly",
        "Embedder pre-load (~3s) + machine boot (~1.5s) means the first "
        "tool call after a Fly cold-stop pays a 5-7s 'thinking...' "
        "latency budget. Acceptable for hobby tier; documented in README.",
    ),
]


def main() -> None:
    print(f"Vault: {EXAMPLE_VAULT}\n")
    store = Store()

    # Save phase. Save() is idempotent on content hash, so re-running
    # the script does not duplicate.
    print("Saving 3 memories...")
    for title, content in MEMORIES:
        doc_id = store.save(title=title, content=content)
        print(f"  #{doc_id}: {title}")

    # Optional: embed the documents so vector search can find them.
    # The MCP server does this automatically post-save; the bare CLI/
    # API caller has to opt in.
    try:
        from mnemon.embedder import embed_document

        print("\nEmbedding for vector search...")
        for title, content in MEMORIES:
            from mnemon.store import _sha256

            embed_document(store, _sha256(content), title, content)
        print("  done\n")
    except ImportError:
        print("\nFastEmbed not available — vector search disabled.\n")

    # Lexical query — exact terms appear in the saved content.
    print("Lexical query: 'SQLite FTS5'")
    for r in search(store, "SQLite FTS5", limit=2):
        print(f"  [{r.composite_score:.3f}] {r.title}")

    # Semantic query — none of these words appear in any saved memory,
    # but the hybrid BM25+vector pipeline still surfaces the right one.
    print("\nSemantic query: 'production rollout error handling'")
    for r in search(store, "production rollout error handling", limit=2):
        print(f"  [{r.composite_score:.3f}] {r.title}")

    # Diagnostic query — covers the cold-start observation.
    print("\nQuery: 'how long does waking up the server take'")
    for r in search(store, "how long does waking up the server take", limit=2):
        print(f"  [{r.composite_score:.3f}] {r.title}")

    store.close()
    print(f"\nDone. Vault preserved at {EXAMPLE_VAULT} for inspection.")


if __name__ == "__main__":
    main()
