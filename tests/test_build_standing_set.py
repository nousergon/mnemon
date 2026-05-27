"""Tests for scripts/build_standing_set.py — vault-derived auto-exemplars.

Focused unit tests for the new ``_sample_vault_exemplars`` function added
in the 2026-05-27 vault-derived-auto-exemplars PR. The function pulls
positive exemplars (high-confidence preference / decision / antipattern)
and negative exemplars (recent handoffs) from the operator's own vault
so the embedding-based scorer adapts per-user without hand-tuning
maintenance.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_standing_set.py"


@pytest.fixture(scope="module")
def bss():
    """Load build_standing_set.py as a module via importlib."""
    spec = importlib.util.spec_from_file_location(
        "build_standing_set", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_standing_set"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def conn():
    """In-memory sqlite seeded with the documents + content schema the
    script's SQL targets. Mirrors the production schema's columns we
    actually query (id, hash, title, content_type, confidence,
    invalidated_at, created_at) without pulling the full Store
    machinery."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            hash TEXT NOT NULL,
            title TEXT,
            content_type TEXT,
            confidence REAL,
            invalidated_at TEXT,
            created_at TEXT
        );
        CREATE TABLE content (
            hash TEXT PRIMARY KEY,
            doc TEXT
        );
    """)
    return db


def _seed(conn, doc_id, content_type, confidence, title, content,
          invalidated=None, created_at="2026-05-01"):
    h = f"h{doc_id}"
    conn.execute(
        "INSERT INTO content (hash, doc) VALUES (?, ?)", (h, content),
    )
    conn.execute(
        """INSERT INTO documents
           (id, hash, title, content_type, confidence, invalidated_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, h, title, content_type, confidence, invalidated, created_at),
    )


class TestSampleVaultExemplars:
    def test_pulls_high_conf_preferences_and_decisions_as_positives(self, bss, conn):
        _seed(conn, 1, "preference", 0.85, "Pref one", "always X")
        _seed(conn, 2, "decision", 0.90, "Dec one", "chose Y over Z")
        _seed(conn, 3, "antipattern", 0.80, "Anti one", "do not Q")
        pos, neg = bss._sample_vault_exemplars(conn, n=10)
        assert len(pos) == 3
        assert any("Pref one" in p for p in pos)
        assert any("Dec one" in p for p in pos)
        assert any("Anti one" in p for p in pos)
        assert neg == []  # no handoffs seeded

    def test_excludes_below_confidence_floor(self, bss, conn):
        _seed(conn, 1, "preference", 0.85, "Strong", "high conf preference")
        _seed(conn, 2, "preference", 0.50, "Weak", "below floor")
        pos, _ = bss._sample_vault_exemplars(conn, n=10)
        assert len(pos) == 1
        assert "Strong" in pos[0]

    def test_excludes_non_durable_types(self, bss, conn):
        _seed(conn, 1, "observation", 0.95, "Obs", "passes confidence but wrong type")
        _seed(conn, 2, "research", 0.90, "Res", "also wrong type")
        _seed(conn, 3, "note", 0.95, "Note", "still wrong type")
        _seed(conn, 4, "project", 0.95, "Proj", "wrong type")
        _seed(conn, 5, "preference", 0.80, "Pref", "right type")
        pos, _ = bss._sample_vault_exemplars(conn, n=10)
        assert len(pos) == 1
        assert "Pref" in pos[0]

    def test_excludes_invalidated_memories(self, bss, conn):
        _seed(conn, 1, "preference", 0.90, "Live", "active",
              invalidated=None)
        _seed(conn, 2, "preference", 0.90, "Dead", "soft-deleted",
              invalidated="2026-05-20")
        pos, _ = bss._sample_vault_exemplars(conn, n=10)
        assert len(pos) == 1
        assert "Live" in pos[0]

    def test_pulls_recent_handoffs_as_negatives(self, bss, conn):
        _seed(conn, 1, "handoff", 0.60, "Session A",
              "first session", created_at="2026-05-20")
        _seed(conn, 2, "handoff", 0.60, "Session B",
              "second session", created_at="2026-05-25")
        _seed(conn, 3, "handoff", 0.60, "Session C",
              "third session", created_at="2026-05-15")
        _, neg = bss._sample_vault_exemplars(conn, n=10)
        assert len(neg) == 3
        # Most-recent-first ordering — recency is the negative signal.
        assert "Session B" in neg[0]
        assert "Session A" in neg[1]
        assert "Session C" in neg[2]

    def test_n_caps_sample_size(self, bss, conn):
        for i in range(20):
            _seed(conn, i + 1, "preference", 0.85,
                  f"Pref {i}", f"content for memory {i}")
        pos, _ = bss._sample_vault_exemplars(conn, n=5)
        assert len(pos) == 5

    def test_format_combines_title_and_snippet(self, bss, conn):
        _seed(conn, 1, "preference", 0.90, "Short title",
              "longer body content with multiple words")
        pos, _ = bss._sample_vault_exemplars(conn, n=10)
        assert pos[0].startswith("Short title: ")
        assert "longer body content" in pos[0]

    def test_empty_vault_returns_empty_lists(self, bss, conn):
        pos, neg = bss._sample_vault_exemplars(conn, n=10)
        assert pos == []
        assert neg == []
