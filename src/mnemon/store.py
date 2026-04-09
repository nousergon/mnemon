"""Storage layer — SQLite + FTS5 with content-addressable storage.

Single-file vault at ~/.mnemon/default.sqlite.
Content-addressable: same content = same SHA-256 hash = no duplicate storage.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    HALF_LIVES,
    MEMORY_TYPE_MAP,
    DEFAULT_CONFIDENCE,
    PIN_BOOST,
    ContentType,
    MemoryType,
    vault_path,
)


@dataclass
class Document:
    id: int
    collection: str | None
    path: str | None
    title: str
    hash: str
    content_type: str
    memory_type: str
    confidence: float
    quality_score: float
    access_count: int
    pinned: int
    source_client: str | None
    invalidated_at: str | None
    invalidated_by: int | None
    created_at: str
    updated_at: str
    content: str = ""  # joined from content table


@dataclass
class SearchResult:
    doc_id: int
    title: str
    content: str
    content_type: str
    memory_type: str
    confidence: float
    created_at: str
    score: float
    source: str = "bm25"


@dataclass
class SweepCandidate:
    id: int
    title: str
    content_type: str
    age_days: int


@dataclass
class RelatedDocument(Document):
    relation_type: str = ""
    weight: float = 0.0


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _row_to_document(row: sqlite3.Row) -> Document:
    d = dict(row)
    return Document(
        id=d["id"],
        collection=d.get("collection"),
        path=d.get("path"),
        title=d["title"],
        hash=d["hash"],
        content_type=d["content_type"],
        memory_type=d["memory_type"],
        confidence=d["confidence"],
        quality_score=d["quality_score"],
        access_count=d["access_count"],
        pinned=d["pinned"],
        source_client=d.get("source_client"),
        invalidated_at=d.get("invalidated_at"),
        invalidated_by=d.get("invalidated_by"),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        content=d.get("doc", ""),
    )


class Store:
    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path) if db_path else vault_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = str(path)
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA busy_timeout = 15000")

        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS content (
                hash TEXT PRIMARY KEY,
                doc TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection TEXT,
                path TEXT,
                title TEXT NOT NULL,
                hash TEXT NOT NULL REFERENCES content(hash),
                content_type TEXT NOT NULL DEFAULT 'note',
                memory_type TEXT NOT NULL DEFAULT 'semantic',
                confidence REAL NOT NULL DEFAULT 0.5,
                quality_score REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                pinned INTEGER NOT NULL DEFAULT 0,
                source_client TEXT,
                invalidated_at TEXT,
                invalidated_by INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(collection, path)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title, body,
                tokenize='porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS relations (
                source_id INTEGER NOT NULL REFERENCES documents(id),
                target_id INTEGER NOT NULL REFERENCES documents(id),
                relation_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (source_id, target_id, relation_type)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                summary TEXT,
                client TEXT
            );

            CREATE TABLE IF NOT EXISTS sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                direction TEXT NOT NULL,
                timestamp TEXT DEFAULT (datetime('now')),
                documents_synced INTEGER,
                source TEXT
            );
        """)
        self.db.commit()

    def save(
        self,
        title: str,
        content: str,
        content_type: str = "note",
        collection: str = "default",
        path: str | None = None,
        source_client: str | None = None,
        confidence: float | None = None,
    ) -> int:
        """Save a memory. Returns the document ID."""
        content_hash = _sha256(content)
        ct = ContentType(content_type)
        mt = MEMORY_TYPE_MAP.get(ct, MemoryType.SEMANTIC)
        conf = confidence if confidence is not None else DEFAULT_CONFIDENCE.get(ct, 0.5)

        # Upsert content (idempotent)
        self.db.execute(
            "INSERT OR IGNORE INTO content (hash, doc) VALUES (?, ?)",
            (content_hash, content),
        )

        # Check for existing document with same hash
        row = self.db.execute(
            "SELECT id FROM documents WHERE hash = ?", (content_hash,)
        ).fetchone()

        if row:
            self.db.execute(
                "UPDATE documents SET access_count = access_count + 1, updated_at = datetime('now') WHERE id = ?",
                (row["id"],),
            )
            self.db.commit()
            return row["id"]

        # Generate path if not provided
        if path is None:
            path = f"{content_type}/{int(time.time() * 1000)}-{content_hash[:8]}"

        cur = self.db.execute(
            """INSERT INTO documents (collection, path, title, hash, content_type, memory_type, confidence, source_client)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (collection, path, title, content_hash, ct.value, mt.value, conf, source_client),
        )
        doc_id = cur.lastrowid

        # Index in FTS5
        self.db.execute(
            "INSERT INTO documents_fts (rowid, title, body) VALUES (?, ?, ?)",
            (doc_id, title, content),
        )
        self.db.commit()
        return doc_id

    def get(self, doc_id: int) -> Document | None:
        """Get a document by ID, including content."""
        row = self.db.execute(
            """SELECT d.*, c.doc
               FROM documents d
               JOIN content c ON d.hash = c.hash
               WHERE d.id = ? AND d.invalidated_at IS NULL""",
            (doc_id,),
        ).fetchone()

        if row:
            self.db.execute(
                "UPDATE documents SET access_count = access_count + 1 WHERE id = ?",
                (doc_id,),
            )
            self.db.commit()
            return _row_to_document(row)
        return None

    def get_by_path(self, path: str, collection: str = "default") -> Document | None:
        """Get a document by path."""
        row = self.db.execute(
            """SELECT d.*, c.doc
               FROM documents d
               JOIN content c ON d.hash = c.hash
               WHERE d.path = ? AND d.collection = ? AND d.invalidated_at IS NULL""",
            (path, collection),
        ).fetchone()
        return _row_to_document(row) if row else None

    def pin(self, doc_id: int) -> bool:
        """Pin a memory (boost confidence)."""
        cur = self.db.execute(
            "UPDATE documents SET pinned = 1, confidence = MIN(1.0, confidence + ?) WHERE id = ?",
            (PIN_BOOST, doc_id),
        )
        self.db.commit()
        return cur.rowcount > 0

    def forget(self, doc_id: int) -> bool:
        """Soft-delete a memory."""
        cur = self.db.execute(
            "UPDATE documents SET invalidated_at = datetime('now') WHERE id = ? AND invalidated_at IS NULL",
            (doc_id,),
        )
        if cur.rowcount > 0:
            self.db.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))
            self.db.commit()
            return True
        return False

    def timeline(self, limit: int = 20, content_type: str | None = None) -> list[Document]:
        """Get recent memories in chronological order."""
        if content_type:
            rows = self.db.execute(
                """SELECT d.*, c.doc
                   FROM documents d
                   JOIN content c ON d.hash = c.hash
                   WHERE d.invalidated_at IS NULL AND d.content_type = ?
                   ORDER BY d.created_at DESC, d.id DESC
                   LIMIT ?""",
                (content_type, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT d.*, c.doc
                   FROM documents d
                   JOIN content c ON d.hash = c.hash
                   WHERE d.invalidated_at IS NULL
                   ORDER BY d.created_at DESC, d.id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [_row_to_document(r) for r in rows]

    def search_bm25(self, query: str, limit: int = 20) -> list[SearchResult]:
        """BM25 full-text search via FTS5."""
        safe_query = " OR ".join(
            f'"{token}"*'
            for token in query.replace("'", "").replace('"', "").split()
            if token
        )
        if not safe_query:
            return []

        try:
            rows = self.db.execute(
                """SELECT
                      d.id AS doc_id,
                      d.title,
                      c.doc AS content,
                      d.content_type,
                      d.memory_type,
                      d.confidence,
                      d.created_at,
                      rank * -1 AS bm25_score
                   FROM documents_fts fts
                   JOIN documents d ON d.id = fts.rowid
                   JOIN content c ON d.hash = c.hash
                   WHERE documents_fts MATCH ?
                     AND d.invalidated_at IS NULL
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, limit),
            ).fetchall()

            return [
                SearchResult(
                    doc_id=r["doc_id"],
                    title=r["title"],
                    content=r["content"],
                    content_type=r["content_type"],
                    memory_type=r["memory_type"],
                    confidence=r["confidence"],
                    created_at=r["created_at"],
                    score=r["bm25_score"],
                    source="bm25",
                )
                for r in rows
            ]
        except sqlite3.OperationalError:
            return []

    def get_related(self, doc_id: int, limit: int = 10) -> list[RelatedDocument]:
        """Find documents related to a given document via the graph."""
        rows = self.db.execute(
            """SELECT d.*, c.doc, r.relation_type, r.weight
               FROM relations r
               JOIN documents d ON d.id = r.target_id
               JOIN content c ON d.hash = c.hash
               WHERE r.source_id = ? AND d.invalidated_at IS NULL
               UNION
               SELECT d.*, c.doc, r.relation_type, r.weight
               FROM relations r
               JOIN documents d ON d.id = r.source_id
               JOIN content c ON d.hash = c.hash
               WHERE r.target_id = ? AND d.invalidated_at IS NULL
               ORDER BY weight DESC
               LIMIT ?""",
            (doc_id, doc_id, limit),
        ).fetchall()

        results = []
        for r in rows:
            doc = _row_to_document(r)
            rd = RelatedDocument(**{k: getattr(doc, k) for k in doc.__dataclass_fields__})
            rd.relation_type = r["relation_type"]
            rd.weight = r["weight"]
            results.append(rd)
        return results

    def add_relation(self, source_id: int, target_id: int, relation_type: str, weight: float = 1.0) -> None:
        """Add a relation between two documents."""
        self.db.execute(
            "INSERT OR REPLACE INTO relations (source_id, target_id, relation_type, weight) VALUES (?, ?, ?, ?)",
            (source_id, target_id, relation_type, weight),
        )
        self.db.commit()

    def status(self) -> dict[str, Any]:
        """Vault health stats."""
        total = self.db.execute(
            "SELECT COUNT(*) as count FROM documents WHERE invalidated_at IS NULL"
        ).fetchone()["count"]

        by_type = self.db.execute(
            "SELECT content_type, COUNT(*) as count FROM documents WHERE invalidated_at IS NULL GROUP BY content_type ORDER BY count DESC"
        ).fetchall()

        invalidated = self.db.execute(
            "SELECT COUNT(*) as count FROM documents WHERE invalidated_at IS NOT NULL"
        ).fetchone()["count"]

        pinned = self.db.execute(
            "SELECT COUNT(*) as count FROM documents WHERE pinned = 1 AND invalidated_at IS NULL"
        ).fetchone()["count"]

        return {
            "total_documents": total,
            "by_type": [{"content_type": r["content_type"], "count": r["count"]} for r in by_type],
            "invalidated": invalidated,
            "pinned": pinned,
            "vault_path": self.db_path,
        }

    def sweep(self, dry_run: bool = True) -> dict[str, Any]:
        """Archive stale documents based on half-life."""
        candidates: list[SweepCandidate] = []

        for ct, half_life in HALF_LIVES.items():
            if half_life is None:
                continue
            rows = self.db.execute(
                """SELECT id, title, content_type,
                          CAST(julianday('now') - julianday(updated_at) AS INTEGER) AS age_days
                   FROM documents
                   WHERE content_type = ?
                     AND invalidated_at IS NULL
                     AND pinned = 0
                     AND julianday('now') - julianday(updated_at) > ?
                   ORDER BY updated_at ASC""",
                (ct.value, half_life),
            ).fetchall()

            candidates.extend(
                SweepCandidate(id=r["id"], title=r["title"], content_type=r["content_type"], age_days=r["age_days"])
                for r in rows
            )

        if not dry_run:
            for c in candidates:
                self.forget(c.id)

        return {
            "archived": 0 if dry_run else len(candidates),
            "candidates": candidates,
        }

    def close(self) -> None:
        self.db.close()
