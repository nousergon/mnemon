"""Storage layer — SQLite + FTS5 + in-process vector store.

Single-file vault at ~/.mnemon/default.sqlite.
Content-addressable: same content = same SHA-256 hash = no duplicate storage.
Vectors stored in a companion .npz file (brute-force cosine search).
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    CAPTURE_ATTENTION_BOOST,
    CAPTURE_ATTENTION_ENABLED,
    CAPTURE_ATTENTION_MIN_HITS,
    CAPTURE_ATTENTION_REQUIRE_DISTINCT_SESSIONS,
    CAPTURE_ATTENTION_THRESHOLD,
    HALF_LIVES,
    HOOK_SOURCE_CLIENTS,
    HOOK_SOURCE_CONFIDENCE_CEILING,
    MEMORY_TYPE_MAP,
    DEFAULT_CONFIDENCE,
    PIN_BOOST,
    ContentType,
    MemoryType,
    vault_path,
)
from .vecstore import VecStore


class CaptureAttentionUnavailableError(RuntimeError):
    """Raised when the capture-attention path can't complete its check.

    Surface conditions: embedder unavailable, vecstore IO failure,
    schema-version mismatch on `recurrence_count`. Caller (typically a
    best-effort hook) is expected to catch + log + continue without
    the attention side effects. Fail-loud per the
    [[feedback_no_silent_fails]] discipline — never silently swallow.
    """


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
    # Provenance — the save's source_client. Carried through search +
    # RRF fusion so composite scoring can apply the Layer 4 provenance
    # demotion (auto-captured transcripts must not outrank deliberate
    # user assertions at equal relevance). None for legacy/unknown.
    source_client: str | None = None


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
    def __init__(self, db_path: str | Path | None = None, vector_dim: int = 384):
        path = Path(db_path) if db_path else vault_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = str(path)
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA busy_timeout = 15000")

        # Vector store in companion file
        vec_path = str(path).replace(".sqlite", ".vec")
        self.vec_store = VecStore(vec_path, vector_dim)

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
        self._migrate_source_key()
        self._migrate_recurrence_count()

    def _migrate_recurrence_count(self) -> None:
        """Additive migration: ``documents.recurrence_count`` counts
        cross-session restatements detected by capture attention Phase A.

        Incremented once per ``_apply_capture_attention`` trigger on the
        canonical neighbor of a detected cluster. Pre-existing rows get
        a count of 0 and recurrence detection starts forward from the
        next save. The column is additive + harmless if
        ``CAPTURE_ATTENTION_ENABLED`` stays off — backout = flip the
        flag; column stays.
        """
        cols = {
            r["name"]
            for r in self.db.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "recurrence_count" not in cols:
            self.db.execute(
                "ALTER TABLE documents ADD COLUMN "
                "recurrence_count INTEGER NOT NULL DEFAULT 0"
            )
            self.db.commit()

    def _migrate_source_key(self) -> None:
        """Additive migration: ``documents.source_key`` is a stable
        caller-supplied identity for upsert-by-slug (the auto-mirror
        path keys it to the local file's frontmatter ``name``). Vaults
        created before this column existed get it added in place;
        ``ADD COLUMN`` with no default is metadata-only and safe on a
        live WAL vault. Pre-existing rows keep ``source_key = NULL`` and
        are treated as un-keyed (insert-only) — exactly the old
        behaviour, so the migration is a no-op for everything that
        predates it."""
        cols = {
            r["name"]
            for r in self.db.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "source_key" not in cols:
            self.db.execute("ALTER TABLE documents ADD COLUMN source_key TEXT")
        # Lookup index for the upsert probe. Non-unique on purpose:
        # uniqueness is enforced at save() time scoped to
        # ``invalidated_at IS NULL`` (a partial UNIQUE index can't
        # express "one *live* row per key" cleanly across the
        # invalidate-prior + insert supersession we do here).
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_source_key "
            "ON documents (collection, source_client, source_key)"
        )
        self.db.commit()

    def save(
        self,
        title: str,
        content: str,
        content_type: str = "note",
        collection: str = "default",
        source_client: str | None = None,
        confidence: float | None = None,
        source_key: str | None = None,
        correction_of: int | None = None,
    ) -> int:
        """Save a memory. Returns the document ID.

        ``source_key`` — when supplied — is a stable caller-owned
        identity (e.g. the auto-mirror path passes the local memory
        file's frontmatter ``name``). At most one *live*
        (non-invalidated) document exists per
        ``(collection, source_client, source_key)``: an unchanged
        re-save is idempotent; a changed re-save invalidates the prior
        live row(s) and inserts a fresh one (supersession recorded via
        ``invalidated_by``). Without ``source_key`` the historical
        insert-only behaviour is preserved exactly.

        ``correction_of`` — when set — is an explicit operator gesture
        that THIS memory corrects/supersedes a prior one (the Phase 2
        promotion signal from the salience-tier plan). Reserved here
        for forward compatibility; today its only effect is to SKIP
        the capture-attention path (operator gesture beats automated
        recurrence detection).
        """
        content_hash = _sha256(content)
        ct = ContentType(content_type)
        mt = MEMORY_TYPE_MAP.get(ct, MemoryType.SEMANTIC)
        conf = confidence if confidence is not None else DEFAULT_CONFIDENCE[ct]
        if source_client in HOOK_SOURCE_CLIENTS:
            conf = min(conf, HOOK_SOURCE_CONFIDENCE_CEILING)

        # Upsert content (idempotent)
        self.db.execute(
            "INSERT OR IGNORE INTO content (hash, doc) VALUES (?, ?)",
            (content_hash, content),
        )

        superseded_ids: list[int] = []

        # Upsert-by-slug: when the caller owns a stable identity key,
        # there is at most one *live* document per
        # (collection, source_client, source_key). Resolve that before
        # the generic hash dedup so a divergent edit supersedes its
        # prior instead of piling up a near-duplicate (the auto-mirror
        # multi-edit-per-session case).
        if source_key is not None:
            priors = self.db.execute(
                """SELECT id, hash FROM documents
                   WHERE collection = ?
                     AND source_client IS ?
                     AND source_key = ?
                     AND invalidated_at IS NULL""",
                (collection, source_client, source_key),
            ).fetchall()
            if len(priors) == 1 and priors[0]["hash"] == content_hash:
                # Unchanged re-save of the same slug — idempotent.
                self.db.execute(
                    "UPDATE documents SET access_count = access_count + 1, "
                    "updated_at = datetime('now') WHERE id = ?",
                    (priors[0]["id"],),
                )
                self.db.commit()
                return priors[0]["id"]
            if priors:
                # Changed (or legacy pile-up) — invalidate every live
                # row for this key; invalidated_by is stamped to the
                # new doc below once we have its id.
                superseded_ids = [p["id"] for p in priors]
                qmarks = ",".join("?" * len(superseded_ids))
                self.db.execute(
                    f"UPDATE documents SET invalidated_at = datetime('now') "
                    f"WHERE id IN ({qmarks})",
                    superseded_ids,
                )
                self.db.executemany(
                    "DELETE FROM documents_fts WHERE rowid = ?",
                    [(i,) for i in superseded_ids],
                )

        # Check for an existing *live* document with the same content.
        # Scoped to invalidated_at IS NULL so a re-save of content that
        # was previously forgotten / superseded creates a fresh visible
        # memory rather than resurrecting a dead row's access_count.
        row = self.db.execute(
            "SELECT id FROM documents WHERE hash = ? AND invalidated_at IS NULL",
            (content_hash,),
        ).fetchone()

        if row:
            self.db.execute(
                "UPDATE documents SET access_count = access_count + 1, updated_at = datetime('now') WHERE id = ?",
                (row["id"],),
            )
            self.db.commit()
            return row["id"]

        # Auto-generate a stable path key — sortable by creation time.
        # Callers have never needed to override this so the parameter was
        # removed in 0.4.2. The uuid salt guarantees the
        # UNIQUE(collection, path) invariant even when the upsert-by-slug
        # path re-inserts identical content within the same millisecond
        # (e.g. an A -> B -> A revert in one session).
        path = (
            f"{content_type}/{int(time.time() * 1000)}"
            f"-{content_hash[:8]}-{uuid.uuid4().hex[:8]}"
        )

        cur = self.db.execute(
            """INSERT INTO documents (collection, path, title, hash, content_type, memory_type, confidence, source_client, source_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (collection, path, title, content_hash, ct.value, mt.value, conf, source_client, source_key),
        )
        doc_id = cur.lastrowid

        # Record supersession: the prior live rows we just invalidated
        # point at the doc that replaces them so the chain is auditable.
        if superseded_ids:
            qmarks = ",".join("?" * len(superseded_ids))
            self.db.execute(
                f"UPDATE documents SET invalidated_by = ? WHERE id IN ({qmarks})",
                (doc_id, *superseded_ids),
            )

        # Index in FTS5
        self.db.execute(
            "INSERT INTO documents_fts (rowid, title, body) VALUES (?, ?, ?)",
            (doc_id, title, content),
        )
        self.db.commit()

        # Capture attention Phase A — preserve+relate+boost. Gated on
        # the feature flag (default off through soak). Skipped when
        # ``correction_of`` is set (operator gesture beats automated
        # recurrence detection per the salience-tier plan composition).
        #
        # Failure is a NAMED swallow per
        # [[feedback_no_silent_fails]] acceptable-category (b) —
        # secondary observability hung off a primary path (the save
        # itself) that survives independently. Mirrors the existing
        # embed_document() WARN pattern in server.py:memory_save.
        if CAPTURE_ATTENTION_ENABLED and correction_of is None:
            try:
                self.apply_capture_attention(
                    new_doc_id=doc_id, content=content,
                    source_client=source_client,
                )
            except CaptureAttentionUnavailableError as exc:
                import logging
                logging.getLogger("mnemon.store").warning(
                    "save: capture-attention skipped for doc_id=%d (%s); "
                    "memory is saved but recurrence-boost was not applied",
                    doc_id, exc,
                )

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
                      d.source_client,
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
                    source_client=r["source_client"],
                )
                for r in rows
            ]
        except sqlite3.OperationalError:
            return []

    def save_embedding(self, content_hash: str, seq: int, embedding: np.ndarray) -> None:
        """Store a vector embedding for a document fragment."""
        vec_id = f"{content_hash}_{seq}"
        self.vec_store.set(vec_id, embedding)

    def flush_vectors(self) -> None:
        """Persist vector store to disk."""
        self.vec_store.save()

    def search_vector(self, embedding: np.ndarray, limit: int = 20) -> list[SearchResult]:
        """Vector similarity search via in-process brute-force cosine."""
        if self.vec_store.size() == 0:
            return []

        vec_results = self.vec_store.search(embedding, limit * 2)
        results: list[SearchResult] = []
        seen_ids: set[int] = set()

        for vr in vec_results:
            content_hash = vr["id"].split("_")[0]
            row = self.db.execute(
                """SELECT d.id AS doc_id, d.title, c.doc AS content, d.content_type,
                          d.memory_type, d.confidence, d.created_at, d.source_client
                   FROM documents d
                   JOIN content c ON d.hash = c.hash
                   WHERE d.hash = ? AND d.invalidated_at IS NULL
                   LIMIT 1""",
                (content_hash,),
            ).fetchone()

            if row and row["doc_id"] not in seen_ids:
                seen_ids.add(row["doc_id"])
                results.append(SearchResult(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    content=row["content"],
                    content_type=row["content_type"],
                    memory_type=row["memory_type"],
                    confidence=row["confidence"],
                    created_at=row["created_at"],
                    score=vr["similarity"],
                    source="vector",
                    source_client=row["source_client"],
                ))

            if len(results) >= limit:
                break

        return results

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

    # ── Capture attention Phase A ────────────────────────────────────
    # private/mnemon-capture-attention-plan-260522.md
    #
    # When a new memory's content is semantically close to ≥
    # CAPTURE_ATTENTION_MIN_HITS prior memories across distinct
    # sessions, boost the canonical neighbor's confidence + insert
    # 'restates' relations + increment its recurrence_count. The new
    # memory itself is preserved unchanged (no information loss — SOTA
    # preserve+relate+boost pattern). Operator-reviewed merge is
    # Phase C; this layer only does the non-destructive auto-apply.

    def apply_capture_attention(
        self, new_doc_id: int, content: str, source_client: str | None = None
    ) -> dict[str, Any]:
        """Run the capture-attention check on a freshly-saved document.

        Returns a dict describing the side effects:
          ``{"fired": bool, "canonical_id": int | None,
             "neighbors": [int, ...], "boost_applied": float}``

        The new doc must already be in the documents table; it does
        NOT need to be in the vec store yet (this method embeds the
        content for its own neighbor query, and excludes the new doc
        by id from the results).

        Raises CaptureAttentionUnavailableError if the embedder or
        vecstore is unreachable. Callers in best-effort paths
        (session_extractor, auto_mirror) must catch + log + continue.
        """
        # Lazy import — keep store.py module-load cheap; FastEmbed's
        # ONNX model only materializes on first embed() call anyway.
        try:
            from .embedder import embed
        except ImportError as e:
            raise CaptureAttentionUnavailableError(
                f"capture attention skipped — embedder import: {e}"
            ) from e

        try:
            query_vec = embed(content)
            vec_results = self.vec_store.search(query_vec, k=20)
        except Exception as e:
            raise CaptureAttentionUnavailableError(
                f"capture attention skipped — embed/vecstore: {e}"
            ) from e

        hits = self._resolve_neighbor_docs(
            vec_results,
            threshold=CAPTURE_ATTENTION_THRESHOLD,
            exclude_doc_id=new_doc_id,
        )

        # Distinct-session gate defends against vault-crowding from a
        # single long session that repeats itself.
        if CAPTURE_ATTENTION_REQUIRE_DISTINCT_SESSIONS:
            distinct_days = {h["created_at"][:10] for h in hits}
            if len(distinct_days) < CAPTURE_ATTENTION_MIN_HITS:
                return {
                    "fired": False, "canonical_id": None,
                    "neighbors": [], "boost_applied": 0.0,
                    "reason": "insufficient_distinct_sessions",
                }

        if len(hits) < CAPTURE_ATTENTION_MIN_HITS:
            return {
                "fired": False, "canonical_id": None,
                "neighbors": [], "boost_applied": 0.0,
                "reason": "insufficient_neighbors",
            }

        canonical = self._pick_canonical(hits)

        # Side effects (auto-apply, non-destructive)
        for hit in hits:
            self.add_relation(
                source_id=new_doc_id,
                target_id=hit["id"],
                relation_type="restates",
                weight=float(hit["similarity"]),
            )
        boost_applied = self._boost_confidence(canonical["id"])
        self._increment_recurrence(canonical["id"])

        return {
            "fired": True,
            "canonical_id": canonical["id"],
            "neighbors": [h["id"] for h in hits],
            "boost_applied": boost_applied,
        }

    def _resolve_neighbor_docs(
        self,
        vec_results: list[dict],
        *,
        threshold: float,
        exclude_doc_id: int,
    ) -> list[dict[str, Any]]:
        """Map vec_store search hits → live document rows above threshold.

        vec_results entries are ``{"id": "{hash}_{seq}", "similarity": float}``.
        Multiple fragments from the same doc collapse to one row (keep the
        highest-similarity hit). Excludes the just-saved doc by id +
        anything invalidated.
        """
        # Group by content_hash, keep max similarity per hash
        best_by_hash: dict[str, float] = {}
        for vr in vec_results:
            if vr["similarity"] < threshold:
                continue
            content_hash = vr["id"].split("_")[0]
            prev = best_by_hash.get(content_hash, 0.0)
            if vr["similarity"] > prev:
                best_by_hash[content_hash] = vr["similarity"]

        if not best_by_hash:
            return []

        # Resolve to live documents, exclude the just-saved one
        hashes = list(best_by_hash.keys())
        qmarks = ",".join("?" * len(hashes))
        rows = self.db.execute(
            f"""SELECT id, hash, confidence, pinned, created_at, source_client
                FROM documents
                WHERE hash IN ({qmarks})
                  AND invalidated_at IS NULL
                  AND id != ?""",
            (*hashes, exclude_doc_id),
        ).fetchall()

        return [
            {
                "id": r["id"],
                "hash": r["hash"],
                "confidence": r["confidence"],
                "pinned": r["pinned"],
                "created_at": r["created_at"],
                "source_client": r["source_client"],
                "similarity": best_by_hash[r["hash"]],
            }
            for r in rows
        ]

    @staticmethod
    def _pick_canonical(hits: list[dict[str, Any]]) -> dict[str, Any]:
        """Select the canonical memory from a near-neighbor cluster.

        Order of preference: pinned (operator gesture) > highest
        confidence > most recent created_at > lowest id (deterministic
        tiebreak). Matches the contradiction.py canonical-selection
        spirit + adds explicit pinned-first per the salience-tier
        invariant that operator gestures beat automated signals.
        """
        return max(
            hits,
            key=lambda h: (
                int(h["pinned"]),
                float(h["confidence"]),
                h["created_at"],
                -int(h["id"]),  # negate so lowest id wins on tie
            ),
        )

    def _boost_confidence(self, doc_id: int) -> float:
        """Increment a canonical's confidence by CAPTURE_ATTENTION_BOOST,
        capped at HOOK_SOURCE_CONFIDENCE_CEILING for hook-sourced docs
        (existing Layer 4 invariant), or 1.0 for user-authored. Returns
        the actual delta applied (zero if already at ceiling).
        """
        row = self.db.execute(
            "SELECT confidence, source_client FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
        if not row:
            return 0.0
        ceiling = (
            HOOK_SOURCE_CONFIDENCE_CEILING
            if row["source_client"] in HOOK_SOURCE_CLIENTS
            else 1.0
        )
        new_conf = min(row["confidence"] + CAPTURE_ATTENTION_BOOST, ceiling)
        delta = new_conf - row["confidence"]
        if delta <= 0:
            return 0.0
        self.db.execute(
            "UPDATE documents SET confidence = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (new_conf, doc_id),
        )
        self.db.commit()
        return delta

    def _increment_recurrence(self, doc_id: int) -> None:
        """Bump the canonical's recurrence_count by 1."""
        self.db.execute(
            "UPDATE documents SET recurrence_count = recurrence_count + 1, "
            "updated_at = datetime('now') WHERE id = ?",
            (doc_id,),
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
            "total_vectors": self.vec_store.size(),
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
        self.vec_store.save()
        self.db.close()
