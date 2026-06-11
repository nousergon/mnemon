"""Storage layer — SQLite + FTS5 + in-process vector store.

Single-file vault at ~/.mnemon/default.sqlite.
Content-addressable: same content = same SHA-256 hash = no duplicate storage.
Vectors stored in a companion .npz file (brute-force cosine search).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    CAPTURE_ATTENTION_BOOST,
    CAPTURE_ATTENTION_MIN_HITS,
    CAPTURE_ATTENTION_REQUIRE_DISTINCT_SESSIONS,
    CAPTURE_ATTENTION_THRESHOLD,
    HALF_LIVES,
    HOOK_SOURCE_CLIENTS,
    HOOK_SOURCE_CONFIDENCE_CEILING,
    MEMORY_TYPE_MAP,
    DEFAULT_CONFIDENCE,
    PIN_BOOST,
    RECENCY_HALF_LIFE_DAYS,
    STANDING_TIER_BLOCKED_SOURCE_CLIENTS,
    STANDING_TIER_DEFAULT_CAP,
    STANDING_TIER_HARD_CEILING,
    ContentType,
    MemoryType,
    vault_path,
)
from .vecstore import VecStore


def _capture_attention_enabled() -> bool:
    """Resolve the capture-attention feature flag (env-var override).

    Truth sources, in order:
      1. ``MNEMON_CAPTURE_ATTENTION_ENABLED`` env var (operator override) —
         lets the operator flip activation on Fly via ``flyctl secrets
         set`` without a code change + redeploy.
      2. ``config.CAPTURE_ATTENTION_ENABLED`` (default-off through soak).

    Mirrors the standing-tier helper in
    ``hooks/context_surfacing.py:_standing_tier_enabled``. Called at
    request time (in ``Store.save``), so secret flips take effect on
    the next save without restarting the server.
    """
    env = os.environ.get("MNEMON_CAPTURE_ATTENTION_ENABLED", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    from .config import CAPTURE_ATTENTION_ENABLED
    return CAPTURE_ATTENTION_ENABLED


class CaptureAttentionUnavailableError(RuntimeError):
    """Raised when the capture-attention path can't complete its check.

    Surface conditions: embedder unavailable, vecstore IO failure,
    schema-version mismatch on `recurrence_count`. Caller (typically a
    best-effort hook) is expected to catch + log + continue without
    the attention side effects. Fail-loud per the
    [[feedback_no_silent_fails]] discipline — never silently swallow.
    """


class StandingTierError(ValueError):
    """Raised when ``promote_to_standing`` rejects a candidate.

    Distinct subclasses surface the reason so MCP / CLI callers can
    render a user-actionable message instead of an opaque False. Per
    the salience-tier plan invariant: "promotion is operator-approved,
    not auto" — operator gets a clear "why not" message.
    """


class StandingTierCapReached(StandingTierError):
    """Already at the runtime cap (STANDING_TIER_DEFAULT_CAP)."""


class StandingTierProvenanceRejected(StandingTierError):
    """Source client is in STANDING_TIER_BLOCKED_SOURCE_CLIENTS.

    Layer 4 composition: hook-sourced memories cannot be promoted —
    operator-explicit gesture only.
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


logger = logging.getLogger("mnemon.store")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Prose-detected supersession: recognise an EXPLICIT "supersedes … id N"
# / "supersedes #N" (also "superseded …") in saved content so the chain
# can be recorded even when the caller forgot the correction_of param.
# This is the non-LLM analogue of Mem0's UPDATE intent-detection (an LLM
# intent classifier could be a future opt-in, composing with the
# --judge anthropic pattern). See ROADMAP "memory_save auto-`supersedes`
# from prose".
#
# The motivating real case had words between the verb and the id
# ("Supersedes the partial financial framings in id 2402"), so the gap is
# allowed — but bounded (≤50 chars) and clause-local: the gap excludes
# sentence/clause punctuation (.;:) so "supersedes the old way; see id 5"
# does NOT link to id 5. Either an "[words] id [#]N" gap or a direct "#N"
# is accepted. Precision-first: when in doubt it does nothing (the caller
# can always pass correction_of explicitly).
_PROSE_SUPERSEDES_RE = re.compile(
    r"\bsupersede(?:s|d)?\s+(?:[^.;:\n]{0,50}?\bid\s+#?|#)(\d+)\b",
    re.IGNORECASE,
)


def _detect_prose_supersedes(content: str) -> int | None:
    """Return the first explicitly-named superseded doc id, or None.

    Returns the first match; multiple targets aren't expressible via the
    single-valued ``correction_of`` param.
    """
    m = _PROSE_SUPERSEDES_RE.search(content)
    return int(m.group(1)) if m else None


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


class LocalVaultInaccessibleError(RuntimeError):
    """Raised when the default local vault is opened while a remote is configured.

    Brian's principle: if a cloud vault exists, the local vault must be
    *inaccessible* — a second reachable source of truth is a silent-divergence
    trap (the 2026-06-04 two-vaults bug). ``serve`` already proxies to the
    remote (PR #188); this closes the residual hole where every *other*
    default-vault open (``rebuild`` / ``forget`` / ``standing`` / ``doctor`` /
    ``sync`` / dashboard / api) would silently re-create + touch a local vault
    in remote mode. Fail loud, never empty.
    """


class Store:
    def __init__(self, db_path: str | Path | None = None, vector_dim: int = 384):
        # Chokepoint guard: opening the DEFAULT local vault while a remote is
        # configured is forbidden (the local vault is out of commission in
        # remote mode). An explicit ``db_path`` (tests, migrations) and the
        # remote server (not in remote mode) are unaffected; the override env
        # is the escape hatch for genuine local maintenance.
        if db_path is None and not os.environ.get("MNEMON_ALLOW_LOCAL_STORE"):
            from .hooks._remote_client import remote_mode_active

            if remote_mode_active():
                raise LocalVaultInaccessibleError(
                    "A remote mnemon vault is configured (MNEMON_REMOTE_URL / "
                    "~/.mnemon/remote_url); the local vault is out of commission "
                    "to prevent a second, divergent source of truth. Use the "
                    "remote (the memory_* MCP tools, or `mnemon status/search/"
                    "save` which route remote). For genuine local-vault "
                    "maintenance set MNEMON_ALLOW_LOCAL_STORE=1."
                )

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
        self._migrate_tier()
        self._migrate_promotion_signals()
        self._migrate_last_injected_at()

    def _migrate_last_injected_at(self) -> None:
        """Additive migration: ``documents.last_injected_at`` records
        the last time a standing-tier memory was injected into a
        recall context (via ``list_standing``).

        Salience tier Phase 3 (added 2026-05-27) — surfaces "this
        memory hasn't fired in N days, still load-bearing?" for
        operator review. Doesn't auto-demote (would re-open the
        noise-floor problem); just observability.

        Stays NULL for non-standing docs and for standing docs that
        haven't yet been injected through ``list_standing``. The
        operator-facing CLI renders "never" for NULL.
        """
        cols = {
            r["name"]
            for r in self.db.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "last_injected_at" not in cols:
            self.db.execute(
                "ALTER TABLE documents ADD COLUMN last_injected_at TEXT"
            )
            self.db.commit()

    def _migrate_promotion_signals(self) -> None:
        """Additive migration: ``documents.correction_count`` and
        ``documents.contradiction_win_count`` count operator-explicit
        corrections + contradiction-classifier wins respectively.
        Salience tier Phase 2 (added 2026-05-27) —
        private/mnemon-salience-tier-plan-260521.md.

        - ``correction_count``: incremented on the TARGET of a
          ``Store.save(correction_of=<id>)`` call. High value means
          this memory keeps getting corrected — load-bearing signal
          for the standing-tier promotion candidate score.
        - ``contradiction_win_count``: incremented on the WINNING
          side (the new doc) when ``contradiction.check_contradictions``
          classifies a pair as ``update`` or ``contradiction``. High
          value means this memory regularly demotes others —
          structurally load-bearing.

        Pre-existing rows get a count of 0. Schema is additive +
        harmless if salience-report is never invoked.
        """
        cols = {
            r["name"]
            for r in self.db.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "correction_count" not in cols:
            self.db.execute(
                "ALTER TABLE documents ADD COLUMN "
                "correction_count INTEGER NOT NULL DEFAULT 0"
            )
        if "contradiction_win_count" not in cols:
            self.db.execute(
                "ALTER TABLE documents ADD COLUMN "
                "contradiction_win_count INTEGER NOT NULL DEFAULT 0"
            )
        self.db.commit()

    def _migrate_tier(self) -> None:
        """Additive migration: ``documents.tier`` distinguishes
        unconditionally-injected standing memories from situational
        recall. Salience tier Phase 1 (added 2026-05-22) —
        private/mnemon-salience-tier-plan-260521.md.

        Values: ``'situational'`` (default; current `memory_search`
        path applies, ranked retrieval) or ``'standing'`` (injected
        into every <mnemon-context> envelope regardless of query
        similarity, capped). Pre-existing rows default to
        ``'situational'`` — every existing memory keeps its current
        behavior. Schema is additive + harmless if
        ``STANDING_TIER_ENABLED`` stays off.
        """
        cols = {
            r["name"]
            for r in self.db.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "tier" not in cols:
            self.db.execute(
                "ALTER TABLE documents ADD COLUMN "
                "tier TEXT NOT NULL DEFAULT 'situational'"
            )
            # Lookup index for the cap-count probe and the search
            # exclusion. Filtered to live rows because invalidated
            # standing-tier members don't count against the cap.
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_tier "
                "ON documents (tier) WHERE invalidated_at IS NULL"
            )
            self.db.commit()

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
        that THIS memory corrects/supersedes a prior one. Two effects:
          1. Inserts a ``'supersedes'`` relation from the new doc to the
             named prior, so the supersession chain is auditable.
          2. Skips the capture-attention path (operator gesture beats
             automated recurrence detection per the salience-tier plan
             composition).

        Raises ``ValueError`` if ``correction_of`` names a non-existent
        document — fail loud per ``[[feedback_no_silent_fails]]``. We
        DON'T require the target to be live (invalidated_at IS NULL):
        an operator may legitimately mark a new memory as superseding
        an already-forgotten one to record the chain. The relation is
        the audit trail.

        If ``correction_of`` is omitted but ``content`` explicitly states
        ``supersedes id N`` / ``supersedes #N``, it is auto-resolved to
        ``correction_of`` (with a WARN, and tolerating a non-existent id
        rather than raising). Pass ``correction_of`` explicitly to opt out
        of the prose scan.
        """
        content_hash = _sha256(content)
        ct = ContentType(content_type)
        mt = MEMORY_TYPE_MAP.get(ct, MemoryType.SEMANTIC)
        conf = confidence if confidence is not None else DEFAULT_CONFIDENCE[ct]
        if source_client in HOOK_SOURCE_CLIENTS:
            conf = min(conf, HOOK_SOURCE_CONFIDENCE_CEILING)

        # Prose-detected supersession (additive, non-breaking). When the
        # caller did NOT pass correction_of but the content explicitly says
        # "supersedes id N" / "supersedes #N", resolve it to correction_of
        # so the chain is recorded — flowing through the same machinery
        # below (relation insert + capture-attention skip). Two deliberate
        # departures from the explicit param: (1) WARN (never silent), so
        # the auto-link is visible per [[feedback_no_silent_fails]] — this
        # is what makes prose-detection safe vs the rejected "hidden
        # auto-insert"; (2) do NOT hard-fail on a non-existent id — the
        # caller didn't request it, so a stray prose mention must not break
        # an otherwise-valid save (the explicit param still raises below).
        if correction_of is None:
            detected = _detect_prose_supersedes(content)
            if detected is not None:
                target_exists = self.db.execute(
                    "SELECT 1 FROM documents WHERE id = ?", (detected,),
                ).fetchone()
                if target_exists is not None:
                    correction_of = detected
                    logger.warning(
                        "save: auto-detected supersession of id %d from "
                        "content prose; recording a 'supersedes' relation. "
                        "Pass correction_of=%d explicitly to silence this.",
                        detected, detected,
                    )
                else:
                    logger.warning(
                        "save: content prose claims it supersedes id %d, but "
                        "no such document exists — NOT recording a relation "
                        "(the save proceeds normally).",
                        detected,
                    )

        # Validate correction_of target exists before we commit anything.
        # Fail loud rather than insert a dangling relation downstream.
        if correction_of is not None:
            row = self.db.execute(
                "SELECT 1 FROM documents WHERE id = ?", (correction_of,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"correction_of={correction_of} names a non-existent "
                    "document"
                )

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

        # Explicit supersession relation. The auto-mirror upsert-by-slug
        # path above records its OWN supersession via invalidated_by; this
        # is the operator-explicit cross-id gesture (different memory
        # entirely, but THIS one corrects the prior). The 'supersedes'
        # relation makes the chain queryable + auditable downstream.
        #
        # Salience Phase 2 (2026-05-27): also bumps the TARGET's
        # correction_count so the operator-explicit correction signal
        # accumulates per memory. High correction_count = "this fact
        # keeps getting corrected" = load-bearing promotion candidate.
        if correction_of is not None:
            self.db.execute(
                "INSERT OR REPLACE INTO relations "
                "(source_id, target_id, relation_type, weight) "
                "VALUES (?, ?, 'supersedes', 1.0)",
                (doc_id, correction_of),
            )
            self.db.execute(
                "UPDATE documents SET correction_count = correction_count + 1 "
                "WHERE id = ?",
                (correction_of,),
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
        if _capture_attention_enabled() and correction_of is None:
            try:
                self.apply_capture_attention(
                    new_doc_id=doc_id, content=content,
                    source_client=source_client,
                )
            except CaptureAttentionUnavailableError as exc:
                logger.warning(
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

    def search_bm25(
        self, query: str, limit: int = 20, *, include_standing: bool = False
    ) -> list[SearchResult]:
        """BM25 full-text search via FTS5.

        ``include_standing`` — when False (default), tier='standing'
        docs are excluded. They're injected unconditionally into the
        <mnemon-context> envelope via ``list_standing()``; including
        them in ranked retrieval would double-count and crowd out the
        situational signal. The dashboard / explicit operator queries
        can opt-in (e.g. an explicit "show all memories" surface).
        """
        safe_query = " OR ".join(
            f'"{token}"*'
            for token in query.replace("'", "").replace('"', "").split()
            if token
        )
        if not safe_query:
            return []

        tier_filter = "" if include_standing else " AND d.tier = 'situational'"

        try:
            rows = self.db.execute(
                f"""SELECT
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
                     AND d.invalidated_at IS NULL{tier_filter}
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

    def search_vector(
        self, embedding: np.ndarray, limit: int = 20, *, include_standing: bool = False
    ) -> list[SearchResult]:
        """Vector similarity search via in-process brute-force cosine.

        ``include_standing`` — see ``search_bm25`` for the rationale.
        Standing-tier docs are excluded from ranked retrieval by default.
        """
        if self.vec_store.size() == 0:
            return []

        vec_results = self.vec_store.search(embedding, limit * 2)
        results: list[SearchResult] = []
        seen_ids: set[int] = set()

        tier_filter = "" if include_standing else " AND d.tier = 'situational'"

        for vr in vec_results:
            content_hash = vr["id"].split("_")[0]
            row = self.db.execute(
                f"""SELECT d.id AS doc_id, d.title, c.doc AS content, d.content_type,
                          d.memory_type, d.confidence, d.created_at, d.source_client
                   FROM documents d
                   JOIN content c ON d.hash = c.hash
                   WHERE d.hash = ? AND d.invalidated_at IS NULL{tier_filter}
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

    def export_relations(self, limit: int = 20000) -> list[dict]:
        """All relation edges between live documents, in one query.

        The Graph page's edge overlay used to call ``get_related`` once per
        document — N round-trips, untenable on a large remote vault. This
        returns every edge whose *both* endpoints are non-invalidated, as
        ``[{source_id, target_id, relation_type, weight}]`` ordered by
        weight desc, capped at ``limit``. Lightweight (ids + type + weight
        only — no document content, so no defang needed).
        """
        rows = self.db.execute(
            """SELECT r.source_id, r.target_id, r.relation_type, r.weight
               FROM relations r
               JOIN documents s ON s.id = r.source_id
               JOIN documents t ON t.id = r.target_id
               WHERE s.invalidated_at IS NULL AND t.invalidated_at IS NULL
               ORDER BY r.weight DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

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
        # Hook-source provenance gate. Hook-sourced saves are best-effort
        # transcripts of a chat session, not deliberate operator
        # assertions — already capped at HOOK_SOURCE_CONFIDENCE_CEILING
        # and demoted by PROVENANCE_DEMOTION_FACTOR at recall. Letting
        # them drive capture-attention means "the hook-extractor recorded
        # N echoes of session noise; boost the first echo," which is the
        # inverse of the mechanism's intent (consolidate operator signal).
        # Composes with Layer 4 (PR #126) + STANDING_TIER_BLOCKED_SOURCE_CLIENTS:
        # the same provenance set that can't be promoted to standing tier
        # also can't drive an auto-boost.
        #
        # Surfaced 2026-05-27 — Phase A soak boost-rate hit 232/325 = 0.714
        # (ceiling 0.25); canonicals were "Session: pr merged, continue"
        # handoff fragments. Skipping here brings firing back to the
        # operator-authored signal the mechanism was designed for.
        if source_client in HOOK_SOURCE_CLIENTS:
            return {
                "fired": False, "canonical_id": None,
                "neighbors": [], "boost_applied": 0.0,
                "reason": "hook_sourced_save",
            }

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

    # ── Salience tier Phase 1 — standing-context recall ──────────────
    # private/mnemon-salience-tier-plan-260521.md
    #
    # Standing-tier memories are injected unconditionally into the
    # <mnemon-context> envelope on every prompt. The cap is the
    # contract: never exceed STANDING_TIER_HARD_CEILING. Per the
    # 2026-05-22 reframing, Phase 1 ships gated; the validation is
    # operator-flips-flag + observes ≥1 week soak for runway-style
    # under-weighting recurrence vs absence.

    def promote_to_standing(self, doc_id: int) -> bool:
        """Promote a memory to the capped standing tier.

        Raises:
            StandingTierCapReached: at the runtime cap
                (``STANDING_TIER_DEFAULT_CAP`` live standing docs)
            StandingTierProvenanceRejected: source_client is in
                ``STANDING_TIER_BLOCKED_SOURCE_CLIENTS`` (Layer 4
                composition — hook-sourced cannot promote)
            StandingTierError: doc not found or invalidated

        Returns True on success. Idempotent — re-promoting an
        already-standing doc returns True without counting against
        the cap. The hard ceiling
        (``STANDING_TIER_HARD_CEILING``) is an invariant the runtime
        cap never exceeds even if an operator overrides
        ``STANDING_TIER_DEFAULT_CAP`` upward.
        """
        row = self.db.execute(
            "SELECT tier, source_client, invalidated_at FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
        if not row:
            raise StandingTierError(f"memory #{doc_id} not found")
        if row["invalidated_at"] is not None:
            raise StandingTierError(
                f"memory #{doc_id} is invalidated — cannot promote"
            )
        if row["source_client"] in STANDING_TIER_BLOCKED_SOURCE_CLIENTS:
            raise StandingTierProvenanceRejected(
                f"memory #{doc_id} is hook-sourced ({row['source_client']}); "
                "only user-authored memories can be promoted "
                "(Layer 4 composition — auto-mirror / session_extractor "
                "captures cannot be elevated to unconditional injection)"
            )
        if row["tier"] == "standing":
            return True  # idempotent re-promote

        # Cap enforcement scoped to live rows.
        current = self.db.execute(
            "SELECT COUNT(*) AS c FROM documents "
            "WHERE tier = 'standing' AND invalidated_at IS NULL"
        ).fetchone()["c"]
        cap = min(STANDING_TIER_DEFAULT_CAP, STANDING_TIER_HARD_CEILING)
        if current >= cap:
            raise StandingTierCapReached(
                f"standing tier at cap ({current}/{cap}); demote an existing "
                "member via memory_demote first. The cap is the contract — "
                "past ~20 it stops being salient and becomes noise again."
            )

        self.db.execute(
            "UPDATE documents SET tier = 'standing', updated_at = datetime('now') "
            "WHERE id = ?",
            (doc_id,),
        )
        self.db.commit()
        return True

    def demote_to_situational(self, doc_id: int) -> bool:
        """Demote a standing-tier memory back to situational.

        Idempotent — demoting an already-situational doc returns False
        (nothing to do). Returns True when an actual demote happened.
        Raises StandingTierError on a missing or invalidated doc.
        """
        row = self.db.execute(
            "SELECT tier, invalidated_at FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
        if not row:
            raise StandingTierError(f"memory #{doc_id} not found")
        if row["invalidated_at"] is not None:
            raise StandingTierError(
                f"memory #{doc_id} is invalidated — nothing to demote"
            )
        if row["tier"] != "standing":
            return False
        self.db.execute(
            "UPDATE documents SET tier = 'situational', updated_at = datetime('now') "
            "WHERE id = ?",
            (doc_id,),
        )
        self.db.commit()
        return True

    def list_standing(self) -> list[Document]:
        """Return all live standing-tier memories, ordered most-recent first.

        Consumed by ``build_context`` (hook injection path) and
        ``mnemon standing list`` (operator CLI). Includes content body
        so callers can render snippets without a second fetch.

        Salience tier Phase 3 (2026-05-27): bumps ``last_injected_at``
        on every returned row to track injection events for the
        ``standing list`` aging surface. Operator can spot stale
        standing-tier members ("this hasn't fired in 90 days, still
        load-bearing?") for review. Single batched UPDATE — cost is one
        round-trip regardless of standing-tier size.
        """
        rows = self.db.execute(
            """SELECT d.*, c.doc
               FROM documents d
               JOIN content c ON d.hash = c.hash
               WHERE d.tier = 'standing' AND d.invalidated_at IS NULL
               ORDER BY d.created_at DESC"""
        ).fetchall()
        if rows:
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            self.db.execute(
                f"UPDATE documents SET last_injected_at = datetime('now') "
                f"WHERE id IN ({placeholders})",
                ids,
            )
            self.db.commit()
        return [_row_to_document(r) for r in rows]

    def find_clusters(
        self,
        *,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 2,
        recent_days: int = 30,
        max_clusters: int = 20,
    ) -> list[list[dict[str, Any]]]:
        """Find vector-similarity clusters in the recent live vault.

        Capture-attention Phase C primitive — surfaces N-memory clusters
        where every pair has cosine ≥ ``similarity_threshold``. Each
        cluster represents a candidate consolidation: multiple
        near-duplicate memories that operator review can fold into a
        canonical.

        Algorithm: walks live memories created within ``recent_days``,
        seeds each as a cluster, expands via vec search neighbors above
        threshold. Avoids re-seeding members that already appeared in
        a discovered cluster (deterministic dedup by member-set).

        Returns ``[[member_dict, ...], ...]`` — each member dict has
        ``{id, title, content_type, confidence, access_count,
        recurrence_count, created_at}``. Members within a cluster are
        ordered by ``recurrence_count DESC, confidence DESC`` so the
        first member is the natural canonical pick.

        Embedding-only, no LLM dependency.
        """
        import datetime as _dt

        cutoff = (
            _dt.datetime.now() - _dt.timedelta(days=recent_days)
        ).isoformat(sep=" ")
        rows = self.db.execute(
            """SELECT d.id, d.hash, d.title, d.content_type, d.confidence,
                      d.access_count, d.recurrence_count, d.created_at,
                      c.doc AS content
               FROM documents d
               JOIN content c ON d.hash = c.hash
               WHERE d.invalidated_at IS NULL
                 AND d.created_at >= ?
               ORDER BY d.id""",
            (cutoff,),
        ).fetchall()
        if len(rows) < min_cluster_size:
            return []

        docs_by_id = {r["id"]: dict(r) for r in rows}

        try:
            from .embedder import embed
        except ImportError:
            return []

        already_clustered: set[int] = set()
        clusters: list[list[int]] = []

        for seed_id in sorted(docs_by_id.keys()):
            if seed_id in already_clustered:
                continue
            seed = docs_by_id[seed_id]
            try:
                seed_emb = embed(
                    f"title: {seed['title']} | text: {seed['content']}"
                )
                neighbors = self.search_vector(seed_emb, 10)
            except Exception:
                continue
            cluster_ids = {seed_id}
            for cand in neighbors:
                if cand.doc_id == seed_id:
                    continue
                if cand.score < similarity_threshold:
                    continue
                if cand.doc_id not in docs_by_id:
                    continue
                cluster_ids.add(cand.doc_id)
            if len(cluster_ids) >= min_cluster_size:
                clusters.append(sorted(cluster_ids))
                already_clustered.update(cluster_ids)
            if len(clusters) >= max_clusters:
                break

        out: list[list[dict[str, Any]]] = []
        for cluster_ids in clusters:
            members = [docs_by_id[i] for i in cluster_ids]
            members.sort(
                key=lambda m: (m["recurrence_count"], m["confidence"]),
                reverse=True,
            )
            out.append([
                {
                    "id": m["id"],
                    "title": m["title"],
                    "content_type": m["content_type"],
                    "confidence": m["confidence"],
                    "access_count": m["access_count"],
                    "recurrence_count": m["recurrence_count"],
                    "created_at": m["created_at"],
                }
                for m in members
            ])
        return out

    def consolidate_cluster(
        self, cluster_ids: list[int],
    ) -> dict[str, Any]:
        """Consolidate a cluster: keep the first id as canonical, mark
        the rest as superseded via 'supersedes' relations + forget.

        Phase C is operator-review-only (ROADMAP invariant); this
        primitive is called by `mnemon consolidate --apply <idx>`
        AFTER explicit operator confirmation. The CLI handles the
        interactive gate.

        Returns ``{canonical_id, superseded_ids, errors: [...]}``.
        Empty / single-member clusters are no-ops.
        """
        result: dict[str, Any] = {
            "canonical_id": None,
            "superseded_ids": [],
            "errors": [],
        }
        if not cluster_ids or len(cluster_ids) < 2:
            return result
        canonical, *rest = cluster_ids
        canonical_doc = self.get(canonical)
        if not canonical_doc:
            result["errors"].append(
                f"canonical #{canonical} not found or invalidated"
            )
            return result
        result["canonical_id"] = canonical

        for victim in rest:
            victim_doc = self.get(victim)
            if not victim_doc:
                result["errors"].append(
                    f"member #{victim} not found or already invalidated"
                )
                continue
            self.add_relation(canonical, victim, "supersedes", 1.0)
            self.forget(victim)
            result["superseded_ids"].append(victim)

        return result

    def standing_tier_aging(self) -> list[dict[str, Any]]:
        """Standing-tier aging surface for Phase 3 observability.

        Returns one row per live standing-tier member with:
          - ``id`` / ``title`` / ``content_type``
          - ``age_days`` (since created_at)
          - ``contradiction_win_count`` (Phase 2 signal, persists for
            review even after demote — but this method returns only
            currently-standing)
          - ``last_injected_at`` (raw timestamp; None until first
            list_standing call hits it)
          - ``days_since_injected`` (None if never, else days since
            last_injected_at)

        DOES NOT bump last_injected_at — this is an OBSERVATION call,
        not an injection event. ``list_standing`` is the injection
        path and owns the bump.

        Operator-facing — surfaces "promoted 90 days ago, never fired"
        as a candidate for ``memory_demote`` (Phase 3 doesn't
        auto-demote per the plan invariant)."""
        import datetime as _dt
        rows = self.db.execute(
            """SELECT id, title, content_type, confidence,
                      created_at, last_injected_at,
                      contradiction_win_count, correction_count
               FROM documents
               WHERE tier = 'standing' AND invalidated_at IS NULL
               ORDER BY created_at DESC"""
        ).fetchall()
        now = _dt.datetime.now()

        def _days_since(ts: str | None) -> float | None:
            if not ts:
                return None
            try:
                t = _dt.datetime.fromisoformat(str(ts).replace("Z", ""))
            except (ValueError, TypeError):
                return None
            return max((now - t).total_seconds() / 86400.0, 0.0)

        out: list[dict[str, Any]] = []
        for r in rows:
            age_days = _days_since(r["created_at"]) or 0.0
            since_inj = _days_since(r["last_injected_at"])
            out.append({
                "id": r["id"],
                "title": r["title"],
                "content_type": r["content_type"],
                "confidence": r["confidence"],
                "age_days": round(age_days, 1),
                "contradiction_win_count": r["contradiction_win_count"],
                "correction_count": r["correction_count"],
                "last_injected_at": r["last_injected_at"],
                "days_since_injected": (
                    round(since_inj, 1) if since_inj is not None else None
                ),
            })
        return out

    def salience_report(
        self, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Rank live situational memories by Phase 2 promotion-signal
        score = ``correction_count + contradiction_win_count``.

        Operator surfaces the candidates that have been corrected or
        won contradictions most often — those are the load-bearing
        facts the standing-tier should consider promoting. Excludes
        memories already on the standing tier (no point recommending
        promotion of something already promoted) and any hook-sourced
        memory (Layer 4 + STANDING_TIER_BLOCKED_SOURCE_CLIENTS apply
        at the recommendation surface too).

        Returns ``[]`` when no candidates meet the
        ``correction_count > 0 OR contradiction_win_count > 0``
        filter — surface stays empty rather than spamming creation-time
        zeros.
        """
        rows = self.db.execute(
            """SELECT id, title, content_type, confidence,
                      correction_count, contradiction_win_count,
                      source_client, created_at
               FROM documents
               WHERE invalidated_at IS NULL
                 AND tier = 'situational'
                 AND (correction_count > 0 OR contradiction_win_count > 0)
               ORDER BY (correction_count + contradiction_win_count) DESC,
                        correction_count DESC,
                        contradiction_win_count DESC,
                        id DESC
               LIMIT ?""",
            (limit * 4,),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            if r["source_client"] in STANDING_TIER_BLOCKED_SOURCE_CLIENTS:
                # Hook-sourced can't be promoted to standing tier anyway —
                # don't recommend them.
                continue
            results.append({
                "id": r["id"],
                "title": r["title"],
                "content_type": r["content_type"],
                "confidence": r["confidence"],
                "correction_count": r["correction_count"],
                "contradiction_win_count": r["contradiction_win_count"],
                "score": r["correction_count"] + r["contradiction_win_count"],
                "created_at": r["created_at"],
            })
            if len(results) >= limit:
                break
        return results

    def attention_report(
        self, limit: int = 20, min_access_count: int = 2,
    ) -> list[dict[str, Any]]:
        """Rank live memories by ``access_count × recency`` for the
        capture-attention Phase B consolidation feedback loop.

        Recency factor: e^(-age_days / RECENCY_HALF_LIFE_DAYS) so a
        memory accessed 30 days ago counts roughly half as much as one
        accessed today, mirroring the search composite-score recency
        decay. High-access fragments are the load-bearing facts the
        standing tier should consider promoting (composes with the
        Salience Phase 2 promotion-signals work — both surface the
        same "this memory is actually being used a lot" signal).

        ``min_access_count`` filters out the tail (every memory has at
        least one access from creation-time get). Default 2 keeps the
        report relevant.
        """
        rows = self.db.execute(
            """SELECT d.id, d.title, d.content_type, d.confidence,
                      d.access_count, d.created_at, d.tier
               FROM documents d
               WHERE d.invalidated_at IS NULL
                 AND d.access_count >= ?
               ORDER BY d.access_count DESC
               LIMIT ?""",
            (min_access_count, limit * 4),  # over-fetch for recency rerank
        ).fetchall()

        import datetime as _dt
        import math as _math
        now = _dt.datetime.now()
        scored: list[dict[str, Any]] = []
        for r in rows:
            try:
                created = _dt.datetime.fromisoformat(
                    str(r["created_at"]).replace("Z", "")
                )
                age_days = max((now - created).total_seconds() / 86400.0, 0.0)
            except (ValueError, TypeError):
                age_days = 0.0
            recency = _math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)
            score = r["access_count"] * recency
            scored.append({
                "id": r["id"],
                "title": r["title"],
                "content_type": r["content_type"],
                "confidence": r["confidence"],
                "access_count": r["access_count"],
                "age_days": round(age_days, 1),
                "recency": round(recency, 3),
                "score": round(score, 2),
                "tier": r["tier"],
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def standing_tier_status(self) -> dict[str, Any]:
        """Stats for ``mnemon status`` + dashboards: current count vs cap."""
        current = self.db.execute(
            "SELECT COUNT(*) AS c FROM documents "
            "WHERE tier = 'standing' AND invalidated_at IS NULL"
        ).fetchone()["c"]
        return {
            "count": current,
            "cap": STANDING_TIER_DEFAULT_CAP,
            "hard_ceiling": STANDING_TIER_HARD_CEILING,
        }

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
