/**
 * Storage layer — SQLite + FTS5 + sqlite-vec.
 *
 * Content-addressable storage with full-text and vector search.
 * Single-file vault at ~/.mnemon/default.sqlite.
 */

import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { createHash } from "node:crypto";
import { mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// ── Types ────────────────────────────────────────────────────────────────────

export type ContentType =
  | "decision"
  | "preference"
  | "antipattern"
  | "observation"
  | "research"
  | "project"
  | "handoff"
  | "note";

export type MemoryType = "episodic" | "semantic" | "procedural";

export interface Document {
  id: number;
  collection: string | null;
  path: string | null;
  title: string;
  hash: string;
  content_type: ContentType;
  memory_type: MemoryType;
  confidence: number;
  quality_score: number;
  access_count: number;
  pinned: number;
  source_client: string | null;
  invalidated_at: string | null;
  invalidated_by: number | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentWithContent extends Document {
  doc: string;
}

export interface SaveOptions {
  title: string;
  content: string;
  content_type?: ContentType;
  memory_type?: MemoryType;
  collection?: string;
  path?: string;
  source_client?: string;
  confidence?: number;
}

export interface SearchResult {
  doc_id: number;
  title: string;
  content: string;
  content_type: ContentType;
  memory_type: MemoryType;
  confidence: number;
  created_at: string;
  score: number;
  source: "bm25" | "vector" | "fused";
}

// ── Half-lives for content types (days, null = never decay) ──────────────────

const HALF_LIVES: Record<ContentType, number | null> = {
  decision: null,
  preference: null,
  antipattern: null,
  observation: 90,
  research: 90,
  project: 120,
  handoff: 30,
  note: 60,
};

// ── Content type → memory type mapping ───────────────────────────────────────

const MEMORY_TYPE_MAP: Record<ContentType, MemoryType> = {
  decision: "semantic",
  preference: "semantic",
  antipattern: "semantic",
  observation: "semantic",
  research: "semantic",
  project: "semantic",
  handoff: "episodic",
  note: "semantic",
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

function defaultVaultDir(): string {
  return join(homedir(), ".mnemon");
}

function defaultVaultPath(): string {
  return join(defaultVaultDir(), "default.sqlite");
}

// ── Store Class ──────────────────────────────────────────────────────────────

export class Store {
  db: Database;
  private vectorDim: number;
  private vectorsEnabled: boolean;

  constructor(dbPath?: string, vectorDim = 768) {
    const path = dbPath ?? defaultVaultPath();
    const dir = join(path, "..");

    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    this.db = new Database(path);
    this.vectorDim = vectorDim;
    this.vectorsEnabled = false;

    // WAL mode for concurrent reads
    this.db.run("PRAGMA journal_mode = WAL");
    this.db.run("PRAGMA busy_timeout = 15000");

    // Try to load sqlite-vec extension (may fail if SQLite doesn't support extensions)
    try {
      sqliteVec.load(this.db);
      this.vectorsEnabled = true;
    } catch {
      console.error("sqlite-vec not available — vector search disabled. BM25 search still works.");
    }

    this._initSchema();
  }

  private _initSchema(): void {
    this.db.run(`
      CREATE TABLE IF NOT EXISTS content (
        hash TEXT PRIMARY KEY,
        doc TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
      )
    `);

    this.db.run(`
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
      )
    `);

    // FTS5 virtual table (external content not needed — we manage sync manually)
    this.db.run(`
      CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        title, body,
        tokenize='porter unicode61'
      )
    `);

    // Vector table (only if sqlite-vec loaded)
    if (this.vectorsEnabled) {
      this.db.run(`
        CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
          id TEXT PRIMARY KEY,
          embedding float[${this.vectorDim}] distance_metric=cosine
        )
      `);
    }

    // Relations graph
    this.db.run(`
      CREATE TABLE IF NOT EXISTS relations (
        source_id INTEGER NOT NULL REFERENCES documents(id),
        target_id INTEGER NOT NULL REFERENCES documents(id),
        relation_type TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        created_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (source_id, target_id, relation_type)
      )
    `);

    // Session log
    this.db.run(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        started_at TEXT,
        ended_at TEXT,
        summary TEXT,
        client TEXT
      )
    `);

    // Sync tracking
    this.db.run(`
      CREATE TABLE IF NOT EXISTS sync_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        direction TEXT NOT NULL,
        timestamp TEXT DEFAULT (datetime('now')),
        documents_synced INTEGER,
        source TEXT
      )
    `);
  }

  // ── Content Operations ───────────────────────────────────────────────────

  /**
   * Save a memory. Content-addressable: same content = same hash = no duplicate.
   * Returns the document ID.
   */
  save(opts: SaveOptions): number {
    const hash = sha256(opts.content);
    const contentType = opts.content_type ?? "note";
    const memoryType = opts.memory_type ?? MEMORY_TYPE_MAP[contentType] ?? "semantic";
    const confidence = opts.confidence ?? this._defaultConfidence(contentType);

    // Upsert content (idempotent)
    this.db.run(
      "INSERT OR IGNORE INTO content (hash, doc) VALUES (?, ?)",
      [hash, opts.content]
    );

    // Check if document already exists with this hash
    const existing = this.db.query<{ id: number }, [string]>(
      "SELECT id FROM documents WHERE hash = ?"
    ).get(hash);

    if (existing) {
      // Update access count and timestamp
      this.db.run(
        "UPDATE documents SET access_count = access_count + 1, updated_at = datetime('now') WHERE id = ?",
        [existing.id]
      );
      return existing.id;
    }

    // Generate path if not provided
    const path = opts.path ?? `${contentType}/${Date.now()}-${hash.slice(0, 8)}`;

    // Insert document
    const result = this.db.run(
      `INSERT INTO documents (collection, path, title, hash, content_type, memory_type, confidence, source_client)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        opts.collection ?? "default",
        path,
        opts.title,
        hash,
        contentType,
        memoryType,
        confidence,
        opts.source_client ?? null,
      ]
    );

    const docId = Number(result.lastInsertRowid);

    // Index in FTS5
    this.db.run(
      "INSERT INTO documents_fts (rowid, title, body) VALUES (?, ?, ?)",
      [docId, opts.title, opts.content]
    );

    return docId;
  }

  /**
   * Store a vector embedding for a document.
   */
  saveEmbedding(contentHash: string, seq: number, embedding: Float32Array): void {
    if (!this.vectorsEnabled) return;
    const id = `${contentHash}_${seq}`;
    this.db.run(
      "INSERT OR REPLACE INTO vectors (id, embedding) VALUES (?, ?)",
      [id, Buffer.from(embedding.buffer)]
    );
  }

  /**
   * Get a document by ID, including content.
   */
  get(docId: number): DocumentWithContent | null {
    const row = this.db.query<DocumentWithContent & { doc: string }, [number]>(`
      SELECT d.*, c.doc
      FROM documents d
      JOIN content c ON d.hash = c.hash
      WHERE d.id = ? AND d.invalidated_at IS NULL
    `).get(docId);

    if (row) {
      // Bump access count
      this.db.run(
        "UPDATE documents SET access_count = access_count + 1 WHERE id = ?",
        [docId]
      );
    }

    return row ?? null;
  }

  /**
   * Get a document by path.
   */
  getByPath(path: string, collection = "default"): DocumentWithContent | null {
    const row = this.db.query<DocumentWithContent & { doc: string }, [string, string]>(`
      SELECT d.*, c.doc
      FROM documents d
      JOIN content c ON d.hash = c.hash
      WHERE d.path = ? AND d.collection = ? AND d.invalidated_at IS NULL
    `).get(path, collection);

    return row ?? null;
  }

  /**
   * Pin a memory (boost confidence by 0.3).
   */
  pin(docId: number): boolean {
    const result = this.db.run(
      "UPDATE documents SET pinned = 1, confidence = MIN(1.0, confidence + 0.3) WHERE id = ?",
      [docId]
    );
    return result.changes > 0;
  }

  /**
   * Soft-delete a memory.
   */
  forget(docId: number): boolean {
    const result = this.db.run(
      "UPDATE documents SET invalidated_at = datetime('now') WHERE id = ? AND invalidated_at IS NULL",
      [docId]
    );
    if (result.changes > 0) {
      // Remove from FTS index
      this.db.run("DELETE FROM documents_fts WHERE rowid = ?", [docId]);
      return true;
    }
    return false;
  }

  /**
   * Get recent memories in chronological order.
   */
  timeline(limit = 20, contentType?: ContentType): DocumentWithContent[] {
    const typeClause = contentType ? "AND d.content_type = ?" : "";
    const params = contentType ? [contentType, limit] : [limit];

    return this.db.query<DocumentWithContent, any[]>(`
      SELECT d.*, c.doc
      FROM documents d
      JOIN content c ON d.hash = c.hash
      WHERE d.invalidated_at IS NULL ${typeClause}
      ORDER BY d.created_at DESC, d.id DESC
      LIMIT ?
    `).all(...params);
  }

  // ── Search ───────────────────────────────────────────────────────────────

  /**
   * BM25 full-text search via FTS5.
   */
  searchBM25(query: string, limit = 20): SearchResult[] {
    // Escape special FTS5 characters and build prefix query
    const safeQuery = query
      .replace(/['"]/g, "")
      .split(/\s+/)
      .filter(Boolean)
      .map((t) => `"${t}"*`)
      .join(" OR ");

    if (!safeQuery) return [];

    try {
      const rows = this.db.query<any, [string, number]>(`
        SELECT
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
        LIMIT ?
      `).all(safeQuery, limit);

      return rows.map((r: any) => ({
        ...r,
        score: r.bm25_score,
        source: "bm25" as const,
      }));
    } catch {
      // FTS5 query syntax can fail on unusual input
      return [];
    }
  }

  /**
   * Vector similarity search via sqlite-vec.
   */
  searchVector(embedding: Float32Array, limit = 20): SearchResult[] {
    if (!this.vectorsEnabled) return [];
    try {
      const rows = this.db.query<any, [Buffer, number]>(`
        SELECT
          v.id AS vec_id,
          v.distance,
          d.id AS doc_id,
          d.title,
          c.doc AS content,
          d.content_type,
          d.memory_type,
          d.confidence,
          d.created_at
        FROM vectors v
        JOIN documents d ON d.hash = SUBSTR(v.id, 1, INSTR(v.id, '_') - 1)
        JOIN content c ON d.hash = c.hash
        WHERE v.embedding MATCH ?
          AND d.invalidated_at IS NULL
          AND k = ?
        ORDER BY v.distance
      `).all(Buffer.from(embedding.buffer), limit);

      return rows.map((r: any) => ({
        doc_id: r.doc_id,
        title: r.title,
        content: r.content,
        content_type: r.content_type,
        memory_type: r.memory_type,
        confidence: r.confidence,
        created_at: r.created_at,
        score: 1 - r.distance, // cosine distance → similarity
        source: "vector" as const,
      }));
    } catch {
      return [];
    }
  }

  // ── Relations ────────────────────────────────────────────────────────────

  /**
   * Find documents related to a given document via the graph.
   */
  getRelated(docId: number, limit = 10): Array<DocumentWithContent & { relation_type: string; weight: number }> {
    return this.db.query<any, [number, number, number]>(`
      SELECT d.*, c.doc, r.relation_type, r.weight
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
      LIMIT ?
    `).all(docId, docId, limit);
  }

  /**
   * Add a relation between two documents.
   */
  addRelation(sourceId: number, targetId: number, relationType: string, weight = 1.0): void {
    this.db.run(
      "INSERT OR REPLACE INTO relations (source_id, target_id, relation_type, weight) VALUES (?, ?, ?, ?)",
      [sourceId, targetId, relationType, weight]
    );
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────

  /**
   * Vault health stats.
   */
  status(): Record<string, any> {
    const totalDocs = this.db.query<{ count: number }, []>(
      "SELECT COUNT(*) as count FROM documents WHERE invalidated_at IS NULL"
    ).get()!.count;

    const byType = this.db.query<{ content_type: string; count: number }, []>(
      "SELECT content_type, COUNT(*) as count FROM documents WHERE invalidated_at IS NULL GROUP BY content_type ORDER BY count DESC"
    ).all();

    let totalVectors = 0;
    if (this.vectorsEnabled) {
      totalVectors = this.db.query<{ count: number }, []>(
        "SELECT COUNT(*) as count FROM vectors"
      ).get()!.count;
    }

    const invalidated = this.db.query<{ count: number }, []>(
      "SELECT COUNT(*) as count FROM documents WHERE invalidated_at IS NOT NULL"
    ).get()!.count;

    const pinned = this.db.query<{ count: number }, []>(
      "SELECT COUNT(*) as count FROM documents WHERE pinned = 1 AND invalidated_at IS NULL"
    ).get()!.count;

    return {
      total_documents: totalDocs,
      by_type: byType,
      total_vectors: totalVectors,
      invalidated,
      pinned,
      vault_path: this.db.filename,
    };
  }

  /**
   * Archive stale documents based on half-life.
   */
  sweep(dryRun = true): { archived: number; candidates: Array<{ id: number; title: string; content_type: string; age_days: number }> } {
    const candidates: Array<{ id: number; title: string; content_type: string; age_days: number }> = [];

    for (const [contentType, halfLife] of Object.entries(HALF_LIVES)) {
      if (halfLife === null) continue;

      const stale = this.db.query<any, [string, number]>(`
        SELECT id, title, content_type,
               CAST(julianday('now') - julianday(updated_at) AS INTEGER) AS age_days
        FROM documents
        WHERE content_type = ?
          AND invalidated_at IS NULL
          AND pinned = 0
          AND julianday('now') - julianday(updated_at) > ?
        ORDER BY updated_at ASC
      `).all(contentType, halfLife);

      candidates.push(...stale);
    }

    if (!dryRun) {
      for (const c of candidates) {
        this.forget(c.id);
      }
    }

    return { archived: dryRun ? 0 : candidates.length, candidates };
  }

  private _defaultConfidence(contentType: ContentType): number {
    const defaults: Record<ContentType, number> = {
      decision: 0.85,
      preference: 0.80,
      antipattern: 0.80,
      observation: 0.70,
      research: 0.70,
      project: 0.65,
      handoff: 0.60,
      note: 0.50,
    };
    return defaults[contentType] ?? 0.5;
  }

  close(): void {
    this.db.close();
  }
}
