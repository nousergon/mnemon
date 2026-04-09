/**
 * In-process vector store — brute-force cosine similarity over Float32Arrays.
 *
 * Stores vectors in a flat binary file alongside the SQLite vault.
 * Sub-millisecond search for <10k documents. No native extensions needed.
 */

import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";

interface VectorEntry {
  id: string; // "{content_hash}_{seq}"
  embedding: Float32Array;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export class VecStore {
  private vectors: Map<string, Float32Array>;
  private filePath: string;
  private dim: number;
  private dirty: boolean;

  constructor(filePath: string, dim = 768) {
    this.filePath = filePath;
    this.dim = dim;
    this.vectors = new Map();
    this.dirty = false;
    this._load();
  }

  /**
   * Add or replace a vector.
   */
  set(id: string, embedding: Float32Array): void {
    this.vectors.set(id, embedding);
    this.dirty = true;
  }

  /**
   * Find the top-k most similar vectors to the query.
   */
  search(query: Float32Array, k = 20): Array<{ id: string; similarity: number }> {
    const results: Array<{ id: string; similarity: number }> = [];

    for (const [id, emb] of this.vectors) {
      results.push({ id, similarity: cosineSimilarity(query, emb) });
    }

    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, k);
  }

  /**
   * Number of stored vectors.
   */
  size(): number {
    return this.vectors.size;
  }

  /**
   * Check if a vector exists.
   */
  has(id: string): boolean {
    return this.vectors.has(id);
  }

  /**
   * Remove a vector.
   */
  delete(id: string): boolean {
    const existed = this.vectors.delete(id);
    if (existed) this.dirty = true;
    return existed;
  }

  /**
   * Persist to disk. Call after batch writes.
   */
  save(): void {
    if (!this.dirty) return;

    const dir = dirname(this.filePath);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

    // Format: [uint32 dim] [uint32 count] then for each entry:
    //   [uint32 id_len] [utf8 id_bytes] [float32[dim] embedding]
    const entries = Array.from(this.vectors.entries());
    const idBuffers = entries.map(([id]) => Buffer.from(id, "utf-8"));

    // Calculate total size
    let totalSize = 8; // dim + count headers
    for (const idBuf of idBuffers) {
      totalSize += 4 + idBuf.length + this.dim * 4;
    }

    const buf = Buffer.alloc(totalSize);
    let offset = 0;

    buf.writeUInt32LE(this.dim, offset);
    offset += 4;
    buf.writeUInt32LE(entries.length, offset);
    offset += 4;

    for (let i = 0; i < entries.length; i++) {
      const [, embedding] = entries[i]!;
      const idBuf = idBuffers[i]!;

      buf.writeUInt32LE(idBuf.length, offset);
      offset += 4;
      idBuf.copy(buf, offset);
      offset += idBuf.length;

      const embBuf = Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength);
      embBuf.copy(buf, offset);
      offset += this.dim * 4;
    }

    writeFileSync(this.filePath, buf);
    this.dirty = false;
  }

  /**
   * Load from disk.
   */
  private _load(): void {
    if (!existsSync(this.filePath)) return;

    try {
      const buf = readFileSync(this.filePath);
      let offset = 0;

      const dim = buf.readUInt32LE(offset);
      offset += 4;
      if (dim !== this.dim) {
        console.error(`Vector dimension mismatch: file has ${dim}, expected ${this.dim}. Ignoring stored vectors.`);
        return;
      }

      const count = buf.readUInt32LE(offset);
      offset += 4;

      for (let i = 0; i < count; i++) {
        const idLen = buf.readUInt32LE(offset);
        offset += 4;
        const id = buf.toString("utf-8", offset, offset + idLen);
        offset += idLen;

        const embedding = new Float32Array(buf.buffer.slice(buf.byteOffset + offset, buf.byteOffset + offset + dim * 4));
        offset += dim * 4;

        this.vectors.set(id, embedding);
      }
    } catch (err) {
      console.error("Failed to load vector store:", err);
    }
  }
}
