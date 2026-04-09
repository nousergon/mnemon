"""In-process vector store — brute-force cosine similarity with numpy.

Stores vectors in a .npy file alongside the SQLite vault.
Sub-millisecond search for <10k documents. No native extensions needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VecStore:
    def __init__(self, file_path: str | Path, dim: int = 384):
        self.file_path = Path(file_path)
        self.dim = dim
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None  # shape: (n, dim)
        self._dirty = False
        self._load()

    def set(self, vec_id: str, embedding: np.ndarray) -> None:
        """Add or replace a vector."""
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (self.dim,):
            raise ValueError(f"Expected dim {self.dim}, got {embedding.shape}")

        if vec_id in self._ids:
            idx = self._ids.index(vec_id)
            self._vectors[idx] = embedding
        else:
            self._ids.append(vec_id)
            if self._vectors is None:
                self._vectors = embedding.reshape(1, -1)
            else:
                self._vectors = np.vstack([self._vectors, embedding.reshape(1, -1)])
        self._dirty = True

    def search(self, query: np.ndarray, k: int = 20) -> list[dict]:
        """Find the top-k most similar vectors to the query."""
        if self._vectors is None or len(self._ids) == 0:
            return []

        query = np.asarray(query, dtype=np.float32)
        # Cosine similarity: dot(q, v) / (||q|| * ||v||)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        vec_norms = np.linalg.norm(self._vectors, axis=1)
        # Avoid division by zero
        nonzero = vec_norms > 0
        similarities = np.zeros(len(self._ids))
        similarities[nonzero] = (
            self._vectors[nonzero] @ query / (vec_norms[nonzero] * query_norm)
        )

        top_k = min(k, len(self._ids))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [
            {"id": self._ids[i], "similarity": float(similarities[i])}
            for i in top_indices
        ]

    def size(self) -> int:
        return len(self._ids)

    def has(self, vec_id: str) -> bool:
        return vec_id in self._ids

    def delete(self, vec_id: str) -> bool:
        if vec_id not in self._ids:
            return False
        idx = self._ids.index(vec_id)
        self._ids.pop(idx)
        if self._vectors is not None:
            self._vectors = np.delete(self._vectors, idx, axis=0)
            if len(self._ids) == 0:
                self._vectors = None
        self._dirty = True
        return True

    def save(self) -> None:
        """Persist to disk."""
        if not self._dirty:
            return
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dim": self.dim,
            "ids": self._ids,
            "vectors": self._vectors if self._vectors is not None else np.empty((0, self.dim), dtype=np.float32),
        }
        np.savez(str(self.file_path), **data)
        self._dirty = False

    def _load(self) -> None:
        """Load from disk."""
        npz_path = Path(str(self.file_path) + ".npz") if not str(self.file_path).endswith(".npz") else self.file_path
        if not npz_path.exists():
            return
        try:
            data = np.load(str(npz_path), allow_pickle=True)
            dim = int(data["dim"])
            if dim != self.dim:
                return
            ids = data["ids"].tolist()
            vectors = data["vectors"]
            if len(ids) > 0 and vectors.shape[0] == len(ids):
                self._ids = ids
                self._vectors = vectors.astype(np.float32)
        except Exception:
            pass
