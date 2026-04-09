"""Embedding pipeline — FastEmbed with bge-small-en-v1.5 (ONNX).

384-dimensional embeddings via ONNX Runtime. ~13MB model, auto-downloaded.
No PyTorch dependency needed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from .store import Store

VECTOR_DIM = 384
_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def _get_model():
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=_MODEL_NAME)
    return _model


def embed(text: str) -> "np.ndarray":
    """Embed a single text string. Returns ndarray of shape (384,)."""
    import numpy as np
    model = _get_model()
    result = list(model.embed([text]))
    return np.asarray(result[0], dtype=np.float32)


def embed_batch(texts: list[str]) -> list["np.ndarray"]:
    """Embed multiple texts."""
    import numpy as np
    model = _get_model()
    results = list(model.embed(texts))
    return [np.asarray(r, dtype=np.float32) for r in results]


def fragmentize(title: str, content: str) -> list[dict]:
    """Split a document into fragments for embedding.

    Returns list of {seq, text} dicts:
      seq=0: full document (title + content, truncated to 2000 chars)
      seq=1-5: individual sections split by markdown headers or double newlines
    """
    fragments = []

    # seq=0: full document
    full_text = f"title: {title} | text: {content}"[:2000]
    fragments.append({"seq": 0, "text": full_text})

    # Split by markdown headers or double newlines
    sections = re.split(r"(?=^#{1,3}\s)", content, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if len(s.strip()) > 50]

    for i, section in enumerate(sections[:5]):
        fragments.append({
            "seq": i + 1,
            "text": f"title: {title} | section: {section[:1000]}",
        })

    return fragments


def embed_document(store: "Store", content_hash: str, title: str, content: str) -> int:
    """Embed and store all fragments for a document. Returns fragment count."""
    fragments = fragmentize(title, content)
    count = 0

    for frag in fragments:
        emb = embed(frag["text"])
        store.save_embedding(content_hash, frag["seq"], emb)
        count += 1

    store.flush_vectors()
    return count
