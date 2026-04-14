"""Local LLM abstraction — QMD-query-expansion-1.7B via llama-cpp-python.

Used for observation extraction, query expansion, and contradiction detection.
Runs on Apple Silicon Metal. Auto-downloads ~1.1GB GGUF on first use.

Phase 3: local LLM integration (replaces Phase 2 regex heuristics).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_REPO = "tobil/qmd-query-expansion-1.7B-gguf"
MODEL_FILE = "qmd-query-expansion-1.7B-q4_k_m.gguf"

_llm: Any = None
_init_lock: Any = None


def _model_dir() -> Path:
    return Path(os.environ.get("MNEMON_MODEL_DIR", Path.home() / ".mnemon" / "models"))


def _ensure_model() -> Any:
    """Lazy-load the LLM (singleton). Downloads model on first use."""
    global _llm, _init_lock

    if _llm is not None:
        return _llm

    # Thread-safe init
    if _init_lock is None:
        import threading
        _init_lock = threading.Lock()

    with _init_lock:
        if _llm is not None:
            return _llm

        model_path = _resolve_model_path()

        from llama_cpp import Llama
        _llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_gpu_layers=-1,  # Use all GPU layers (Metal on macOS)
            verbose=False,
        )
        return _llm


def _resolve_model_path() -> Path:
    """Find or download the GGUF model file."""
    model_dir = _model_dir()
    local_path = model_dir / MODEL_FILE

    if local_path.exists():
        return local_path

    # Try huggingface_hub download
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(model_dir),
        )
        return Path(downloaded)
    except ImportError:
        pass

    # Fallback: check if model exists in HF cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for p in hf_cache.rglob(MODEL_FILE):
        return p

    raise FileNotFoundError(
        f"Model not found. Install huggingface-hub and run: "
        f"python -c \"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download('{MODEL_REPO}', '{MODEL_FILE}', local_dir='{model_dir}')\""
    )


def generate(system_prompt: str, user_message: str, max_tokens: int = 2000) -> str:
    """Generate text from a system prompt + user message using the local 1.7B model."""
    llm = _ensure_model()

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )

    return response["choices"][0]["message"]["content"].strip()


def is_available() -> bool:
    """Check if the LLM backend is available (llama-cpp-python installed + model exists)."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False

    try:
        _resolve_model_path()
        return True
    except FileNotFoundError:
        return False


def try_generate(
    system_prompt: str, user_message: str, max_tokens: int = 2000
) -> str | None:
    """Try to generate an LLM response, returning None if unavailable.

    Unifies the check-available / generate / fall-back plumbing shared
    by the LLM-using callers (``search.expand_query``,
    ``hooks.session_extractor``, ``hooks.handoff_generator``). Each
    caller still does its own parsing of the returned text; this helper
    only covers the "is the LLM usable right now" plumbing.

    Returns:
        The raw LLM output string, or None if the backend is unavailable
        or the call raises any exception. Callers should treat None as
        "fall back to regex/empty" rather than an error.
    """
    try:
        if not is_available():
            return None
        return generate(system_prompt, user_message, max_tokens=max_tokens)
    except Exception as exc:
        # LLM is optional infra — callers treat None as "fall back to
        # regex/empty". But silent failure hides model crashes, OOMs,
        # and llama-cpp version mismatches from the operator. Log once
        # per failure so degradation is visible.
        logger.warning("try_generate: LLM call failed (%s: %s)", type(exc).__name__, exc)
        return None
