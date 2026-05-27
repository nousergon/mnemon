"""NLI (Natural Language Inference) classifier for pair-wise memory
relationships — used by contradiction detection.

Replaces the prior LLM-based classifier (``llm.generate``) for the
``check_contradictions`` path. Same operational shape as FastEmbed:
ONNX runtime + tokenizer, lazy-loaded singleton, pre-warm at lifespan
startup, no PyTorch dependency. All transitive deps (``onnxruntime``,
``tokenizers``, ``huggingface_hub``) ship with FastEmbed already, so
this module adds zero new requirements on the Fly image.

Model: ``cross-encoder/nli-deberta-v3-xsmall`` (22M params, ~87 MB
INT8 quantized). Trained on MNLI / SNLI / FEVER. Outputs three
labels: ``contradiction``, ``entailment``, ``neutral``.

Per Brian's 2026-05-21 "no LLM in mnemon" decision: this is the
embedded-ML primitive for contradiction detection. Public-release
operators get it out of the box (auto-downloads on first use OR
baked into the Fly Docker image). Composes with the same SOTA-for-
public-release-constraint reasoning that drove the embedding-based
exemplar scorer in ``scripts/build_standing_set.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# Model identity. INT8 quantized variant is the operational default —
# ~87 MB, x86 AVX-512 optimized (Fly hosts are x86), ~10ms / pair on
# CPU inference. FP32 (``onnx/model.onnx``) is larger but available
# via the ``MNEMON_NLI_ONNX_VARIANT`` env override.
MODEL_REPO = "cross-encoder/nli-deberta-v3-xsmall"
ONNX_FILE_DEFAULT = "onnx/model_qint8_avx512.onnx"
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "config.json"

# Label canonical from model config; verified at load.
EXPECTED_LABELS = {"contradiction", "entailment", "neutral"}


class NLIResult(NamedTuple):
    """Single-pair classification result.

    ``label`` is the argmax of the three probabilities. ``probs`` is a
    dict keyed by canonical label so callers don't depend on the
    model's internal index order.
    """
    label: str
    probs: dict[str, float]


_session: Any = None
_tokenizer: Any = None
_id2label: dict[int, str] | None = None
_init_lock: Any = None


def _model_dir() -> Path:
    """Where downloaded NLI files live. Honors ``MNEMON_NLI_MODEL_DIR``
    so an operator can point at a baked-in location (e.g., the Fly
    image's ``/app/.cache/huggingface``).

    Cache resolution interacts with ``HF_HOME`` (set on Fly to
    ``/app/.cache/huggingface``): the huggingface_hub library writes
    its downloads under ``{HF_HOME}/hub``; this function's default
    resolves to ``$HOME/.cache/huggingface/hub`` which coincides
    with the HF_HOME path when HOME=/root. Override both env vars
    together if you need to point at a non-default location — see
    the audit comment in ``Dockerfile`` (2026-05-27)."""
    return Path(os.environ.get(
        "MNEMON_NLI_MODEL_DIR",
        Path.home() / ".cache" / "huggingface" / "hub",
    ))


def _ensure_loaded() -> None:
    """Lazy-load model + tokenizer (singleton). Auto-downloads from
    HuggingFace on first use if not already cached.

    Raises ``NLIUnavailableError`` if the model can't be loaded —
    surface to callers rather than silently degrading.
    """
    global _session, _tokenizer, _id2label, _init_lock

    if _session is not None:
        return

    if _init_lock is None:
        import threading
        _init_lock = threading.Lock()

    with _init_lock:
        if _session is not None:
            return

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise NLIUnavailableError(
                f"huggingface_hub not installed — should ship with FastEmbed: {e}"
            ) from e

        variant = os.environ.get("MNEMON_NLI_ONNX_VARIANT", ONNX_FILE_DEFAULT)

        try:
            onnx_path = hf_hub_download(repo_id=MODEL_REPO, filename=variant)
            tokenizer_path = hf_hub_download(repo_id=MODEL_REPO, filename=TOKENIZER_FILE)
            config_path = hf_hub_download(repo_id=MODEL_REPO, filename=CONFIG_FILE)
        except Exception as e:
            raise NLIUnavailableError(
                f"NLI model download failed ({MODEL_REPO}): {e}"
            ) from e

        # Verify the model's label space matches our expectations. If
        # someone overrides MODEL_REPO to a model with a different
        # label set, we want a fast, named failure here — not a silent
        # mis-classification downstream.
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        raw_id2label = cfg.get("id2label", {})
        id2label = {int(k): v for k, v in raw_id2label.items()}
        if set(id2label.values()) != EXPECTED_LABELS:
            raise NLIUnavailableError(
                f"unexpected label set in {MODEL_REPO}: {id2label.values()}; "
                f"expected {EXPECTED_LABELS}"
            )

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise NLIUnavailableError(
                f"onnxruntime or tokenizers missing — should ship with FastEmbed: {e}"
            ) from e

        try:
            _session = ort.InferenceSession(onnx_path)
            _tokenizer = Tokenizer.from_file(tokenizer_path)
            _id2label = id2label
            logger.info(
                "nli: loaded %s (%s) — labels=%s",
                MODEL_REPO, variant, list(id2label.values()),
            )
        except Exception as e:
            raise NLIUnavailableError(
                f"NLI model load failed: {e}"
            ) from e


class NLIUnavailableError(RuntimeError):
    """Raised when NLI inference can't run — typically a download /
    load failure on first use. Callers in best-effort paths
    (``check_contradictions``) catch + degrade with a clear message;
    fail-loud per ``feedback_no_silent_fails``."""


def classify_pair(premise: str, hypothesis: str) -> NLIResult:
    """Classify a single premise / hypothesis pair.

    Returns ``NLIResult(label, probs)`` with the argmax label and a
    canonical-keyed probability dict.

    Raises ``NLIUnavailableError`` on download / load failure.
    """
    _ensure_loaded()
    assert _session is not None and _tokenizer is not None and _id2label is not None

    enc = _tokenizer.encode(premise, hypothesis)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)

    inputs: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    # token_type_ids only required by some architectures (BERT-family);
    # DeBERTa-v3-xsmall doesn't take it, but include defensively for
    # the operator-override case where MODEL_REPO points at a BERT-style
    # model. The session check makes this a no-op for DeBERTa.
    input_names = {i.name for i in _session.get_inputs()}
    if "token_type_ids" in input_names:
        inputs["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)

    logits = _session.run(None, inputs)[0]
    # Softmax for interpretable probabilities
    exp = np.exp(logits[0] - np.max(logits[0]))
    probs_arr = exp / exp.sum()
    probs = {_id2label[i]: float(probs_arr[i]) for i in range(len(probs_arr))}
    label = max(probs, key=probs.__getitem__)
    return NLIResult(label=label, probs=probs)


class BidirectionalResult(NamedTuple):
    """Result of running NLI in both directions on a pair (a, b).

    Disambiguates ``same`` from ``update`` for mnemon's
    contradiction-detection taxonomy:
      - both directions entail → ``same`` (semantic equivalence)
      - one direction entails, other neutral → ``update``
        (the entailing direction "supersedes" the other)
      - either direction is contradiction → ``contradiction``
      - both neutral → ``unrelated``
    """
    mnemon_label: str  # "same" | "update" | "contradiction" | "unrelated"
    a_implies_b: NLIResult
    b_implies_a: NLIResult


# Mnemon's contradiction taxonomy maps from NLI by combining both
# directions of the pair. Bidirectional inference is cheap — same
# model, two inferences, ~10-20ms total on CPU INT8.
def classify_pair_bidirectional(a: str, b: str) -> BidirectionalResult:
    """Classify the relationship between two memories in both directions.

    Conventions: ``a`` is treated as the "existing" memory, ``b`` as
    the "new" one. ``b_implies_a == entailment`` while
    ``a_implies_b != entailment`` means ``b`` is a stronger /
    more-detailed restatement → ``update`` (the new supersedes the
    old). ``a_implies_b == contradiction`` (either direction) means
    direct conflict.
    """
    a_to_b = classify_pair(a, b)
    b_to_a = classify_pair(b, a)

    # Contradiction in either direction is contradiction
    if a_to_b.label == "contradiction" or b_to_a.label == "contradiction":
        mnemon = "contradiction"
    elif a_to_b.label == "entailment" and b_to_a.label == "entailment":
        # Both directions entail → semantic equivalence
        mnemon = "same"
    elif b_to_a.label == "entailment":
        # New entails old (b → a) but not vice versa → new is stronger
        # / more-detailed → update
        mnemon = "update"
    elif a_to_b.label == "entailment":
        # Old entails new (a → b) but not vice versa → new is a
        # weaker subset / less-detailed restatement → still "same"
        # rather than "update"; the existing memory dominates
        mnemon = "same"
    else:
        # Both neutral → unrelated
        mnemon = "unrelated"

    return BidirectionalResult(
        mnemon_label=mnemon,
        a_implies_b=a_to_b,
        b_implies_a=b_to_a,
    )


def is_available() -> bool:
    """Check if NLI inference can run. Returns True iff the model is
    already loaded OR the dependencies + network needed to download it
    are present. Cheap probe — doesn't actually load if unloaded;
    callers that need a guaranteed working model should call
    ``_ensure_loaded()`` directly and catch ``NLIUnavailableError``.
    """
    if _session is not None:
        return True
    try:
        import onnxruntime  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401
        return True
    except ImportError:
        return False


def prewarm() -> None:
    """Pre-load the model at lifespan startup so the first
    classification doesn't pay the cold-load cost. Mirrors FastEmbed's
    pre-warm pattern in ``server_remote.py``. Safe to call multiple
    times — idempotent via the singleton check in ``_ensure_loaded``.

    Swallowed errors: pre-warm is best-effort. Failures here just
    mean the first real classification will see the cold-load cost
    (and surface any persistent failure as ``NLIUnavailableError``
    there).
    """
    try:
        _ensure_loaded()
    except NLIUnavailableError as e:
        # Pre-warm is secondary observability — primary deliverable
        # (the server) survives without it; first real call will
        # surface the named exception. Acceptable swallow per
        # feedback_no_silent_fails category (b).
        logger.warning("nli: pre-warm failed: %s; will retry on first call", e)
