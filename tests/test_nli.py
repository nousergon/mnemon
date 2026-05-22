"""Tests for the NLI cross-encoder module — model layer + label mapping.

Real inference is exercised by ``tests/fixtures/nli_real_inference_pairs.py``
(skipped by default; opt-in via env var). These tests cover the mnemon-
side logic (bidirectional → taxonomy mapping, error handling) with the
classifier mocked so they run fast and offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mnemon.nli import (
    BidirectionalResult,
    NLIResult,
    NLIUnavailableError,
    classify_pair_bidirectional,
    is_available,
)


def _result(label: str, **probs) -> NLIResult:
    """Build an NLIResult with explicit probs (defaults sum to 1.0)."""
    full = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}
    full.update(probs)
    full[label] = max(full[label], 1.0 - sum(v for k, v in full.items() if k != label))
    return NLIResult(label=label, probs=full)


class TestBidirectionalMapping:
    """The mnemon taxonomy is derived from the pair of unidirectional
    NLI classifications. These tests lock the mapping logic."""

    def _patch(self, a_to_b: str, b_to_a: str):
        """Return a context manager that stubs classify_pair to
        produce a_to_b on the first call and b_to_a on the second."""
        results = iter([_result(a_to_b), _result(b_to_a)])
        return patch("mnemon.nli.classify_pair",
                     side_effect=lambda p, h: next(results))

    def test_contradiction_in_either_direction_wins(self):
        # a→b contradiction
        with self._patch("contradiction", "neutral"):
            r = classify_pair_bidirectional("a", "b")
            assert r.mnemon_label == "contradiction"
        # b→a contradiction
        with self._patch("neutral", "contradiction"):
            r = classify_pair_bidirectional("a", "b")
            assert r.mnemon_label == "contradiction"
        # both directions contradict
        with self._patch("contradiction", "contradiction"):
            r = classify_pair_bidirectional("a", "b")
            assert r.mnemon_label == "contradiction"

    def test_both_entail_is_same(self):
        with self._patch("entailment", "entailment"):
            r = classify_pair_bidirectional("a", "b")
            assert r.mnemon_label == "same"

    def test_b_entails_a_only_is_update(self):
        """New (b) entails old (a) but not vice versa → new is the
        stronger / more-detailed statement → 'update'."""
        with self._patch("neutral", "entailment"):
            r = classify_pair_bidirectional("a", "b")
            assert r.mnemon_label == "update"

    def test_a_entails_b_only_is_same_not_update(self):
        """Old (a) entails new (b) but not vice versa → new is a
        weaker subset → existing memory dominates → 'same', not
        'update'. Composes with the salience-tier 'don't crowd with
        weaker restatements' invariant."""
        with self._patch("entailment", "neutral"):
            r = classify_pair_bidirectional("a", "b")
            assert r.mnemon_label == "same"

    def test_both_neutral_is_unrelated(self):
        with self._patch("neutral", "neutral"):
            r = classify_pair_bidirectional("a", "b")
            assert r.mnemon_label == "unrelated"

    def test_result_carries_subdirections(self):
        """The BidirectionalResult preserves both NLI sub-results
        for caller-side inspection (logging, observability)."""
        with self._patch("contradiction", "entailment"):
            r = classify_pair_bidirectional("a", "b")
            assert r.a_implies_b.label == "contradiction"
            assert r.b_implies_a.label == "entailment"


class TestAvailability:
    def test_is_available_when_deps_present(self):
        # In a normal test env all FastEmbed transitive deps are present
        assert is_available() is True

    def test_is_available_false_when_onnxruntime_missing(self):
        # Simulate missing onnxruntime
        import builtins
        original_import = builtins.__import__

        def raise_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("simulated missing")
            return original_import(name, *args, **kwargs)

        # Reset the session singleton so is_available actually probes
        import mnemon.nli
        original_session = mnemon.nli._session
        mnemon.nli._session = None
        try:
            with patch("builtins.__import__", side_effect=raise_import):
                assert is_available() is False
        finally:
            mnemon.nli._session = original_session


class TestErrorSurfacing:
    def test_unavailable_error_has_descriptive_message(self):
        e = NLIUnavailableError("model load failed: connection refused")
        # The message survives string conversion (needed for the MCP
        # tool's clear-error path).
        assert "model load failed" in str(e)

    def test_ensure_loaded_raises_on_hub_download_failure(self):
        """Network / 403 / 404 errors during model download must
        surface as NLIUnavailableError, not bubble unchanged."""
        import mnemon.nli
        original_session = mnemon.nli._session
        mnemon.nli._session = None
        try:
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=ConnectionError("simulated network failure"),
            ):
                with pytest.raises(NLIUnavailableError) as exc:
                    mnemon.nli._ensure_loaded()
            assert "simulated network failure" in str(exc.value)
        finally:
            mnemon.nli._session = original_session

    def test_ensure_loaded_raises_on_unexpected_label_set(self, tmp_path):
        """A model with a different label space than the expected
        contradiction/entailment/neutral triple must fail fast at
        load time, not produce mis-classifications downstream."""
        import json as json_mod
        import mnemon.nli

        # Fake config + tokenizer files
        config_path = tmp_path / "config.json"
        config_path.write_text(json_mod.dumps({"id2label": {"0": "happy", "1": "sad"}}))
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text('{"version":"1.0"}')  # minimal stub
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"")  # not actually loaded; rejected earlier

        original_session = mnemon.nli._session
        mnemon.nli._session = None
        try:
            def fake_download(repo_id, filename):
                if filename == "config.json":
                    return str(config_path)
                if filename == "tokenizer.json":
                    return str(tokenizer_path)
                return str(onnx_path)

            with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
                with pytest.raises(NLIUnavailableError) as exc:
                    mnemon.nli._ensure_loaded()
            assert "unexpected label set" in str(exc.value)
        finally:
            mnemon.nli._session = original_session


class TestPrewarm:
    def test_prewarm_swallows_unavailability(self):
        """prewarm() is best-effort observability — must NOT raise
        even when the underlying model can't load. (Acceptable
        swallow per feedback_no_silent_fails — pre-warm is secondary;
        the first real call surfaces the named error.)"""
        from mnemon.nli import prewarm
        with patch("mnemon.nli._ensure_loaded",
                   side_effect=NLIUnavailableError("simulated")):
            prewarm()  # must not raise

    def test_prewarm_calls_ensure_loaded(self):
        from mnemon.nli import prewarm
        with patch("mnemon.nli._ensure_loaded") as ensure:
            prewarm()
            assert ensure.called


class TestClassifyPairTokenization:
    """Smoke-test the input-building path with a stubbed session +
    tokenizer. Exercises lines 164-189 (the input dict construction
    + softmax over logits) without paying the real model load cost."""

    def test_classify_pair_returns_argmax_label(self):
        """Given fake logits that favor 'contradiction', the result
        label is 'contradiction' and probs sum to ~1.0."""
        import numpy as np
        import mnemon.nli

        # Stub the session: return fixed logits favoring contradiction (idx 0)
        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        # Configure .name attrs explicitly (MagicMock(name=...) doesn't
        # set the attribute, just the repr)
        for input_mock, real_name in zip(
            fake_session.get_inputs.return_value, ["input_ids", "attention_mask"]
        ):
            input_mock.name = real_name
        fake_session.run.return_value = [np.array([[5.0, -2.0, 0.0]], dtype=np.float32)]

        fake_tokenizer = MagicMock()
        fake_enc = MagicMock()
        fake_enc.ids = [101, 1000, 102, 2000, 102]
        fake_enc.attention_mask = [1, 1, 1, 1, 1]
        fake_enc.type_ids = [0, 0, 0, 1, 1]
        fake_tokenizer.encode.return_value = fake_enc

        original_session = mnemon.nli._session
        original_tokenizer = mnemon.nli._tokenizer
        original_id2label = mnemon.nli._id2label
        mnemon.nli._session = fake_session
        mnemon.nli._tokenizer = fake_tokenizer
        mnemon.nli._id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        try:
            result = mnemon.nli.classify_pair("premise", "hypothesis")
            assert result.label == "contradiction"
            # Softmax probs sum to 1
            assert abs(sum(result.probs.values()) - 1.0) < 1e-5
            # Contradiction has highest prob
            assert result.probs["contradiction"] > result.probs["entailment"]
            assert result.probs["contradiction"] > result.probs["neutral"]
        finally:
            mnemon.nli._session = original_session
            mnemon.nli._tokenizer = original_tokenizer
            mnemon.nli._id2label = original_id2label


