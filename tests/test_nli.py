"""Tests for the NLI cross-encoder module — model layer + label mapping.

Real inference is exercised by ``tests/fixtures/nli_real_inference_pairs.py``
(skipped by default; opt-in via env var). These tests cover the mnemon-
side logic (bidirectional → taxonomy mapping, error handling) with the
classifier mocked so they run fast and offline.
"""

from __future__ import annotations

from unittest.mock import patch

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
