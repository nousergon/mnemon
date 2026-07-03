"""Tests for scripts/build_standing_set.py — vault-derived auto-exemplars.

Focused unit tests for the new ``_sample_vault_exemplars`` function added
in the 2026-05-27 vault-derived-auto-exemplars PR. The function pulls
positive exemplars (high-confidence preference / decision / antipattern)
and negative exemplars (recent handoffs) from the operator's own vault
so the embedding-based scorer adapts per-user without hand-tuning
maintenance.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_standing_set.py"


@pytest.fixture(scope="module")
def bss():
    """Load build_standing_set.py as a module via importlib."""
    spec = importlib.util.spec_from_file_location(
        "build_standing_set", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_standing_set"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def conn():
    """In-memory sqlite seeded with the documents + content schema the
    script's SQL targets. Mirrors the production schema's columns we
    actually query (id, hash, title, content_type, confidence,
    invalidated_at, created_at) without pulling the full Store
    machinery."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            hash TEXT NOT NULL,
            title TEXT,
            content_type TEXT,
            confidence REAL,
            invalidated_at TEXT,
            created_at TEXT
        );
        CREATE TABLE content (
            hash TEXT PRIMARY KEY,
            doc TEXT
        );
    """)
    return db


def _seed(conn, doc_id, content_type, confidence, title, content,
          invalidated=None, created_at="2026-05-01"):
    h = f"h{doc_id}"
    conn.execute(
        "INSERT INTO content (hash, doc) VALUES (?, ?)", (h, content),
    )
    conn.execute(
        """INSERT INTO documents
           (id, hash, title, content_type, confidence, invalidated_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, h, title, content_type, confidence, invalidated, created_at),
    )


class TestSampleVaultExemplars:
    def test_pulls_high_conf_preferences_and_decisions_as_positives(self, bss, conn):
        _seed(conn, 1, "preference", 0.85, "Pref one", "always X")
        _seed(conn, 2, "decision", 0.90, "Dec one", "chose Y over Z")
        _seed(conn, 3, "antipattern", 0.80, "Anti one", "do not Q")
        pos, neg = bss._sample_vault_exemplars(conn, n=10)
        assert len(pos) == 3
        assert any("Pref one" in p for p in pos)
        assert any("Dec one" in p for p in pos)
        assert any("Anti one" in p for p in pos)
        assert neg == []  # no handoffs seeded

    def test_excludes_below_confidence_floor(self, bss, conn):
        _seed(conn, 1, "preference", 0.85, "Strong", "high conf preference")
        _seed(conn, 2, "preference", 0.50, "Weak", "below floor")
        pos, _ = bss._sample_vault_exemplars(conn, n=10)
        assert len(pos) == 1
        assert "Strong" in pos[0]

    def test_excludes_non_durable_types(self, bss, conn):
        _seed(conn, 1, "observation", 0.95, "Obs", "passes confidence but wrong type")
        _seed(conn, 2, "research", 0.90, "Res", "also wrong type")
        _seed(conn, 3, "note", 0.95, "Note", "still wrong type")
        _seed(conn, 4, "project", 0.95, "Proj", "wrong type")
        _seed(conn, 5, "preference", 0.80, "Pref", "right type")
        pos, _ = bss._sample_vault_exemplars(conn, n=10)
        assert len(pos) == 1
        assert "Pref" in pos[0]

    def test_excludes_invalidated_memories(self, bss, conn):
        _seed(conn, 1, "preference", 0.90, "Live", "active",
              invalidated=None)
        _seed(conn, 2, "preference", 0.90, "Dead", "soft-deleted",
              invalidated="2026-05-20")
        pos, _ = bss._sample_vault_exemplars(conn, n=10)
        assert len(pos) == 1
        assert "Live" in pos[0]

    def test_pulls_recent_handoffs_as_negatives(self, bss, conn):
        _seed(conn, 1, "handoff", 0.60, "Session A",
              "first session", created_at="2026-05-20")
        _seed(conn, 2, "handoff", 0.60, "Session B",
              "second session", created_at="2026-05-25")
        _seed(conn, 3, "handoff", 0.60, "Session C",
              "third session", created_at="2026-05-15")
        _, neg = bss._sample_vault_exemplars(conn, n=10)
        assert len(neg) == 3
        # Most-recent-first ordering — recency is the negative signal.
        assert "Session B" in neg[0]
        assert "Session A" in neg[1]
        assert "Session C" in neg[2]

    def test_n_caps_sample_size(self, bss, conn):
        for i in range(20):
            _seed(conn, i + 1, "preference", 0.85,
                  f"Pref {i}", f"content for memory {i}")
        pos, _ = bss._sample_vault_exemplars(conn, n=5)
        assert len(pos) == 5

    def test_format_combines_title_and_snippet(self, bss, conn):
        _seed(conn, 1, "preference", 0.90, "Short title",
              "longer body content with multiple words")
        pos, _ = bss._sample_vault_exemplars(conn, n=10)
        assert pos[0].startswith("Short title: ")
        assert "longer body content" in pos[0]

    def test_empty_vault_returns_empty_lists(self, bss, conn):
        pos, neg = bss._sample_vault_exemplars(conn, n=10)
        assert pos == []
        assert neg == []


class TestParseJudgeResponse:
    """The judge parses Haiku's JSON response. Robust against preamble
    text, missing keys, or invalid JSON — never raises, returns {}
    on failure so caller can default missing keys."""

    def test_pure_json_object(self, bss):
        text = '{"generality": 4, "durability": 5, "imperative_shape": 3, "cross_domain": 4, "rationale": "Multi-year preference"}'
        parsed = bss._parse_judge_response(text)
        assert parsed["generality"] == 4
        assert parsed["durability"] == 5
        assert parsed["rationale"] == "Multi-year preference"

    def test_json_with_preamble(self, bss):
        text = (
            "Here is my assessment:\n\n"
            '{"generality": 5, "durability": 5, "imperative_shape": 5, '
            '"cross_domain": 5, "rationale": "perfect rule"}'
        )
        parsed = bss._parse_judge_response(text)
        assert parsed["generality"] == 5
        assert parsed["rationale"] == "perfect rule"

    def test_invalid_json_returns_empty(self, bss):
        text = "This is not JSON at all { not-valid"
        assert bss._parse_judge_response(text) == {}

    def test_no_braces_returns_empty(self, bss):
        text = "no object here, just prose"
        assert bss._parse_judge_response(text) == {}

    def test_nested_json_returns_outermost(self, bss):
        # Nested object inside a value — the bracket counter handles it.
        text = '{"a": {"b": 1}, "c": 2}'
        parsed = bss._parse_judge_response(text)
        assert parsed == {"a": {"b": 1}, "c": 2}


class TestScoreViaLlmJudge:
    """Tests for the LLM-judge backend (opt-in --judge llm / anthropic).
    The krepis adapter is faked in sys.modules — no real key, SDK, or
    network needed (krepis is an operator-side opt-in install)."""

    DOC = {"id": 1, "title": "T", "content": "C", "content_type": "preference"}
    RUBRIC_08 = (
        '{"generality": 4, "durability": 4, "imperative_shape": 4, '
        '"cross_domain": 4, "rationale": "solid"}'
    )

    def _inject_fake_krepis(self, monkeypatch, complete_fn, captured=None):
        """Register fake krepis.{llm,llm_capture,llm_config} modules so
        the script's lazy imports resolve to stubs."""
        import sys as _sys
        import types as _types

        key_envs = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }

        constructed = []

        class FakeClient:
            def __init__(self, spec, **kw):
                self.spec = spec
                constructed.append(spec)

            def complete(self, **kw):
                return complete_fn(**kw)

        def parse_model_spec(value, source=""):
            provider, model = value.split(":", 1)
            return _types.SimpleNamespace(
                provider=provider,
                model=model,
                resolved_api_key_env=lambda p=provider: key_envs[p],
            )

        def capture_llm_call(result, **kw):
            if captured is not None:
                captured.append((result, kw))
            return captured is not None

        pkg = _types.ModuleType("krepis")
        llm_mod = _types.ModuleType("krepis.llm")
        llm_mod.LLMClient = FakeClient
        cap_mod = _types.ModuleType("krepis.llm_capture")
        cap_mod.capture_llm_call = capture_llm_call
        cfg_mod = _types.ModuleType("krepis.llm_config")
        cfg_mod.parse_model_spec = parse_model_spec
        pkg.llm, pkg.llm_capture, pkg.llm_config = llm_mod, cap_mod, cfg_mod

        monkeypatch.setitem(_sys.modules, "krepis", pkg)
        monkeypatch.setitem(_sys.modules, "krepis.llm", llm_mod)
        monkeypatch.setitem(_sys.modules, "krepis.llm_capture", cap_mod)
        monkeypatch.setitem(_sys.modules, "krepis.llm_config", cfg_mod)
        return constructed

    def _result(self, text):
        import types as _types

        return _types.SimpleNamespace(text=text, raw_request={}, raw_response=None)

    def test_missing_krepis_raises_runtime_error(self, bss, monkeypatch):
        import sys as _sys

        monkeypatch.setitem(_sys.modules, "krepis", None)
        monkeypatch.setitem(_sys.modules, "krepis.llm", None)
        import pytest as _p
        with _p.raises(RuntimeError, match=r"mnemon-memory\[judge\]"):
            bss._score_via_llm_judge(
                [dict(self.DOC)], default_spec=bss.JUDGE_LLM_DEFAULT_SPEC
            )

    def test_missing_api_key_raises_runtime_error(self, bss, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv(bss.JUDGE_LLM_ENV, raising=False)
        self._inject_fake_krepis(monkeypatch, lambda **kw: self._result("{}"))
        import pytest as _p
        with _p.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            bss._score_via_llm_judge(
                [dict(self.DOC)], default_spec=bss.JUDGE_LLM_DEFAULT_SPEC
            )

    def test_anthropic_alias_requires_anthropic_key(self, bss, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv(bss.JUDGE_LLM_ENV, raising=False)
        self._inject_fake_krepis(monkeypatch, lambda **kw: self._result("{}"))
        import pytest as _p
        with _p.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            bss._score_via_llm_judge(
                [dict(self.DOC)], default_spec=bss.JUDGE_LLM_ANTHROPIC_SPEC
            )

    def test_happy_path_scores_each_doc(self, bss, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
        monkeypatch.delenv(bss.JUDGE_LLM_ENV, raising=False)
        seen_calls = []

        def complete(**kw):
            seen_calls.append(kw)
            return self._result(self.RUBRIC_08)

        constructed = self._inject_fake_krepis(monkeypatch, complete)
        scores = bss._score_via_llm_judge(
            [
                {"id": 1, "title": "A", "content": "first", "content_type": "preference"},
                {"id": 2, "title": "B", "content": "second", "content_type": "decision"},
            ],
            default_spec=bss.JUDGE_LLM_DEFAULT_SPEC,
        )
        assert scores[1] == pytest.approx(0.8)
        assert scores[2] == pytest.approx(0.8)
        assert len(seen_calls) == 2
        # rubric rides the system prompt; the memory rides user_content
        assert seen_calls[0]["system"] == bss.JUDGE_RUBRIC_PROMPT
        assert "first" in seen_calls[0]["user_content"]
        assert seen_calls[0]["max_tokens"] == bss.JUDGE_MAX_TOKENS
        # default spec resolved to the open-weight OpenRouter judge
        assert constructed[0].provider == "openrouter"

    def test_env_override_wins(self, bss, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
        monkeypatch.setenv(bss.JUDGE_LLM_ENV, "anthropic:claude-haiku-4-5")
        constructed = self._inject_fake_krepis(
            monkeypatch, lambda **kw: self._result(self.RUBRIC_08)
        )
        bss._score_via_llm_judge(
            [dict(self.DOC)], default_spec=bss.JUDGE_LLM_DEFAULT_SPEC
        )
        assert constructed[0].provider == "anthropic"

    def test_classify_failure_falls_back_to_zero(self, bss, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
        monkeypatch.delenv(bss.JUDGE_LLM_ENV, raising=False)

        def complete(**kw):
            raise RuntimeError("rate limit")

        self._inject_fake_krepis(monkeypatch, complete)
        scores = bss._score_via_llm_judge(
            [{"id": 1, "title": "A", "content": "x", "content_type": "preference"}],
            default_spec=bss.JUDGE_LLM_DEFAULT_SPEC,
        )
        assert scores[1] == 0.0  # fallback, doesn't crash the run

    def test_missing_dims_default_to_neutral(self, bss, monkeypatch):
        """When the rubric JSON is missing a dimension, the caller
        defaults it to 3 (neutral). Score = (5 + 1 + 3 + 3) / 4 / 5 = 0.6"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
        monkeypatch.delenv(bss.JUDGE_LLM_ENV, raising=False)
        self._inject_fake_krepis(
            monkeypatch,
            lambda **kw: self._result('{"generality": 5, "durability": 1}'),
        )
        scores = bss._score_via_llm_judge(
            [dict(self.DOC)], default_spec=bss.JUDGE_LLM_DEFAULT_SPEC
        )
        assert scores[1] == pytest.approx(0.6)

    def test_sft_capture_invoked_per_call(self, bss, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
        monkeypatch.delenv(bss.JUDGE_LLM_ENV, raising=False)
        captured = []
        self._inject_fake_krepis(
            monkeypatch, lambda **kw: self._result(self.RUBRIC_08),
            captured=captured,
        )
        bss._score_via_llm_judge(
            [dict(self.DOC)], default_spec=bss.JUDGE_LLM_DEFAULT_SPEC
        )
        assert len(captured) == 1
        _result_obj, kw = captured[0]
        assert kw["producer"] == "mnemon_judge"
        assert kw["meta"]["memory_id"] == 1
        assert kw["meta"]["score"] == pytest.approx(0.8)
