"""Extended tests for hook framework, context surfacing, session extractor, and handoff generator."""

import json
import sys
import time
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mnemon.hooks.framework import is_duplicate, read_stdin, read_transcript, write_output


# ── framework.py: is_duplicate ────────────────────────────────────────────────


class TestIsDuplicate:
    def test_first_call_is_not_duplicate(self, tmp_path):
        dedup_file = tmp_path / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            assert is_duplicate("hello world") is False

    def test_second_call_is_duplicate(self, tmp_path):
        dedup_file = tmp_path / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            is_duplicate("hello world")
            assert is_duplicate("hello world") is True

    def test_different_text_is_not_duplicate(self, tmp_path):
        dedup_file = tmp_path / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            is_duplicate("hello world")
            assert is_duplicate("goodbye world") is False

    def test_expired_entry_is_not_duplicate(self, tmp_path):
        dedup_file = tmp_path / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            is_duplicate("hello world")
            # Manually backdate the entry beyond the 600s window
            entries = json.loads(dedup_file.read_text())
            entries[0]["timestamp"] = time.time() - 700
            dedup_file.write_text(json.dumps(entries))
            assert is_duplicate("hello world") is False

    def test_corrupt_dedup_file_handled(self, tmp_path):
        dedup_file = tmp_path / "dedup.json"
        dedup_file.parent.mkdir(parents=True, exist_ok=True)
        dedup_file.write_text("not valid json!!!")
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            assert is_duplicate("hello world") is False

    def test_creates_parent_directory(self, tmp_path):
        dedup_file = tmp_path / "subdir" / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            is_duplicate("hello world")
            assert dedup_file.exists()


# ── framework.py: read_stdin ──────────────────────────────────────────────────


class TestReadStdin:
    def test_reads_json_from_stdin(self):
        payload = {"prompt": "test prompt", "key": 42}
        with patch("sys.stdin", StringIO(json.dumps(payload))):
            result = read_stdin()
        assert result == payload

    def test_reads_empty_object(self):
        with patch("sys.stdin", StringIO("{}")):
            result = read_stdin()
        assert result == {}


# ── framework.py: write_output ────────────────────────────────────────────────


class TestWriteOutput:
    def test_writes_json_to_stdout(self):
        buf = StringIO()
        with patch("sys.stdout", buf):
            write_output({"result": "ok"})
        assert json.loads(buf.getvalue()) == {"result": "ok"}

    def test_flushes_stdout(self):
        mock_stdout = MagicMock()
        with patch("sys.stdout", mock_stdout):
            write_output({"a": 1})
        mock_stdout.flush.assert_called_once()


# ── framework.py: read_transcript ─────────────────────────────────────────────


class TestReadTranscript:
    def test_returns_empty_for_none_path(self):
        assert read_transcript(None) == ""
        assert read_transcript("") == ""

    def test_returns_empty_for_missing_file(self):
        assert read_transcript("/nonexistent/path/transcript.jsonl") == ""

    def test_reads_user_and_assistant_messages(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "How does X work?"}),
            json.dumps({"role": "assistant", "content": "X works by doing Y."}),
        ]
        transcript.write_text("\n".join(lines))
        result = read_transcript(str(transcript))
        assert "[user]: How does X work?" in result
        assert "[assistant]: X works by doing Y." in result

    def test_filters_non_user_assistant_roles(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps({"role": "system", "content": "You are helpful."}),
            json.dumps({"role": "user", "content": "Hello"}),
        ]
        transcript.write_text("\n".join(lines))
        result = read_transcript(str(transcript))
        assert "system" not in result.lower() or "[user]" in result
        assert "[user]: Hello" in result
        assert "You are helpful" not in result

    def test_handles_list_content(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "First part."},
                {"type": "text", "text": "Second part."},
                {"type": "image", "data": "binary"},
            ],
        }
        transcript.write_text(json.dumps(msg))
        result = read_transcript(str(transcript))
        assert "First part." in result
        assert "Second part." in result
        assert "binary" not in result

    def test_respects_max_chars_budget(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "A" * 500})
            for _ in range(20)
        ]
        transcript.write_text("\n".join(lines))
        result = read_transcript(str(transcript), max_chars=1000)
        # Should have stopped reading before exhausting all messages
        assert len(result) < 1500 + 200  # some overhead for role prefixes

    def test_skips_malformed_json_lines(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            "not json at all",
            json.dumps({"role": "user", "content": "Valid line"}),
        ]
        transcript.write_text("\n".join(lines))
        result = read_transcript(str(transcript))
        assert "[user]: Valid line" in result

    def test_reads_from_end_most_recent_first(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "First message"}),
            json.dumps({"role": "user", "content": "Second message"}),
        ]
        transcript.write_text("\n".join(lines))
        result = read_transcript(str(transcript), max_chars=30)
        # With a budget of 30 chars, should pick up the last message first
        assert "Second message" in result


# ── context_surfacing.py: build_context ───────────────────────────────────────


def _make_result(score, content_type="observation", title="Test", content="Test content"):
    r = MagicMock()
    r.composite_score = score
    r.content_type = content_type
    r.title = title
    r.content = content
    return r


class TestBuildContext:
    def test_empty_results_returns_empty(self):
        from mnemon.hooks.context_surfacing import build_context

        assert build_context([]) == ""
        assert build_context(None) == ""

    def test_hot_result_includes_300_char_snippet(self):
        from mnemon.hooks.context_surfacing import build_context

        long_content = "A" * 500
        r = _make_result(0.20, "decision", "Big Decision", long_content)
        ctx = build_context([r])
        assert "<mnemon-context>" in ctx
        assert "[decision] Big Decision:" in ctx
        assert "A" * 300 in ctx
        assert "..." in ctx

    def test_hot_result_short_content_no_ellipsis(self):
        from mnemon.hooks.context_surfacing import build_context

        r = _make_result(0.20, "note", "Short", "Brief content")
        ctx = build_context([r])
        # Content is under 300 chars so no ellipsis
        assert "Brief content" in ctx
        # The entry should not end with "..."
        lines = ctx.split("\n")
        content_line = [l for l in lines if "Short" in l][0]
        assert not content_line.endswith("...")

    def test_warm_result_includes_150_char_snippet(self):
        from mnemon.hooks.context_surfacing import build_context

        long_content = "B" * 300
        r = _make_result(0.12, "preference", "My Pref", long_content)
        ctx = build_context([r])
        assert "[preference] My Pref:" in ctx
        assert "B" * 150 in ctx
        # Should not include full 300 chars
        assert "B" * 200 not in ctx

    def test_cold_result_title_only(self):
        from mnemon.hooks.context_surfacing import build_context

        r = _make_result(0.05, "observation", "Cold Fact", "Lots of detail here")
        ctx = build_context([r])
        assert "[observation] Cold Fact" in ctx
        assert "Lots of detail" not in ctx

    def test_respects_char_budget(self):
        from mnemon.hooks.context_surfacing import build_context, CHAR_BUDGET

        results = [
            _make_result(0.20, "note", f"Item {i}", "X" * 400)
            for i in range(50)
        ]
        ctx = build_context(results)
        # Total context should be within budget (plus the wrapper lines)
        inner_lines = [l for l in ctx.split("\n") if l.startswith("[")]
        inner_text = "\n".join(inner_lines)
        assert len(inner_text) <= CHAR_BUDGET + 100

    def test_wraps_in_mnemon_context_tags(self):
        from mnemon.hooks.context_surfacing import build_context

        r = _make_result(0.20, "note", "Title", "Content")
        ctx = build_context([r])
        assert ctx.startswith("<mnemon-context>")
        assert ctx.endswith("</mnemon-context>")
        assert "Relevant memories from previous sessions:" in ctx

    def test_mixed_tiers(self):
        from mnemon.hooks.context_surfacing import build_context

        results = [
            _make_result(0.20, "decision", "Hot", "Hot content here"),
            _make_result(0.12, "preference", "Warm", "Warm content here"),
            _make_result(0.05, "observation", "Cold", "Cold content here"),
        ]
        ctx = build_context(results)
        assert "[decision] Hot: Hot content here" in ctx
        assert "[preference] Warm: Warm content here..." in ctx
        assert "[observation] Cold" in ctx
        assert "Cold content here" not in ctx.split("[observation] Cold")[-1].split("\n")[0]


# ── context_surfacing.py: main ────────────────────────────────────────────────


class TestContextSurfacingMain:
    def test_full_pipeline(self):
        from mnemon.hooks.context_surfacing import main

        mock_store = MagicMock()
        mock_result = _make_result(0.20, "note", "Pipeline", "It works via Step Functions")
        with patch("mnemon.hooks.framework.read_stdin", return_value={"prompt": "how does the pipeline work?"}), \
             patch("mnemon.hooks.framework.is_noise", return_value=False), \
             patch("mnemon.hooks.framework.is_duplicate", return_value=False), \
             patch("mnemon.hooks.framework.write_output") as mock_write, \
             patch("mnemon.store.Store", return_value=mock_store), \
             patch("mnemon.search.search", return_value=[mock_result]):
            main()
        mock_write.assert_called_once()
        output = mock_write.call_args[0][0]
        assert "additionalContext" in output["hookSpecificOutput"]
        assert "Pipeline" in output["hookSpecificOutput"]["additionalContext"]

    def test_skips_noise(self):
        from mnemon.hooks.context_surfacing import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"prompt": "hi"}), \
             patch("mnemon.hooks.framework.is_noise", return_value=True), \
             patch("mnemon.hooks.framework.write_output") as mock_write:
            main()
        mock_write.assert_not_called()

    def test_skips_duplicate(self):
        from mnemon.hooks.context_surfacing import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"prompt": "how does the pipeline work?"}), \
             patch("mnemon.hooks.framework.is_noise", return_value=False), \
             patch("mnemon.hooks.framework.is_duplicate", return_value=True), \
             patch("mnemon.hooks.framework.write_output") as mock_write:
            main()
        mock_write.assert_not_called()

    def test_no_results_no_output(self):
        from mnemon.hooks.context_surfacing import main

        mock_store = MagicMock()
        with patch("mnemon.hooks.framework.read_stdin", return_value={"prompt": "something obscure"}), \
             patch("mnemon.hooks.framework.is_noise", return_value=False), \
             patch("mnemon.hooks.framework.is_duplicate", return_value=False), \
             patch("mnemon.hooks.framework.write_output") as mock_write, \
             patch("mnemon.store.Store", return_value=mock_store), \
             patch("mnemon.search.search", return_value=[]):
            main()
        mock_write.assert_not_called()


# ── session_extractor.py: extract_with_llm ───────────────────────────────────


class TestExtractWithLLM:
    def test_returns_none_when_llm_unavailable(self):
        from mnemon.hooks.session_extractor import extract_with_llm

        with patch("mnemon.llm.is_available", return_value=False), \
             patch("mnemon.llm.generate"):
            result = extract_with_llm("some transcript")
        assert result is None

    def test_returns_empty_list_for_none_tag(self):
        from mnemon.hooks.session_extractor import extract_with_llm

        with patch("mnemon.llm.is_available", return_value=True), \
             patch("mnemon.llm.generate", return_value="<none/>"):
            result = extract_with_llm("some transcript")
        assert result == []

    def test_returns_parsed_observations(self):
        from mnemon.hooks.session_extractor import extract_with_llm

        llm_response = (
            "<observation>\n"
            "  <type>decision</type>\n"
            "  <title>Use PostgreSQL</title>\n"
            "  <content>Chose PostgreSQL for JSON support.</content>\n"
            "</observation>"
        )
        with patch("mnemon.llm.is_available", return_value=True), \
             patch("mnemon.llm.generate", return_value=llm_response):
            result = extract_with_llm("some transcript")
        assert len(result) == 1
        assert result[0]["title"] == "Use PostgreSQL"

    def test_returns_none_on_exception(self):
        from mnemon.hooks.session_extractor import extract_with_llm

        with patch("mnemon.llm.is_available", side_effect=Exception("boom")):
            result = extract_with_llm("some transcript")
        assert result is None


# ── session_extractor.py: is_duplicate (vector dedup) ─────────────────────────


class TestSessionExtractorIsDuplicate:
    def test_not_duplicate_when_low_similarity(self):
        from mnemon.hooks.session_extractor import is_duplicate as vec_is_dup

        mock_store = MagicMock()
        low_result = MagicMock()
        low_result.score = 0.80
        mock_store.search_vector.return_value = [low_result]
        with patch("mnemon.embedder.embed", return_value="fake_emb"):
            assert vec_is_dup(mock_store, "title", "content") is False

    def test_duplicate_when_high_similarity(self):
        from mnemon.hooks.session_extractor import is_duplicate as vec_is_dup

        mock_store = MagicMock()
        high_result = MagicMock()
        high_result.score = 0.95
        mock_store.search_vector.return_value = [high_result]
        with patch("mnemon.embedder.embed", return_value="fake_emb"):
            assert vec_is_dup(mock_store, "title", "content") is True

    def test_returns_false_on_exception(self):
        from mnemon.hooks.session_extractor import is_duplicate as vec_is_dup

        mock_store = MagicMock()
        with patch("mnemon.embedder.embed", side_effect=Exception("no model")):
            assert vec_is_dup(mock_store, "title", "content") is False

    def test_no_results_not_duplicate(self):
        from mnemon.hooks.session_extractor import is_duplicate as vec_is_dup

        mock_store = MagicMock()
        mock_store.search_vector.return_value = []
        with patch("mnemon.embedder.embed", return_value="fake_emb"):
            assert vec_is_dup(mock_store, "title", "content") is False


# ── session_extractor.py: main ────────────────────────────────────────────────


class TestSessionExtractorMain:
    def test_skips_short_transcript(self):
        from mnemon.hooks.session_extractor import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="short"):
            main()

    def test_falls_back_to_regex(self):
        from mnemon.hooks import session_extractor
        from mnemon.hooks.session_extractor import main

        mock_store = MagicMock()
        mock_store.save.return_value = "doc-123"
        mock_store.get.return_value = MagicMock(hash="abc123")
        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 200), \
             patch.object(session_extractor, "extract_with_llm", return_value=None), \
             patch.object(session_extractor, "extract_with_regex", return_value=[{"type": "decision", "title": "Use Redis", "content": "Chose Redis for caching."}]) as mock_regex, \
             patch.object(session_extractor, "is_duplicate", return_value=False), \
             patch("mnemon.store.Store", return_value=mock_store), \
             patch("mnemon.embedder.embed_document"):
            main()
        mock_regex.assert_called_once()
        mock_store.save.assert_called_once()

    def test_skips_duplicate_observations(self):
        from mnemon.hooks import session_extractor
        from mnemon.hooks.session_extractor import main

        mock_store = MagicMock()
        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 200), \
             patch.object(session_extractor, "extract_with_llm", return_value=[{"type": "decision", "title": "Dup", "content": "Already saved."}]), \
             patch.object(session_extractor, "is_duplicate", return_value=True), \
             patch("mnemon.store.Store", return_value=mock_store):
            main()
        mock_store.save.assert_not_called()

    def test_no_observations_exits_early(self):
        from mnemon.hooks import session_extractor
        from mnemon.hooks.session_extractor import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 200), \
             patch.object(session_extractor, "extract_with_llm", return_value=[]), \
             patch("mnemon.store.Store") as mock_store_cls:
            main()
        mock_store_cls.assert_not_called()


# ── handoff_generator.py: generate_with_llm ──────────────────────────────────


class TestGenerateWithLLM:
    def test_returns_none_when_llm_unavailable(self):
        from mnemon.hooks.handoff_generator import generate_with_llm

        with patch("mnemon.llm.is_available", return_value=False), \
             patch("mnemon.llm.generate"):
            result = generate_with_llm("some transcript")
        assert result is None

    def test_returns_skip_for_none_tag(self):
        from mnemon.hooks.handoff_generator import generate_with_llm

        with patch("mnemon.llm.is_available", return_value=True), \
             patch("mnemon.llm.generate", return_value="<none/>"):
            result = generate_with_llm("some transcript")
        assert result == {"skip": True}

    def test_returns_parsed_handoff(self):
        from mnemon.hooks.handoff_generator import generate_with_llm

        llm_response = (
            "<handoff>\n"
            "  <title>Fixed auth bug</title>\n"
            "  <summary>- Fixed JWT validation\n- Added tests</summary>\n"
            "</handoff>"
        )
        with patch("mnemon.llm.is_available", return_value=True), \
             patch("mnemon.llm.generate", return_value=llm_response):
            result = generate_with_llm("some transcript")
        assert result["title"] == "Fixed auth bug"
        assert "JWT" in result["summary"]

    def test_returns_none_on_exception(self):
        from mnemon.hooks.handoff_generator import generate_with_llm

        with patch("mnemon.llm.is_available", side_effect=Exception("boom")):
            result = generate_with_llm("some transcript")
        assert result is None

    def test_returns_none_for_unparseable_response(self):
        from mnemon.hooks.handoff_generator import generate_with_llm

        with patch("mnemon.llm.is_available", return_value=True), \
             patch("mnemon.llm.generate", return_value="just some text"):
            result = generate_with_llm("some transcript")
        assert result is None


# ── handoff_generator.py: main ────────────────────────────────────────────────


class TestHandoffGeneratorMain:
    def test_skips_short_transcript(self):
        from mnemon.hooks.handoff_generator import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="short"), \
             patch("mnemon.store.Store") as mock_store_cls:
            main()
        mock_store_cls.assert_not_called()

    def test_falls_back_to_regex(self):
        from mnemon.hooks import handoff_generator
        from mnemon.hooks.handoff_generator import main

        mock_store = MagicMock()
        mock_store.save.return_value = "doc-456"
        mock_store.get.return_value = MagicMock(hash="def456")
        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 300), \
             patch.object(handoff_generator, "generate_with_llm", return_value=None), \
             patch.object(handoff_generator, "generate_with_regex", return_value={"title": "Regex handoff", "summary": "- Did stuff"}) as mock_regex, \
             patch("mnemon.store.Store", return_value=mock_store), \
             patch("mnemon.embedder.embed_document"):
            main()
        mock_regex.assert_called_once()
        mock_store.save.assert_called_once()
        call_kwargs = mock_store.save.call_args[1]
        assert call_kwargs["content_type"] == "handoff"
        assert "Regex handoff" in call_kwargs["title"]

    def test_skips_when_llm_says_none(self):
        from mnemon.hooks import handoff_generator
        from mnemon.hooks.handoff_generator import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 300), \
             patch.object(handoff_generator, "generate_with_llm", return_value={"skip": True}), \
             patch("mnemon.store.Store") as mock_store_cls:
            main()
        mock_store_cls.assert_not_called()

    def test_saves_llm_handoff(self):
        from mnemon.hooks import handoff_generator
        from mnemon.hooks.handoff_generator import main

        mock_store = MagicMock()
        mock_store.save.return_value = "doc-789"
        mock_store.get.return_value = MagicMock(hash="ghi789")
        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 300), \
             patch.object(handoff_generator, "generate_with_llm", return_value={"title": "LLM summary", "summary": "- Deployed feature X"}), \
             patch("mnemon.store.Store", return_value=mock_store), \
             patch("mnemon.embedder.embed_document"):
            main()
        mock_store.save.assert_called_once()
        call_kwargs = mock_store.save.call_args[1]
        assert call_kwargs["title"] == "Session: LLM summary"
        assert call_kwargs["content"] == "- Deployed feature X"
