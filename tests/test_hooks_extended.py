"""Extended tests for hook framework, context surfacing, session extractor, and handoff generator."""

import json
import sys
import time
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mnemon.hooks.framework import (
    is_duplicate,
    mark_seen,
    read_stdin,
    read_transcript,
    write_output,
)


# ── framework.py: is_duplicate + mark_seen ────────────────────────────────────
#
# is_duplicate is read-only: it reports whether a prompt was previously
# marked as seen but does not itself persist anything. mark_seen is the
# write side. The split lets hooks postpone dedup marking until after
# their downstream work succeeds — a failed remote call no longer locks
# out a prompt for 10 minutes.


class TestIsDuplicate:
    def test_first_call_is_not_duplicate(self, tmp_path):
        """An empty dedup store always reports not-duplicate."""
        dedup_file = tmp_path / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            assert is_duplicate("hello world") is False

    def test_is_duplicate_does_not_write(self, tmp_path):
        """is_duplicate must be purely read-only — no persistence side
        effects. Without this guarantee the dedup-after-success fix
        regresses silently."""
        dedup_file = tmp_path / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            is_duplicate("hello world")
        assert not dedup_file.exists()

    def test_mark_seen_then_is_duplicate_true(self, tmp_path):
        """After mark_seen, the same text is reported as duplicate."""
        dedup_file = tmp_path / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            mark_seen("hello world")
            assert is_duplicate("hello world") is True

    def test_different_text_is_not_duplicate(self, tmp_path):
        dedup_file = tmp_path / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            mark_seen("hello world")
            assert is_duplicate("goodbye world") is False

    def test_expired_entry_is_not_duplicate(self, tmp_path):
        dedup_file = tmp_path / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            mark_seen("hello world")
            # Manually backdate the entry beyond the 600s window.
            entries = json.loads(dedup_file.read_text())
            entries[0]["timestamp"] = time.time() - 700
            dedup_file.write_text(json.dumps(entries))
            assert is_duplicate("hello world") is False

    def test_corrupt_dedup_file_handled(self, tmp_path):
        """is_duplicate must tolerate a corrupt file — returning False
        rather than crashing, so hooks don't hard-fail on a bad cache."""
        dedup_file = tmp_path / "dedup.json"
        dedup_file.parent.mkdir(parents=True, exist_ok=True)
        dedup_file.write_text("not valid json!!!")
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            assert is_duplicate("hello world") is False

    def test_mark_seen_creates_parent_directory(self, tmp_path):
        """mark_seen must create ~/.mnemon/ if it doesn't exist —
        otherwise the first hook invocation on a fresh install would
        fail to persist dedup state."""
        dedup_file = tmp_path / "subdir" / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            mark_seen("hello world")
            assert dedup_file.exists()

    def test_mark_seen_idempotent(self, tmp_path):
        """Calling mark_seen twice on the same text should not create
        duplicate entries — keeps the dedup file from growing
        unboundedly in edge cases."""
        dedup_file = tmp_path / "dedup.json"
        with patch("mnemon.hooks.framework._dedup_path", return_value=dedup_file):
            mark_seen("hello world")
            mark_seen("hello world")
            entries = json.loads(dedup_file.read_text())
            assert len(entries) == 1


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
#
# After Phase 3 unification, build_context takes the raw string output of
# the remote memory_search tool and wraps it in mnemon-context tags — the
# old HOT/WARM/COLD tiered logic moved to the server (which always returns
# 300-char snippets per match). build_context is now a thin wrapper with
# budget-enforcement as a safety net.


class TestBuildContext:
    def test_empty_input_returns_empty(self):
        from mnemon.hooks.context_surfacing import build_context

        assert build_context("") == ""
        assert build_context(None) == ""

    def test_whitespace_only_returns_empty(self):
        from mnemon.hooks.context_surfacing import build_context

        assert build_context("   \n  \t  ") == ""

    def test_no_results_sentinel_returns_empty(self):
        """The server returns a specific sentinel when nothing matches —
        we must not wrap it in mnemon-context tags and inject noise."""
        from mnemon.hooks.context_surfacing import (
            NO_RESULTS_SENTINEL,
            build_context,
        )

        assert build_context(NO_RESULTS_SENTINEL) == ""

    def test_wraps_in_mnemon_context_tags(self):
        from mnemon.hooks.context_surfacing import build_context

        raw = "1. [note] **Title** (score: 0.80)\n   content here"
        ctx = build_context(raw)
        assert ctx.startswith("<mnemon-context>")
        assert ctx.endswith("</mnemon-context>")
        assert "Relevant memories from previous sessions:" in ctx
        assert raw in ctx

    def test_preserves_server_formatting_unchanged(self):
        """The server's pre-formatted snippets should pass through
        untouched — no regex re-parsing, no re-ranking."""
        from mnemon.hooks.context_surfacing import build_context

        raw = (
            "1. [decision] **Use PostgreSQL** (score: 0.950, confidence: 0.90)\n"
            "   Chose PostgreSQL for JSON support.\n"
            "   _id: 1 | created: 2026-04-08_\n"
            "\n"
            "2. [preference] **Tabs over spaces** (score: 0.320, confidence: 0.70)\n"
            "   User prefers tabs.\n"
            "   _id: 2 | created: 2026-04-07_"
        )
        ctx = build_context(raw)
        # All score/confidence/id metadata preserved.
        assert "score: 0.950" in ctx
        assert "confidence: 0.70" in ctx
        assert "_id: 1" in ctx
        assert "_id: 2" in ctx

    def test_truncates_at_char_budget(self):
        from mnemon.hooks.context_surfacing import CHAR_BUDGET, build_context

        raw = "X" * (CHAR_BUDGET * 2)
        ctx = build_context(raw)
        # Output includes the wrapper + truncation marker, but the inner
        # payload is capped at CHAR_BUDGET chars.
        assert "[truncated]" in ctx
        # Sanity: total length bounded (wrapper adds <200 chars).
        assert len(ctx) < CHAR_BUDGET + 300

    def test_no_truncation_when_within_budget(self):
        from mnemon.hooks.context_surfacing import build_context

        raw = "1. [note] **Small** (score: 0.5)\n   Small content"
        ctx = build_context(raw)
        assert "[truncated]" not in ctx


# ── context_surfacing.py: main ────────────────────────────────────────────────


class TestContextSurfacingMain:
    def test_full_pipeline_calls_remote_and_emits_context(self):
        from mnemon.hooks.context_surfacing import main

        raw_tool_output = (
            "1. [note] **Pipeline** (score: 0.750, confidence: 0.80)\n"
            "   It works via Step Functions\n"
            "   _id: 42 | created: 2026-04-08_"
        )
        with patch(
            "mnemon.hooks.framework.read_stdin",
            return_value={"prompt": "how does the pipeline work?"},
        ), patch(
            "mnemon.hooks.framework.is_noise", return_value=False
        ), patch(
            "mnemon.hooks.framework.is_duplicate", return_value=False
        ), patch(
            "mnemon.hooks.framework.mark_seen"
        ) as mock_mark_seen, patch(
            "mnemon.hooks.framework.write_output"
        ) as mock_write, patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            return_value=(raw_tool_output, 0.5),
        ) as mock_call:
            main()

        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert call_args[0][0] == "memory_search"
        assert call_args[0][1] == {
            "query": "how does the pipeline work?",
            "limit": 8,
        }
        # Client label makes attribution possible in server-side logs.
        assert call_args.kwargs.get("client_label") == "claude-code-context-surfacing"

        # Successful remote call must mark the prompt as seen so an
        # immediate resubmit within the dedup window is suppressed.
        mock_mark_seen.assert_called_once_with("how does the pipeline work?")

        mock_write.assert_called_once()
        output = mock_write.call_args[0][0]
        assert "additionalContext" in output["hookSpecificOutput"]
        injected = output["hookSpecificOutput"]["additionalContext"]
        assert "Pipeline" in injected
        assert "Step Functions" in injected

    def test_skips_noise_without_calling_remote(self):
        from mnemon.hooks.context_surfacing import main

        with patch(
            "mnemon.hooks.framework.read_stdin", return_value={"prompt": "hi"}
        ), patch(
            "mnemon.hooks.framework.is_noise", return_value=True
        ), patch(
            "mnemon.hooks.framework.write_output"
        ) as mock_write, patch(
            "mnemon.hooks._remote_client.call_tool_sync"
        ) as mock_call:
            main()
        mock_write.assert_not_called()
        mock_call.assert_not_called()

    def test_skips_duplicate_without_calling_remote(self):
        from mnemon.hooks.context_surfacing import main

        with patch(
            "mnemon.hooks.framework.read_stdin",
            return_value={"prompt": "how does the pipeline work?"},
        ), patch(
            "mnemon.hooks.framework.is_noise", return_value=False
        ), patch(
            "mnemon.hooks.framework.is_duplicate", return_value=True
        ), patch(
            "mnemon.hooks.framework.write_output"
        ) as mock_write, patch(
            "mnemon.hooks._remote_client.call_tool_sync"
        ) as mock_call:
            main()
        mock_write.assert_not_called()
        mock_call.assert_not_called()

    def test_no_results_still_marks_seen(self):
        """Even when memory_search returns the no-results sentinel, we
        mark the prompt as seen so an immediate identical resubmit does
        not re-hit the network. The remote call succeeded — the prompt
        just has no matches — and dedup is about suppressing redundant
        work, not about gating on output."""
        from mnemon.hooks.context_surfacing import (
            NO_RESULTS_SENTINEL,
            main,
        )

        with patch(
            "mnemon.hooks.framework.read_stdin",
            return_value={"prompt": "something obscure"},
        ), patch(
            "mnemon.hooks.framework.is_noise", return_value=False
        ), patch(
            "mnemon.hooks.framework.is_duplicate", return_value=False
        ), patch(
            "mnemon.hooks.framework.mark_seen"
        ) as mock_mark_seen, patch(
            "mnemon.hooks.framework.write_output"
        ) as mock_write, patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            return_value=(NO_RESULTS_SENTINEL, 0.3),
        ):
            main()
        mock_write.assert_not_called()
        mock_mark_seen.assert_called_once_with("something obscure")

    def test_remote_error_degrades_gracefully_and_does_not_mark_seen(
        self, capsys
    ):
        """Network errors must not crash the hook — log to stderr, exit 0,
        and emit a visible warning context block so the user sees the
        outage without having to watch logs. Critically, mark_seen must NOT
        fire on failure so the exact same prompt can be retried immediately
        once the transient failure clears (wifi reconnect, Fly cold-start
        wakes, etc.). The exception type is included in the stderr line so
        empty-str exceptions like asyncio.TimeoutError still produce a
        debuggable trace."""
        from mnemon.hooks.context_surfacing import main

        with patch(
            "mnemon.hooks.framework.read_stdin",
            return_value={"prompt": "real prompt"},
        ), patch(
            "mnemon.hooks.framework.is_noise", return_value=False
        ), patch(
            "mnemon.hooks.framework.is_duplicate", return_value=False
        ), patch(
            "mnemon.hooks.framework.mark_seen"
        ) as mock_mark_seen, patch(
            "mnemon.hooks.framework.write_output"
        ) as mock_write, patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            side_effect=ConnectionError("connection refused"),
        ):
            main()
        # Warning context block emitted so user sees it in the prompt.
        mock_write.assert_called_once()
        injected = mock_write.call_args[0][0]["hookSpecificOutput"]["additionalContext"]
        assert "⚠ mnemon unavailable" in injected
        assert "ConnectionError" in injected
        assert "connection refused" in injected
        mock_mark_seen.assert_not_called()
        captured = capsys.readouterr()
        assert "remote error" in captured.err
        assert "ConnectionError" in captured.err
        assert "connection refused" in captured.err

    def test_empty_exception_still_surfaces_type(self, capsys):
        """asyncio.TimeoutError has an empty str() which caused the
        original 'remote error: ' empty message during smoke testing.
        The rewritten error handler includes the exception class name so
        even empty-message exceptions produce a debuggable log line and
        a visible warning context block."""
        import asyncio

        from mnemon.hooks.context_surfacing import main

        with patch(
            "mnemon.hooks.framework.read_stdin",
            return_value={"prompt": "real prompt"},
        ), patch(
            "mnemon.hooks.framework.is_noise", return_value=False
        ), patch(
            "mnemon.hooks.framework.is_duplicate", return_value=False
        ), patch(
            "mnemon.hooks.framework.mark_seen"
        ), patch(
            "mnemon.hooks.framework.write_output"
        ) as mock_write, patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            side_effect=asyncio.TimeoutError(),
        ):
            main()
        captured = capsys.readouterr()
        assert "TimeoutError" in captured.err
        mock_write.assert_called_once()
        injected = mock_write.call_args[0][0]["hookSpecificOutput"]["additionalContext"]
        assert "⚠ mnemon unavailable" in injected
        assert "TimeoutError" in injected

    def test_config_error_degrades_gracefully(self, capsys):
        """Missing URL/token should log a specific config error, emit a
        visible warning context block, exit 0, and must NOT mark the prompt
        as seen — the user can fix the config and retry immediately."""
        from mnemon.hooks._remote_client import RemoteClientConfigError
        from mnemon.hooks.context_surfacing import main

        with patch(
            "mnemon.hooks.framework.read_stdin",
            return_value={"prompt": "real prompt"},
        ), patch(
            "mnemon.hooks.framework.is_noise", return_value=False
        ), patch(
            "mnemon.hooks.framework.is_duplicate", return_value=False
        ), patch(
            "mnemon.hooks.framework.mark_seen"
        ) as mock_mark_seen, patch(
            "mnemon.hooks.framework.write_output"
        ) as mock_write, patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            side_effect=RemoteClientConfigError("no token"),
        ):
            main()
        mock_write.assert_called_once()
        injected = mock_write.call_args[0][0]["hookSpecificOutput"]["additionalContext"]
        assert "⚠ mnemon config error" in injected
        assert "no token" in injected
        mock_mark_seen.assert_not_called()
        captured = capsys.readouterr()
        assert "config error" in captured.err


    def test_slow_success_prepends_warning(self):
        """When the remote call succeeds but takes >3s, the context block
        must start with a ⚠ mnemon slow: warning so the user sees the
        latency degradation without watching logs."""
        from mnemon.hooks.context_surfacing import SLOW_THRESHOLD_SEC, main

        raw_tool_output = (
            "1. [note] **Thing** (score: 0.80)\n"
            "   Some content\n"
            "   _id: 1 | created: 2026-04-11_"
        )
        slow_elapsed = SLOW_THRESHOLD_SEC + 1.0

        with patch(
            "mnemon.hooks.framework.read_stdin",
            return_value={"prompt": "how does it work?"},
        ), patch(
            "mnemon.hooks.framework.is_noise", return_value=False
        ), patch(
            "mnemon.hooks.framework.is_duplicate", return_value=False
        ), patch(
            "mnemon.hooks.framework.mark_seen"
        ), patch(
            "mnemon.hooks.framework.write_output"
        ) as mock_write, patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            return_value=(raw_tool_output, slow_elapsed),
        ):
            main()

        mock_write.assert_called_once()
        injected = mock_write.call_args[0][0]["hookSpecificOutput"]["additionalContext"]
        assert "⚠ mnemon slow:" in injected
        assert f"{slow_elapsed:.1f}s" in injected
        # Memories still present after the warning.
        assert "Thing" in injected
        assert "Some content" in injected

    def test_fast_success_no_slow_warning(self):
        """When elapsed is within the threshold, no slow warning is emitted —
        the context block contains only the memories header and results."""
        from mnemon.hooks.context_surfacing import SLOW_THRESHOLD_SEC, main

        raw_tool_output = (
            "1. [note] **Fast** (score: 0.90)\n"
            "   Quick response\n"
            "   _id: 2 | created: 2026-04-11_"
        )
        fast_elapsed = SLOW_THRESHOLD_SEC - 0.5

        with patch(
            "mnemon.hooks.framework.read_stdin",
            return_value={"prompt": "quick query"},
        ), patch(
            "mnemon.hooks.framework.is_noise", return_value=False
        ), patch(
            "mnemon.hooks.framework.is_duplicate", return_value=False
        ), patch(
            "mnemon.hooks.framework.mark_seen"
        ), patch(
            "mnemon.hooks.framework.write_output"
        ) as mock_write, patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            return_value=(raw_tool_output, fast_elapsed),
        ):
            main()

        mock_write.assert_called_once()
        injected = mock_write.call_args[0][0]["hookSpecificOutput"]["additionalContext"]
        assert "⚠ mnemon slow" not in injected
        assert "Fast" in injected


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


# ── session_extractor.py: is_duplicate_remote (remote vector dedup) ───────────


class TestSessionExtractorIsDuplicateRemote:
    def test_not_duplicate_when_low_similarity(self):
        from mnemon.hooks.session_extractor import is_duplicate_remote

        raw = json.dumps([{"doc_id": 1, "title": "Existing", "composite_score": 0.456}])
        with patch("mnemon.hooks._remote_client.call_tool_sync", return_value=(raw, 0.3)):
            assert is_duplicate_remote("title", "content") is False

    def test_duplicate_when_high_similarity(self):
        from mnemon.hooks.session_extractor import is_duplicate_remote

        raw = json.dumps([{"doc_id": 1, "title": "Same thing", "composite_score": 0.950}])
        with patch("mnemon.hooks._remote_client.call_tool_sync", return_value=(raw, 0.3)):
            assert is_duplicate_remote("title", "content") is True

    def test_returns_false_on_exception(self):
        from mnemon.hooks.session_extractor import is_duplicate_remote

        with patch("mnemon.hooks._remote_client.call_tool_sync", side_effect=Exception("network")):
            assert is_duplicate_remote("title", "content") is False

    def test_returns_false_on_invalid_json(self):
        """Dedup must not crash if the server returns malformed JSON —
        treat as 'not duplicate' and let the save proceed."""
        from mnemon.hooks.session_extractor import is_duplicate_remote

        with patch("mnemon.hooks._remote_client.call_tool_sync", return_value=("not json", 0.2)):
            assert is_duplicate_remote("title", "content") is False

    def test_no_results_not_duplicate(self):
        from mnemon.hooks.session_extractor import is_duplicate_remote

        raw = json.dumps([])
        with patch("mnemon.hooks._remote_client.call_tool_sync", return_value=(raw, 0.2)):
            assert is_duplicate_remote("title", "content") is False

    def test_multiple_scores_only_needs_one_above_threshold(self):
        from mnemon.hooks.session_extractor import is_duplicate_remote

        raw = json.dumps([
            {"doc_id": 1, "title": "A", "composite_score": 0.800},
            {"doc_id": 2, "title": "B", "composite_score": 0.930},
        ])
        with patch("mnemon.hooks._remote_client.call_tool_sync", return_value=(raw, 0.3)):
            assert is_duplicate_remote("title", "content") is True

    def test_calls_structured_tool_not_text(self):
        """Guard against regression to text-parsing dedup. The hook must
        use memory_search_structured so scores aren't parsed from
        human-readable formatted output."""
        from mnemon.hooks.session_extractor import is_duplicate_remote

        raw = json.dumps([])
        with patch(
            "mnemon.hooks._remote_client.call_tool_sync",
            return_value=(raw, 0.1),
        ) as mock_call:
            is_duplicate_remote("title", "content")
        assert mock_call.call_args[0][0] == "memory_search_structured"


# ── session_extractor.py: main ────────────────────────────────────────────────


class TestSessionExtractorMain:
    def test_skips_short_transcript(self):
        from mnemon.hooks.session_extractor import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="short"):
            main()

    def test_falls_back_to_regex_and_saves_remotely(self):
        from mnemon.hooks import session_extractor
        from mnemon.hooks.session_extractor import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 200), \
             patch.object(session_extractor, "extract_with_llm", return_value=None), \
             patch.object(session_extractor, "extract_with_regex", return_value=[{"type": "decision", "title": "Use Redis", "content": "Chose Redis for caching."}]) as mock_regex, \
             patch.object(session_extractor, "is_duplicate_remote", return_value=False), \
             patch("mnemon.hooks._remote_client.call_tool_sync", return_value=("Saved doc-123", 0.5)) as mock_call:
            main()
        mock_regex.assert_called_once()
        mock_call.assert_called_once()
        args = mock_call.call_args[0]
        assert args[0] == "memory_save"
        assert args[1]["title"] == "Use Redis"
        assert args[1]["content"] == "Chose Redis for caching."
        assert args[1]["content_type"] == "decision"
        assert args[1]["source_client"] == "claude-code-hook"

    def test_skips_duplicate_observations(self):
        from mnemon.hooks import session_extractor
        from mnemon.hooks.session_extractor import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 200), \
             patch.object(session_extractor, "extract_with_llm", return_value=[{"type": "decision", "title": "Dup", "content": "Already saved."}]), \
             patch.object(session_extractor, "is_duplicate_remote", return_value=True), \
             patch("mnemon.hooks._remote_client.call_tool_sync") as mock_call:
            main()
        mock_call.assert_not_called()

    def test_no_observations_exits_early(self):
        from mnemon.hooks import session_extractor
        from mnemon.hooks.session_extractor import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 200), \
             patch.object(session_extractor, "extract_with_llm", return_value=[]), \
             patch("mnemon.hooks._remote_client.call_tool_sync") as mock_call:
            main()
        mock_call.assert_not_called()

    def test_config_error_stops_immediately(self, capsys):
        from mnemon.hooks import session_extractor
        from mnemon.hooks._remote_client import RemoteClientConfigError
        from mnemon.hooks.session_extractor import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 200), \
             patch.object(session_extractor, "extract_with_llm", return_value=[{"type": "decision", "title": "X", "content": "Y"}]), \
             patch.object(session_extractor, "is_duplicate_remote", return_value=False), \
             patch("mnemon.hooks._remote_client.call_tool_sync", side_effect=RemoteClientConfigError("no token")):
            main()
        captured = capsys.readouterr()
        assert "config error" in captured.err

    def test_network_error_continues_to_next_observation(self, capsys):
        from mnemon.hooks import session_extractor
        from mnemon.hooks.session_extractor import main

        observations = [
            {"type": "decision", "title": "First", "content": "Content 1"},
            {"type": "observation", "title": "Second", "content": "Content 2"},
        ]
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("refused")
            return ("Saved", 0.3)

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 200), \
             patch.object(session_extractor, "extract_with_llm", return_value=observations), \
             patch.object(session_extractor, "is_duplicate_remote", return_value=False), \
             patch("mnemon.hooks._remote_client.call_tool_sync", side_effect=side_effect):
            main()
        captured = capsys.readouterr()
        assert "save error" in captured.err
        assert "saved [observation]" in captured.err


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
             patch("mnemon.hooks._remote_client.call_tool_sync") as mock_call:
            main()
        mock_call.assert_not_called()

    def test_falls_back_to_regex_and_saves_remotely(self):
        from mnemon.hooks import handoff_generator
        from mnemon.hooks.handoff_generator import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 300), \
             patch.object(handoff_generator, "generate_with_llm", return_value=None), \
             patch.object(handoff_generator, "generate_with_regex", return_value={"title": "Regex handoff", "summary": "- Did stuff"}) as mock_regex, \
             patch("mnemon.hooks._remote_client.call_tool_sync", return_value=("Saved doc-456", 0.4)) as mock_call:
            main()
        mock_regex.assert_called_once()
        mock_call.assert_called_once()
        args = mock_call.call_args[0]
        assert args[0] == "memory_save"
        assert args[1]["title"] == "Session: Regex handoff"
        assert args[1]["content"] == "- Did stuff"
        assert args[1]["content_type"] == "handoff"
        assert args[1]["source_client"] == "claude-code-hook"

    def test_skips_when_llm_says_none(self):
        from mnemon.hooks import handoff_generator
        from mnemon.hooks.handoff_generator import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 300), \
             patch.object(handoff_generator, "generate_with_llm", return_value={"skip": True}), \
             patch("mnemon.hooks._remote_client.call_tool_sync") as mock_call:
            main()
        mock_call.assert_not_called()

    def test_saves_llm_handoff_remotely(self):
        from mnemon.hooks import handoff_generator
        from mnemon.hooks.handoff_generator import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 300), \
             patch.object(handoff_generator, "generate_with_llm", return_value={"title": "LLM summary", "summary": "- Deployed feature X"}), \
             patch("mnemon.hooks._remote_client.call_tool_sync", return_value=("Saved doc-789", 0.5)) as mock_call:
            main()
        mock_call.assert_called_once()
        args = mock_call.call_args[0]
        assert args[0] == "memory_save"
        assert args[1]["title"] == "Session: LLM summary"
        assert args[1]["content"] == "- Deployed feature X"

    def test_config_error_logged(self, capsys):
        from mnemon.hooks import handoff_generator
        from mnemon.hooks._remote_client import RemoteClientConfigError
        from mnemon.hooks.handoff_generator import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 300), \
             patch.object(handoff_generator, "generate_with_llm", return_value={"title": "X", "summary": "Y"}), \
             patch("mnemon.hooks._remote_client.call_tool_sync", side_effect=RemoteClientConfigError("no url")):
            main()
        captured = capsys.readouterr()
        assert "config error" in captured.err

    def test_network_error_logged(self, capsys):
        from mnemon.hooks import handoff_generator
        from mnemon.hooks.handoff_generator import main

        with patch("mnemon.hooks.framework.read_stdin", return_value={"transcript_path": "/tmp/t.jsonl"}), \
             patch("mnemon.hooks.framework.read_transcript", return_value="A" * 300), \
             patch.object(handoff_generator, "generate_with_llm", return_value={"title": "X", "summary": "Y"}), \
             patch("mnemon.hooks._remote_client.call_tool_sync", side_effect=ConnectionError("timeout")):
            main()
        captured = capsys.readouterr()
        assert "save error" in captured.err
        assert "ConnectionError" in captured.err
