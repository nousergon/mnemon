"""Tests for the PostToolUse auto-mirror hook (0.6.0rc7).

Exercises the JSON-stdin → file_path filter → mirror dispatch path
with the trigger-tool whitelist (Write/Edit/MultiEdit). Uses the
shared :func:`mnemon.mirror.mirror_path` so most logic is already
covered by ``test_mirror.py``; this file locks the hook's integration
shape.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mnemon.hooks import auto_mirror as hook_mod
from mnemon.hooks.auto_mirror import _extract_file_path, main


# ── _extract_file_path filter ───────────────────────────────────────────────
class TestExtractFilePath:
    def test_write_tool_extracts_path(self):
        payload = {
            "tool_name": "Write",
            "tool_input": {"file_path": "/abs/path.md", "content": "x"},
        }
        assert _extract_file_path(payload) == "/abs/path.md"

    def test_edit_tool_extracts_path(self):
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": "/abs/path.md"},
        }
        assert _extract_file_path(payload) == "/abs/path.md"

    def test_multiedit_tool_extracts_path(self):
        payload = {
            "tool_name": "MultiEdit",
            "tool_input": {"file_path": "/abs/path.md"},
        }
        assert _extract_file_path(payload) == "/abs/path.md"

    def test_unrelated_tool_returns_none(self):
        for tool in ("Bash", "Read", "Grep", "Glob", "WebSearch"):
            payload = {
                "tool_name": tool,
                "tool_input": {"file_path": "/abs/path.md"},
            }
            assert _extract_file_path(payload) is None

    def test_missing_tool_input_returns_none(self):
        assert _extract_file_path({"tool_name": "Write"}) is None

    def test_missing_file_path_returns_none(self):
        payload = {"tool_name": "Write", "tool_input": {"content": "x"}}
        assert _extract_file_path(payload) is None

    def test_empty_file_path_returns_none(self):
        payload = {"tool_name": "Write", "tool_input": {"file_path": ""}}
        assert _extract_file_path(payload) is None


# ── main() integration ──────────────────────────────────────────────────────
def _stub_stdin(monkeypatch, payload):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))


def _stub_stdout(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    return buf


def _stub_stderr(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    return buf


class TestMainIntegration:
    def test_write_to_memory_path_triggers_mirror_save(self, tmp_path, monkeypatch):
        # Build a real auto-memory file the hook can mirror
        monkeypatch.setenv("HOME", str(tmp_path))
        memory = tmp_path / ".claude" / "projects" / "x" / "memory" / "h.md"
        memory.parent.mkdir(parents=True)
        memory.write_text(
            "---\nname: HookTest\ntype: handoff\n---\nThe body.\n"
        )

        # Stub the dispatch client so the test doesn't hit the real vault
        fake_client = MagicMock()
        fake_client.call_tool.return_value = (
            'Saved memory #99: "HookTest"',
            0.05,
        )
        from mnemon.hooks import _client as client_mod

        monkeypatch.setattr(client_mod, "get_client", lambda: fake_client)

        _stub_stdin(
            monkeypatch,
            {
                "tool_name": "Write",
                "tool_input": {"file_path": str(memory)},
            },
        )
        _stub_stdout(monkeypatch)
        err = _stub_stderr(monkeypatch)

        rc = main()
        assert rc == 0
        # Client was called once with memory_save
        assert fake_client.call_tool.call_count == 1
        assert fake_client.call_tool.call_args[0][0] == "memory_save"
        # Stderr confirms the save
        assert "saved 'HookTest'" in err.getvalue()
        assert "#99" in err.getvalue()

    def test_write_to_non_memory_path_is_silent_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        unrelated = tmp_path / "src" / "main.py"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("print('hi')\n")

        fake_client = MagicMock()
        from mnemon.hooks import _client as client_mod

        monkeypatch.setattr(client_mod, "get_client", lambda: fake_client)

        _stub_stdin(
            monkeypatch,
            {
                "tool_name": "Write",
                "tool_input": {"file_path": str(unrelated)},
            },
        )
        _stub_stdout(monkeypatch)
        err = _stub_stderr(monkeypatch)

        rc = main()
        assert rc == 0
        # The mirror skip-no-match path is silent — no client call, no
        # stderr output. The bulk of Write events are unrelated source
        # edits; noisy logs would drown out real saves.
        assert fake_client.call_tool.call_count == 0
        assert err.getvalue() == ""

    def test_unrelated_tool_is_no_op(self, tmp_path, monkeypatch):
        # Bash + Read + Grep all trigger PostToolUse but should never
        # fire the mirror path.
        monkeypatch.setenv("HOME", str(tmp_path))
        fake_client = MagicMock()
        from mnemon.hooks import _client as client_mod

        monkeypatch.setattr(client_mod, "get_client", lambda: fake_client)

        for tool_name in ("Bash", "Read", "Grep"):
            _stub_stdin(
                monkeypatch,
                {
                    "tool_name": tool_name,
                    "tool_input": {"file_path": "/dev/null"},
                },
            )
            _stub_stdout(monkeypatch)
            _stub_stderr(monkeypatch)
            rc = main()
            assert rc == 0
        assert fake_client.call_tool.call_count == 0

    def test_malformed_stdin_does_not_block(self, monkeypatch):
        # Hook framework's read_stdin raises on parse error. The hook
        # must absorb it (exit 0) per "never block Claude Code's
        # continued operation". Surface the error via stderr.
        monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
        out = _stub_stdout(monkeypatch)
        err = _stub_stderr(monkeypatch)
        rc = main()
        assert rc == 0
        # Expect a structured error line on stderr
        assert "auto_mirror" in err.getvalue()

    def test_mirror_error_is_logged_and_does_not_block(
        self, tmp_path, monkeypatch
    ):
        # File matches the memory regex but has invalid frontmatter
        # → MirrorError. Hook must surface to stderr + exit 0.
        monkeypatch.setenv("HOME", str(tmp_path))
        bad = tmp_path / ".claude" / "projects" / "x" / "memory" / "bad.md"
        bad.parent.mkdir(parents=True)
        bad.write_text("no frontmatter here at all\n")

        fake_client = MagicMock()
        from mnemon.hooks import _client as client_mod

        monkeypatch.setattr(client_mod, "get_client", lambda: fake_client)

        _stub_stdin(
            monkeypatch,
            {
                "tool_name": "Write",
                "tool_input": {"file_path": str(bad)},
            },
        )
        _stub_stdout(monkeypatch)
        err = _stub_stderr(monkeypatch)

        rc = main()
        assert rc == 0
        assert fake_client.call_tool.call_count == 0
        assert "auto_mirror" in err.getvalue()
        assert "missing the YAML frontmatter" in err.getvalue()

    def test_unexpected_exception_is_swallowed_and_logged(
        self, tmp_path, monkeypatch
    ):
        # Simulate an unexpected failure (e.g. RemoteMemoryClient timeout
        # or auth error) — hook must NEVER block Claude. The error should
        # be surfaced via stderr so the operator + Claude see it per
        # feedback_surface_mnemon_unreachable.
        monkeypatch.setenv("HOME", str(tmp_path))
        good = tmp_path / ".claude" / "projects" / "x" / "memory" / "g.md"
        good.parent.mkdir(parents=True)
        good.write_text(
            "---\nname: G\ntype: note\n---\nbody\n"
        )

        # Fake client that raises a non-MirrorError
        class _Boom:
            def call_tool(self, *args, **kwargs):
                raise RuntimeError("simulated remote failure")

        from mnemon.hooks import _client as client_mod

        monkeypatch.setattr(client_mod, "get_client", lambda: _Boom())

        _stub_stdin(
            monkeypatch,
            {
                "tool_name": "Write",
                "tool_input": {"file_path": str(good)},
            },
        )
        _stub_stdout(monkeypatch)
        err = _stub_stderr(monkeypatch)

        rc = main()
        assert rc == 0
        log = err.getvalue()
        assert "auto_mirror" in log
        assert "RuntimeError" in log
        assert "simulated remote failure" in log
