"""Tests for ``mnemon.mirror`` and the ``mnemon mirror`` CLI subcommand
introduced in 0.6.0rc7 to close the 2026-04-28 auto-memory gap (Claude
wrote to local memory but failed to mirror to mnemon).

Tests the parsing + dispatch logic in isolation (no real network or
filesystem state outside the temp dir), the auto-mode path filter, the
sync-source loop guard, the dedup window, and the CLI exit codes.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mnemon import mirror as mirror_mod
from mnemon.mirror import (
    MirrorError,
    MirrorResult,
    _is_auto_memory_path,
    _parse_frontmatter,
    mirror_path,
    run_cli,
)


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    """Build a memory dir under ``tmp_path/.claude/projects/.../memory/``
    so the auto-memory regex matches. Also redirects ``$HOME`` so the
    dedup state file goes to a temp location instead of the user's real
    ``~/.mnemon/`` (and so subsequent test runs don't see hits)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / ".claude" / "projects" / "myproject" / "memory"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def memory_file(memory_dir):
    """A sample auto-memory file with the standard frontmatter shape
    Claude Code writes."""
    p = memory_dir / "handoff_test.md"
    p.write_text(
        "---\n"
        "name: Test handoff\n"
        "description: A test memory file\n"
        "type: handoff\n"
        "---\n"
        "Body content goes here.\n\n"
        "Multiple paragraphs are fine.\n"
    )
    return p


@pytest.fixture
def fake_client():
    """A MemoryClient stub that records calls and returns a stub
    save-result. The real :func:`mirror_path` accepts a client kwarg
    so tests don't have to monkeypatch the resolver."""
    client = MagicMock()
    client.call_tool.return_value = ('Saved memory #42: "Test handoff" [handoff]', 0.05)
    return client


# ── _is_auto_memory_path ────────────────────────────────────────────────────
class TestAutoMemoryPathFilter:
    def test_claude_code_memory_path_matches(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        p = tmp_path / ".claude" / "projects" / "x" / "memory" / "h.md"
        p.parent.mkdir(parents=True)
        p.write_text("test")
        assert _is_auto_memory_path(p) is True

    def test_random_source_file_does_not_match(self, tmp_path):
        p = tmp_path / "src" / "main.py"
        p.parent.mkdir(parents=True)
        p.write_text("test")
        assert _is_auto_memory_path(p) is False

    def test_memory_file_under_other_project_dir_does_not_match(self, tmp_path):
        # memory subdir but NOT under .claude/projects — must not match.
        p = tmp_path / "some_repo" / "memory" / "notes.md"
        p.parent.mkdir(parents=True)
        p.write_text("test")
        assert _is_auto_memory_path(p) is False

    def test_generic_mnemon_auto_memory_path_matches(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        p = tmp_path / ".config" / "mnemon" / "auto-memory" / "n.md"
        p.parent.mkdir(parents=True)
        p.write_text("test")
        assert _is_auto_memory_path(p) is True


# ── _parse_frontmatter ──────────────────────────────────────────────────────
class TestFrontmatterParse:
    def test_parses_basic_keys(self):
        text = (
            "---\n"
            "name: Hello\n"
            "type: handoff\n"
            "description: A short summary\n"
            "---\n"
            "Body here\n"
        )
        fm, body = _parse_frontmatter(text)
        assert fm["name"] == "Hello"
        assert fm["type"] == "handoff"
        assert fm["description"] == "A short summary"
        assert body.strip() == "Body here"

    def test_missing_frontmatter_raises(self):
        with pytest.raises(MirrorError, match="missing the YAML frontmatter"):
            _parse_frontmatter("Just body text, no frontmatter\n")


# ── mirror_path: happy paths ────────────────────────────────────────────────
class TestMirrorPathSaved:
    def test_saves_with_correct_arguments(self, memory_file, fake_client):
        result = mirror_path(memory_file, client=fake_client)
        assert result.status == "saved"
        assert result.title == "Test handoff"
        assert result.doc_id == 42
        # Verify the call shape — arguments must include title, content,
        # content_type, source_client.
        args, kwargs = fake_client.call_tool.call_args
        assert args[0] == "memory_save"
        payload = args[1]
        assert payload["title"] == "Test handoff"
        assert payload["content_type"] == "handoff"
        assert payload["source_client"] == "mnemon-mirror"
        # Stable upsert identity = the slug (frontmatter `name`), so a
        # multi-edit session updates one doc instead of piling up dups.
        assert payload["source_key"] == "Test handoff"
        # Body must be present; description should also be merged in.
        assert "Body content goes here." in payload["content"]
        assert "_A test memory file_" in payload["content"]

    def test_saves_without_description(self, memory_dir, fake_client):
        p = memory_dir / "h.md"
        p.write_text(
            "---\n"
            "name: NoDesc\n"
            "type: note\n"
            "---\n"
            "Just a body.\n"
        )
        result = mirror_path(p, client=fake_client)
        assert result.status == "saved"
        payload = fake_client.call_tool.call_args[0][1]
        assert payload["content"] == "Just a body."
        # No leading italicized description line
        assert not payload["content"].startswith("_")

    def test_default_content_type_when_missing(self, memory_dir, fake_client):
        p = memory_dir / "h.md"
        p.write_text("---\nname: X\n---\nBody\n")
        mirror_path(p, client=fake_client)
        assert fake_client.call_tool.call_args[0][1]["content_type"] == "note"


# ── mirror_path: skip paths ─────────────────────────────────────────────────
class TestMirrorPathSkipped:
    def test_auto_skips_non_memory_paths(self, tmp_path, fake_client):
        p = tmp_path / "src" / "module.py"
        p.parent.mkdir()
        p.write_text("---\nname: X\n---\nBody")
        result = mirror_path(p, auto=True, client=fake_client)
        assert result.status == "skipped_no_match"
        # Must NOT call the client when skipping
        assert fake_client.call_tool.call_count == 0

    def test_sync_source_marker_short_circuits(self, memory_dir, fake_client):
        p = memory_dir / "synced.md"
        p.write_text(
            "---\n"
            "name: Synced\n"
            "type: note\n"
            "mnemon_sync_source: 281\n"
            "---\n"
            "Body\n"
        )
        result = mirror_path(p, client=fake_client)
        assert result.status == "skipped_sync_source"
        assert fake_client.call_tool.call_count == 0

    def test_dedup_skips_same_content_within_window(self, memory_file, fake_client):
        # First call saves
        first = mirror_path(memory_file, client=fake_client)
        assert first.status == "saved"
        # Second call with identical content within the dedup window
        # should short-circuit
        second = mirror_path(memory_file, client=fake_client)
        assert second.status == "skipped_duplicate"
        assert fake_client.call_tool.call_count == 1


# ── mirror_path: error paths ────────────────────────────────────────────────
class TestMirrorPathErrors:
    def test_missing_file_raises(self, tmp_path, fake_client):
        with pytest.raises(MirrorError, match="does not exist"):
            mirror_path(tmp_path / "no.md", client=fake_client)

    def test_missing_name_raises(self, memory_dir, fake_client):
        p = memory_dir / "h.md"
        p.write_text("---\ntype: note\n---\nBody\n")
        with pytest.raises(MirrorError, match="missing a 'name'"):
            mirror_path(p, client=fake_client)

    def test_empty_body_raises(self, memory_dir, fake_client):
        p = memory_dir / "h.md"
        p.write_text("---\nname: X\n---\n\n")
        with pytest.raises(MirrorError, match="body is empty"):
            mirror_path(p, client=fake_client)


# ── run_cli (CLI exit codes) ────────────────────────────────────────────────
class TestRunCli:
    def test_no_args_prints_usage_and_returns_2(self, capsys):
        rc = run_cli([])
        assert rc == 2
        err = capsys.readouterr().err
        assert "Usage:" in err

    def test_unknown_flag_returns_2(self, capsys):
        rc = run_cli(["--bogus"])
        assert rc == 2

    def test_too_many_paths_returns_2(self, capsys):
        rc = run_cli(["a.md", "b.md"])
        assert rc == 2

    def test_invalid_timeout_returns_2(self, capsys):
        rc = run_cli(["--timeout", "abc", "a.md"])
        assert rc == 2

    def test_missing_file_returns_1(self, tmp_path, capsys):
        rc = run_cli([str(tmp_path / "missing.md")])
        assert rc == 1
        err = capsys.readouterr().err
        assert "does not exist" in err

    def test_auto_mode_skip_is_quiet(self, tmp_path, capsys, monkeypatch):
        # Build a non-memory path and run with --auto. Should exit 0
        # silently — the hook fires on every Write event so verbose
        # skip-output would be log noise.
        monkeypatch.setenv("HOME", str(tmp_path))
        p = tmp_path / "src" / "x.py"
        p.parent.mkdir(parents=True)
        p.write_text("---\nname: X\n---\nBody")
        rc = run_cli(["--auto", str(p)])
        out = capsys.readouterr()
        assert rc == 0
        assert out.out == ""
        assert out.err == ""

    def test_save_path_returns_0_and_prints_title(
        self, memory_file, capsys, monkeypatch, fake_client
    ):
        # Patch get_client to return our stub
        from mnemon.hooks import _client as client_mod

        monkeypatch.setattr(client_mod, "get_client", lambda: fake_client)
        rc = run_cli([str(memory_file)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Mirrored" in out
        assert "Test handoff" in out
        assert "#42" in out
