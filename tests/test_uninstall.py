"""Tests for ``mnemon uninstall``.

Covers: plan detection, confirmation behavior, JSON scrubbing (mcpServers
+ hooks), vault removal, --keep-vault preserves memories, --yes bypasses
prompt, non-TTY stdin defaults to abort, claude CLI missing is tolerated.
"""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from mnemon import uninstall as un_mod
from mnemon.uninstall import UninstallError, uninstall


def _ok(returncode: int = 0) -> CompletedProcess:
    return CompletedProcess(args=[], returncode=returncode, stdout="", stderr="")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr("mnemon.uninstall.Path.home", lambda: tmp_path)
    monkeypatch.setenv("MNEMON_VAULT_DIR", str(tmp_path / ".mnemon"))
    yield


def _seed_full_install(tmp_path, *, with_remote: bool = False) -> dict:
    """Create a machine in a 'fully installed' state so we can assert
    the uninstall scrubs every expected location."""
    mdir = tmp_path / ".mnemon"
    mdir.mkdir()
    (mdir / "default.sqlite").write_bytes(b"stub vault")
    if with_remote:
        (mdir / "remote_url").write_text("https://x.fly.dev/mcp")
        (mdir / "local_token").write_text("tok-abc")

    # Claude Code settings.json with mnemon hooks + stale mcpServers
    cc = tmp_path / ".claude" / "settings.json"
    cc.parent.mkdir()
    cc.write_text(json.dumps({
        "mcpServers": {
            "mnemon": {"command": "python", "args": ["-m", "mnemon", "serve"]},
            "other": {"command": "keepme"},
        },
        "hooks": {
            "UserPromptSubmit": [
                {"matcher": "", "hooks": [
                    {"type": "command",
                     "command": "python -m mnemon.hooks.context_surfacing"},
                ]},
            ],
            "Stop": [
                {"matcher": "", "hooks": [
                    {"type": "command",
                     "command": "python -m mnemon.hooks.session_extractor"},
                    {"type": "command", "command": "other-tool"},
                ]},
            ],
        },
        "customSetting": True,
    }))

    # Cursor
    cursor = tmp_path / ".cursor" / "mcp.json"
    cursor.parent.mkdir()
    cursor.write_text(json.dumps({
        "mcpServers": {
            "mnemon": {"url": "https://x.fly.dev/mcp"},
            "another": {"command": "keepme"},
        }
    }))

    return {"mnemon_dir": mdir, "cc": cc, "cursor": cursor}


class TestUninstallHappyPath:
    def test_removes_everything_with_yes(self, tmp_path):
        seeded = _seed_full_install(tmp_path)

        with patch(
            "mnemon.uninstall.subprocess.run", return_value=_ok(0)
        ) as mock_run:
            result = uninstall(yes=True)

        # Vault gone
        assert not seeded["mnemon_dir"].exists()
        # claude mcp remove mnemon was called
        assert any(
            "remove" in c.args[0] and "mnemon" in c.args[0]
            for c in mock_run.call_args_list
        )
        # Claude Code settings.json scrubbed of mnemon; other keys survive
        cc_after = json.loads(seeded["cc"].read_text())
        assert "mnemon" not in cc_after.get("mcpServers", {})
        assert cc_after["mcpServers"]["other"] == {"command": "keepme"}
        assert cc_after["customSetting"] is True
        assert "UserPromptSubmit" not in cc_after.get("hooks", {})
        # The Stop entry's non-mnemon inner hook should survive
        stop = cc_after["hooks"]["Stop"]
        assert len(stop) == 1
        assert stop[0]["hooks"][0]["command"] == "other-tool"
        # Cursor: mnemon gone, other survives
        cursor_after = json.loads(seeded["cursor"].read_text())
        assert "mnemon" not in cursor_after["mcpServers"]
        assert cursor_after["mcpServers"]["another"] == {"command": "keepme"}

        assert "Uninstall complete" in result
        assert "Restart Claude Code" in result

    def test_keep_vault_preserves_memories(self, tmp_path, capsys):
        seeded = _seed_full_install(tmp_path)
        with patch("mnemon.uninstall.subprocess.run", return_value=_ok(0)):
            result = uninstall(yes=True, keep_vault=True)
        # Vault directory still exists
        assert seeded["mnemon_dir"].exists()
        assert (seeded["mnemon_dir"] / "default.sqlite").exists()
        # Client configs scrubbed
        cc_after = json.loads(seeded["cc"].read_text())
        assert "mnemon" not in cc_after.get("mcpServers", {})
        assert "Uninstall complete" in result
        # The "KEPT" marker is in the plan printed to stderr before the
        # action, not the post-action summary.
        assert "KEPT" in capsys.readouterr().err


class TestConfirmation:
    def test_no_tty_without_yes_aborts(self, tmp_path):
        """Non-interactive context without --yes must not delete data."""
        _seed_full_install(tmp_path)
        with patch("mnemon.uninstall._confirm", return_value=False), \
             patch("mnemon.uninstall.subprocess.run") as mock_run:
            result = uninstall(yes=False)
        assert "aborted" in result.lower()
        # No destructive actions fired
        mock_run.assert_not_called()
        assert (tmp_path / ".mnemon").exists()

    def test_interactive_yes_proceeds(self, tmp_path):
        _seed_full_install(tmp_path)
        with patch("mnemon.uninstall._confirm", return_value=True), \
             patch("mnemon.uninstall.subprocess.run", return_value=_ok(0)):
            result = uninstall(yes=False)
        assert "Uninstall complete" in result
        assert not (tmp_path / ".mnemon").exists()


class TestClaudeCliMissing:
    def test_tolerates_missing_claude_cli(self, tmp_path):
        _seed_full_install(tmp_path)
        with patch(
            "mnemon.uninstall.subprocess.run", side_effect=FileNotFoundError
        ):
            result = uninstall(yes=True)
        # Should still clean everything else
        assert not (tmp_path / ".mnemon").exists()
        assert "claude CLI not on PATH" in result


class TestRemoteWarning:
    def test_warns_when_remote_configured(self, tmp_path, capsys):
        _seed_full_install(tmp_path, with_remote=True)
        with patch("mnemon.uninstall.subprocess.run", return_value=_ok(0)):
            uninstall(yes=True)
        # Warning goes to stderr, not the return value
        err = capsys.readouterr().err
        assert "remote URL is configured" in err
        assert "mnemon downgrade local" in err


class TestEmptyMachine:
    def test_nothing_installed_completes_cleanly(self, tmp_path):
        """A machine with no mnemon state should complete with no
        destructive actions and no errors."""
        with patch("mnemon.uninstall.subprocess.run", return_value=_ok(0)):
            result = uninstall(yes=True)
        assert "Uninstall complete" in result