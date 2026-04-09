"""Tests for setup integrations."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mnemon.setup import (
    run_setup,
    setup_claude_code,
    setup_cursor,
    setup_gemini,
    setup_hooks,
    _mcp_config,
    _hooks_config,
    TARGETS,
)


class TestMcpConfig:
    def test_mcp_config_uses_current_python(self):
        config = _mcp_config()
        assert config["command"] == os.path.realpath(config["command"]) or "python" in config["command"]
        assert config["args"] == ["-m", "mnemon", "serve"]

    def test_hooks_config_has_all_events(self):
        hooks = _hooks_config()
        assert "UserPromptSubmit" in hooks
        assert "Stop" in hooks
        assert len(hooks["Stop"][0]["hooks"]) == 2  # extractor + handoff


class TestSetupClaudeCode:
    def test_creates_settings_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)):
                result = setup_claude_code()

            assert settings_path.exists()
            settings = json.loads(settings_path.read_text())
            assert "mnemon" in settings["mcpServers"]
            assert "UserPromptSubmit" in settings["hooks"]
            assert "Stop" in settings["hooks"]
            assert "Restart" in result

    def test_preserves_existing_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            settings_path.parent.mkdir(parents=True)
            settings_path.write_text(json.dumps({
                "mcpServers": {"other-server": {"command": "other"}},
                "customSetting": True,
            }))

            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)):
                setup_claude_code()

            settings = json.loads(settings_path.read_text())
            assert "other-server" in settings["mcpServers"]
            assert "mnemon" in settings["mcpServers"]
            assert settings["customSetting"] is True


class TestSetupCursor:
    def test_creates_mcp_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cursor_path = Path(tmpdir) / ".cursor" / "mcp.json"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)):
                result = setup_cursor()

            assert cursor_path.exists()
            config = json.loads(cursor_path.read_text())
            assert "mnemon" in config["mcpServers"]
            assert "Restart" in result


class TestSetupGemini:
    def test_returns_config_snippet(self):
        result = setup_gemini()
        assert "mnemon" in result
        assert "-m" in result


class TestSetupHooks:
    def test_writes_hooks_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)):
                setup_hooks()

            settings = json.loads(settings_path.read_text())
            assert "hooks" in settings
            # Should not have mcpServers (hooks-only)
            assert "mcpServers" not in settings


class TestRunSetup:
    def test_unknown_target_returns_error(self):
        result = run_setup("unknown")
        assert "Unknown target" in result

    def test_all_targets_registered(self):
        assert set(TARGETS.keys()) == {"claude-code", "cursor", "gemini", "hooks"}
