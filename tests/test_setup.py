"""Tests for setup integrations."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mnemon.setup import (
    run_setup,
    setup_claude_code,
    setup_cursor,
    setup_gemini,
    setup_hooks,
    _mcp_config,
    _hooks_config,
    _ensure_local_token,
    _ensure_remote_url,
    _parse_setup_args,
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

    def test_hooks_config_no_session_start_without_remote_url(self):
        hooks = _hooks_config()
        assert "SessionStart" not in hooks

    def test_hooks_config_adds_session_start_with_remote_url(self):
        hooks = _hooks_config(remote_url="https://example.fly.dev/mcp")
        assert "SessionStart" in hooks
        cmd = hooks["SessionStart"][0]["hooks"][0]["command"]
        assert "https://example.fly.dev/health" in cmd
        assert "curl" in cmd


class TestEnsureRemoteUrl:
    def test_writes_url_file(self, tmp_path):
        url_file = tmp_path / "remote_url"
        with patch("mnemon.setup.MNEMON_DIR", tmp_path), \
             patch("mnemon.setup.REMOTE_URL_FILE", url_file):
            result = _ensure_remote_url("https://test.fly.dev/mcp")
        assert result == "https://test.fly.dev/mcp"
        assert url_file.read_text() == "https://test.fly.dev/mcp"


class TestEnsureLocalToken:
    def test_writes_provided_token(self, tmp_path):
        token_file = tmp_path / "local_token"
        with patch("mnemon.setup.MNEMON_DIR", tmp_path), \
             patch("mnemon.setup.LOCAL_TOKEN_FILE", token_file):
            result = _ensure_local_token("my-secret-token")
        assert result == "my-secret-token"
        assert token_file.read_text() == "my-secret-token"
        assert oct(token_file.stat().st_mode)[-3:] == "600"

    def test_keeps_existing_token(self, tmp_path):
        token_file = tmp_path / "local_token"
        token_file.write_text("existing-token")
        with patch("mnemon.setup.MNEMON_DIR", tmp_path), \
             patch("mnemon.setup.LOCAL_TOKEN_FILE", token_file):
            result = _ensure_local_token()
        assert result == "existing-token"

    def test_generates_new_token_when_missing(self, tmp_path):
        token_file = tmp_path / "local_token"
        with patch("mnemon.setup.MNEMON_DIR", tmp_path), \
             patch("mnemon.setup.LOCAL_TOKEN_FILE", token_file):
            result = _ensure_local_token()
        assert len(result) == 43  # url-safe base64 of 32 bytes
        assert token_file.exists()
        assert oct(token_file.stat().st_mode)[-3:] == "600"


class TestParseSetupArgs:
    def test_parses_remote_url(self):
        result = _parse_setup_args(["--remote-url", "https://example.fly.dev/mcp"])
        assert result["remote_url"] == "https://example.fly.dev/mcp"

    def test_parses_token(self):
        result = _parse_setup_args(["--token", "abc123"])
        assert result["token"] == "abc123"

    def test_parses_both(self):
        result = _parse_setup_args(["--remote-url", "https://x.fly.dev/mcp", "--token", "tok"])
        assert result["remote_url"] == "https://x.fly.dev/mcp"
        assert result["token"] == "tok"

    def test_returns_none_when_missing(self):
        result = _parse_setup_args([])
        assert result["remote_url"] is None
        assert result["token"] is None


class TestSetupClaudeCode:
    def test_creates_settings_file_local_mode(self):
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

    def test_remote_mode_registers_via_claude_cli_and_cleans_stale_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            settings_path.parent.mkdir(parents=True)
            settings_path.write_text(json.dumps({
                "mcpServers": {
                    "mnemon": {"type": "http", "url": "https://old.fly.dev/mcp"},
                    "other": {"command": "keepme"},
                },
            }))
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
                result = setup_claude_code(
                    remote_url="https://test.fly.dev/mcp",
                    token="test-tok",
                )

            add_calls = [c for c in mock_run.call_args_list if "add" in c.args[0]]
            assert len(add_calls) == 1
            cmd = add_calls[0].args[0]
            assert cmd[:3] == ["claude", "mcp", "add"]
            assert "--scope" in cmd and "user" in cmd
            assert "--transport" in cmd and "http" in cmd
            assert "https://test.fly.dev/mcp" in cmd
            assert "Authorization: Bearer test-tok" in cmd

            remove_calls = [c for c in mock_run.call_args_list if "remove" in c.args[0]]
            assert len(remove_calls) == 1

            settings = json.loads(settings_path.read_text())
            assert "mnemon" not in settings.get("mcpServers", {})
            assert settings["mcpServers"]["other"]["command"] == "keepme"
            assert "UserPromptSubmit" in settings["hooks"]
            assert "SessionStart" in settings["hooks"]
            assert (mnemon_dir / "remote_url").read_text() == "https://test.fly.dev/mcp"
            assert "Remote URL" in result
            assert "claude mcp add" in result

    def test_remote_mode_raises_when_claude_cli_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup.subprocess.run", side_effect=FileNotFoundError):
                with pytest.raises(RuntimeError, match="claude.*CLI was not found"):
                    setup_claude_code(remote_url="https://test.fly.dev/mcp", token="tok")

    def test_remote_mode_adds_session_start_hook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
                setup_claude_code(remote_url="https://test.fly.dev/mcp")

            settings = json.loads(settings_path.read_text())
            session_start = settings["hooks"]["SessionStart"]
            cmd = session_start[0]["hooks"][0]["command"]
            assert "curl" in cmd
            assert "https://test.fly.dev/health" in cmd

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
    def test_creates_mcp_json_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cursor_path = Path(tmpdir) / ".cursor" / "mcp.json"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)):
                result = setup_cursor()

            assert cursor_path.exists()
            config = json.loads(cursor_path.read_text())
            assert "mnemon" in config["mcpServers"]
            assert "command" in config["mcpServers"]["mnemon"]
            assert "Restart" in result

    def test_remote_mode_uses_url_and_bearer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cursor_path = Path(tmpdir) / ".cursor" / "mcp.json"
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"):
                setup_cursor(remote_url="https://test.fly.dev/mcp", token="test-tok")

            config = json.loads(cursor_path.read_text())
            mnemon_cfg = config["mcpServers"]["mnemon"]
            assert mnemon_cfg["url"] == "https://test.fly.dev/mcp"
            assert "Bearer test-tok" in mnemon_cfg["headers"]["Authorization"]


class TestSetupGemini:
    def test_returns_config_snippet(self):
        result = setup_gemini()
        assert "mnemon" in result
        assert "-m" in result


class TestSetupHooks:
    def test_writes_hooks_only_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)):
                setup_hooks()

            settings = json.loads(settings_path.read_text())
            assert "hooks" in settings
            assert "mcpServers" not in settings
            assert "SessionStart" not in settings["hooks"]

    def test_writes_hooks_with_session_start_when_remote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"):
                setup_hooks(remote_url="https://test.fly.dev/mcp")

            settings = json.loads(settings_path.read_text())
            assert "SessionStart" in settings["hooks"]


class TestRunSetup:
    def test_unknown_target_returns_error(self):
        result = run_setup("unknown")
        assert "Unknown target" in result

    def test_all_targets_registered(self):
        assert set(TARGETS.keys()) == {"claude-code", "cursor", "gemini", "hooks"}

    def test_passes_remote_url_from_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
                result = run_setup("claude-code", ["--remote-url", "https://my.fly.dev/mcp"])
        assert "Remote URL" in result
        assert "https://my.fly.dev/mcp" in result

    def test_gemini_ignores_remote_url(self):
        result = run_setup("gemini", ["--remote-url", "https://ignored.fly.dev/mcp"])
        assert "mnemon" in result

    def test_no_hardcoded_fly_url(self):
        """CRITICAL GUARDRAIL: setup must never contain a hardcoded default
        pointing at any specific Fly deployment. A forked user running
        ``mnemon setup`` without ``--remote-url`` would accidentally send
        memories to the wrong vault."""
        import inspect
        from mnemon import setup as setup_module
        source = inspect.getsource(setup_module)
        assert "mnemon-memory.fly.dev" not in source
        assert "fly.dev/mcp" not in source
