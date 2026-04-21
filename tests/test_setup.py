"""Tests for setup integrations."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mnemon.setup import (
    SetupError,
    run_setup,
    setup_claude_code,
    setup_cursor,
    setup_gemini,
    setup_hooks,
    _hooks_config,
    _mcp_config,
    _ensure_local_token,
    _ensure_remote_url,
    _next_steps_block,
    _parse_setup_args,
    _strip_mnemon_hooks,
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
    def test_creates_settings_file_local_mode_installs_local_hooks(self):
        """P1b: local-mode setup installs UserPromptSubmit + Stop hooks
        that dispatch via :class:`LocalMemoryClient`. P0 skipped them
        because the hook client was HTTP-only; P1a fixed that.
        SessionStart is intentionally omitted in local mode — no remote
        machine to pre-warm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)):
                result = setup_claude_code()

            assert settings_path.exists()
            settings = json.loads(settings_path.read_text())
            assert "mnemon" in settings["mcpServers"]
            assert "UserPromptSubmit" in settings["hooks"]
            assert "Stop" in settings["hooks"]
            assert "SessionStart" not in settings["hooks"]
            assert "local (in-process)" in result
            assert "Restart" in result

    def test_local_mode_drops_stale_session_start(self):
        """Re-running setup in local mode after a prior remote install
        must strip the mnemon SessionStart pre-warm hook (it polls a
        remote URL that's no longer authoritative). UserPromptSubmit /
        Stop are overwritten with the local variants; SessionStart is
        removed entirely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            settings_path.parent.mkdir(parents=True)
            settings_path.write_text(json.dumps({
                "hooks": {
                    "SessionStart": [
                        {"matcher": "", "hooks": [
                            {"type": "command",
                             "command": "curl mnemon.fly.dev/health"},
                        ]},
                    ],
                },
            }))
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)):
                setup_claude_code()
            settings = json.loads(settings_path.read_text())
            hooks = settings.get("hooks", {})
            assert "SessionStart" not in hooks
            # Local UserPromptSubmit + Stop were installed
            assert "UserPromptSubmit" in hooks
            assert "Stop" in hooks

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
                 patch("mnemon.setup._preflight_remote_endpoint") as mock_preflight, \
                 patch("mnemon.setup.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
                result = setup_claude_code(
                    remote_url="https://test.fly.dev/mcp",
                    token="test-tok",
                )
            mock_preflight.assert_called_once_with(
                "https://test.fly.dev/mcp", "test-tok"
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
                 patch("mnemon.setup._preflight_remote_endpoint"), \
                 patch("mnemon.setup.subprocess.run", side_effect=FileNotFoundError):
                with pytest.raises(RuntimeError, match="claude.*CLI was not found"):
                    setup_claude_code(remote_url="https://test.fly.dev/mcp", token="tok")

    def test_remote_preflight_failure_aborts_setup_cleanly(self):
        """Preflight must run BEFORE any destructive step. On failure, no
        config files or settings.json edits are left behind."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup._preflight_remote_endpoint",
                       side_effect=SetupError("endpoint unreachable")), \
                 patch("mnemon.setup.subprocess.run") as mock_run:
                with pytest.raises(SetupError, match="endpoint unreachable"):
                    setup_claude_code(remote_url="https://test.fly.dev/mcp", token="tok")
            # claude CLI never invoked (preflight aborted before it)
            mock_run.assert_not_called()
            # settings.json never written
            assert not settings_path.exists()
            # remote_url file never written (we bail before _ensure_remote_url)
            assert not (mnemon_dir / "remote_url").exists()

    def test_remote_mode_adds_session_start_hook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup._preflight_remote_endpoint"), \
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
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup._preflight_remote_endpoint"):
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
    def test_installs_local_hooks_without_remote_url(self):
        """P1b: setup_hooks works in local mode now that LocalMemoryClient
        exists. UserPromptSubmit + Stop are written; SessionStart is
        skipped (only meaningful for remote cold-start)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)):
                out = setup_hooks()
            assert settings_path.exists()
            settings = json.loads(settings_path.read_text())
            assert "UserPromptSubmit" in settings["hooks"]
            assert "Stop" in settings["hooks"]
            assert "SessionStart" not in settings["hooks"]
            assert "local (in-process)" in out

    def test_writes_hooks_with_session_start_when_remote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup._preflight_remote_endpoint"):
                setup_hooks(remote_url="https://test.fly.dev/mcp")

            settings = json.loads(settings_path.read_text())
            assert "SessionStart" in settings["hooks"]

    def test_preflight_failure_leaves_no_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / ".claude" / "settings.json"
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup._preflight_remote_endpoint",
                       side_effect=SetupError("endpoint 502")):
                with pytest.raises(SetupError, match="502"):
                    setup_hooks(remote_url="https://test.fly.dev/mcp")
            assert not settings_path.exists()
            assert not (mnemon_dir / "remote_url").exists()


class TestRunSetup:
    def test_unknown_target_returns_error(self):
        result = run_setup("unknown")
        assert "Unknown target" in result

    def test_all_targets_registered(self):
        assert set(TARGETS.keys()) == {
            "claude-code",
            "claude-desktop",
            "cursor",
            "gemini",
            "hooks",
        }

    def test_passes_remote_url_from_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup._preflight_remote_endpoint"), \
                 patch("mnemon.doctor.run_doctor", return_value=0), \
                 patch("mnemon.setup.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
                result = run_setup(
                    "claude-code",
                    ["--remote-url", "https://my.fly.dev/mcp", "--skip-doctor"],
                )
        assert "Remote URL" in result
        assert "https://my.fly.dev/mcp" in result

    def test_gemini_ignores_remote_url(self):
        result = run_setup(
            "gemini",
            ["--remote-url", "https://ignored.fly.dev/mcp", "--skip-doctor"],
        )
        assert "mnemon" in result

    def test_auto_runs_doctor_after_successful_setup(self):
        """Without --skip-doctor, a successful setup invokes run_doctor
        and appends its output to the returned message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_doctor = MagicMock(return_value=0)

            def _doctor_side_effect(out, **_kw):
                out.write("mnemon doctor — local mode (FAKE)\n")
                out.write("  OK fake_check: everything fine\n")
                return 0

            fake_doctor.side_effect = _doctor_side_effect
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.doctor.run_doctor", fake_doctor):
                result = run_setup("claude-code", [])

            fake_doctor.assert_called_once()
            assert "Running mnemon doctor" in result
            assert "fake_check: everything fine" in result
            assert "Next steps:" in result

    def test_skip_doctor_bypasses_invocation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.doctor.run_doctor") as mock_doctor:
                result = run_setup("claude-code", ["--skip-doctor"])
            mock_doctor.assert_not_called()
            assert "Running mnemon doctor" not in result

    def test_setup_error_returned_as_failure_message(self):
        """A SetupError must be surfaced as a user-facing string, not a
        crash. Keeps the CLI from dumping a traceback on the user."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mnemon_dir = Path(tmpdir) / ".mnemon"
            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.setup.MNEMON_DIR", mnemon_dir), \
                 patch("mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"), \
                 patch("mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"), \
                 patch("mnemon.setup._preflight_remote_endpoint",
                       side_effect=SetupError("cold fly machine")), \
                 patch("mnemon.setup.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
                result = run_setup(
                    "claude-code",
                    ["--remote-url", "https://x.fly.dev/mcp", "--skip-doctor"],
                )
        # Per-target prefix disambiguates in auto-detect mode.
        assert result.startswith("setup failed (claude-code):")
        assert "cold fly machine" in result

    def test_doctor_failure_appends_note_but_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def _failing_doctor(out, **_kw):
                out.write("mnemon doctor — local mode\n")
                out.write("  FAIL vault: missing\n")
                return 1

            with patch("mnemon.setup.Path.home", return_value=Path(tmpdir)), \
                 patch("mnemon.doctor.run_doctor", side_effect=_failing_doctor):
                result = run_setup("claude-code", [])

        assert "doctor reported issues" in result


class TestStripMnemonHooks:
    def test_drops_mnemon_entries_keeps_others(self):
        hooks = {
            "UserPromptSubmit": [
                {"matcher": "", "hooks": [
                    {"type": "command",
                     "command": "python -m mnemon.hooks.context_surfacing"},
                    {"type": "command", "command": "other-tool"},
                ]},
            ],
            "Stop": [
                {"matcher": "", "hooks": [
                    {"type": "command",
                     "command": "python -m mnemon.hooks.session_extractor"},
                ]},
            ],
        }
        _strip_mnemon_hooks(hooks)
        assert "Stop" not in hooks  # only had a mnemon entry
        ups = hooks["UserPromptSubmit"][0]["hooks"]
        assert len(ups) == 1
        assert ups[0]["command"] == "other-tool"

    def test_leaves_unrelated_events_untouched(self):
        hooks = {
            "CustomEvent": [
                {"matcher": "", "hooks": [
                    {"type": "command", "command": "python -m mnemon.hooks.foo"},
                ]},
            ],
        }
        _strip_mnemon_hooks(hooks)
        # CustomEvent is not one of mnemon's three hook events, so the
        # filter leaves it alone even if the command references mnemon.
        assert "CustomEvent" in hooks


class TestNextStepsBlock:
    def test_local_claude_code_mentions_upgrade_web(self):
        out = _next_steps_block("claude-code", None)
        assert "upgrade web" in out
        assert "Restart Claude Code" in out

    def test_remote_claude_code_omits_upgrade_suggestion(self):
        out = _next_steps_block("claude-code", "https://x.fly.dev/mcp")
        assert "upgrade web" not in out
        assert "Restart Claude Code" in out

    def test_cursor_targets_cursor_restart(self):
        out = _next_steps_block("cursor", None)
        assert "Restart Cursor" in out


class TestPreflightRemoteEndpoint:
    def test_restores_env_vars_on_success(self):
        os.environ.pop("MNEMON_REMOTE_URL", None)
        os.environ.pop("MNEMON_LOCAL_TOKEN", None)

        from mnemon.setup import _preflight_remote_endpoint

        with patch("mnemon.hooks._remote_client.call_tool_sync",
                   return_value=("ok", 0.01)):
            _preflight_remote_endpoint("https://x.fly.dev/mcp", "tok")
        # Env vars set during preflight must not leak into the process
        assert "MNEMON_REMOTE_URL" not in os.environ
        assert "MNEMON_LOCAL_TOKEN" not in os.environ

    def test_restores_env_vars_on_failure(self):
        os.environ["MNEMON_REMOTE_URL"] = "https://existing/mcp"
        os.environ["MNEMON_LOCAL_TOKEN"] = "existing-token"
        try:
            from mnemon.setup import _preflight_remote_endpoint

            with patch(
                "mnemon.hooks._remote_client.call_tool_sync",
                side_effect=TimeoutError("no response"),
            ):
                with pytest.raises(SetupError, match="did not respond"):
                    _preflight_remote_endpoint(
                        "https://new.fly.dev/mcp", "new-tok"
                    )
            # Prior values must be restored, not overwritten
            assert os.environ["MNEMON_REMOTE_URL"] == "https://existing/mcp"
            assert os.environ["MNEMON_LOCAL_TOKEN"] == "existing-token"
        finally:
            os.environ.pop("MNEMON_REMOTE_URL", None)
            os.environ.pop("MNEMON_LOCAL_TOKEN", None)


class TestSetupClaudeDesktop:
    def test_writes_platform_config_local_mode(self, tmp_path, monkeypatch):
        """Claude Desktop has no hook system — only MCP is written."""
        from mnemon.setup import setup_claude_desktop

        # Force the darwin path so the test is platform-deterministic.
        monkeypatch.setattr("mnemon.setup.sys.platform", "darwin")
        monkeypatch.setattr("mnemon.setup.Path.home", lambda: tmp_path)

        out = setup_claude_desktop()
        expected = (
            tmp_path
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
        assert expected.exists()
        config = json.loads(expected.read_text())
        assert "mnemon" in config["mcpServers"]
        # Hooks are not Claude Desktop's concern
        assert "hooks" not in config
        assert "Mode: stdio (local)" in out

    def test_remote_mode_writes_http_transport(self, tmp_path, monkeypatch):
        from mnemon.setup import setup_claude_desktop

        monkeypatch.setattr("mnemon.setup.sys.platform", "darwin")
        monkeypatch.setattr("mnemon.setup.Path.home", lambda: tmp_path)
        mnemon_dir = tmp_path / ".mnemon"
        monkeypatch.setattr("mnemon.setup.MNEMON_DIR", mnemon_dir)
        monkeypatch.setattr(
            "mnemon.setup.LOCAL_TOKEN_FILE", mnemon_dir / "local_token"
        )
        monkeypatch.setattr(
            "mnemon.setup.REMOTE_URL_FILE", mnemon_dir / "remote_url"
        )
        with patch("mnemon.setup._preflight_remote_endpoint"):
            setup_claude_desktop(
                remote_url="https://x.fly.dev/mcp", token="tok-123"
            )

        expected = (
            tmp_path
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
        config = json.loads(expected.read_text())
        entry = config["mcpServers"]["mnemon"]
        assert entry["url"] == "https://x.fly.dev/mcp"
        assert entry["headers"]["Authorization"] == "Bearer tok-123"


class TestDetectInstalledClients:
    def test_detects_claude_code_only(self, tmp_path, monkeypatch):
        from mnemon.setup import detect_installed_clients

        monkeypatch.setattr("mnemon.setup.Path.home", lambda: tmp_path)
        # Force Linux so Claude Desktop probe looks under ~/.config/Claude/
        monkeypatch.setattr("mnemon.setup.sys.platform", "linux")
        (tmp_path / ".claude").mkdir()
        detected = detect_installed_clients()
        assert detected == ["claude-code"]

    def test_detects_multiple(self, tmp_path, monkeypatch):
        from mnemon.setup import detect_installed_clients

        monkeypatch.setattr("mnemon.setup.Path.home", lambda: tmp_path)
        monkeypatch.setattr("mnemon.setup.sys.platform", "linux")
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".cursor").mkdir()
        detected = detect_installed_clients()
        assert "claude-code" in detected
        assert "cursor" in detected
        # Stable order per _AUTODETECT_ORDER
        assert detected == ["claude-code", "cursor"]

    def test_detects_nothing_when_no_client_dirs(self, tmp_path, monkeypatch):
        from mnemon.setup import detect_installed_clients

        monkeypatch.setattr("mnemon.setup.Path.home", lambda: tmp_path)
        monkeypatch.setattr("mnemon.setup.sys.platform", "linux")
        assert detect_installed_clients() == []


class TestRunSetupAutodetect:
    def test_no_target_no_clients_returns_helpful_message(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("mnemon.setup.Path.home", lambda: tmp_path)
        monkeypatch.setattr("mnemon.setup.sys.platform", "linux")
        out = run_setup(None, ["--skip-doctor"])
        assert "No MCP clients detected" in out
        assert "mnemon setup <target>" in out

    def test_no_target_configures_detected_clients(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("mnemon.setup.Path.home", lambda: tmp_path)
        monkeypatch.setattr("mnemon.setup.sys.platform", "linux")
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".cursor").mkdir()
        out = run_setup(None, ["--skip-doctor"])
        assert "Detected MCP clients: claude-code, cursor" in out
        # Each detected target produced its own block
        assert "── claude-code ──" in out
        assert "── cursor ──" in out
        # Gemini tail is always printed as a manual-step reminder
        assert "── gemini (manual) ──" in out
        # Aggregate footer (singular "Next steps:"), not per-target
        assert out.count("Next steps:") == 1
        # And the real files were written
        assert (tmp_path / ".claude" / "settings.json").exists()
        assert (tmp_path / ".cursor" / "mcp.json").exists()


class TestFailOnWarnPropagation:
    def test_setup_surfaces_doctor_warning_as_failure(
        self, tmp_path, monkeypatch
    ):
        """With fail_on_warn plumbed through, a warning-only doctor run
        after setup appends the "doctor reported issues" NOTE so users
        don't ignore it."""
        monkeypatch.setattr("mnemon.setup.Path.home", lambda: tmp_path)
        monkeypatch.setattr("mnemon.setup.sys.platform", "linux")

        def _warning_doctor(out, *, fail_on_warn=False, **_):
            # Simulate a doctor run that warns but doesn't fail. With
            # fail_on_warn=True, run_doctor returns 1 anyway.
            out.write("mnemon doctor — local mode\n")
            out.write("  WARN something: heads up\n")
            return 1 if fail_on_warn else 0

        with patch("mnemon.doctor.run_doctor", side_effect=_warning_doctor):
            result = run_setup("claude-code", [])

        assert "doctor reported issues" in result
        assert "including warnings" in result


class TestParseSkipDoctor:
    def test_skip_doctor_flag(self):
        result = _parse_setup_args(["--skip-doctor"])
        assert result["skip_doctor"] is True

    def test_skip_doctor_default_false(self):
        result = _parse_setup_args([])
        assert result["skip_doctor"] is False

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
