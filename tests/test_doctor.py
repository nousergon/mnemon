"""Tests for mnemon doctor — the remote-vault self-diagnostic."""

from __future__ import annotations

import io
import json
import os
import socket
import stat
from unittest.mock import patch, MagicMock

import pytest

from mnemon import doctor
from mnemon.hooks._remote_client import RemoteClientConfigError


# ── check_remote_url ────────────────────────────────────────────────────────


class TestCheckRemoteUrl:
    def test_passes_when_env_var_set(self, monkeypatch):
        monkeypatch.setenv("MNEMON_REMOTE_URL", "https://example.fly.dev/mcp")
        result = doctor.check_remote_url()
        assert result.ok
        assert "example.fly.dev" in result.detail
        assert "env" in result.detail

    def test_fails_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
        # Redirect the file lookup to a non-existent path
        monkeypatch.setattr(doctor, "REMOTE_URL_FILE", tmp_path / "does-not-exist")
        with patch("mnemon.doctor.get_remote_url",
                   side_effect=RemoteClientConfigError("not configured")):
            result = doctor.check_remote_url()
        assert not result.ok
        assert "not configured" in result.detail


# ── check_local_token ───────────────────────────────────────────────────────


class TestCheckLocalToken:
    def test_passes_when_env_var_set(self, monkeypatch):
        fake_token = "fake-test-token"  # 15 bytes, hyphenated to dodge secret scanners
        monkeypatch.setenv("MNEMON_LOCAL_TOKEN", fake_token)
        result = doctor.check_local_token()
        assert result.ok
        assert f"{len(fake_token)} bytes" in result.detail
        assert "env" in result.detail

    def test_fails_when_unset(self, monkeypatch):
        monkeypatch.delenv("MNEMON_LOCAL_TOKEN", raising=False)
        with patch("mnemon.doctor.get_local_token",
                   side_effect=RemoteClientConfigError("no token")):
            result = doctor.check_local_token()
        assert not result.ok
        assert "no token" in result.detail


# ── check_token_file_perms ──────────────────────────────────────────────────


class TestCheckTokenFilePerms:
    def test_skipped_when_env_var_used(self, monkeypatch):
        monkeypatch.setenv("MNEMON_LOCAL_TOKEN", "fromenv")
        result = doctor.check_token_file_perms()
        assert result.ok
        assert not result.warn
        assert "env" in result.detail

    def test_skipped_when_no_token_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MNEMON_LOCAL_TOKEN", raising=False)
        monkeypatch.setattr(doctor, "LOCAL_TOKEN_FILE", tmp_path / "nope")
        result = doctor.check_token_file_perms()
        assert result.ok
        assert not result.warn

    def test_warns_on_group_readable(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MNEMON_LOCAL_TOKEN", raising=False)
        token_file = tmp_path / "local_token"
        token_file.write_text("secret")
        token_file.chmod(0o640)  # group-readable — not safe
        monkeypatch.setattr(doctor, "LOCAL_TOKEN_FILE", token_file)
        result = doctor.check_token_file_perms()
        assert result.ok
        assert result.warn
        assert "chmod 600" in result.detail

    def test_passes_on_0600(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MNEMON_LOCAL_TOKEN", raising=False)
        token_file = tmp_path / "local_token"
        token_file.write_text("secret")
        token_file.chmod(0o600)
        monkeypatch.setattr(doctor, "LOCAL_TOKEN_FILE", token_file)
        result = doctor.check_token_file_perms()
        assert result.ok
        assert not result.warn
        assert "0600" in result.detail


# ── check_health_endpoint ───────────────────────────────────────────────────


class TestCheckHealthEndpoint:
    def test_passes_on_ok_response(self, monkeypatch):
        monkeypatch.setenv("MNEMON_REMOTE_URL", "https://example.fly.dev/mcp")

        fake_resp = MagicMock()
        fake_resp.status = 200
        fake_resp.read.return_value = json.dumps({"status": "ok"}).encode()
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=fake_resp) as m:
            result = doctor.check_health_endpoint()

        assert result.ok
        # Should have stripped the /mcp suffix before hitting /health
        called_url = m.call_args[0][0]
        assert called_url == "https://example.fly.dev/health"

    def test_fails_on_non_200(self, monkeypatch):
        monkeypatch.setenv("MNEMON_REMOTE_URL", "https://example.fly.dev/mcp")

        fake_resp = MagicMock()
        fake_resp.status = 503
        fake_resp.read.return_value = b"{}"
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = doctor.check_health_endpoint()

        assert not result.ok
        assert "503" in result.detail

    def test_fails_on_connection_error(self, monkeypatch):
        monkeypatch.setenv("MNEMON_REMOTE_URL", "https://example.fly.dev/mcp")
        import urllib.error
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            result = doctor.check_health_endpoint()
        assert not result.ok
        assert "connection refused" in result.detail

    def test_fails_on_non_ok_status_payload(self, monkeypatch):
        monkeypatch.setenv("MNEMON_REMOTE_URL", "https://example.fly.dev/mcp")

        fake_resp = MagicMock()
        fake_resp.status = 200
        fake_resp.read.return_value = json.dumps({"status": "degraded"}).encode()
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = doctor.check_health_endpoint()

        assert not result.ok
        assert "degraded" in result.detail


# ── check_auth_and_tool_call ────────────────────────────────────────────────


class TestCheckAuthAndToolCall:
    def test_passes_on_successful_call(self):
        with patch("mnemon.doctor.call_tool_sync",
                   return_value=("some text result", 0.123)) as m:
            result = doctor.check_auth_and_tool_call()
        assert result.ok
        assert "memory_search" in result.detail
        # Ensure we passed a recognizable client label
        assert m.call_args.kwargs["client_label"] == "mnemon-doctor"

    def test_fails_on_timeout(self):
        with patch("mnemon.doctor.call_tool_sync",
                   side_effect=TimeoutError()):
            result = doctor.check_auth_and_tool_call()
        assert not result.ok
        assert "timed out" in result.detail

    def test_fails_on_auth_error(self):
        with patch("mnemon.doctor.call_tool_sync",
                   side_effect=RuntimeError("401 Unauthorized")):
            result = doctor.check_auth_and_tool_call()
        assert not result.ok
        assert "401" in result.detail


# ── check_round_trip ────────────────────────────────────────────────────────


class TestCheckRoundTrip:
    def test_full_round_trip_passes(self):
        save_response = 'Saved memory #999: "mnemon-doctor-probe-abcd1234"'
        # The search response must contain the probe title
        def fake_call(tool, args, **kwargs):
            if tool == "memory_save":
                return (save_response, 0.1)
            if tool == "memory_search":
                return (f"1. {args['query']} (some metadata)", 0.1)
            if tool == "memory_forget":
                return ("Forgot memory #999.", 0.05)
            raise AssertionError(f"unexpected tool: {tool}")

        with patch("mnemon.doctor.call_tool_sync", side_effect=fake_call):
            result = doctor.check_round_trip()

        assert result.ok
        assert not result.warn
        assert "999" in result.detail

    def test_save_failure_short_circuits(self):
        with patch("mnemon.doctor.call_tool_sync",
                   side_effect=RuntimeError("save exploded")):
            result = doctor.check_round_trip()
        assert not result.ok
        assert "save failed" in result.detail

    def test_unparseable_save_response_fails(self):
        with patch("mnemon.doctor.call_tool_sync",
                   return_value=("no doc id anywhere here", 0.1)):
            result = doctor.check_round_trip()
        assert not result.ok
        assert "could not parse" in result.detail

    def test_search_failure_cleans_up(self):
        save_response = 'Saved memory #999: "mnemon-doctor-probe-xx"'
        forget_calls: list = []

        def fake_call(tool, args, **kwargs):
            if tool == "memory_save":
                return (save_response, 0.1)
            if tool == "memory_search":
                raise RuntimeError("search exploded")
            if tool == "memory_forget":
                forget_calls.append(args)
                return ("Forgot memory #999.", 0.05)
            raise AssertionError(tool)

        with patch("mnemon.doctor.call_tool_sync", side_effect=fake_call):
            result = doctor.check_round_trip()

        assert not result.ok
        assert "search failed" in result.detail
        # Best-effort cleanup should still have fired
        assert forget_calls == [{"document_id": 999}]

    def test_forget_failure_warns_but_passes(self):
        save_response = 'Saved memory #999: "mnemon-doctor-probe-xx"'

        def fake_call(tool, args, **kwargs):
            if tool == "memory_save":
                return (save_response, 0.1)
            if tool == "memory_search":
                return (f"found: {args['query']}", 0.1)
            if tool == "memory_forget":
                raise RuntimeError("forget exploded")
            raise AssertionError(tool)

        with patch("mnemon.doctor.call_tool_sync", side_effect=fake_call):
            result = doctor.check_round_trip()

        assert result.ok
        assert result.warn
        assert "leaked" in result.detail

    def test_saved_memory_not_found_fails(self):
        save_response = 'Saved memory #999: "mnemon-doctor-probe-xx"'
        forget_calls: list = []

        def fake_call(tool, args, **kwargs):
            if tool == "memory_save":
                return (save_response, 0.1)
            if tool == "memory_search":
                return ("some other unrelated result", 0.1)
            if tool == "memory_forget":
                forget_calls.append(args)
                return ("ok", 0.05)
            raise AssertionError(tool)

        with patch("mnemon.doctor.call_tool_sync", side_effect=fake_call):
            result = doctor.check_round_trip()

        assert not result.ok
        assert "not found by search" in result.detail
        assert forget_calls == [{"document_id": 999}]


# ── run_doctor end-to-end ───────────────────────────────────────────────────


class TestRunDoctor:
    def test_all_pass_returns_zero(self):
        def ok(name: str) -> doctor.CheckResult:
            return doctor.CheckResult(name, True, "fine")

        fake_checks = [
            lambda: ok("A"),
            lambda: ok("B"),
        ]
        buf = io.StringIO()
        with patch("mnemon.doctor.CHECKS", fake_checks):
            code = doctor.run_doctor(out=buf)
        assert code == 0
        output = buf.getvalue()
        assert "All 2 checks passed" in output
        assert doctor.PASS in output

    def test_any_fail_returns_one(self):
        fake_checks = [
            lambda: doctor.CheckResult("A", True, "ok"),
            lambda: doctor.CheckResult("B", False, "broken"),
        ]
        buf = io.StringIO()
        with patch("mnemon.doctor.CHECKS", fake_checks):
            code = doctor.run_doctor(out=buf)
        assert code == 1
        output = buf.getvalue()
        assert "1/2 checks failed" in output
        assert doctor.FAIL in output

    def test_warn_only_returns_zero_with_note(self):
        fake_checks = [
            lambda: doctor.CheckResult("A", True, "ok"),
            lambda: doctor.CheckResult("B", True, "heads-up", warn=True),
        ]
        buf = io.StringIO()
        with patch("mnemon.doctor.CHECKS", fake_checks):
            code = doctor.run_doctor(out=buf)
        assert code == 0
        output = buf.getvalue()
        assert "1 warning" in output

    def test_cli_dispatches_to_doctor(self, monkeypatch):
        """`mnemon doctor` on the CLI should call run_doctor and exit with its code."""
        from mnemon.cli import main

        monkeypatch.setattr("sys.argv", ["mnemon", "doctor"])
        with patch("mnemon.doctor.run_doctor", return_value=0) as mock_run:
            with pytest.raises(SystemExit) as excinfo:
                main()
        mock_run.assert_called_once()
        assert excinfo.value.code == 0

    def test_cli_propagates_nonzero_exit(self, monkeypatch):
        from mnemon.cli import main

        monkeypatch.setattr("sys.argv", ["mnemon", "doctor"])
        with patch("mnemon.doctor.run_doctor", return_value=1):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1
