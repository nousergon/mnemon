"""Layer 1 unit tests for :mod:`mnemon.downgrade` — ``mnemon downgrade local``.

Covers: remote-required guard, sync pull failure aborts, client
reconfigure iterates in local mode, --destroy-fly-app prompt + confirm
+ unattended paths, non-fly.dev URL can't be auto-destroyed, doctor
runs against local after.
"""

from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from mnemon import downgrade as dg
from mnemon.downgrade import (
    DowngradeError,
    _client_config_root,
    _confirm,
    _extract_app_name,
    _reconfigure_clients_local,
    _resolve_remote_url,
    downgrade_local,
)


def _ok_completed():
    return CompletedProcess(args=[], returncode=0, stdout="", stderr="")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    monkeypatch.setattr("mnemon.downgrade.Path.home", lambda: tmp_path)
    # Also override the module-level constants that resolved at import time.
    monkeypatch.setattr(
        "mnemon.downgrade.MNEMON_DIR", tmp_path / ".mnemon"
    )
    monkeypatch.setattr(
        "mnemon.downgrade.REMOTE_URL_FILE",
        tmp_path / ".mnemon" / "remote_url",
    )
    monkeypatch.setattr(
        "mnemon.downgrade.LOCAL_TOKEN_FILE",
        tmp_path / ".mnemon" / "local_token",
    )
    monkeypatch.setenv("MNEMON_VAULT_DIR", str(tmp_path / ".mnemon"))
    monkeypatch.setenv("MNEMON_S3_BUCKET", "test-bucket")
    monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
    monkeypatch.delenv("MNEMON_CLIENT_CONFIG_ROOT", raising=False)
    yield


# ── _extract_app_name ────────────────────────────────────────────────────────


class TestExtractAppName:
    def test_parses_standard_fly_url(self):
        assert (
            _extract_app_name("https://mnemon-memory.fly.dev/mcp")
            == "mnemon-memory"
        )

    def test_parses_without_path(self):
        assert _extract_app_name("https://my-app.fly.dev") == "my-app"

    def test_none_for_custom_domain(self):
        assert (
            _extract_app_name("https://mnemon.example.com/mcp") is None
        )

    def test_none_for_invalid_shape(self):
        assert _extract_app_name("not-a-url") is None


# ── downgrade_local ──────────────────────────────────────────────────────────


class TestRequireRemote:
    def test_no_remote_config_raises(self, tmp_path):
        # Neither env var nor the file exists
        with pytest.raises(DowngradeError, match="nothing to downgrade"):
            downgrade_local(skip_doctor=True)


class TestSyncPull:
    def _seed_remote_config(self, tmp_path, url="https://mnemon-test.fly.dev/mcp"):
        mnemon_dir = tmp_path / ".mnemon"
        mnemon_dir.mkdir()
        (mnemon_dir / "remote_url").write_text(url)

    def test_sync_pull_failure_aborts(self, tmp_path):
        self._seed_remote_config(tmp_path)
        with patch(
            "mnemon.sync.pull",
            return_value={"pulled": [], "errors": ["access denied"]},
        ), patch("mnemon.downgrade._fly_dump_vault"):
            with pytest.raises(DowngradeError, match="S3 pull failed"):
                downgrade_local(skip_doctor=True)
        # Remote config was NOT cleared — downgrade aborted.
        assert (tmp_path / ".mnemon" / "remote_url").exists()

    def test_bucket_required(self, tmp_path, monkeypatch):
        self._seed_remote_config(tmp_path)
        monkeypatch.delenv("MNEMON_S3_BUCKET", raising=False)
        with pytest.raises(DowngradeError, match="MNEMON_S3_BUCKET"):
            downgrade_local(skip_doctor=True)


class TestReconfigureLocal:
    def _seed_remote_config(self, tmp_path):
        mnemon_dir = tmp_path / ".mnemon"
        mnemon_dir.mkdir()
        (mnemon_dir / "remote_url").write_text(
            "https://mnemon-test.fly.dev/mcp"
        )

    def test_happy_path_clears_remote_and_reconfigures(
        self, tmp_path, monkeypatch
    ):
        self._seed_remote_config(tmp_path)

        with patch(
            "mnemon.sync.pull",
            return_value={"pulled": ["sqlite"], "errors": []},
        ), \
            patch(
                "mnemon.setup.detect_installed_clients",
                return_value=["claude-code", "cursor"],
            ), \
            patch(
                "mnemon.downgrade._reconfigure_clients_local",
                return_value=["claude-code", "cursor"],
            ) as mock_reconfig, \
            patch("mnemon.downgrade._fly_dump_vault"), \
            patch("mnemon.doctor.run_doctor", return_value=0):
            result = downgrade_local(skip_doctor=True)

        mock_reconfig.assert_called_once()
        # Remote config cleared
        assert not (tmp_path / ".mnemon" / "remote_url").exists()
        assert "Downgrade to local complete" in result
        assert "claude-code, cursor" in result


class TestDestroyFlyApp:
    def _seed_remote_config(self, tmp_path, url="https://mnemon-test-999.fly.dev/mcp"):
        mnemon_dir = tmp_path / ".mnemon"
        mnemon_dir.mkdir()
        (mnemon_dir / "remote_url").write_text(url)

    def test_destroy_without_yes_prompts(self, tmp_path):
        self._seed_remote_config(tmp_path)
        with patch(
            "mnemon.sync.pull",
            return_value={"pulled": [], "errors": []},
        ), \
            patch(
                "mnemon.setup.detect_installed_clients",
                return_value=[],
            ), \
            patch(
                "mnemon.downgrade._reconfigure_clients_local",
                return_value=[],
            ), \
            patch(
                "mnemon.downgrade._confirm", return_value=True
            ) as mock_confirm, \
            patch(
                "mnemon.downgrade.subprocess.run",
                return_value=_ok_completed(),
            ) as mock_run:
            result = downgrade_local(
                destroy_fly_app=True, skip_doctor=True
            )
        mock_confirm.assert_called_once()
        # flyctl apps destroy was invoked
        destroy_calls = [
            c for c in mock_run.call_args_list if "destroy" in c.args[0]
        ]
        assert len(destroy_calls) == 1
        assert destroy_calls[0].args[0][:3] == ["flyctl", "apps", "destroy"]
        assert "destroyed (mnemon-test-999)" in result

    def test_destroy_with_yes_skips_prompt(self, tmp_path):
        self._seed_remote_config(tmp_path)
        with patch(
            "mnemon.sync.pull",
            return_value={"pulled": [], "errors": []},
        ), \
            patch(
                "mnemon.setup.detect_installed_clients",
                return_value=[],
            ), \
            patch(
                "mnemon.downgrade._reconfigure_clients_local",
                return_value=[],
            ), \
            patch(
                "mnemon.downgrade._confirm"
            ) as mock_confirm, \
            patch(
                "mnemon.downgrade.subprocess.run",
                return_value=_ok_completed(),
            ) as mock_run:
            downgrade_local(
                destroy_fly_app=True, yes=True, skip_doctor=True
            )
        mock_confirm.assert_not_called()
        destroy_calls = [
            c for c in mock_run.call_args_list if "destroy" in c.args[0]
        ]
        assert len(destroy_calls) == 1

    def test_destroy_declined_leaves_app_running(self, tmp_path):
        self._seed_remote_config(tmp_path)
        with patch(
            "mnemon.sync.pull",
            return_value={"pulled": [], "errors": []},
        ), \
            patch(
                "mnemon.setup.detect_installed_clients",
                return_value=[],
            ), \
            patch(
                "mnemon.downgrade._reconfigure_clients_local",
                return_value=[],
            ), \
            patch(
                "mnemon.downgrade._confirm", return_value=False
            ), \
            patch(
                "mnemon.downgrade.subprocess.run"
            ) as mock_run:
            # skip_fly_push=True because the Fly→S3 pre-pull push isn't
            # what this test exercises; it tests destroy-decline behavior.
            result = downgrade_local(
                destroy_fly_app=True, skip_doctor=True, skip_fly_push=True
            )
        # No flyctl destroy call
        mock_run.assert_not_called()
        assert "mnemon-test-999 is still running" in result

    def test_destroy_custom_domain_requires_app_name_override(
        self, tmp_path
    ):
        # Non-fly.dev URL — can't auto-infer the app name.
        self._seed_remote_config(
            tmp_path, url="https://mnemon.example.com/mcp"
        )
        with patch(
            "mnemon.sync.pull",
            return_value={"pulled": [], "errors": []},
        ), \
            patch(
                "mnemon.setup.detect_installed_clients",
                return_value=[],
            ), \
            patch(
                "mnemon.downgrade._reconfigure_clients_local",
                return_value=[],
            ):
            with pytest.raises(DowngradeError, match="Could not infer"):
                downgrade_local(
                    destroy_fly_app=True, yes=True, skip_doctor=True
                )

    def test_destroy_with_override_succeeds_on_custom_domain(
        self, tmp_path
    ):
        self._seed_remote_config(
            tmp_path, url="https://mnemon.example.com/mcp"
        )
        with patch(
            "mnemon.sync.pull",
            return_value={"pulled": [], "errors": []},
        ), \
            patch(
                "mnemon.setup.detect_installed_clients",
                return_value=[],
            ), \
            patch(
                "mnemon.downgrade._reconfigure_clients_local",
                return_value=[],
            ), \
            patch(
                "mnemon.downgrade.subprocess.run",
                return_value=_ok_completed(),
            ) as mock_run:
            result = downgrade_local(
                destroy_fly_app=True,
                yes=True,
                skip_doctor=True,
                app_name_override="my-custom-app",
            )
        destroy_calls = [
            c for c in mock_run.call_args_list if "destroy" in c.args[0]
        ]
        assert destroy_calls[0].args[0] == [
            "flyctl",
            "apps",
            "destroy",
            "my-custom-app",
            "-y",
        ]
        assert "destroyed (my-custom-app)" in result


class TestFlyDumpVaultBeforePull:
    """Regression for the 2026-05-21 Layer-3 bug: downgrade only did
    S3→local pull, never Fly→S3 push first. Any memory added on the
    Fly side after upgrade time was silently lost on downgrade because
    the local pull got the stale upgrade-time S3 snapshot. Surfaced
    as "expected 4 docs after downgrade, got '3'" — the 4th doc was
    added via remote after upgrade, only existed in Fly's SQLite, and
    didn't make it back to local.

    For prod operators this would have been a quiet, severe data-loss
    bug: any work done via remote between upgrade and downgrade would
    vanish. Fix: SSH into Fly and run `mnemon sync push` before the
    local `mnemon sync pull`.
    """

    def _seed_remote_config(self, tmp_path, url="https://mnemon-test-999.fly.dev/mcp"):
        mnemon_dir = tmp_path / ".mnemon"
        mnemon_dir.mkdir()
        (mnemon_dir / "remote_url").write_text(url)

    def test_fly_dump_runs_before_s3_pull_by_default(self, tmp_path):
        self._seed_remote_config(tmp_path)
        call_order: list[str] = []

        def _record_dump(app_name):
            call_order.append("fly_dump")

        def _record_pull():
            call_order.append("s3_pull")
            return {"pulled": ["sqlite"], "errors": []}

        with patch("mnemon.downgrade._fly_dump_vault", side_effect=_record_dump), \
            patch("mnemon.sync.pull", side_effect=_record_pull), \
            patch("mnemon.setup.detect_installed_clients", return_value=[]), \
            patch("mnemon.downgrade._reconfigure_clients_local", return_value=[]):
            downgrade_local(skip_doctor=True)

        assert call_order == ["fly_dump", "s3_pull"], (
            f"expected fly_dump → s3_pull, got {call_order}"
        )

    def test_skip_fly_push_omits_dump(self, tmp_path):
        self._seed_remote_config(tmp_path)
        with patch("mnemon.downgrade._fly_dump_vault") as mock_dump, \
            patch("mnemon.sync.pull", return_value={"pulled": ["sqlite"], "errors": []}), \
            patch("mnemon.setup.detect_installed_clients", return_value=[]), \
            patch("mnemon.downgrade._reconfigure_clients_local", return_value=[]):
            downgrade_local(skip_doctor=True, skip_fly_push=True)
        mock_dump.assert_not_called()

    def test_fly_dump_failure_aborts_downgrade(self, tmp_path):
        self._seed_remote_config(tmp_path)
        from subprocess import CalledProcessError
        with patch(
            "mnemon.downgrade._fly_dump_vault",
            side_effect=CalledProcessError(1, ["flyctl"]),
        ), patch("mnemon.sync.pull") as mock_pull:
            with pytest.raises(DowngradeError, match="Fly→S3 dump failed"):
                downgrade_local(skip_doctor=True)
        # Pull never ran — abort happened first, so no stale-snapshot
        # restore of the local vault.
        mock_pull.assert_not_called()

    def test_fly_dump_skipped_when_app_name_not_extractable(self, tmp_path):
        # Custom (non-fly.dev) domain → _extract_app_name returns None →
        # we can't SSH so we skip the dump. Pull still runs.
        self._seed_remote_config(tmp_path, url="https://mnemon.example.com/mcp")
        with patch("mnemon.downgrade._fly_dump_vault") as mock_dump, \
            patch("mnemon.sync.pull", return_value={"pulled": ["sqlite"], "errors": []}), \
            patch("mnemon.setup.detect_installed_clients", return_value=[]), \
            patch("mnemon.downgrade._reconfigure_clients_local", return_value=[]):
            downgrade_local(skip_doctor=True)
        mock_dump.assert_not_called()

    def test_fly_dump_runs_ssh_console_with_mnemon_sync_push(self, tmp_path):
        # Direct test that _fly_dump_vault issues the expected flyctl
        # command. Simplified 2026-05-27: the inlined ~40-line Python
        # snapshot script collapsed to `mnemon sync push` once the
        # backup-API push was canonical on Fly (mnemon-memory==0.6.0+
        # ships sync.push using Connection.backup() per PR #142).
        # The installed mnemon on the Fly machine is the source of
        # truth for the snapshot primitive; this function just routes.
        from mnemon import downgrade as dgmod
        with patch("mnemon.downgrade.subprocess.run") as mock_run:
            dgmod._fly_dump_vault("mnemon-test-abc")
        mock_run.assert_called_once()
        call_args = mock_run.call_args.args[0]
        # Outer shape: flyctl ssh console --app NAME -C "mnemon sync push"
        assert call_args[0] == "flyctl"
        assert call_args[1:5] == ["ssh", "console", "--app", "mnemon-test-abc"]
        assert call_args[5] == "-C"
        assert call_args[6] == "mnemon sync push", (
            f"expected `mnemon sync push`, got: {call_args[6]!r}"
        )


# ── _resolve_remote_url ──────────────────────────────────────────────────────


class TestResolveRemoteUrl:
    def test_env_var_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("MNEMON_REMOTE_URL", "https://env-app.fly.dev/mcp")
        assert _resolve_remote_url() == "https://env-app.fly.dev/mcp"

    def test_falls_back_to_remote_url_file(self, tmp_path):
        mnemon_dir = tmp_path / ".mnemon"
        mnemon_dir.mkdir()
        (mnemon_dir / "remote_url").write_text("https://file-app.fly.dev/mcp\n")
        assert _resolve_remote_url() == "https://file-app.fly.dev/mcp"

    def test_oserror_reading_file_falls_through_to_raise(self, monkeypatch):
        # exists() True but read_text() raises OSError → the except OSError
        # branch swallows it and we fall through to the "no remote" raise.
        fake_file = MagicMock()
        fake_file.exists.return_value = True
        fake_file.read_text.side_effect = OSError("permission denied")
        monkeypatch.setattr("mnemon.downgrade.REMOTE_URL_FILE", fake_file)
        with pytest.raises(DowngradeError, match="nothing to downgrade"):
            _resolve_remote_url()


# ── _client_config_root ──────────────────────────────────────────────────────


class TestClientConfigRoot:
    def test_default_is_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MNEMON_CLIENT_CONFIG_ROOT", raising=False)
        # _isolate_env patches Path.home → tmp_path
        assert _client_config_root() == tmp_path

    def test_override_env_wins(self, monkeypatch, tmp_path):
        override = tmp_path / "custom-root"
        monkeypatch.setenv("MNEMON_CLIENT_CONFIG_ROOT", str(override))
        assert _client_config_root() == override


# ── _reconfigure_clients_local ───────────────────────────────────────────────


class TestReconfigureClientsLocalDirect:
    """Exercise the real _reconfigure_clients_local (the downgrade_local
    tests patch it out). Verifies it calls each detected client's setup
    function in local (stdio) mode, skips the 'hooks' pseudo-target, and
    restores setup.Path.home afterward."""

    def test_iterates_targets_in_local_mode_and_restores_home(
        self, monkeypatch, tmp_path
    ):
        from mnemon import setup as setup_mod

        prior_home = setup_mod.Path.home
        cc = MagicMock()
        cursor = MagicMock()
        fake_targets = {"claude-code": cc, "cursor": cursor}
        monkeypatch.setattr(setup_mod, "TARGETS", fake_targets)
        monkeypatch.setenv("MNEMON_CLIENT_CONFIG_ROOT", str(tmp_path / "root"))

        # 'hooks' must be skipped even if it sneaks into the detected list.
        result = _reconfigure_clients_local(["claude-code", "hooks", "cursor"])

        assert result == ["claude-code", "cursor"]
        cc.assert_called_once_with(remote_url=None, token=None)
        cursor.assert_called_once_with(remote_url=None, token=None)
        # Path.home restored to whatever it was on entry.
        assert setup_mod.Path.home is prior_home

    def test_restores_home_even_when_setup_raises(self, monkeypatch, tmp_path):
        from mnemon import setup as setup_mod

        prior_home = setup_mod.Path.home
        boom = MagicMock(side_effect=RuntimeError("setup blew up"))
        monkeypatch.setattr(setup_mod, "TARGETS", {"claude-code": boom})
        monkeypatch.setenv("MNEMON_CLIENT_CONFIG_ROOT", str(tmp_path / "root"))

        with pytest.raises(RuntimeError, match="setup blew up"):
            _reconfigure_clients_local(["claude-code"])
        # finally: block must restore Path.home despite the exception.
        assert setup_mod.Path.home is prior_home


# ── _confirm ─────────────────────────────────────────────────────────────────


class TestConfirm:
    def test_non_tty_returns_false(self, monkeypatch):
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = False
        monkeypatch.setattr("mnemon.downgrade.sys.stdin", fake_stdin)
        assert _confirm("destroy? ") is False

    def test_tty_yes_returns_true(self, monkeypatch):
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = True
        monkeypatch.setattr("mnemon.downgrade.sys.stdin", fake_stdin)
        monkeypatch.setattr("builtins.input", lambda _prompt: "  YES  ")
        assert _confirm("destroy? ") is True

    def test_tty_no_returns_false(self, monkeypatch):
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = True
        monkeypatch.setattr("mnemon.downgrade.sys.stdin", fake_stdin)
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        assert _confirm("destroy? ") is False

    def test_tty_eoferror_returns_false(self, monkeypatch):
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = True
        monkeypatch.setattr("mnemon.downgrade.sys.stdin", fake_stdin)

        def _raise(_prompt):
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise)
        assert _confirm("destroy? ") is False


# ── doctor step (skip_doctor=False) ──────────────────────────────────────────


class TestDoctorStep:
    """The existing downgrade_local tests all pass skip_doctor=True. These
    exercise the Step-5 doctor block (run_doctor against the restored local
    vault) including the clean, issues-found, and crash paths."""

    def _seed_remote_config(self, tmp_path, url="https://mnemon-test.fly.dev/mcp"):
        mnemon_dir = tmp_path / ".mnemon"
        mnemon_dir.mkdir()
        (mnemon_dir / "remote_url").write_text(url)

    def _happy_path_patches(self):
        return (
            patch("mnemon.downgrade._fly_dump_vault"),
            patch(
                "mnemon.sync.pull",
                return_value={"pulled": ["sqlite"], "errors": []},
            ),
            patch("mnemon.setup.detect_installed_clients", return_value=[]),
            patch("mnemon.downgrade._reconfigure_clients_local", return_value=[]),
        )

    def test_doctor_clean_no_note(self, tmp_path):
        self._seed_remote_config(tmp_path)
        p1, p2, p3, p4 = self._happy_path_patches()
        with p1, p2, p3, p4, patch(
            "mnemon.doctor.run_doctor", return_value=0
        ) as mock_doc:
            result = downgrade_local(skip_doctor=False)
        mock_doc.assert_called_once()
        assert "Running mnemon doctor against the restored local vault" in result
        assert "doctor reported issues" not in result

    def test_doctor_issues_appends_note(self, tmp_path):
        self._seed_remote_config(tmp_path)
        p1, p2, p3, p4 = self._happy_path_patches()
        with p1, p2, p3, p4, patch("mnemon.doctor.run_doctor", return_value=1):
            result = downgrade_local(skip_doctor=False)
        assert "doctor reported issues against the local vault" in result

    def test_doctor_crash_is_caught(self, tmp_path):
        self._seed_remote_config(tmp_path)
        p1, p2, p3, p4 = self._happy_path_patches()
        with p1, p2, p3, p4, patch(
            "mnemon.doctor.run_doctor", side_effect=RuntimeError("doctor boom")
        ):
            result = downgrade_local(skip_doctor=False)
        # Crash is captured into the summary, not propagated.
        assert "doctor invocation crashed" in result
        assert "RuntimeError" in result
