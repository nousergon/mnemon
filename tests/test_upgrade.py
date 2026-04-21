"""Layer 1 unit tests for :mod:`mnemon.upgrade` — ``mnemon upgrade web``.

Covers: prereq validation, prod-app-name collision guard, S3 push
failure aborts before any flyctl call, happy path argument construction,
archive-on-upgrade invariant, client reconfigure iterates detected
targets, MNEMON_FLY_ENDPOINT_OVERRIDE bypass for Layer 2 integration.

Layer 2 (MinIO + local serve-remote) and Layer 3 (real Fly + AWS) tests
are NOT exercised here — see ``private/e2e-test-runbook-260421.md``.
"""

from __future__ import annotations

from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from mnemon import upgrade
from mnemon.upgrade import (
    UpgradeError,
    _archive_local_vault,
    _validate_app_name,
    upgrade_web,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _ok_completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Isolate every test from real $HOME / $MNEMON_* so accidents can't
    write to the user's real vault or configs."""
    monkeypatch.setattr("mnemon.upgrade.Path.home", lambda: tmp_path)
    monkeypatch.setenv("MNEMON_VAULT_DIR", str(tmp_path / ".mnemon"))
    monkeypatch.delenv("MNEMON_FLY_ENDPOINT_OVERRIDE", raising=False)
    monkeypatch.delenv("MNEMON_S3_ENDPOINT_OVERRIDE", raising=False)
    monkeypatch.delenv("MNEMON_CLIENT_CONFIG_ROOT", raising=False)
    monkeypatch.delenv("MNEMON_PROD_APP_NAMES", raising=False)
    monkeypatch.delenv("MNEMON_REMOTE_URL", raising=False)
    monkeypatch.delenv("MNEMON_LOCAL_TOKEN", raising=False)
    monkeypatch.setenv("MNEMON_S3_BUCKET", "test-bucket")
    # Provide AWS creds so _fly_set_secrets doesn't error out.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRETTEST")
    yield


# ── _validate_app_name ───────────────────────────────────────────────────────


class TestValidateAppName:
    def test_accepts_valid_dns_label(self):
        _validate_app_name("my-mnemon-test")
        _validate_app_name("mnemon1")
        _validate_app_name("a")

    def test_rejects_empty(self):
        with pytest.raises(UpgradeError, match="Invalid Fly app name"):
            _validate_app_name("")

    def test_rejects_uppercase(self):
        with pytest.raises(UpgradeError, match="Invalid Fly app name"):
            _validate_app_name("Mnemon-Prod")

    def test_rejects_leading_hyphen(self):
        with pytest.raises(UpgradeError, match="Invalid Fly app name"):
            _validate_app_name("-mnemon")

    def test_rejects_64_chars(self):
        with pytest.raises(UpgradeError, match="Invalid Fly app name"):
            _validate_app_name("a" * 64)

    def test_prod_app_name_collision_blocked(self, monkeypatch):
        monkeypatch.setenv("MNEMON_PROD_APP_NAMES", "mnemon-memory, other-prod")
        with pytest.raises(UpgradeError, match="MNEMON_PROD_APP_NAMES"):
            _validate_app_name("mnemon-memory")

    def test_prod_guard_allows_non_matching(self, monkeypatch):
        monkeypatch.setenv("MNEMON_PROD_APP_NAMES", "mnemon-memory")
        _validate_app_name("mnemon-test-9999")  # not in list → ok


# ── prereq helpers ───────────────────────────────────────────────────────────


class TestPrereqValidation:
    def test_require_flyctl_missing_raises(self):
        with patch(
            "mnemon.upgrade.subprocess.run", side_effect=FileNotFoundError
        ):
            with pytest.raises(UpgradeError, match="flyctl not found"):
                upgrade._require_flyctl()

    def test_require_flyctl_unauthenticated_raises(self):
        err = CalledProcessError(
            1, ["flyctl", "auth", "whoami"], stderr="not signed in"
        )
        with patch("mnemon.upgrade.subprocess.run", side_effect=err):
            with pytest.raises(UpgradeError, match="not authenticated"):
                upgrade._require_flyctl()

    def test_require_flyctl_skipped_when_override_set(self, monkeypatch):
        monkeypatch.setenv(
            "MNEMON_FLY_ENDPOINT_OVERRIDE", "http://localhost:8502/mcp"
        )
        # Should not invoke subprocess at all
        with patch("mnemon.upgrade.subprocess.run") as mock_run:
            upgrade._require_flyctl()
        mock_run.assert_not_called()

    def test_require_aws_missing_raises(self):
        with patch(
            "mnemon.upgrade.subprocess.run", side_effect=FileNotFoundError
        ):
            with pytest.raises(UpgradeError, match="aws CLI not found"):
                upgrade._require_aws()

    def test_require_aws_bad_creds_raises(self):
        err = CalledProcessError(
            255, ["aws"], stderr="The security token included in the request is invalid"
        )
        with patch("mnemon.upgrade.subprocess.run", side_effect=err):
            with pytest.raises(UpgradeError, match="AWS credentials"):
                upgrade._require_aws()

    def test_require_bucket_from_env(self, monkeypatch):
        monkeypatch.setenv("MNEMON_S3_BUCKET", "from-env")
        assert upgrade._require_bucket(None) == "from-env"

    def test_require_bucket_arg_wins(self, monkeypatch):
        monkeypatch.setenv("MNEMON_S3_BUCKET", "from-env")
        assert upgrade._require_bucket("explicit") == "explicit"

    def test_require_bucket_missing_raises(self, monkeypatch):
        monkeypatch.delenv("MNEMON_S3_BUCKET", raising=False)
        with pytest.raises(UpgradeError, match="S3 bucket not specified"):
            upgrade._require_bucket(None)


# ── _archive_local_vault ─────────────────────────────────────────────────────


class TestArchiveLocalVault:
    def test_returns_none_when_no_local_vault(self):
        assert _archive_local_vault() is None

    def test_renames_into_archive_dir(self, tmp_path, monkeypatch):
        vdir = tmp_path / ".mnemon"
        vdir.mkdir()
        sqlite = vdir / "default.sqlite"
        sqlite.write_bytes(b"stub sqlite bytes")
        vec = vdir / "default.vec.npz"
        vec.write_bytes(b"stub vec bytes")

        monkeypatch.setattr("mnemon.config.vault_dir", lambda: vdir)

        archived = _archive_local_vault()
        assert archived is not None
        assert archived.name.startswith("pre-web-")
        assert archived.suffix == ".sqlite"
        assert archived.parent == vdir / "archive"
        # Source files were renamed, not copied
        assert not sqlite.exists()
        assert not vec.exists()
        # Vector companion rode along
        vec_archive = archived.parent / archived.with_suffix(".vec.npz").name
        assert vec_archive.exists()

    def test_collision_gets_suffix(self, tmp_path, monkeypatch):
        vdir = tmp_path / ".mnemon"
        vdir.mkdir()

        monkeypatch.setattr("mnemon.config.vault_dir", lambda: vdir)

        # First archive (creates pre-web-<today>.sqlite)
        (vdir / "default.sqlite").write_bytes(b"run 1")
        first = _archive_local_vault()
        # Second archive on the same day must not clobber the first
        (vdir / "default.sqlite").write_bytes(b"run 2")
        second = _archive_local_vault()

        assert first is not None and second is not None
        assert first != second
        assert first.read_bytes() == b"run 1"
        assert second.read_bytes() == b"run 2"


# ── upgrade_web orchestration ────────────────────────────────────────────────


class TestUpgradeWebHappyPath:
    """With all subprocess calls mocked, verify the orchestration runs
    the expected sequence of operations."""

    def _seed_local_vault(self, tmp_path):
        vdir = tmp_path / ".mnemon"
        vdir.mkdir()
        (vdir / "default.sqlite").write_bytes(b"seeded")
        return vdir

    def test_orchestration_sequence(self, tmp_path, monkeypatch):
        vdir = self._seed_local_vault(tmp_path)
        monkeypatch.setattr("mnemon.config.vault_dir", lambda: vdir)

        # Fake sync push reports success
        with patch(
            "mnemon.sync.push",
            return_value={"pushed": ["sqlite"], "errors": []},
        ) as mock_push, \
            patch(
                "mnemon.upgrade.subprocess.run",
                return_value=_ok_completed(),
            ) as mock_run, \
            patch("mnemon.upgrade._require_flyctl"), \
            patch("mnemon.upgrade._require_aws"), \
            patch(
                "mnemon.upgrade._reconfigure_clients",
                return_value=["claude-code"],
            ) as mock_reconfig, \
            patch("mnemon.doctor.run_doctor", return_value=0):
            result = upgrade_web(
                app_name="mnemon-test-abc",
                s3_bucket="test-bucket",
                token="fixed-test-token",
                region="sjc",
                mnemon_version="0.5.0",
                skip_doctor=True,
            )

        mock_push.assert_called_once()
        mock_reconfig.assert_called_once()
        args, _kwargs = mock_reconfig.call_args
        remote_url, token_arg, _detected = args
        assert remote_url == "https://mnemon-test-abc.fly.dev/mcp"
        assert token_arg == "fixed-test-token"

        # flyctl was invoked for: launch, volumes create, secrets set,
        # deploy, ssh console. Five subprocess.run calls through
        # mnemon.upgrade.subprocess.run.
        commands = [call.args[0][:2] for call in mock_run.call_args_list]
        assert ["flyctl", "launch"] in commands
        assert ["flyctl", "volumes"] in commands
        assert ["flyctl", "secrets"] in commands
        assert ["flyctl", "deploy"] in commands
        assert ["flyctl", "ssh"] in commands

        # Local vault was archived
        assert not (vdir / "default.sqlite").exists()
        assert (vdir / "archive").exists()
        assert "Upgrade to web complete" in result
        assert "https://mnemon-test-abc.fly.dev/mcp" in result

    def test_s3_push_failure_aborts_before_flyctl(self, tmp_path, monkeypatch):
        vdir = self._seed_local_vault(tmp_path)
        monkeypatch.setattr("mnemon.config.vault_dir", lambda: vdir)

        with patch(
            "mnemon.sync.push",
            return_value={"pushed": [], "errors": ["access denied"]},
        ), \
            patch("mnemon.upgrade._require_flyctl"), \
            patch("mnemon.upgrade._require_aws"), \
            patch("mnemon.upgrade.subprocess.run") as mock_run:
            with pytest.raises(UpgradeError, match="S3 push failed"):
                upgrade_web(
                    app_name="mnemon-test",
                    s3_bucket="test-bucket",
                    skip_doctor=True,
                )

        # flyctl was never touched — orchestration aborted at step 2
        mock_run.assert_not_called()
        # Local vault still in place (archive step is after flyctl)
        assert (vdir / "default.sqlite").exists()

    def test_fly_endpoint_override_skips_flyctl_calls(
        self, tmp_path, monkeypatch
    ):
        """Layer 2 integration: a local serve-remote can stand in for
        Fly. With the override set, no flyctl commands fire."""
        vdir = self._seed_local_vault(tmp_path)
        monkeypatch.setattr("mnemon.config.vault_dir", lambda: vdir)
        monkeypatch.setenv(
            "MNEMON_FLY_ENDPOINT_OVERRIDE", "http://localhost:8502/mcp"
        )

        with patch(
            "mnemon.sync.push",
            return_value={"pushed": [], "errors": []},
        ), \
            patch("mnemon.upgrade._require_aws"), \
            patch(
                "mnemon.upgrade._reconfigure_clients",
                return_value=[],
            ) as mock_reconfig, \
            patch("mnemon.upgrade.subprocess.run") as mock_run:
            result = upgrade_web(
                app_name="mnemon-test",
                s3_bucket="test-bucket",
                skip_doctor=True,
            )

        mock_run.assert_not_called()
        args, _ = mock_reconfig.call_args
        assert args[0] == "http://localhost:8502/mcp"
        assert "http://localhost:8502/mcp" in result


class TestRequireAwsSecrets:
    def test_errors_when_aws_creds_cannot_be_resolved(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        # aws configure get returns empty
        with patch(
            "mnemon.upgrade.subprocess.run",
            return_value=_ok_completed(stdout=""),
        ):
            with pytest.raises(UpgradeError, match="AWS credentials not resolvable"):
                upgrade._fly_set_secrets(
                    "mnemon-test", "some-token", "bucket"
                )
