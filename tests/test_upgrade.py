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
    _fly_app_exists,
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
    # Disable the post-deploy settle wait so Layer 1 tests don't pay
    # 30s of real wall-clock time on every doctor-invoking path.
    # The settle test class re-enables it explicitly.
    monkeypatch.setenv("MNEMON_UPGRADE_SETTLE_SECONDS", "0")
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
            patch("mnemon.upgrade._fly_app_exists", return_value=False), \
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
            patch("mnemon.upgrade._fly_app_exists", return_value=False), \
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


class TestFlySecretsForwardS3PrefixAndVaultName:
    """Regression for the 2026-05-21 Layer-3 bug: `_fly_set_secrets`
    forwarded `MNEMON_S3_BUCKET` but not `MNEMON_S3_PREFIX` / `MNEMON_VAULT_NAME`,
    so the Fly container's `mnemon sync pull` fell back to defaults
    and seeded from the wrong S3 prefix (operator's prod prefix instead
    of the test-scoped override the operator set locally).
    """

    def _capture_secrets_args(self, monkeypatch, env_overrides=None):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-test")
        for k, v in (env_overrides or {}).items():
            monkeypatch.setenv(k, v)

        with patch("mnemon.upgrade.subprocess.run") as mock_run:
            mock_run.return_value = _ok_completed()
            upgrade._fly_set_secrets(
                "mnemon-test", "some-token", "mnemon-memory"
            )
        # The subprocess.run call carries the secrets-set CLI args.
        call_args = mock_run.call_args.args[0]
        # Pull just the KEY=VALUE pairs from the CLI args.
        return [a for a in call_args if "=" in a]

    def test_forwards_s3_prefix_override_to_fly_secrets(self, monkeypatch):
        kvs = self._capture_secrets_args(
            monkeypatch,
            {"MNEMON_S3_PREFIX": "test-upgrade/abc123"},
        )
        assert "MNEMON_S3_PREFIX=test-upgrade/abc123" in kvs

    def test_forwards_vault_name_override_to_fly_secrets(self, monkeypatch):
        kvs = self._capture_secrets_args(
            monkeypatch,
            {"MNEMON_VAULT_NAME": "custom-vault"},
        )
        assert "MNEMON_VAULT_NAME=custom-vault" in kvs

    def test_defaults_match_sync_module_constants_when_env_unset(self, monkeypatch):
        # Unset both env vars + verify Fly gets the same defaults sync.py uses.
        # Preserves prod-redeploy behavior (no override → default both sides).
        monkeypatch.delenv("MNEMON_S3_PREFIX", raising=False)
        monkeypatch.delenv("MNEMON_VAULT_NAME", raising=False)
        from mnemon.sync import S3_PREFIX_DEFAULT, VAULT_NAME_DEFAULT
        kvs = self._capture_secrets_args(monkeypatch)
        assert f"MNEMON_S3_PREFIX={S3_PREFIX_DEFAULT}" in kvs
        assert f"MNEMON_VAULT_NAME={VAULT_NAME_DEFAULT}" in kvs

    def test_still_forwards_token_and_bucket(self, monkeypatch):
        # Belt + suspenders: this fix didn't regress the existing
        # forwarded secrets.
        kvs = self._capture_secrets_args(monkeypatch)
        assert "MNEMON_LOCAL_TOKEN=some-token" in kvs
        assert "MNEMON_S3_BUCKET=mnemon-memory" in kvs
        assert any(kv.startswith("AWS_ACCESS_KEY_ID=") for kv in kvs)
        assert any(kv.startswith("AWS_SECRET_ACCESS_KEY=") for kv in kvs)


# ── _fly_app_exists detection ────────────────────────────────────────────────


class TestFlyAppExists:
    def test_returns_true_when_status_exits_zero(self):
        with patch(
            "mnemon.upgrade.subprocess.run",
            return_value=_ok_completed(),
        ) as mock_run:
            assert _fly_app_exists("mnemon-test") is True
        args, _kwargs = mock_run.call_args
        assert args[0] == ["flyctl", "status", "--app", "mnemon-test"]

    def test_returns_false_when_flyctl_missing(self):
        with patch(
            "mnemon.upgrade.subprocess.run", side_effect=FileNotFoundError
        ):
            assert _fly_app_exists("mnemon-test") is False

    def test_returns_false_when_status_exits_nonzero(self):
        err = CalledProcessError(
            1, ["flyctl", "status"], stderr="Could not find App"
        )
        with patch("mnemon.upgrade.subprocess.run", side_effect=err):
            assert _fly_app_exists("mnemon-test") is False


# ── upgrade_web redeploy path (idempotent version bump) ──────────────────────


class TestUpgradeWebRedeploy:
    """When the Fly app already exists, ``upgrade_web`` takes a
    redeploy-only path: skip S3 push, volume create, secrets set, seed,
    archive, and client reconfigure. Just rebuild + deploy with the new
    mnemon version pinned."""

    def test_redeploy_skips_every_first_time_step(self, tmp_path, monkeypatch):
        """Core contract of the idempotent path: when flyctl says the
        app exists, none of the first-time-only helpers fire. Only
        ``flyctl deploy`` runs."""
        monkeypatch.setattr(
            "mnemon.config.vault_dir", lambda: tmp_path / ".mnemon"
        )

        with patch("mnemon.upgrade._require_flyctl"), \
            patch("mnemon.upgrade._fly_app_exists", return_value=True), \
            patch("mnemon.upgrade._require_aws") as mock_req_aws, \
            patch("mnemon.upgrade._require_bucket") as mock_req_bucket, \
            patch("mnemon.sync.push") as mock_push, \
            patch("mnemon.upgrade._fly_launch") as mock_launch, \
            patch("mnemon.upgrade._fly_create_volume") as mock_vol, \
            patch("mnemon.upgrade._fly_set_secrets") as mock_secrets, \
            patch("mnemon.upgrade._fly_seed_vault") as mock_seed, \
            patch("mnemon.upgrade._archive_local_vault") as mock_archive, \
            patch("mnemon.upgrade._reconfigure_clients") as mock_reconfig, \
            patch("mnemon.upgrade._fly_deploy") as mock_deploy:
            result = upgrade_web(
                app_name="mnemon-test-existing",
                mnemon_version="0.6.0rc3",
                skip_doctor=True,
            )

        # First-time-only helpers never called
        mock_req_aws.assert_not_called()
        mock_req_bucket.assert_not_called()
        mock_push.assert_not_called()
        mock_launch.assert_not_called()
        mock_vol.assert_not_called()
        mock_secrets.assert_not_called()
        mock_seed.assert_not_called()
        mock_archive.assert_not_called()
        mock_reconfig.assert_not_called()

        # Only flyctl deploy fired
        mock_deploy.assert_called_once()
        _workdir, app_arg = mock_deploy.call_args.args
        assert app_arg == "mnemon-test-existing"

        assert "Redeploy complete" in result
        assert "https://mnemon-test-existing.fly.dev/mcp" in result
        assert "0.6.0rc3" in result

    def test_redeploy_pins_provided_version_in_dockerfile(
        self, tmp_path, monkeypatch
    ):
        """The Dockerfile written to the tempdir pins the exact mnemon
        version the caller passed — this is the version bump knob."""
        monkeypatch.setattr(
            "mnemon.config.vault_dir", lambda: tmp_path / ".mnemon"
        )
        captured: dict[str, str] = {}

        def _capture(workdir: Path, app_name: str) -> None:
            captured["dockerfile"] = (workdir / "Dockerfile").read_text()
            captured["fly_toml"] = (workdir / "fly.toml").read_text()

        with patch("mnemon.upgrade._require_flyctl"), \
            patch("mnemon.upgrade._fly_app_exists", return_value=True), \
            patch("mnemon.upgrade._fly_deploy", side_effect=_capture):
            upgrade_web(
                app_name="mnemon-test-existing",
                mnemon_version="0.6.0rc3",
                region="iad",
                skip_doctor=True,
            )

        assert "mnemon-memory[server]==0.6.0rc3" in captured["dockerfile"]
        assert 'app = "mnemon-test-existing"' in captured["fly_toml"]
        assert 'primary_region = "iad"' in captured["fly_toml"]

    def test_redeploy_ignores_missing_aws_and_bucket(
        self, tmp_path, monkeypatch
    ):
        """Users bumping the version shouldn't need AWS creds or an S3
        bucket — those are only required for the first-time vault seed.
        A redeploy with neither set must succeed."""
        monkeypatch.setattr(
            "mnemon.config.vault_dir", lambda: tmp_path / ".mnemon"
        )
        monkeypatch.delenv("MNEMON_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        with patch("mnemon.upgrade._require_flyctl"), \
            patch("mnemon.upgrade._fly_app_exists", return_value=True), \
            patch("mnemon.upgrade._fly_deploy"):
            result = upgrade_web(
                app_name="mnemon-test-existing",
                mnemon_version="0.6.0rc3",
                skip_doctor=True,
            )

        assert "Redeploy complete" in result

    def test_redeploy_runs_doctor_when_not_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mnemon.config.vault_dir", lambda: tmp_path / ".mnemon"
        )

        with patch("mnemon.upgrade._require_flyctl"), \
            patch("mnemon.upgrade._fly_app_exists", return_value=True), \
            patch("mnemon.upgrade._fly_deploy"), \
            patch("mnemon.doctor.run_doctor", return_value=0) as mock_doctor:
            result = upgrade_web(
                app_name="mnemon-test-existing",
                mnemon_version="0.6.0rc3",
                skip_doctor=False,
            )

        mock_doctor.assert_called_once()
        _args, kwargs = mock_doctor.call_args
        assert kwargs.get("fail_on_warn") is True
        assert "Redeploy complete" in result

    def test_fly_endpoint_override_takes_first_time_path(
        self, tmp_path, monkeypatch
    ):
        """Integration-test override bypasses the existence check so
        Layer 2 harnesses still exercise the full orchestration. Even
        if an existing app is "visible", the override means we skip
        all flyctl and treat the endpoint as the remote."""
        vdir = tmp_path / ".mnemon"
        vdir.mkdir()
        (vdir / "default.sqlite").write_bytes(b"seeded")
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
                "mnemon.upgrade._fly_app_exists", return_value=True
            ) as mock_exists, \
            patch(
                "mnemon.upgrade._reconfigure_clients",
                return_value=[],
            ), \
            patch("mnemon.upgrade._fly_deploy") as mock_deploy, \
            patch("mnemon.upgrade.subprocess.run"):
            result = upgrade_web(
                app_name="mnemon-test",
                s3_bucket="test-bucket",
                skip_doctor=True,
            )

        # Override path doesn't even ask — redeploy branch never reached
        mock_exists.assert_not_called()
        mock_deploy.assert_not_called()
        assert "http://localhost:8502/mcp" in result


class TestPostDeploySettleWindow:
    """The post-deploy settle wait gives a freshly-redeployed Fly machine
    a quiet window before doctor's 7 rapid-fire probes hit it. This is
    the rc11-deploy fix from 2026-05-06 — without it, doctor wedges
    against an in-progress FastEmbed pre-load + cold session manager.
    """

    def test_settle_called_between_deploy_and_doctor_on_redeploy(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "mnemon.config.vault_dir", lambda: tmp_path / ".mnemon"
        )

        order: list[str] = []

        def _record_deploy(workdir, app_name):  # noqa: ARG001
            order.append("deploy")

        def _record_settle():
            order.append("settle")

        def _record_doctor(*_args, **_kwargs):
            order.append("doctor")
            return 0

        with patch("mnemon.upgrade._require_flyctl"), \
            patch("mnemon.upgrade._fly_app_exists", return_value=True), \
            patch("mnemon.upgrade._fly_deploy", side_effect=_record_deploy), \
            patch(
                "mnemon.upgrade._settle_after_deploy",
                side_effect=_record_settle,
            ), \
            patch(
                "mnemon.doctor.run_doctor",
                side_effect=_record_doctor,
            ):
            upgrade_web(
                app_name="mnemon-test-existing",
                mnemon_version="0.6.0rc12",
                skip_doctor=False,
            )

        assert order == ["deploy", "settle", "doctor"]

    def test_settle_skipped_when_skip_doctor_true(self, tmp_path, monkeypatch):
        """If the user opted out of doctor, there's no probe to settle
        before — don't make them wait 30s for nothing."""
        monkeypatch.setattr(
            "mnemon.config.vault_dir", lambda: tmp_path / ".mnemon"
        )

        with patch("mnemon.upgrade._require_flyctl"), \
            patch("mnemon.upgrade._fly_app_exists", return_value=True), \
            patch("mnemon.upgrade._fly_deploy"), \
            patch("mnemon.upgrade._settle_after_deploy") as mock_settle:
            upgrade_web(
                app_name="mnemon-test-existing",
                mnemon_version="0.6.0rc12",
                skip_doctor=True,
            )

        mock_settle.assert_not_called()

    def test_settle_honors_env_override_to_zero(self, monkeypatch):
        """Tests + scripted upgrades disable the wait via env var.
        ``0`` returns immediately without sleeping."""
        import time as _time

        monkeypatch.setenv("MNEMON_UPGRADE_SETTLE_SECONDS", "0")
        with patch.object(_time, "sleep") as mock_sleep:
            upgrade._settle_after_deploy()
        mock_sleep.assert_not_called()

    def test_settle_honors_env_override_to_custom_seconds(self, monkeypatch):
        """Non-zero env override is forwarded to ``time.sleep``."""
        import time as _time

        monkeypatch.setenv("MNEMON_UPGRADE_SETTLE_SECONDS", "3")
        with patch.object(_time, "sleep") as mock_sleep:
            upgrade._settle_after_deploy()
        mock_sleep.assert_called_once_with(3)

    def test_settle_falls_back_to_default_on_garbage_env(self, monkeypatch):
        """Malformed env value falls back to the default — never
        crashes the upgrade flow over a typo."""
        import time as _time

        monkeypatch.setenv("MNEMON_UPGRADE_SETTLE_SECONDS", "not-a-number")
        with patch.object(_time, "sleep") as mock_sleep:
            upgrade._settle_after_deploy()
        mock_sleep.assert_called_once()
        (called_with,), _ = mock_sleep.call_args
        assert called_with == upgrade._DEFAULT_DEPLOY_SETTLE_SECONDS

    def test_settle_called_on_first_time_upgrade_path_too(
        self, tmp_path, monkeypatch
    ):
        """The post-deploy doctor probe in the first-time upgrade path
        has the same fragility window — settle there too."""
        vdir = tmp_path / ".mnemon"
        vdir.mkdir()
        (vdir / "default.sqlite").write_bytes(b"seeded")
        monkeypatch.setattr("mnemon.config.vault_dir", lambda: vdir)
        monkeypatch.setenv(
            "MNEMON_FLY_ENDPOINT_OVERRIDE", "http://localhost:8502/mcp"
        )

        with patch(
            "mnemon.sync.push",
            return_value={"pushed": [], "errors": []},
        ), \
            patch("mnemon.upgrade._require_aws"), \
            patch("mnemon.upgrade._fly_app_exists", return_value=False), \
            patch(
                "mnemon.upgrade._reconfigure_clients",
                return_value=[],
            ), \
            patch(
                "mnemon.upgrade._settle_after_deploy"
            ) as mock_settle, \
            patch(
                "mnemon.doctor.run_doctor", return_value=0
            ):
            upgrade_web(
                app_name="mnemon-test",
                s3_bucket="test-bucket",
                skip_doctor=False,
            )

        mock_settle.assert_called_once()
