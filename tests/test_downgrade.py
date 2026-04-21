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
    _extract_app_name,
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
        ):
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
            result = downgrade_local(
                destroy_fly_app=True, skip_doctor=True
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
