"""Tests for CLI command dispatcher."""

import sys
from unittest.mock import patch, MagicMock

import pytest

from mnemon import __version__
from mnemon.cli import main, _print_usage


class TestVersionAndHelp:
    def test_version_flag(self, capsys):
        with patch("sys.argv", ["mnemon", "--version"]):
            main()
        out = capsys.readouterr().out
        assert f"mnemon v{__version__}" in out

    def test_version_short_flag(self, capsys):
        with patch("sys.argv", ["mnemon", "-v"]):
            main()
        out = capsys.readouterr().out
        assert f"mnemon v{__version__}" in out

    def test_help_flag(self, capsys):
        with patch("sys.argv", ["mnemon", "--help"]):
            main()
        out = capsys.readouterr().out
        assert "mnemon serve" in out
        assert "mnemon setup" in out

    def test_help_short_flag(self, capsys):
        with patch("sys.argv", ["mnemon", "-h"]):
            main()
        out = capsys.readouterr().out
        assert "mnemon" in out

    def test_no_args_prints_usage(self, capsys):
        with patch("sys.argv", ["mnemon"]):
            main()
        out = capsys.readouterr().out
        assert "mnemon" in out

    def test_print_usage_contains_all_commands(self, capsys):
        _print_usage()
        out = capsys.readouterr().out
        for cmd in ["serve", "serve-remote", "status", "search", "save",
                     "forget", "setup", "sync push", "sync pull"]:
            assert cmd in out

    def test_print_usage_contains_env_vars(self, capsys):
        _print_usage()
        out = capsys.readouterr().out
        assert "MNEMON_REMOTE_URL" in out
        assert "MNEMON_LOCAL_TOKEN" in out
        assert "MNEMON_VAULT_DIR" in out
        assert "MNEMON_S3_BUCKET" in out


class TestServe:
    @patch("mnemon.server.run_stdio")
    def test_serve_calls_run_stdio(self, mock_run):
        with patch("sys.argv", ["mnemon", "serve"]):
            main()
        mock_run.assert_called_once()

    @patch("mnemon.server_remote.run_remote")
    def test_serve_remote_calls_run_remote(self, mock_run):
        with patch("sys.argv", ["mnemon", "serve-remote"]):
            main()
        mock_run.assert_called_once()


class TestStatus:
    @patch("mnemon.store.Store")
    def test_status_prints_vault_stats(self, MockStore, capsys):
        mock_store = MagicMock()
        mock_store.status.return_value = {
            "vault_path": "/home/user/.mnemon/default.sqlite",
            "total_documents": 42,
            "total_vectors": 40,
            "pinned": 3,
            "invalidated": 1,
            "by_type": [
                {"content_type": "note", "count": 30},
                {"content_type": "decision", "count": 12},
            ],
        }
        MockStore.return_value = mock_store

        with patch("sys.argv", ["mnemon", "status"]):
            main()

        out = capsys.readouterr().out
        assert "Vault: /home/user/.mnemon/default.sqlite" in out
        assert "Total memories: 42" in out
        assert "Vectors: 40" in out
        assert "Pinned: 3" in out
        assert "Invalidated: 1" in out
        assert "note: 30" in out
        assert "decision: 12" in out
        mock_store.close.assert_called_once()


class TestSearch:
    @patch("mnemon.search.search")
    @patch("mnemon.store.Store")
    def test_search_with_results(self, MockStore, mock_search, capsys):
        mock_store = MagicMock()
        MockStore.return_value = mock_store

        result = MagicMock()
        result.content = "Some memory content here"
        result.content_type = "note"
        result.title = "My Note"
        result.composite_score = 0.875
        mock_search.return_value = [result]

        with patch("sys.argv", ["mnemon", "search", "test", "query"]):
            main()

        out = capsys.readouterr().out
        assert "[note] My Note (score: 0.875)" in out
        assert "Some memory content here" in out
        mock_search.assert_called_once_with(mock_store, "test query", limit=10)
        mock_store.close.assert_called_once()

    @patch("mnemon.search.search")
    @patch("mnemon.store.Store")
    def test_search_no_results(self, MockStore, mock_search, capsys):
        MockStore.return_value = MagicMock()
        mock_search.return_value = []

        with patch("sys.argv", ["mnemon", "search", "nothing"]):
            main()

        out = capsys.readouterr().out
        assert "No memories found." in out

    def test_search_without_query_exits(self, capsys):
        with patch("sys.argv", ["mnemon", "search"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Usage: mnemon search <query>" in err

    @patch("mnemon.search.search")
    @patch("mnemon.store.Store")
    def test_search_long_content_truncated(self, MockStore, mock_search, capsys):
        mock_store = MagicMock()
        MockStore.return_value = mock_store

        result = MagicMock()
        result.content = "x" * 300
        result.content_type = "note"
        result.title = "Long"
        result.composite_score = 0.5
        mock_search.return_value = [result]

        with patch("sys.argv", ["mnemon", "search", "long"]):
            main()

        out = capsys.readouterr().out
        assert "..." in out
        # Content should be truncated to 200 chars
        assert "x" * 200 in out


class TestSave:
    @patch("mnemon.store.Store")
    def test_save_with_title_and_content(self, MockStore, capsys):
        mock_store = MagicMock()
        mock_store.save.return_value = 7
        # embed_document is called post-save; returning None from get()
        # makes the `if doc:` branch skip embedding so this test doesn't
        # depend on a FastEmbed ONNX model being in the cache.
        mock_store.get.return_value = None
        MockStore.return_value = mock_store

        with patch("sys.argv", ["mnemon", "save", "My Title", "some", "content"]):
            main()

        out = capsys.readouterr().out
        assert 'Saved memory #7: "My Title"' in out
        mock_store.save.assert_called_once_with(
            title="My Title", content="some content", source_client="cli"
        )
        mock_store.close.assert_called_once()

    def test_save_without_enough_args_exits(self, capsys):
        with patch("sys.argv", ["mnemon", "save", "titleonly"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Usage: mnemon save <title> <content>" in err

    def test_save_no_args_exits(self, capsys):
        with patch("sys.argv", ["mnemon", "save"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1


class TestForget:
    @patch("mnemon.store.Store")
    def test_forget_found(self, MockStore, capsys):
        mock_store = MagicMock()
        mock_store.forget.return_value = True
        MockStore.return_value = mock_store

        with patch("sys.argv", ["mnemon", "forget", "42"]):
            main()

        out = capsys.readouterr().out
        assert "Forgot memory #42." in out
        mock_store.forget.assert_called_once_with(42)
        mock_store.close.assert_called_once()

    @patch("mnemon.store.Store")
    def test_forget_not_found_exits(self, MockStore, capsys):
        mock_store = MagicMock()
        mock_store.forget.return_value = False
        MockStore.return_value = mock_store

        with patch("sys.argv", ["mnemon", "forget", "999"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Memory #999 not found or already forgotten." in err

    def test_forget_without_id_exits(self, capsys):
        with patch("sys.argv", ["mnemon", "forget"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Usage: mnemon forget <id>" in err

    def test_forget_non_numeric_id_exits(self, capsys):
        with patch("sys.argv", ["mnemon", "forget", "abc"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Usage: mnemon forget <id>" in err


class TestSync:
    @patch("mnemon.sync.push")
    def test_sync_push_calls_push(self, mock_push, capsys):
        mock_push.return_value = {"pushed": ["vault.sqlite"], "errors": []}

        with patch("sys.argv", ["mnemon", "sync", "push"]):
            main()

        mock_push.assert_called_once()
        out = capsys.readouterr().out
        assert "Pushed:" in out
        assert "vault.sqlite" in out

    @patch("mnemon.sync.push")
    def test_sync_push_shows_errors(self, mock_push, capsys):
        mock_push.return_value = {
            "pushed": [],
            "errors": ["S3 bucket not found"],
        }

        with patch("sys.argv", ["mnemon", "sync", "push"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Errors:" in err
        assert "S3 bucket not found" in err

    @patch("mnemon.sync.push")
    def test_sync_push_nothing_to_push(self, mock_push, capsys):
        mock_push.return_value = {"pushed": [], "errors": []}

        with patch("sys.argv", ["mnemon", "sync", "push"]):
            main()

        out = capsys.readouterr().out
        assert "No vault files found to push." in out

    @patch("mnemon.sync.pull")
    def test_sync_pull_calls_pull(self, mock_pull, capsys):
        mock_pull.return_value = {"pulled": ["vault.sqlite"], "errors": []}

        with patch("sys.argv", ["mnemon", "sync", "pull"]):
            main()

        mock_pull.assert_called_once()
        out = capsys.readouterr().out
        assert "Pulled:" in out
        assert "vault.sqlite" in out

    @patch("mnemon.sync.pull")
    def test_sync_pull_shows_errors(self, mock_pull, capsys):
        mock_pull.return_value = {
            "pulled": [],
            "errors": ["Access denied"],
        }

        with patch("sys.argv", ["mnemon", "sync", "pull"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Access denied" in err

    @patch("mnemon.sync.pull")
    def test_sync_pull_nothing_on_s3(self, mock_pull, capsys):
        mock_pull.return_value = {"pulled": [], "errors": []}

        with patch("sys.argv", ["mnemon", "sync", "pull"]):
            main()

        out = capsys.readouterr().out
        assert "No vault files found on S3." in out

    def test_sync_without_subcommand_exits(self, capsys):
        with patch("sys.argv", ["mnemon", "sync"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        combined = capsys.readouterr()
        assert "Usage: mnemon sync <push|pull>" in combined.err
        # Env var help lines go to stdout (no file=sys.stderr)
        assert "MNEMON_S3_BUCKET" in combined.out

    def test_sync_invalid_subcommand_exits(self, capsys):
        with patch("sys.argv", ["mnemon", "sync", "bogus"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1


class TestSetup:
    @patch("mnemon.setup.run_setup")
    def test_setup_calls_run_setup(self, mock_run, capsys):
        mock_run.return_value = "Configured claude-code successfully."

        with patch("sys.argv", ["mnemon", "setup", "claude-code"]):
            main()

        mock_run.assert_called_once_with("claude-code", [])
        out = capsys.readouterr().out
        assert "Configured claude-code successfully." in out

    @patch("mnemon.setup.run_setup")
    def test_setup_without_target_invokes_autodetect(
        self, mock_run, capsys
    ):
        """P1b: `mnemon setup` with no target auto-detects clients
        instead of printing usage and exiting. CLI passes ``None`` for
        target when no positional arg is given (only flags, or nothing)."""
        mock_run.return_value = "auto-detect output"
        with patch("sys.argv", ["mnemon", "setup"]):
            main()
        mock_run.assert_called_once_with(None, [])
        assert "auto-detect output" in capsys.readouterr().out

    @patch("mnemon.setup.run_setup")
    def test_setup_with_only_flags_is_autodetect(self, mock_run, capsys):
        """Flags like --remote-url without a target should still be
        treated as auto-detect, not as a target name."""
        mock_run.return_value = "ok"
        with patch(
            "sys.argv",
            ["mnemon", "setup", "--remote-url", "https://x/mcp", "--skip-doctor"],
        ):
            main()
        mock_run.assert_called_once_with(
            None, ["--remote-url", "https://x/mcp", "--skip-doctor"]
        )


class TestUpgradeCli:
    @patch("mnemon.upgrade.upgrade_web")
    def test_happy_path_passes_parsed_flags(self, mock_upgrade, capsys):
        mock_upgrade.return_value = "upgrade output"
        with patch(
            "sys.argv",
            [
                "mnemon",
                "upgrade",
                "web",
                "--app-name",
                "mnemon-test-cli",
                "--s3-bucket",
                "my-bucket",
                "--region",
                "sjc",
                "--skip-doctor",
            ],
        ):
            main()
        mock_upgrade.assert_called_once_with(
            app_name="mnemon-test-cli",
            s3_bucket="my-bucket",
            token=None,
            region="sjc",
            mnemon_version=None,
            skip_doctor=True,
        )
        assert "upgrade output" in capsys.readouterr().out

    @patch("mnemon.upgrade.upgrade_web")
    def test_mnemon_version_flag_passes_through(self, mock_upgrade, capsys):
        """--mnemon-version pins the version in the deployed Dockerfile,
        sidestepping the local-install-must-match-PyPI gotcha."""
        mock_upgrade.return_value = "ok"
        with patch(
            "sys.argv",
            [
                "mnemon",
                "upgrade",
                "web",
                "--app-name",
                "mnemon-test-cli",
                "--mnemon-version",
                "0.6.0rc5",
                "--skip-doctor",
            ],
        ):
            main()
        kwargs = mock_upgrade.call_args.kwargs
        assert kwargs["mnemon_version"] == "0.6.0rc5"
        assert kwargs["app_name"] == "mnemon-test-cli"

    def test_malformed_mnemon_version_exits_1(self, capsys):
        """Refuse anything outside [A-Za-z0-9._+-] — the string is
        interpolated into a Dockerfile shell context."""
        with patch(
            "sys.argv",
            [
                "mnemon",
                "upgrade",
                "web",
                "--app-name",
                "mnemon-test",
                "--mnemon-version",
                "0.6.0'; rm -rf /; '",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "upgrade failed" in err
        assert "--mnemon-version must match" in err

    def test_web_subcommand_missing_prints_usage(self, capsys):
        with patch("sys.argv", ["mnemon", "upgrade"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        assert "Usage: mnemon upgrade web" in capsys.readouterr().err

    def test_missing_app_name_exits(self, capsys):
        with patch(
            "sys.argv",
            ["mnemon", "upgrade", "web", "--s3-bucket", "b"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        assert "--app-name is required" in capsys.readouterr().err

    @patch("mnemon.upgrade.upgrade_web")
    def test_upgrade_error_surfaces_as_exit_1(self, mock_upgrade, capsys):
        from mnemon.upgrade import UpgradeError

        mock_upgrade.side_effect = UpgradeError("boom")
        with patch(
            "sys.argv",
            [
                "mnemon",
                "upgrade",
                "web",
                "--app-name",
                "mnemon-test",
                "--skip-doctor",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "upgrade failed: boom" in err


class TestDoctorCli:
    @patch("mnemon.doctor.run_doctor")
    def test_default_invocation_does_not_fail_on_warn(self, mock_doctor):
        mock_doctor.return_value = 0
        with patch("sys.argv", ["mnemon", "doctor"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0
        mock_doctor.assert_called_once_with(fail_on_warn=False)

    @patch("mnemon.doctor.run_doctor")
    def test_fail_on_warn_flag_propagates(self, mock_doctor):
        mock_doctor.return_value = 1
        with patch("sys.argv", ["mnemon", "doctor", "--fail-on-warn"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        mock_doctor.assert_called_once_with(fail_on_warn=True)


class TestUninstallCli:
    @patch("mnemon.uninstall.uninstall")
    def test_happy_path_passes_flags(self, mock_uninstall, capsys):
        mock_uninstall.return_value = "uninstall output"
        with patch(
            "sys.argv",
            ["mnemon", "uninstall", "--yes", "--keep-vault"],
        ):
            main()
        mock_uninstall.assert_called_once_with(yes=True, keep_vault=True)
        assert "uninstall output" in capsys.readouterr().out

    @patch("mnemon.uninstall.uninstall")
    def test_no_flags_defaults_are_false(self, mock_uninstall):
        mock_uninstall.return_value = ""
        with patch("sys.argv", ["mnemon", "uninstall"]):
            main()
        mock_uninstall.assert_called_once_with(yes=False, keep_vault=False)

    @patch("mnemon.uninstall.uninstall")
    def test_uninstall_error_surfaces_as_exit_1(self, mock_uninstall, capsys):
        from mnemon.uninstall import UninstallError

        mock_uninstall.side_effect = UninstallError("disk full")
        with patch("sys.argv", ["mnemon", "uninstall", "--yes"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        assert "uninstall failed: disk full" in capsys.readouterr().err


class TestDowngradeCli:
    @patch("mnemon.downgrade.downgrade_local")
    def test_happy_path_passes_parsed_flags(self, mock_downgrade, capsys):
        mock_downgrade.return_value = "downgrade output"
        with patch(
            "sys.argv",
            [
                "mnemon",
                "downgrade",
                "local",
                "--destroy-fly-app",
                "--yes",
                "--skip-doctor",
            ],
        ):
            main()
        mock_downgrade.assert_called_once_with(
            destroy_fly_app=True,
            yes=True,
            skip_doctor=True,
            app_name_override=None,
            skip_fly_push=False,
        )
        assert "downgrade output" in capsys.readouterr().out

    def test_local_subcommand_missing_prints_usage(self, capsys):
        with patch("sys.argv", ["mnemon", "downgrade"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        assert "Usage: mnemon downgrade local" in capsys.readouterr().err

    @patch("mnemon.downgrade.downgrade_local")
    def test_downgrade_error_surfaces_as_exit_1(self, mock_downgrade, capsys):
        from mnemon.downgrade import DowngradeError

        mock_downgrade.side_effect = DowngradeError("no remote")
        with patch(
            "sys.argv",
            ["mnemon", "downgrade", "local", "--skip-doctor"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        assert "downgrade failed: no remote" in capsys.readouterr().err


class TestUnknownCommand:
    def test_unknown_command_exits(self, capsys):
        with patch("sys.argv", ["mnemon", "foobar"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Unknown command: foobar" in err
        out = capsys.readouterr().out
        # Usage is printed to stdout by _print_usage

    def test_unknown_command_prints_usage(self, capsys):
        with patch("sys.argv", ["mnemon", "badcmd"]):
            with pytest.raises(SystemExit):
                main()
        # _print_usage prints to stdout
        combined = capsys.readouterr()
        assert "Unknown command: badcmd" in combined.err
        assert "mnemon setup" in combined.out


class TestAttentionStatusStrict:
    """Regression: --strict exits 1 when boost-rate > ceiling so the
    soak gate can drive periodic health checks. Without this, a soak
    regression like 2026-05-27 (boost-rate hit 0.714) requires manual
    eyeball during a "close the soak" prompt to surface."""

    def test_default_exits_zero_even_when_over_ceiling(self):
        with patch("mnemon.cli._print_attention_status") as mock_status:
            mock_status.return_value = (0.714, 0.25)
            with patch("mnemon.store.Store") as mock_store_cls:
                mock_store_cls.return_value = MagicMock()
                with patch("sys.argv", ["mnemon", "attention-status"]):
                    # Default mode = print + return 0 even when over ceiling.
                    main()

    def test_strict_exits_one_when_over_ceiling(self):
        with patch("mnemon.cli._print_attention_status") as mock_status:
            mock_status.return_value = (0.714, 0.25)
            with patch("mnemon.store.Store") as mock_store_cls:
                mock_store_cls.return_value = MagicMock()
                with patch("sys.argv", ["mnemon", "attention-status", "--strict"]):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
        assert exc_info.value.code == 1

    def test_strict_exits_zero_when_under_ceiling(self):
        with patch("mnemon.cli._print_attention_status") as mock_status:
            mock_status.return_value = (0.12, 0.25)
            with patch("mnemon.store.Store") as mock_store_cls:
                mock_store_cls.return_value = MagicMock()
                with patch("sys.argv", ["mnemon", "attention-status", "--strict"]):
                    # Under-ceiling rate → exits 0 (no SystemExit raised).
                    main()

    def test_strict_exits_zero_at_exact_ceiling(self):
        """Boundary: rate == ceiling is still passing (≤, not <)."""
        with patch("mnemon.cli._print_attention_status") as mock_status:
            mock_status.return_value = (0.25, 0.25)
            with patch("mnemon.store.Store") as mock_store_cls:
                mock_store_cls.return_value = MagicMock()
                with patch("sys.argv", ["mnemon", "attention-status", "--strict"]):
                    main()


class TestStandingCli:
    """Coverage for the `mnemon standing list|promote|demote` paths.
    Closes ROADMAP P3 follow-up to push cli.py above the 80% module
    floor — the _handle_standing block was the largest uncovered region."""

    @patch("mnemon.store.Store")
    def test_list_populated(self, MockStore, capsys):
        from mnemon.store import Document
        mock_store = MockStore.return_value
        mock_store.standing_tier_status.return_value = {
            "count": 2, "cap": 15, "hard_ceiling": 20,
        }
        d1 = Document(
            id=1, collection="default", path=None, title="Rule 1",
            hash="h1", content_type="preference", memory_type="semantic",
            confidence=0.85, quality_score=0.0, access_count=0, pinned=0,
            source_client=None, invalidated_at=None, invalidated_by=None,
            created_at="2026-05-01", updated_at="2026-05-01",
            content="some constraint content",
        )
        d2 = Document(
            id=2, collection="default", path=None, title="Rule 2",
            hash="h2", content_type="decision", memory_type="semantic",
            confidence=0.9, quality_score=0.0, access_count=0, pinned=0,
            source_client=None, invalidated_at=None, invalidated_by=None,
            created_at="2026-05-02", updated_at="2026-05-02",
            content="another constraint",
        )
        mock_store.list_standing.return_value = [d1, d2]
        with patch("sys.argv", ["mnemon", "standing", "list"]):
            main()
        out = capsys.readouterr().out
        assert "Standing tier: 2/15" in out
        assert "Rule 1" in out
        assert "Rule 2" in out

    @patch("mnemon.store.Store")
    def test_list_empty(self, MockStore, capsys):
        mock_store = MockStore.return_value
        mock_store.standing_tier_status.return_value = {
            "count": 0, "cap": 15, "hard_ceiling": 20,
        }
        mock_store.list_standing.return_value = []
        with patch("sys.argv", ["mnemon", "standing", "list"]):
            main()
        out = capsys.readouterr().out
        assert "Standing tier: 0/15" in out
        assert "(empty" in out

    @patch("mnemon.store.Store")
    def test_promote_success(self, MockStore, capsys):
        mock_store = MockStore.return_value
        mock_store.standing_tier_status.return_value = {
            "count": 3, "cap": 15, "hard_ceiling": 20,
        }
        with patch("sys.argv", ["mnemon", "standing", "promote", "42"]):
            main()
        out = capsys.readouterr().out
        assert "Promoted memory #42" in out
        mock_store.promote_to_standing.assert_called_once_with(42)

    @patch("mnemon.store.Store")
    def test_promote_cap_reached_exits_1(self, MockStore, capsys):
        from mnemon.store import StandingTierCapReached
        mock_store = MockStore.return_value
        mock_store.promote_to_standing.side_effect = StandingTierCapReached(
            "at cap 15/15"
        )
        with patch("sys.argv", ["mnemon", "standing", "promote", "7"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 1
        assert "Cap reached" in capsys.readouterr().err

    @patch("mnemon.store.Store")
    def test_promote_provenance_rejected_exits_1(self, MockStore, capsys):
        from mnemon.store import StandingTierProvenanceRejected
        mock_store = MockStore.return_value
        mock_store.promote_to_standing.side_effect = (
            StandingTierProvenanceRejected("hook-sourced")
        )
        with patch("sys.argv", ["mnemon", "standing", "promote", "7"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 1
        assert "Provenance rejected" in capsys.readouterr().err

    @patch("mnemon.store.Store")
    def test_promote_missing_id_exits_2(self, MockStore, capsys):
        with patch("sys.argv", ["mnemon", "standing", "promote"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 2
        assert "Usage:" in capsys.readouterr().err

    @patch("mnemon.store.Store")
    def test_promote_non_integer_id_exits_2(self, MockStore, capsys):
        with patch("sys.argv", ["mnemon", "standing", "promote", "abc"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 2
        assert "must be an integer" in capsys.readouterr().err

    @patch("mnemon.store.Store")
    def test_demote_success(self, MockStore, capsys):
        mock_store = MockStore.return_value
        mock_store.demote_to_situational.return_value = True
        mock_store.standing_tier_status.return_value = {
            "count": 2, "cap": 15, "hard_ceiling": 20,
        }
        with patch("sys.argv", ["mnemon", "standing", "demote", "42"]):
            main()
        out = capsys.readouterr().out
        assert "Demoted memory #42" in out

    @patch("mnemon.store.Store")
    def test_demote_idempotent_when_not_standing(self, MockStore, capsys):
        mock_store = MockStore.return_value
        mock_store.demote_to_situational.return_value = False
        mock_store.standing_tier_status.return_value = {
            "count": 3, "cap": 15, "hard_ceiling": 20,
        }
        with patch("sys.argv", ["mnemon", "standing", "demote", "7"]):
            main()
        out = capsys.readouterr().out
        assert "not on the standing tier" in out

    @patch("mnemon.store.Store")
    def test_demote_missing_id_exits_2(self, MockStore, capsys):
        with patch("sys.argv", ["mnemon", "standing", "demote"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 2

    @patch("mnemon.store.Store")
    def test_unknown_standing_subcommand_exits_2(self, MockStore, capsys):
        with patch("sys.argv", ["mnemon", "standing", "frobnicate"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "Unknown subcommand" in err


class TestAttentionStatusPrint:
    """Coverage for _print_attention_status — the largest uncovered
    cli.py block after _handle_standing."""

    @patch("mnemon.store.Store")
    def test_status_with_no_recent_activity(self, MockStore, capsys):
        mock_store = MockStore.return_value
        mock_cursor = MagicMock()
        # Three SQL queries: boosts, saves, recurrence_count histogram, top, recent
        # First call: COUNT(*) FROM relations → boosts_7d
        # Second call: COUNT(*) FROM documents → saves_7d
        # Third call: histogram
        # Fourth: top canonicals
        # Fifth: recent restates
        mock_cursor.fetchone.side_effect = [{"c": 0}, {"c": 5}]
        mock_cursor.fetchall.side_effect = [
            [{"recurrence_count": 0, "n": 5}],  # histogram
            [],  # top canonicals (none yet)
            [],  # recent relations
        ]
        mock_store.db.execute.return_value = mock_cursor
        with patch("sys.argv", ["mnemon", "attention-status"]):
            main()
        out = capsys.readouterr().out
        assert "Capture attention" in out
        assert "Boost-rate 7d      : 0 / 5 = 0.000" in out
        assert "Recurrence count distribution" in out
        assert "No canonicals" in out

    @patch("mnemon.store.Store")
    def test_status_with_canonicals_and_recent_restates(self, MockStore, capsys):
        mock_store = MockStore.return_value
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [{"c": 2}, {"c": 10}]
        mock_cursor.fetchall.side_effect = [
            # histogram
            [{"recurrence_count": 0, "n": 8}, {"recurrence_count": 1, "n": 2}],
            # top canonicals
            [{"id": 42, "title": "Some canonical fact", "recurrence_count": 1,
              "confidence": 0.85}],
            # recent restates
            [{"source_id": 99, "target_id": 42, "weight": 0.91,
              "created_at": "2026-05-27 10:00:00"}],
        ]
        mock_store.db.execute.return_value = mock_cursor
        with patch("sys.argv", ["mnemon", "attention-status"]):
            main()
        out = capsys.readouterr().out
        assert "Boost-rate 7d      : 2 / 10 = 0.200" in out
        assert "Top canonicals" in out
        assert "Some canonical fact" in out
        assert "Last 10 'restates' relations" in out
        assert "#   99 → #   42" in out
