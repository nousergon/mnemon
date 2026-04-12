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

    def test_setup_without_target_exits(self, capsys):
        with patch("sys.argv", ["mnemon", "setup"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Usage: mnemon setup" in err


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
