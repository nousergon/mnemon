"""Tests for S3 vault sync."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mnemon.sync import push, pull, _vault_files, _s3_path


class TestConfig:
    def test_s3_path_default_prefix(self):
        with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "my-bucket"}):
            assert _s3_path("default.sqlite") == "s3://my-bucket/mnemon/vaults/default.sqlite"

    def test_s3_path_custom_prefix(self):
        with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "b", "MNEMON_S3_PREFIX": "custom/path"}):
            assert _s3_path("default.sqlite") == "s3://b/custom/path/default.sqlite"

    def test_vault_files_returns_sqlite_and_vec(self):
        files = _vault_files()
        assert "sqlite" in files
        assert "vec" in files
        assert str(files["sqlite"]).endswith(".sqlite")
        assert str(files["vec"]).endswith(".vec.npz")


class TestPush:
    def test_push_fails_without_bucket(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MNEMON_S3_BUCKET", None)
            result = push()
            assert result["errors"] == ["MNEMON_S3_BUCKET not set"]

    @patch("mnemon.sync._checkpoint_wal", return_value=None)
    @patch("mnemon.sync._run_cmd")
    def test_push_uploads_existing_files(self, mock_run, _mock_ckpt):
        mock_run.return_value = (True, "")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake vault files
            sqlite_path = Path(tmpdir) / "default.sqlite"
            sqlite_path.write_bytes(b"fake sqlite data")

            with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "test-bucket"}), \
                 patch("mnemon.sync._vault_files", return_value={
                     "sqlite": sqlite_path,
                     "vec": Path(tmpdir) / "default.vec.npz",  # doesn't exist
                 }):
                result = push()

            assert len(result["pushed"]) == 1
            assert "sqlite" in result["pushed"][0]
            assert result["errors"] == []

    @patch("mnemon.sync._checkpoint_wal", return_value=None)
    @patch("mnemon.sync._run_cmd")
    def test_push_reports_errors(self, mock_run, _mock_ckpt):
        mock_run.return_value = (False, "access denied")

        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / "default.sqlite"
            sqlite_path.write_bytes(b"data")

            with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "test-bucket"}), \
                 patch("mnemon.sync._vault_files", return_value={
                     "sqlite": sqlite_path,
                     "vec": Path(tmpdir) / "nonexistent.vec.npz",
                 }):
                result = push()

            assert len(result["errors"]) == 1
            assert "access denied" in result["errors"][0]


class TestPull:
    def test_pull_fails_without_bucket(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MNEMON_S3_BUCKET", None)
            result = pull()
            assert result["errors"] == ["MNEMON_S3_BUCKET not set"]

    @patch("mnemon.sync._run_cmd")
    def test_pull_skips_missing_s3_files(self, mock_run):
        # s3 ls returns nothing = file doesn't exist
        mock_run.return_value = (False, "")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "test-bucket"}), \
                 patch("mnemon.sync._vault_files", return_value={
                     "sqlite": Path(tmpdir) / "default.sqlite",
                     "vec": Path(tmpdir) / "default.vec.npz",
                 }):
                result = pull()

            assert result["pulled"] == []
            assert result["errors"] == []

    @patch("mnemon.sync._run_cmd")
    def test_pull_downloads_existing_files(self, mock_run):
        def side_effect(cmd):
            if "s3 ls" in cmd:
                return (True, "2026-01-01 00:00:00  1234 default.sqlite")
            if "s3 cp" in cmd:
                return (True, "")
            return (False, "")

        mock_run.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / "default.sqlite"

            with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "test-bucket"}), \
                 patch("mnemon.sync._vault_files", return_value={
                     "sqlite": sqlite_path,
                     "vec": Path(tmpdir) / "default.vec.npz",
                 }):
                result = pull()

            # Both files were listed, but only sqlite had content in the ls response
            # The vec ls also returns True in our mock, so both get attempted
            assert len(result["pulled"]) >= 1


class TestPushCheckpointsWAL:
    """Regression for the 2026-05-21 Layer-3 downgrade bug: sync.push
    was uploading the main sqlite file without first flushing the WAL.
    For long-lived server processes (mnemon serve-remote on Fly) that
    hold an open SQLite connection, recent commits accumulate in WAL
    without auto-checkpoint (which only fires at 1000 pages by default).
    `aws s3 cp default.sqlite` then uploaded a stale main file missing
    all the recent writes — silently lost on the downstream sync pull.

    Short-lived CLI processes (`mnemon save`) auto-checkpointed WAL
    on connection close, so this bug only manifested for the
    serve-remote → upgrade-web-dump path (and would have manifested
    similarly for any long-lived-process → sync-push flow).
    """

    def test_checkpoint_wal_flushes_data_from_wal_to_main(self, tmp_path):
        # Reproduce a long-lived-connection scenario, then call the
        # checkpoint helper and verify main file now contains the data.
        import sqlite3
        import shutil
        from mnemon.sync import _checkpoint_wal

        db = tmp_path / "test.sqlite"
        # Open a persistent connection, set WAL, insert without closing.
        conn = sqlite3.connect(str(db))
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);")
        for i in range(3):
            conn.execute("INSERT INTO t (v) VALUES (?)", (f"row-{i}",))
        conn.commit()
        # Don't close — simulates serve-remote holding the connection.

        # Confirm the bug: copy of main file alone has no rows (or no table)
        main_only = tmp_path / "main_only.sqlite"
        shutil.copy(db, main_only)
        try:
            cnt_pre = sqlite3.connect(str(main_only)).execute(
                "SELECT COUNT(*) FROM t"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            cnt_pre = -1  # no table at all — even worse symptom
        assert cnt_pre != 3, (
            f"WAL bug didn't reproduce — main file already has all 3 rows "
            f"(got {cnt_pre}). Test setup is wrong; SQLite WAL semantics changed."
        )

        # Apply the fix.
        err = _checkpoint_wal(db)
        assert err is None, f"checkpoint failed: {err}"

        # Verify: main file alone now has 3 rows.
        main_only2 = tmp_path / "main_after.sqlite"
        shutil.copy(db, main_only2)
        cnt_post = sqlite3.connect(str(main_only2)).execute(
            "SELECT COUNT(*) FROM t"
        ).fetchone()[0]
        assert cnt_post == 3, f"expected 3 rows in main file post-checkpoint, got {cnt_post}"

        conn.close()

    def test_checkpoint_wal_returns_error_string_on_locked_file(self, tmp_path):
        # Failure mode: returns a string instead of raising, so push()
        # can record it as an error and skip the cp without aborting
        # the whole push call.
        from mnemon.sync import _checkpoint_wal

        # Non-existent path → sqlite will create + fail on missing table,
        # but checkpoint of an empty DB is actually fine — so we test
        # the contract by passing a path that's not a SQLite file.
        bad = tmp_path / "not_a_db.bin"
        bad.write_bytes(b"\x00\x01\x02not sqlite header at all")
        err = _checkpoint_wal(bad)
        assert err is not None, "expected error string for non-SQLite file"
        assert "checkpoint" in err.lower() or "sqlite" in err.lower()

    def test_push_calls_checkpoint_before_aws_cp(self, tmp_path):
        # Integration: push() should call _checkpoint_wal for the
        # sqlite file before invoking aws s3 cp. Verifies the call
        # order via a side-effect-recording mock.
        from unittest.mock import MagicMock
        import shutil

        sqlite_path = tmp_path / "default.sqlite"
        sqlite_path.write_bytes(b"fake sqlite data")

        call_order: list[str] = []

        def _record_checkpoint(_path):
            call_order.append("checkpoint")
            return None

        def _record_cp(cmd):
            if "s3 cp" in cmd:
                call_order.append("aws_cp")
            return (True, "")

        with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "test-bucket"}), \
             patch("mnemon.sync._vault_files", return_value={
                 "sqlite": sqlite_path,
                 "vec": tmp_path / "default.vec.npz",  # doesn't exist
             }), \
             patch("mnemon.sync._checkpoint_wal", side_effect=_record_checkpoint), \
             patch("mnemon.sync._run_cmd", side_effect=_record_cp):
            result = push()

        # Checkpoint must come before the s3 cp call for the sqlite file.
        assert call_order == ["checkpoint", "aws_cp"], (
            f"expected checkpoint → aws_cp, got {call_order}"
        )
        assert not result["errors"], f"unexpected errors: {result['errors']}"
