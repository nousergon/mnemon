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

    @patch("mnemon.sync._snapshot_sqlite", return_value=None)
    @patch("mnemon.sync._run_cmd")
    def test_push_uploads_existing_files(self, mock_run, _mock_snap):
        mock_run.return_value = (True, "")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake vault files
            sqlite_path = Path(tmpdir) / "default.sqlite"
            sqlite_path.write_bytes(b"fake sqlite data")
            # _snapshot_sqlite is mocked, so we need the snapshot file
            # to exist for the aws cp to "succeed" against it.
            (sqlite_path.with_suffix(".sqlite.snapshot")).write_bytes(b"snap")

            with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "test-bucket"}), \
                 patch("mnemon.sync._vault_files", return_value={
                     "sqlite": sqlite_path,
                     "vec": Path(tmpdir) / "default.vec.npz",  # doesn't exist
                 }):
                result = push()

            assert len(result["pushed"]) == 1
            assert "sqlite" in result["pushed"][0]
            assert result["errors"] == []

    @patch("mnemon.sync._snapshot_sqlite", return_value=None)
    @patch("mnemon.sync._run_cmd")
    def test_push_reports_errors(self, mock_run, _mock_snap):
        mock_run.return_value = (False, "access denied")

        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / "default.sqlite"
            sqlite_path.write_bytes(b"data")
            (sqlite_path.with_suffix(".sqlite.snapshot")).write_bytes(b"snap")

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


class TestPushUsesBackupAPI:
    """Canonical regression for the 2026-05-21 sync-correctness arc.

    Replaces the prior `TestPushCheckpointsWAL` — the checkpoint-based
    approach was a wrong mental model: `PRAGMA wal_checkpoint(TRUNCATE)`
    returns `(busy=0, total=0, checkpointed=0)` (success, zero frames
    flushed) when another process holds the connection open in WAL mode.
    The actual correct primitive is SQLite's online-backup API
    (`Connection.backup()`), which captures all committed writes
    regardless of WAL state — even with concurrent writers.

    These tests cover the new `_snapshot_sqlite` helper and the
    backup-then-upload behavior of `push()`.
    """

    def test_snapshot_captures_writes_from_long_lived_holder(self, tmp_path):
        # The cross-process scenario PRAGMA wal_checkpoint can't handle.
        # Open a persistent connection in WAL mode, insert without
        # closing, then snapshot from a separate code path. Backup must
        # capture all 3 rows even though the main file is empty.
        import sqlite3
        import shutil
        from mnemon.sync import _snapshot_sqlite

        db = tmp_path / "test.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);")
        for i in range(3):
            conn.execute("INSERT INTO t (v) VALUES (?)", (f"row-{i}",))
        conn.commit()
        # Don't close — simulates a long-lived serve-remote process.

        # Confirm the underlying scenario: a main-only copy is empty.
        main_only = tmp_path / "main_only.sqlite"
        shutil.copy(db, main_only)
        try:
            cnt_main_only = sqlite3.connect(str(main_only)).execute(
                "SELECT COUNT(*) FROM t"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            cnt_main_only = -1
        assert cnt_main_only != 3, (
            f"setup wrong — main file already has 3 rows, can't test "
            f"the long-lived-holder scenario (got {cnt_main_only})"
        )

        # Snapshot via backup API.
        snap = tmp_path / "snap.sqlite"
        err = _snapshot_sqlite(db, snap)
        assert err is None, f"snapshot failed: {err}"

        # Snapshot must have all 3 rows.
        cnt_snap = sqlite3.connect(str(snap)).execute(
            "SELECT COUNT(*) FROM t"
        ).fetchone()[0]
        assert cnt_snap == 3, f"snapshot missing rows: got {cnt_snap}, expected 3"

        conn.close()

    def test_snapshot_does_not_modify_source(self, tmp_path):
        # Backup is read-side / non-destructive. Source DB byte-equal
        # before and after snapshot (modulo SQLite's own auto-checkpoint
        # behavior, which we don't control). At minimum: source still
        # opens and contains the same data.
        import sqlite3
        from mnemon.sync import _snapshot_sqlite

        db = tmp_path / "source.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (id INT)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        conn.close()

        size_before = db.stat().st_size
        err = _snapshot_sqlite(db, tmp_path / "snap.sqlite")
        assert err is None
        # Source still readable + intact
        cnt = sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert cnt == 1
        # Source size unchanged (within a small tolerance for any incidental writes — none expected)
        assert db.stat().st_size == size_before

    def test_snapshot_returns_error_string_on_invalid_source(self, tmp_path):
        # Non-SQLite file → backup() raises sqlite3.DatabaseError. The
        # helper catches and returns the error string so push() can
        # record it and skip this entry without aborting the whole call.
        from mnemon.sync import _snapshot_sqlite

        bad = tmp_path / "not_a_db.bin"
        bad.write_bytes(b"\x00\x01\x02not sqlite at all")
        err = _snapshot_sqlite(bad, tmp_path / "snap.sqlite")
        assert err is not None, "expected error string for non-SQLite source"
        assert "sqlite" in err.lower() or "backup" in err.lower()

    def test_push_invokes_snapshot_before_aws_cp_for_sqlite(self, tmp_path):
        # Integration: push() snapshots the sqlite file via the backup
        # API, then aws-cp's the snapshot (NOT the live sqlite file).
        sqlite_path = tmp_path / "default.sqlite"
        sqlite_path.write_bytes(b"fake")

        call_order: list[str] = []
        aws_cp_paths: list[str] = []

        def _record_snapshot(src, snap):
            call_order.append("snapshot")
            # Real backup would write content here. Simulate so the
            # subsequent aws cp has a file to "upload".
            snap.write_bytes(b"snap content")
            return None

        def _record_cp(cmd):
            if "s3 cp" in cmd:
                call_order.append("aws_cp")
                # Extract the local path arg from the cp command for
                # the source-file assertion below.
                tokens = cmd.split('"')
                aws_cp_paths.append(tokens[1])  # first quoted arg
            return (True, "")

        with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "test-bucket"}), \
             patch("mnemon.sync._vault_files", return_value={
                 "sqlite": sqlite_path,
                 "vec": tmp_path / "default.vec.npz",  # doesn't exist
             }), \
             patch("mnemon.sync._snapshot_sqlite", side_effect=_record_snapshot), \
             patch("mnemon.sync._run_cmd", side_effect=_record_cp):
            result = push()

        assert call_order == ["snapshot", "aws_cp"], (
            f"expected snapshot → aws_cp, got {call_order}"
        )
        # Crucial: aws cp uploaded the SNAPSHOT, not the live sqlite file.
        # The snapshot has the `.snapshot` extension our push code uses.
        assert ".sqlite.snapshot" in aws_cp_paths[0], (
            f"aws cp should upload the snapshot file, got: {aws_cp_paths[0]}"
        )
        assert not result["errors"], f"unexpected errors: {result['errors']}"

    def test_push_cleans_up_snapshot_after_upload(self, tmp_path):
        # The snapshot file is a transient — it must not survive a push
        # call (would accumulate in the operator's .mnemon/ directory).
        sqlite_path = tmp_path / "default.sqlite"
        sqlite_path.write_bytes(b"fake")
        snap_path = sqlite_path.with_suffix(".sqlite.snapshot")

        def _record_snapshot(src, snap):
            snap.write_bytes(b"snap")
            return None

        with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "test-bucket"}), \
             patch("mnemon.sync._vault_files", return_value={
                 "sqlite": sqlite_path,
                 "vec": tmp_path / "default.vec.npz",
             }), \
             patch("mnemon.sync._snapshot_sqlite", side_effect=_record_snapshot), \
             patch("mnemon.sync._run_cmd", return_value=(True, "")):
            push()

        assert not snap_path.exists(), (
            "snapshot file should be removed after push completes"
        )

    def test_push_vec_npz_is_not_snapshot(self, tmp_path):
        # vec.npz is a binary numpy file, no SQLite semantics. push()
        # should aws cp it directly without going through the snapshot
        # path (which is sqlite-specific).
        vec_path = tmp_path / "default.vec.npz"
        vec_path.write_bytes(b"npz bytes")

        cp_paths: list[str] = []

        def _record_cp(cmd):
            if "s3 cp" in cmd:
                tokens = cmd.split('"')
                cp_paths.append(tokens[1])
            return (True, "")

        with patch.dict(os.environ, {"MNEMON_S3_BUCKET": "test-bucket"}), \
             patch("mnemon.sync._vault_files", return_value={
                 "sqlite": tmp_path / "missing.sqlite",  # doesn't exist
                 "vec": vec_path,
             }), \
             patch("mnemon.sync._snapshot_sqlite") as mock_snap, \
             patch("mnemon.sync._run_cmd", side_effect=_record_cp):
            push()

        # snapshot should not be called for vec
        mock_snap.assert_not_called()
        # aws cp should upload the vec file directly
        assert len(cp_paths) == 1
        assert cp_paths[0] == str(vec_path)
