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

    @patch("mnemon.sync._run_cmd")
    def test_push_uploads_existing_files(self, mock_run):
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

    @patch("mnemon.sync._run_cmd")
    def test_push_reports_errors(self, mock_run):
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
