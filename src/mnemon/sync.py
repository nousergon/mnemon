"""S3 vault sync — push/pull SQLite vault + vector store to/from S3.

Uses AWS CLI (no SDK dependency). Content-addressable storage
prevents content duplication. Last-write-wins for metadata.

Usage:
    MNEMON_S3_BUCKET=my-bucket mnemon sync push
    MNEMON_S3_BUCKET=my-bucket mnemon sync pull

Env vars:
    MNEMON_S3_BUCKET    S3 bucket name (required)
    MNEMON_S3_PREFIX    S3 key prefix (default: mnemon/vaults)
    MNEMON_VAULT_NAME   vault name (default: default)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import vault_dir

S3_PREFIX_DEFAULT = "mnemon/vaults"
VAULT_NAME_DEFAULT = "default"


def _s3_bucket() -> str:
    return os.environ.get("MNEMON_S3_BUCKET", "")


def _s3_prefix() -> str:
    return os.environ.get("MNEMON_S3_PREFIX", S3_PREFIX_DEFAULT)


def _vault_name() -> str:
    return os.environ.get("MNEMON_VAULT_NAME", VAULT_NAME_DEFAULT)


def _vault_files() -> dict[str, Path]:
    """Return local vault file paths."""
    vdir = vault_dir()
    name = _vault_name()
    return {
        "sqlite": vdir / f"{name}.sqlite",
        "vec": vdir / f"{name}.vec.npz",
    }


def _s3_path(filename: str) -> str:
    return f"s3://{_s3_bucket()}/{_s3_prefix()}/{filename}"


def _run_cmd(cmd: str) -> tuple[bool, str]:
    """Run a shell command. Returns (success, output)."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return True, result.stdout.strip()
    return False, result.stderr.strip()


def _snapshot_sqlite(src_path: Path, snap_path: Path) -> str | None:
    """Produce an atomic consistent snapshot of a SQLite database via
    the online-backup API.

    SOTA primitive for "copy a SQLite database file safely while
    something else might be writing it." ``Connection.backup()`` uses
    SQLite's WAL-aware backup protocol — captures all committed writes
    even when another process holds the connection open with frames
    only in the WAL.

    Why not ``PRAGMA wal_checkpoint`` + raw ``aws s3 cp``: the
    checkpoint is a *cooperative* request that returns
    ``(busy=0, total=0, checkpointed=0)`` when another process holds
    the WAL (verified 2026-05-21 against a long-running serve-remote
    scenario — checkpoint reported success but flushed zero frames).
    The backup API is the canonical primitive for this; the checkpoint
    band-aid (formerly ``_checkpoint_wal``) was a wrong mental model.

    Why not litestream/LiteFS: mnemon's use case is one-shot cross-host
    transfer at upgrade/downgrade time, not continuous replication.
    The backup API matches the actual operational model.

    Returns ``None`` on success, an error string on failure (matches
    the existing best-effort error contract in ``push()``).
    """
    import sqlite3
    try:
        # Clean any stale snapshot from a prior run.
        try:
            snap_path.unlink()
        except FileNotFoundError:
            pass
        src = sqlite3.connect(str(src_path), timeout=10.0)
        try:
            dst = sqlite3.connect(str(snap_path))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    except sqlite3.Error as exc:
        return f"sqlite backup failed: {exc}"
    except OSError as exc:
        return f"snapshot file IO failed: {exc}"
    return None


def push() -> dict[str, list[str]]:
    """Push local vault to S3.

    Uses SQLite's online-backup API to produce an atomic consistent
    snapshot of the live database before uploading — handles the
    long-lived-server case (mnemon serve-remote on Fly) where
    raw ``aws s3 cp`` of the sqlite file would upload a stale main
    file missing all WAL-resident writes.

    Vec store (default.vec.npz) is a binary numpy file with no
    concurrent-writer semantics — uploaded directly without snapshot.

    Returns {"pushed": [...], "errors": [...]}.
    """
    bucket = _s3_bucket()
    if not bucket:
        return {"pushed": [], "errors": ["MNEMON_S3_BUCKET not set"]}

    files = _vault_files()
    pushed: list[str] = []
    errors: list[str] = []

    for label, local_path in files.items():
        if not local_path.exists():
            continue

        # SQLite vault: snapshot via backup API → upload the snapshot.
        # vec.npz: raw file → upload directly.
        if label == "sqlite":
            snap_path = local_path.with_suffix(".sqlite.snapshot")
            snap_err = _snapshot_sqlite(local_path, snap_path)
            if snap_err is not None:
                errors.append(snap_err)
                continue
            upload_path = snap_path
            ext = "sqlite"
        else:
            upload_path = local_path
            ext = "vec.npz"

        s3_target = _s3_path(f"{_vault_name()}.{ext}")
        ok, output = _run_cmd(f'aws s3 cp "{upload_path}" "{s3_target}" --only-show-errors')

        # Clean up sqlite snapshot regardless of upload success — it's a
        # transient artifact tied to this push call.
        if label == "sqlite":
            try:
                snap_path.unlink()
            except FileNotFoundError:
                pass

        if ok:
            size_kb = local_path.stat().st_size / 1024
            pushed.append(f"{label}: {size_kb:.1f}KB → {s3_target}")
        else:
            errors.append(f"{label}: {output}")

    return {"pushed": pushed, "errors": errors}


def pull() -> dict[str, list[str]]:
    """Pull vault from S3 to local.

    Returns {"pulled": [...], "errors": [...]}.
    """
    bucket = _s3_bucket()
    if not bucket:
        return {"pulled": [], "errors": ["MNEMON_S3_BUCKET not set"]}

    files = _vault_files()
    pulled: list[str] = []
    errors: list[str] = []

    for label, local_path in files.items():
        ext = "sqlite" if label == "sqlite" else "vec.npz"
        s3_source = _s3_path(f"{_vault_name()}.{ext}")

        # Check if file exists on S3
        ok, output = _run_cmd(f'aws s3 ls "{s3_source}" 2>/dev/null')
        if not ok or not output:
            continue

        # Ensure parent dir exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        ok, output = _run_cmd(f'aws s3 cp "{s3_source}" "{local_path}" --only-show-errors')

        if ok:
            size_kb = local_path.stat().st_size / 1024 if local_path.exists() else 0
            pulled.append(f"{label}: {s3_source} → {size_kb:.1f}KB")
        else:
            errors.append(f"{label}: {output}")

    return {"pulled": pulled, "errors": errors}
