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


def _checkpoint_wal(sqlite_path: Path) -> str | None:
    """Force a WAL checkpoint so the main sqlite file contains all
    committed writes before ``aws s3 cp`` reads it.

    Short-lived CLI processes (e.g. ``mnemon save``) auto-checkpoint
    on connection close, so the WAL file disappears and the main file
    contains all data. Long-lived processes (e.g. ``mnemon
    serve-remote`` on Fly) hold an open SQLite connection — new
    commits accumulate in the WAL and SQLite only auto-checkpoints
    once the WAL grows past 1000 pages (default). Without an explicit
    checkpoint here, ``mnemon sync push`` running against a
    long-running server uploads a stale main file missing the latest
    writes, and the downstream ``sync pull`` silently restores
    the stale state.

    Surfaced 2026-05-21 during the 0.6.0 Layer-3 test: Step 4 added a
    doc via remote (committed to Fly serve-remote's WAL but never
    flushed to main); downgrade's Fly→S3 dump (added in 0.6.0 too)
    then uploaded the stale main and lost the doc on local restore.

    TRUNCATE mode: most thorough, blocks briefly if other readers
    hold locks. Acceptable for the (push, sync) flow — we don't
    expect concurrent traffic during a deliberate sync push. Failures
    are returned as a string so the caller can log without raising
    (sync push remains best-effort per error).
    """
    import sqlite3
    try:
        conn = sqlite3.connect(str(sqlite_path), timeout=10.0)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return f"sqlite checkpoint failed: {exc}"
    return None


def push() -> dict[str, list[str]]:
    """Push local vault to S3.

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

        # Force WAL → main flush before reading the sqlite file as bytes.
        # See _checkpoint_wal docstring for the rationale.
        if label == "sqlite":
            ckpt_err = _checkpoint_wal(local_path)
            if ckpt_err is not None:
                errors.append(ckpt_err)
                continue

        ext = "sqlite" if label == "sqlite" else "vec.npz"
        s3_target = _s3_path(f"{_vault_name()}.{ext}")
        ok, output = _run_cmd(f'aws s3 cp "{local_path}" "{s3_target}" --only-show-errors')

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
