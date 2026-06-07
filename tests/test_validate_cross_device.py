"""Smoke tests for scripts/validate_cross_device.sh.

The script's real path deploys a Fly app (operator-only, can't run in CI),
but its syntax, arg-parsing, dry-run plan, and the prod-app safety guard
are all CI-testable via subprocess. This catches the
`AttributeError-on-first-run` defect class for the bash operator scripts
(the python ones are covered by test_scripts_smoke.py).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_cross_device.sh"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_script_exists_and_executable():
    assert SCRIPT.exists(), f"{SCRIPT} missing"
    assert SCRIPT.stat().st_mode & 0o111, "script is not executable"


def test_bash_syntax_valid():
    # `bash -n` parses without executing — catches syntax errors.
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, f"syntax error:\n{r.stderr}"


def test_help_exits_zero_and_documents_usage():
    r = _run("--help")
    assert r.returncode == 0
    assert "validate_cross_device.sh" in r.stdout
    assert "--dry-run" in r.stdout


def test_dry_run_prints_plan_no_side_effects():
    r = _run("--dry-run")
    assert r.returncode == 0
    out = r.stdout
    assert "DRY RUN" in out
    # The plan must name the load-bearing safety + deploy steps.
    assert "upgrade web" in out
    assert "OAuth AS metadata" in out
    assert "MNEMON_PROD_APP_NAMES=mnemon-memory" in out
    assert "restore ~/.mnemon snapshot" in out


def test_dry_run_keep_changes_teardown():
    default = _run("--dry-run").stdout
    keep = _run("--dry-run", "--keep").stdout
    assert "pause for Enter, then teardown" in default
    assert "leave the app running" in keep
    assert "skip destroy" in keep


def test_prod_app_guard_refuses():
    r = _run("--app-name", "mnemon-memory")
    assert r.returncode != 0
    assert "refusing to run against the prod app" in r.stderr


def test_unknown_arg_errors():
    r = _run("--bogus-flag")
    assert r.returncode != 0
    assert "unknown arg" in r.stderr
