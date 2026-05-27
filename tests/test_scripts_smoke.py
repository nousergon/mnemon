"""Smoke tests for ``scripts/*.py`` operator-facing CLIs.

Closes the 2026-05-24 ROADMAP P2 follow-up: scripts live outside
``src/`` and aren't picked up by the coverage gate, so import-time
and arg-parse defects (like the 2026-05-24 ``VecStore.get`` bug
that shipped in PR #153 because the call site was never exercised
end-to-end before merge) don't surface until an operator runs the
script for real.

Each public script under ``scripts/`` (excluding underscore-prefixed
helpers like ``_layer3_remote_helper.py``) is invoked with ``--help``.
``--help`` exits 0 if the script's module loads cleanly + ``argparse``
constructs without error. That's the minimum bar — it catches:

- Import errors from a missing dependency
- Attribute errors at module scope (e.g. referencing ``VecStore.get``
  before it existed)
- Syntax errors from a typo
- ``argparse`` mis-configuration

It does NOT catch logic errors inside the script's runtime path —
those need full integration tests (see ``test_promote_stable.sh`` for
the bash-harness pattern). This is the lightweight gate-keeper layer.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _public_scripts() -> list[Path]:
    """Every ``scripts/*.py`` that isn't a helper (underscore-prefixed).

    Underscore-prefixed files are internal helpers invoked by other
    scripts (e.g. ``_layer3_remote_helper.py`` is exec'd inside the
    promote_stable.sh layer3 step); they aren't user-facing CLIs and
    don't carry ``--help``.
    """
    return sorted(
        p for p in SCRIPTS_DIR.glob("*.py")
        if not p.name.startswith("_")
    )


@pytest.mark.parametrize(
    "script_path",
    _public_scripts(),
    ids=lambda p: p.name,
)
def test_script_help_exits_zero(script_path: Path):
    """Running ``python <script> --help`` must exit 0.

    Catches: import errors, syntax errors, argparse mis-configuration,
    module-scope attribute errors. Caught the
    2026-05-24 ``VecStore.get`` regression pattern at PR-review time
    rather than first-operator-run time."""
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"{script_path.name} --help exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # Sanity — argparse always renders a "usage:" line in --help output.
    assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower(), (
        f"{script_path.name} --help produced no recognizable usage block\n"
        f"stdout:\n{result.stdout}"
    )


def test_smoke_covers_every_public_script():
    """Meta-check: confirm the parameterization actually picks up every
    .py file we'd expect. Catches the case where a script lands but
    its name happens to start with `_` (would be silently skipped) or
    where a refactor moves something out of scripts/."""
    found = {p.name for p in _public_scripts()}
    # Anchor: scripts that exist today. Update this set when adding /
    # removing public scripts; the smoke harness should grow with them.
    expected_minimum = {
        "build_standing_set.py",
        "calibrate_capture_threshold.py",
        "check_health.py",
    }
    missing = expected_minimum - found
    assert not missing, (
        f"smoke harness lost coverage of: {missing}. "
        f"Either restore the scripts or update the expected_minimum set."
    )
