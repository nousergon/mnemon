"""Tests for ``scripts/phase_a_resoak.sh``.

Covers syntax + the network-free banner/help paths. The Fly-dependent
subcommands (preflight, activate, status, close, deactivate) are
exercised by the operator against the real Fly app — unit tests cover
shape; integration is operator-driven.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "phase_a_resoak.sh"


def _env_with_venv_bin() -> dict[str, str]:
    """Point MNEMON_VENV_BIN at the test interpreter's bin dir so the
    script's prerequisite check passes in both .venv (local) and
    bare-pip (CI) environments. Same pattern as
    tests/test_mnemon_ops.py."""
    env = {**os.environ}
    env["MNEMON_VENV_BIN"] = str(Path(sys.executable).parent)
    return env


class TestSyntax:
    def test_script_is_syntactically_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"bash -n flagged a syntax error:\n{result.stderr}"
        )


class TestHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "help"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT), env=_env_with_venv_bin(),
        )
        assert result.returncode == 0, result.stderr

    def test_no_arg_prints_usage(self):
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT), env=_env_with_venv_bin(),
        )
        assert result.returncode == 0, result.stderr
        assert "Subcommands:" in result.stdout

    def test_help_lists_every_subcommand(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "help"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT), env=_env_with_venv_bin(),
        )
        for sub in ("preflight", "activate", "status", "close", "deactivate"):
            assert sub in result.stdout, f"help text missing {sub}"


class TestUnknownSubcommand:
    def test_unknown_subcommand_exits_2(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "frobnicate"],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT), env=_env_with_venv_bin(),
        )
        assert result.returncode == 2
        assert "unknown subcommand" in result.stderr.lower()
