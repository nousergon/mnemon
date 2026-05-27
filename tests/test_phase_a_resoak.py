"""Tests for ``scripts/phase_a_resoak.sh``.

Covers syntax + the network-free banner/help paths. The Fly-dependent
subcommands (preflight, activate, status, close, deactivate) are
exercised by the operator against the real Fly app — unit tests cover
shape; integration is operator-driven.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "phase_a_resoak.sh"


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
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0

    def test_no_arg_prints_usage(self):
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "Subcommands:" in result.stdout

    def test_help_lists_every_subcommand(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "help"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        # Sanity — each named subcommand appears in the help block.
        for sub in ("preflight", "activate", "status", "close", "deactivate"):
            assert sub in result.stdout, f"help text missing {sub}"


class TestUnknownSubcommand:
    def test_unknown_subcommand_exits_2(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "frobnicate"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 2
        assert "unknown subcommand" in result.stderr.lower()
