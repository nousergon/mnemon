"""Tests for ``scripts/mnemon_ops.sh``.

Covers the network-free subcommands (``help``, ``changelog-extract``,
unknown-subcommand error path). Network/flyctl-dependent subcommands
(``cleanup-test-apps``, ``recover-token``, ``restart-machine``,
``vault-stats``) are exercised by operators end-to-end and only get
a syntax-check + help-text presence assertion here.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "mnemon_ops.sh"


def _run(*args, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


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
        result = _run("help")
        assert result.returncode == 0

    def test_no_arg_prints_usage(self):
        # Empty subcommand falls through to `help` per the case branch.
        result = _run()
        assert result.returncode == 0
        assert "Subcommands:" in result.stdout

    def test_help_lists_every_subcommand(self):
        result = _run("help")
        # Sanity — each named subcommand appears in the help block. Catches
        # silent drift where a subcommand is added but the usage block isn't
        # updated.
        for sub in (
            "cleanup-test-apps",
            "recover-token",
            "restart-machine",
            "vault-stats",
            "changelog-extract",
        ):
            assert sub in result.stdout, f"help text missing {sub}"


class TestUnknownSubcommand:
    def test_unknown_subcommand_exits_2(self):
        result = _run("nonsense-subcommand")
        assert result.returncode == 2
        assert "unknown subcommand" in result.stderr.lower()


class TestChangelogExtract:
    @pytest.fixture
    def fake_changelog(self, tmp_path, monkeypatch):
        """Substitute a synthetic CHANGELOG.md so the test doesn't depend
        on the repo's real history."""
        cl = tmp_path / "CHANGELOG.md"
        cl.write_text(textwrap.dedent("""\
            # Changelog

            ## [0.9.0] - 2026-06-01

            ### Some feature

            - bullet one
            - bullet two

            ## [0.8.0] - 2026-05-15

            ### Another feature

            - older bullet
        """))
        # The script reads $REPO_ROOT/CHANGELOG.md — make the temp dir
        # masquerade as the repo root and point the script at it via the
        # explicit env vars its `set -u` reads.
        return cl

    def _run_in_synthetic_repo(self, tmp_path, *args, env_extra=None):
        """Run the script with REPO_ROOT pointing at a synthetic tmp dir
        but MNEMON_VENV_BIN pointing at the real venv (the script needs
        a working python). Returns the CompletedProcess."""
        import os
        script_dir = tmp_path / "scripts"
        script_dir.mkdir(exist_ok=True)
        (script_dir / "mnemon_ops.sh").write_bytes(SCRIPT.read_bytes())
        env = {**os.environ}
        env["MNEMON_VENV_BIN"] = str(REPO_ROOT / ".venv" / "bin")
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            ["bash", str(script_dir / "mnemon_ops.sh"), *args],
            capture_output=True, text=True, cwd=str(tmp_path), env=env,
        )

    def test_extracts_named_version(self, fake_changelog, tmp_path):
        result = self._run_in_synthetic_repo(
            tmp_path, "changelog-extract", "0.9.0",
        )
        assert result.returncode == 0, result.stderr
        assert "## [0.9.0] - 2026-06-01" in result.stdout
        assert "Some feature" in result.stdout
        assert "bullet one" in result.stdout
        # The next-version section must NOT bleed into the output.
        assert "0.8.0" not in result.stdout
        assert "Another feature" not in result.stdout

    def test_missing_version_exits_1(self, fake_changelog, tmp_path):
        result = self._run_in_synthetic_repo(
            tmp_path, "changelog-extract", "9.9.9",
        )
        assert result.returncode == 1
        assert "no `## [9.9.9]`" in result.stderr or "9.9.9" in result.stderr

    def test_no_version_arg_exits_2(self):
        result = _run("changelog-extract")
        assert result.returncode == 2
        assert "usage:" in result.stderr.lower()
