"""The coverage gate's *scope* is asserted here, not only its number.

repository-baseline-policy.md §4.2 C5: the way a coverage gate stops being
honest is by narrowing what it measures rather than by lowering the number —
which reads as an improvement in every report. Measured on symposion, removing
one flag moved the reported figure from 34.76% to 92.36% with no new test
code.

So these tests assert what a passing suite cannot otherwise notice:

* the measured source is the WHOLE ``mnemon`` package (C1), never a path or
  submodule narrower than that;
* the floor is enforced by a non-zero exit (C2) and is a ratchet that may be
  raised and never lowered (C3);
* the ``omit`` list is pinned to its current, individually-justified members
  (dashboard UI, the entry-point shim, the deprecated optional LLM path — see
  the comments above ``[tool.coverage.run]`` in pyproject.toml) — a future PR
  that widens it silently shrinks the denominator without this test noticing
  the change in words, so this test forces the diff to be reviewed instead of
  waved through as "coverage went up";
* every source module under ``src/mnemon`` is inside the measured package or
  on that pinned omit list — nothing is invisible to the gate by omission of
  a different kind.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_ROOT = REPO_ROOT / "src" / "mnemon"

#: The floor may be RAISED here as coverage improves. Lowering it is a policy
#: amendment (repository-baseline-policy.md §4.2 C3), not a code change.
MINIMUM_FLOOR = 90

#: Individually justified in pyproject.toml's [tool.coverage.run] comments.
#: Widening this set is a scope decision, not a drive-by coverage bump —
#: update the comment there and this list together.
EXPECTED_OMIT = {
    "src/mnemon/dashboard/*",
    "src/mnemon/__main__.py",
    "src/mnemon/llm.py",
}


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_coverage_source_is_the_whole_package() -> None:
    """C1 — ``source`` names the package root, so unimported modules still count."""
    sources = _pyproject()["tool"]["coverage"]["run"]["source"]
    assert sources == ["src/mnemon"], (
        f"coverage source must be exactly the mnemon package, got {sources!r}. "
        "Narrowing it to a submodule or a path measures the tested subset and "
        "reports it as the repository."
    )


def test_coverage_floor_is_enforced_and_never_lowered() -> None:
    """C2 + C3 — the gate exits non-zero below a floor that only ratchets up."""
    fail_under = _pyproject()["tool"]["coverage"]["report"]["fail_under"]
    assert isinstance(fail_under, int), (
        f"fail_under must be a single integer floor, got {fail_under!r}"
    )
    assert fail_under >= MINIMUM_FLOOR, (
        f"coverage floor {fail_under} is below the ratchet {MINIMUM_FLOOR}. "
        "A floor is raised as coverage improves and never lowered to make a "
        "change pass (repository-baseline-policy.md §4.2 C3)."
    )


def test_coverage_omit_matches_the_pinned_justified_set() -> None:
    """A shrunk denominator is a narrowing this test forces into review."""
    omit = set(_pyproject()["tool"]["coverage"]["run"].get("omit", []))
    added = omit - EXPECTED_OMIT
    removed = EXPECTED_OMIT - omit
    assert not added, (
        f"coverage omit gained unreviewed entries: {sorted(added)}. "
        "Each omitted path removes files from the denominator, raising the "
        "reported figure without adding a test — update EXPECTED_OMIT here "
        "alongside its justification in pyproject.toml if this is deliberate."
    )
    assert not removed, (
        f"coverage omit lost tracked entries: {sorted(removed)}. If these "
        "modules are now measured, good — also narrow EXPECTED_OMIT here."
    )


def test_every_source_module_is_inside_the_measured_package_or_pinned_omit() -> None:
    """No source file is invisible to the gate by living outside src/mnemon."""
    src = REPO_ROOT / "src"
    stray = sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in src.rglob("*.py")
        if PACKAGE_ROOT not in p.parents and p != PACKAGE_ROOT
    )
    assert not stray, (
        f"source modules outside the measured package are invisible to the "
        f"coverage gate: {stray}"
    )


def test_no_cov_fail_under_flag_shadows_the_pyproject_gate() -> None:
    """A CI-passed --cov-fail-under could silently override pyproject.toml's."""
    for workflow in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        for match in re.findall(r"--cov-fail-under=(\d+)", text):
            assert int(match) >= MINIMUM_FLOOR, (
                f"{workflow.name} passes --cov-fail-under={match} directly, "
                "bypassing the pyproject.toml ratchet this test protects."
            )
