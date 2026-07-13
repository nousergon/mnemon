# Contributing to mnemon

mnemon is in alpha. Bug reports, PRs, and design discussion are all welcome. Issues that reproduce on a fresh `pip install` get prioritized.

## Quick start

```bash
git clone https://github.com/nousergon/mnemon.git
cd mnemon
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not integration"
```

You should see ~1160 tests pass in a few seconds. The 4 integration tests are excluded from this run because they bind a local socket — run them separately with `pytest -m integration` if you're touching `serve-remote` or auth.

## What to work on

- **Roadmap / open scope:** see [`README.md`](README.md) for the user-facing pitch and the public roadmap; design discussions happen in [GitHub Issues](https://github.com/nousergon/mnemon/issues).
- **Good first issues:** anything tagged [`good first issue`](https://github.com/nousergon/mnemon/labels/good%20first%20issue) — usually doc fixes, test coverage, or minor CLI polish.
- **New features:** open an issue first to align on shape before writing code. mnemon's interface is intentionally narrow (19 MCP tools) and adding to it has compounding cost.

## Style

- **Linter:** `ruff check src/ tests/` — see `[tool.ruff]` in `pyproject.toml` for active rules. Run before pushing.
- **Type hints:** required on new public functions and class methods. Internal helpers can skip them when the type is obvious from context.
- **Tests:** any behavior change ships with a test. Pure-Python tests live in `tests/`; tests that spawn `serve-remote` go behind the `integration` marker (see `tests/test_integration_remote.py` for the pattern).
- **Docstrings:** load-bearing public APIs get a docstring explaining *why* the behavior exists, not what the code does. The `persistent_sessions.py` and `auth.py` docstrings are good references.

## Pull requests

1. Branch from `main`. Never push directly to `main` — branch protection rejects it.
2. Open the PR early (draft is fine) and push commits as you go.
3. PR title matches Conventional Commits style: `feat(...)`, `fix(...)`, `chore(...)`, `docs(...)`, `ci(...)`, `refactor(...)`. CI uses this for changelog suggestions.
4. **Bump the version** (`pyproject.toml` + `src/mnemon/__init__.py`) for any non-trivial change. We avoid the "merged seven PRs without a version bump" problem that bit us in 2026-04-21 — see `CHANGELOG.md` and the release-discipline note in `README.md`.
5. CI must be green before merge. The matrix runs Python 3.10 / 3.12 / 3.13.

## Reporting bugs

Open a [GitHub Issue](https://github.com/nousergon/mnemon/issues/new) with:
- The exact command you ran
- The output of `mnemon --version` and `mnemon doctor`
- Your OS + Python version
- Anything you've already tried

If `mnemon doctor` itself fails, include `MNEMON_LOG_LEVEL=debug` output.

## Security

Do **not** open public issues for security reports. See [`SECURITY.md`](SECURITY.md) for the disclosure path (private GitHub Security Advisory or `security@nousergon.ai`).

## License

By submitting a contribution, you agree it is licensed under the project's [MIT license](LICENSE).
