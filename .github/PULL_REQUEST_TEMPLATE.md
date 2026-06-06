## What & why

<!-- What does this change and why? Link any related issue. -->

## Checklist

- [ ] Tests added/updated for the behavior change
- [ ] `pytest` passes locally and coverage stays ≥ 80%
- [ ] `ruff check src/ tests/` is clean for files I touched
- [ ] `CHANGELOG.md` updated (under `[Unreleased]`)
- [ ] Schema changes (if any) are **additive** — new nullable columns + migration, never rename/drop
- [ ] No secrets, vault data, or `private/` content committed
- [ ] Fail-loud preserved — no new silent `except: pass` swallows

## Test plan

<!-- How you verified this works. -->
