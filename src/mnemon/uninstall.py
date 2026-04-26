"""Remove all mnemon state from this machine.

Designed for users who want to test the full install-from-scratch
experience — e.g. to validate the `mnemon setup` happy path after
previously running `mnemon upgrade web` or a historical setup that
left stale config behind. Also useful as a clean exit for users who
decide mnemon isn't for them.

What gets removed
-----------------
- ``~/.mnemon/`` — vault (SQLite + vectors), archive/, remote_url,
  local_token, models cache. Everything. Irreversibly.
- ``claude mcp remove mnemon`` — drops the Claude Code MCP registration
  whether it was stdio or http.
- ``~/.claude/settings.json`` — removes mnemon hook entries and the
  (never-effective, but confusing) ``mcpServers.mnemon`` entry.
- ``~/.cursor/mcp.json`` — removes ``mcpServers.mnemon``.
- Claude Desktop config — removes ``mcpServers.mnemon``.

What does NOT get removed
-------------------------
- The ``mnemon-memory`` Python package itself (``pip uninstall`` is
  the user's package manager's job; we never touch it).
- Fly.io apps — user-owned infra. If the user had deployed web via
  ``mnemon upgrade web``, they should run ``mnemon downgrade local
  --destroy-fly-app`` first to preserve their memories (S3 backup)
  and destroy the app cleanly. We warn loudly if this state is
  detected.
- S3 bucket contents — user-owned; deleting someone's bucket data
  is never our call.
- claude.ai / Claude mobile MCP entries — live in Anthropic's UI,
  can't be auto-removed. Output tells the user to remove manually.

CLI
---
``mnemon uninstall [--yes] [--keep-vault]``

``--yes`` bypasses the confirmation prompt (for scripted teardowns).
``--keep-vault`` removes client configs and ``claude mcp`` registration
but preserves ``~/.mnemon/`` — useful for "I want to redo setup without
losing my memories."
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


class UninstallError(Exception):
    """Raised when uninstall cannot proceed. Message is user-facing."""


MNEMON_DIR_DEFAULT = Path.home() / ".mnemon"
REMOTE_URL_FILE = MNEMON_DIR_DEFAULT / "remote_url"


def _mnemon_dir() -> Path:
    """Honor MNEMON_VAULT_DIR if the user relocated the vault."""
    override = os.environ.get("MNEMON_VAULT_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return MNEMON_DIR_DEFAULT


def _claude_desktop_config_path() -> Path:
    """Same logic as setup._claude_desktop_config_path — duplicated here
    to avoid a circular import and to keep uninstall self-contained."""
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
        return (
            Path.home()
            / "AppData"
            / "Roaming"
            / "Claude"
            / "claude_desktop_config.json"
        )
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def detect_claude_ai_mnemon() -> bool:
    """True if ``claude mcp list`` shows a mnemon entry that originated
    from the claude.ai web UI (Settings → Connected Apps).

    Claude Code's CLI tags such entries with a ``claude.ai`` prefix:

        claude.ai mnemon: https://mnemon-memory.fly.dev/mcp - ✓ Connected

    Those registrations live in the user's Anthropic account (synced
    from claude.ai), not in any local filesystem that
    ``claude mcp remove --scope user`` can reach. This command can
    detect them but cannot remove them — the user has to delete the
    mnemon entry in claude.ai's web UI manually.

    Returns False if the ``claude`` CLI isn't on PATH, list fails, or
    no claude.ai-scoped mnemon entry is present.
    """
    try:
        out = subprocess.run(
            ["claude", "mcp", "list"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    if out.returncode != 0:
        return False
    for line in out.stdout.splitlines():
        # Match exactly the prefix Claude Code emits for claude.ai-synced
        # entries. Narrow match on purpose — don't false-positive on a
        # server literally named "mnemon" in some other context.
        if line.lstrip().startswith("claude.ai mnemon:") or line.lstrip().startswith("claude.ai mnemon "):
            return True
    return False


def _detect_state() -> dict:
    """Return a dict describing what mnemon state is present on this
    machine. Used to build the confirmation prompt and the summary."""
    mdir = _mnemon_dir()
    state = {
        "mnemon_dir": mdir if mdir.exists() else None,
        "remote_url_configured": (mdir / "remote_url").exists(),
        "claude_code_settings": (
            Path.home() / ".claude" / "settings.json"
        )
        if (Path.home() / ".claude" / "settings.json").exists()
        else None,
        "cursor_config": (Path.home() / ".cursor" / "mcp.json")
        if (Path.home() / ".cursor" / "mcp.json").exists()
        else None,
        "claude_desktop_config": _claude_desktop_config_path()
        if _claude_desktop_config_path().exists()
        else None,
        "claude_ai_registration": detect_claude_ai_mnemon(),
    }
    return state


def _confirm(prompt: str) -> bool:
    """y/N confirmation. Stdin-aware: no TTY → False (force --yes)."""
    if not sys.stdin.isatty():
        return False
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _claude_mcp_remove() -> str | None:
    """Run ``claude mcp remove mnemon``. Returns a status line or None
    if the claude CLI isn't on PATH."""
    try:
        out = subprocess.run(
            ["claude", "mcp", "remove", "--scope", "user", "mnemon"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if out.returncode == 0:
        return "  claude mcp:    user-scope mnemon registration removed"
    # Non-zero is the common case — the user-scope registration was
    # already gone (fresh setup, prior uninstall, or claude.ai-only).
    # Surface stderr if it looks like a real error (anything not in
    # the "not found" family) so unusual failures aren't swallowed.
    detail = (out.stderr or out.stdout or "").strip()
    if detail and not any(
        marker in detail.lower()
        for marker in ("not found", "no such", "no mcp server", "does not exist")
    ):
        first_line = detail.splitlines()[0]
        return f"  claude mcp:    skipped — {first_line}"
    return "  claude mcp:    no user-scope mnemon registration to remove"


def _strip_from_json(path: Path, keys_to_strip: dict[str, list[str]]) -> bool:
    """Remove nested keys from a JSON file. Returns True if anything
    was actually removed (i.e. the file was modified).

    ``keys_to_strip`` maps top-level key → list of subkeys to delete.
    Top-level keys that become empty after stripping are also removed
    so the resulting file doesn't accumulate empty containers.
    """
    import json

    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict):
        return False

    changed = False
    for top_key, subkeys in keys_to_strip.items():
        container = data.get(top_key)
        if not isinstance(container, dict):
            continue
        for sub in subkeys:
            if sub in container:
                del container[sub]
                changed = True
        if not container:
            del data[top_key]
            changed = True

    # Also strip mnemon hook entries from Claude Code settings.json hooks.
    # Hooks are a nested shape: hooks.<EventName>[{hooks: [{command: ...}]}].
    # We delete any inner hook dict whose command references mnemon.
    hooks = data.get("hooks")
    if isinstance(hooks, dict):
        for event in list(hooks.keys()):
            entries = hooks[event]
            if not isinstance(entries, list):
                continue
            filtered = []
            for entry in entries:
                inner = [
                    h
                    for h in entry.get("hooks", [])
                    if "mnemon" not in (h.get("command") or "")
                ]
                if inner:
                    filtered.append({**entry, "hooks": inner})
                else:
                    changed = True
            if filtered:
                hooks[event] = filtered
            else:
                del hooks[event]
                changed = True
        if not hooks:
            del data["hooks"]
            changed = True

    if changed:
        path.write_text(__import__("json").dumps(data, indent=2) + "\n")
    return changed


def uninstall(*, yes: bool = False, keep_vault: bool = False) -> str:
    """Remove all mnemon state from this machine.

    Returns a user-facing summary. Raises :class:`UninstallError` only
    for genuine failures — the "nothing to remove" case is handled as
    a normal return with a no-op summary.
    """
    state = _detect_state()

    # Warn if this looks like a web install — user should downgrade
    # first to preserve their remote vault.
    if state["remote_url_configured"] and not keep_vault:
        warning = (
            "⚠ A remote URL is configured at ~/.mnemon/remote_url. "
            "This machine may have a live Fly deployment whose vault "
            "you are about to wipe locally. If you want to preserve "
            "that data:\n"
            "  1. Abort now (Ctrl-C).\n"
            "  2. Run `mnemon downgrade local --destroy-fly-app` to "
            "pull the remote vault back and destroy the Fly app.\n"
            "  3. Then run `mnemon uninstall` to remove everything.\n"
        )
        print(warning, file=sys.stderr)

    # Warn loudly if there is a claude.ai-synced mnemon MCP registration.
    # This command literally cannot remove it — it lives in the user's
    # Anthropic account, not on this machine. Surfacing the detection
    # prominently lets the user act on it; otherwise `claude mcp list`
    # after uninstall would still show mnemon and they'd (reasonably)
    # think the command didn't work.
    if state["claude_ai_registration"]:
        ca_warning = (
            "⚠ claude.ai-synced mnemon MCP detected.\n"
            "  `claude mcp list` shows a `claude.ai mnemon: …` entry. "
            "This registration lives in your Anthropic account and is "
            "synced to Claude Code from claude.ai — it CANNOT be removed "
            "by `mnemon uninstall` or any `claude mcp remove` command.\n"
            "  To finish the uninstall, open claude.ai → Settings → "
            "Connected Apps and remove the mnemon entry there.\n"
            "  If you leave it in place: it will shadow any local stdio "
            "registration from a future `mnemon setup`, and Claude Code "
            "will keep talking to the remote vault.\n"
        )
        print(ca_warning, file=sys.stderr)

    # Build the plan so the user knows what's about to happen.
    plan_lines = ["mnemon uninstall will remove:"]
    if state["mnemon_dir"] and not keep_vault:
        plan_lines.append(
            f"  • Vault directory: {state['mnemon_dir']} "
            "(SQLite vault, vectors, archive, models, config files)"
        )
    elif state["mnemon_dir"] and keep_vault:
        plan_lines.append(
            f"  • Vault directory: {state['mnemon_dir']} — KEPT "
            "(--keep-vault specified)"
        )
    plan_lines.append(
        "  • Claude Code MCP registration (`claude mcp remove mnemon`)"
    )
    if state["claude_code_settings"]:
        plan_lines.append(
            f"  • mnemon hook + mcpServers entries in {state['claude_code_settings']}"
        )
    if state["cursor_config"]:
        plan_lines.append(
            f"  • mnemon entry in {state['cursor_config']}"
        )
    if state["claude_desktop_config"]:
        plan_lines.append(
            f"  • mnemon entry in {state['claude_desktop_config']}"
        )
    plan_lines.append("")
    plan_lines.append("What this command does NOT touch:")
    plan_lines.append("  • The `mnemon-memory` Python package (use `pip uninstall` separately)")
    plan_lines.append("  • Any Fly.io apps you own")
    plan_lines.append("  • Any S3 bucket contents")
    plan_lines.append("  • claude.ai / Claude mobile MCP entries (remove manually in Anthropic's UI)")

    print("\n".join(plan_lines), file=sys.stderr)

    if not yes:
        if not _confirm("\nProceed? [y/N]: "):
            return "Uninstall aborted by user."

    # Execute plan.
    summary: list[str] = ["Uninstall complete."]

    # 1. Claude Code MCP registration.
    line = _claude_mcp_remove()
    if line:
        summary.append(line)
    else:
        summary.append(
            "  claude mcp:    skipped (claude CLI not on PATH)"
        )

    # 2. Claude Code settings.json — strip mnemon mcpServers + hooks.
    cc_settings = Path.home() / ".claude" / "settings.json"
    if _strip_from_json(
        cc_settings, {"mcpServers": ["mnemon"]}
    ):
        summary.append(
            f"  Claude Code:   scrubbed mnemon entries in {cc_settings}"
        )

    # 3. Cursor.
    cursor = Path.home() / ".cursor" / "mcp.json"
    if _strip_from_json(cursor, {"mcpServers": ["mnemon"]}):
        summary.append(
            f"  Cursor:        scrubbed mnemon entry in {cursor}"
        )

    # 4. Claude Desktop.
    cdesktop = _claude_desktop_config_path()
    if _strip_from_json(cdesktop, {"mcpServers": ["mnemon"]}):
        summary.append(
            f"  Claude Desktop: scrubbed mnemon entry in {cdesktop}"
        )

    # 5. Vault directory.
    if not keep_vault:
        mdir = _mnemon_dir()
        if mdir.exists():
            try:
                shutil.rmtree(mdir)
                summary.append(f"  Vault:         removed {mdir}")
            except OSError as exc:
                raise UninstallError(
                    f"Failed to remove vault directory {mdir}: {exc}"
                ) from exc

    summary.extend(
        [
            "",
            "Next steps:",
            "  • Restart Claude Code / Cursor / Claude Desktop to drop the cached MCP connections.",
        ]
    )
    # Promote the claude.ai note to its own bullet when we know one
    # actually exists. Otherwise it reads as a hypothetical reminder
    # and users skip it.
    if state["claude_ai_registration"]:
        summary.append(
            "  • ⚠ REQUIRED: remove the mnemon MCP entry in claude.ai → "
            "Settings → Connected Apps. `mnemon uninstall` cannot touch "
            "claude.ai-synced registrations (they live in your Anthropic "
            "account, not on this machine)."
        )
        summary.append(
            "  • Then remove the mnemon entry in the Claude mobile app "
            "(Settings → Connected Apps) if you use it."
        )
    else:
        summary.append(
            "  • If you use claude.ai or the Claude mobile app, remove "
            "any mnemon entry there manually (Settings → Connected Apps)."
        )
    summary.extend(
        [
            "  • `pip uninstall mnemon-memory` to remove the Python package itself.",
            "  • Re-run `mnemon setup` at any time to reinstall from scratch.",
        ]
    )
    return "\n".join(summary)
