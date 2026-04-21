"""Setup integrations — configure Claude Code, Cursor, Gemini CLI, and hooks.

Auto-detects the Python interpreter and generates MCP server configs
that work whether mnemon is installed globally or in a virtualenv.

Phase 3 unification: setup now requires ``--remote-url`` for Claude Code
and Cursor targets, writes the URL + token to ``~/.mnemon/`` config files,
and generates hook configs that hit the remote vault. The ``--remote-url``
flag has **no default** — this is the highest-risk guardrail in the
unification plan. If it defaulted to the author's Fly URL, a forked user
running ``mnemon setup`` would accidentally send their memories into the
wrong vault.

P0 simplification (2026-04-21, mnemon memory #109): hooks are only ever
written when ``--remote-url`` is supplied, because the hook code path
(``hooks/_remote_client.py``) talks exclusively over HTTP. Installing
hooks against a local stdio MCP produced silent config errors on every
prompt. A remote endpoint is now validated end-to-end before any config
is written, and a summary + ``mnemon doctor`` run fire at the end of
every successful setup so config gaps surface loudly. See
``private/mnemon-simplification-plan-260421.md``.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path


MNEMON_DIR = Path.home() / ".mnemon"
LOCAL_TOKEN_FILE = MNEMON_DIR / "local_token"
REMOTE_URL_FILE = MNEMON_DIR / "remote_url"

# Timeout for the setup-time remote preflight. Generous vs. the hook's
# 8s budget because a brand-new Fly machine may be cold and the first
# FastEmbed load adds ~15s to the first call.
REMOTE_PREFLIGHT_TIMEOUT_SEC = 30.0


class SetupError(Exception):
    """Raised when setup cannot proceed — surfaces a user-facing message."""


def _python_path() -> str:
    """Return the path to the Python interpreter running mnemon."""
    return sys.executable


def _mcp_config() -> dict:
    """Generate MCP server config for the current Python environment."""
    return {
        "command": _python_path(),
        "args": ["-m", "mnemon", "serve"],
    }


def _hooks_config(remote_url: str | None = None) -> dict:
    """Generate Claude Code hooks config.

    When ``remote_url`` is provided, adds a SessionStart polling hook that
    pre-warms the Fly machine so subsequent hooks don't pay cold-start
    latency. The polling hook runs in the background and exits 0 in all
    cases — it never blocks session startup.
    """
    py = _python_path()
    hooks: dict = {
        "UserPromptSubmit": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{py} -m mnemon.hooks.context_surfacing",
                        "timeout": 8,
                    },
                ],
            },
        ],
        "Stop": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{py} -m mnemon.hooks.session_extractor",
                        "timeout": 30,
                    },
                    {
                        "type": "command",
                        "command": f"{py} -m mnemon.hooks.handoff_generator",
                        "timeout": 30,
                    },
                ],
            },
        ],
    }

    if remote_url:
        # Extract base URL (strip /mcp suffix for health endpoint)
        base_url = remote_url.rstrip("/")
        if base_url.endswith("/mcp"):
            base_url = base_url[:-4]
        health_url = f"{base_url}/health"

        hooks["SessionStart"] = [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": (
                            f"for i in $(seq 1 60); do "
                            f"curl -sf -m 2 {health_url} > /dev/null 2>&1 && exit 0; "
                            f"sleep 1; done; exit 0"
                        ),
                        "timeout": 90,
                    },
                ],
            },
        ]

    return hooks


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning empty dict if missing or invalid."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    """Write JSON to a file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _ensure_remote_url(remote_url: str) -> str:
    """Write remote URL to ~/.mnemon/remote_url and return it."""
    MNEMON_DIR.mkdir(parents=True, exist_ok=True)
    REMOTE_URL_FILE.write_text(remote_url)
    return remote_url


def _ensure_local_token(token: str | None = None) -> str:
    """Ensure a local token exists at ~/.mnemon/local_token.

    If ``token`` is provided, writes it. If the file already exists and
    no token is provided, keeps the existing value. If neither, generates
    a new token.

    Returns the token value (for display/logging — never log the full
    token in production, but setup is a one-time interactive command).
    """
    MNEMON_DIR.mkdir(parents=True, exist_ok=True)

    if token:
        LOCAL_TOKEN_FILE.write_text(token)
        LOCAL_TOKEN_FILE.chmod(0o600)
        return token

    if LOCAL_TOKEN_FILE.exists():
        existing = LOCAL_TOKEN_FILE.read_text().strip()
        if existing:
            return existing

    new_token = secrets.token_urlsafe(32)
    LOCAL_TOKEN_FILE.write_text(new_token)
    LOCAL_TOKEN_FILE.chmod(0o600)
    return new_token


def _strip_mnemon_hooks(hooks: dict) -> None:
    """Remove any hook entries whose command references ``mnemon.hooks.*``.

    Used when re-running setup in local mode to clean up stale remote-era
    hook entries that would otherwise keep firing and surfacing config
    errors. Mutates ``hooks`` in place and drops keys whose list is empty
    after filtering.
    """

    def _keeps(entry: dict) -> dict | None:
        inner = [
            h
            for h in entry.get("hooks", [])
            if "mnemon.hooks." not in (h.get("command") or "")
        ]
        if not inner:
            return None
        return {**entry, "hooks": inner}

    for event in ("UserPromptSubmit", "Stop", "SessionStart"):
        entries = hooks.get(event)
        if not isinstance(entries, list):
            continue
        filtered = [kept for kept in (_keeps(e) for e in entries) if kept]
        if filtered:
            hooks[event] = filtered
        else:
            del hooks[event]


def _preflight_remote_endpoint(remote_url: str, local_token: str) -> None:
    """Validate that ``remote_url`` + token round-trip a tool call.

    Runs the real MCP ``memory_status`` call through the hook client path
    so we exercise auth, transport, and server readiness — exactly what the
    hooks will do at runtime. Env vars are set directly (not the config
    files) so we do not leave partial state behind on failure.

    Raises :class:`SetupError` with a concrete message on any failure.
    """
    # Import late: _remote_client pulls in the MCP SDK, which we do not
    # want to load for users running local-only setup.
    from .hooks._remote_client import (
        RemoteClientConfigError,
        call_tool_sync,
    )

    prior_url = os.environ.get("MNEMON_REMOTE_URL")
    prior_token = os.environ.get("MNEMON_LOCAL_TOKEN")
    os.environ["MNEMON_REMOTE_URL"] = remote_url
    os.environ["MNEMON_LOCAL_TOKEN"] = local_token
    try:
        call_tool_sync(
            "memory_status",
            {},
            timeout=REMOTE_PREFLIGHT_TIMEOUT_SEC,
            client_label="mnemon-setup",
        )
    except RemoteClientConfigError as exc:
        raise SetupError(f"Remote preflight failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — any failure aborts setup
        raise SetupError(
            f"Remote endpoint {remote_url} did not respond to "
            f"memory_status within {REMOTE_PREFLIGHT_TIMEOUT_SEC:.0f}s. "
            f"Root cause: {type(exc).__name__}: {exc}. "
            "No configuration was written."
        ) from exc
    finally:
        if prior_url is None:
            os.environ.pop("MNEMON_REMOTE_URL", None)
        else:
            os.environ["MNEMON_REMOTE_URL"] = prior_url
        if prior_token is None:
            os.environ.pop("MNEMON_LOCAL_TOKEN", None)
        else:
            os.environ["MNEMON_LOCAL_TOKEN"] = prior_token


def _register_claude_code_mcp(remote_url: str, local_token: str) -> None:
    """Register the mnemon MCP server with Claude Code via the `claude` CLI.

    Claude Code loads MCP server registrations from its own config
    (managed by ``claude mcp add``), not from ``settings.json.mcpServers``.
    Writing an HTTP transport entry to ``settings.json`` has no effect — the
    CLI silently ignores it. Shelling out to ``claude mcp add`` is the only
    supported registration path.
    """
    try:
        subprocess.run(
            ["claude", "mcp", "remove", "--scope", "user", "mnemon"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            [
                "claude", "mcp", "add",
                "--scope", "user",
                "--transport", "http",
                "mnemon",
                remote_url,
                "--header", f"Authorization: Bearer {local_token}",
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The `claude` CLI was not found on PATH. Install Claude Code "
            "before running `mnemon setup claude-code` with --remote-url."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode(errors="replace").strip()
        raise RuntimeError(
            f"`claude mcp add` failed (exit {exc.returncode}): {stderr}"
        ) from exc


def setup_claude_code(*, remote_url: str | None = None, token: str | None = None) -> str:
    """Configure Claude Code (MCP server, and hooks only when remote).

    When ``remote_url`` is provided:
      - Preflight the endpoint (aborts setup if unreachable).
      - Write URL + token to ``~/.mnemon/`` config files.
      - Register the HTTP MCP server via ``claude mcp add``.
      - Install UserPromptSubmit / Stop / SessionStart hooks.

    When ``remote_url`` is not provided:
      - Register a local stdio MCP server in ``settings.json``.
      - **Hooks are NOT installed** — the hook code path is HTTP-only
        (``hooks/_remote_client.py``) and has no local fallback. Writing
        them against stdio would produce silent config errors on every
        prompt. Users who want hooks should run ``mnemon upgrade web``
        (forthcoming) or pass ``--remote-url``.
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    settings = _read_json(settings_path)

    lines = []

    if remote_url:
        local_token = _ensure_local_token(token)
        _preflight_remote_endpoint(remote_url, local_token)
        _ensure_remote_url(remote_url)
        lines.append(f"  Remote URL: {remote_url}")
        lines.append(f"  Token file: {LOCAL_TOKEN_FILE} (chmod 600)")
        lines.append(f"  Preflight:  OK (memory_status round-trip < {REMOTE_PREFLIGHT_TIMEOUT_SEC:.0f}s)")

        _register_claude_code_mcp(remote_url, local_token)
        lines.append("  MCP server: mnemon (http, remote) — registered via `claude mcp add`")

        mcp_servers = settings.get("mcpServers")
        if isinstance(mcp_servers, dict) and "mnemon" in mcp_servers:
            del mcp_servers["mnemon"]
            if not mcp_servers:
                del settings["mcpServers"]
            lines.append("  Cleaned up stale settings.json mcpServers.mnemon entry")

        hooks = _hooks_config(remote_url=remote_url)
        if "hooks" not in settings:
            settings["hooks"] = {}
        settings["hooks"]["UserPromptSubmit"] = hooks["UserPromptSubmit"]
        settings["hooks"]["Stop"] = hooks["Stop"]
        settings["hooks"]["SessionStart"] = hooks["SessionStart"]
        lines.append("  UserPromptSubmit: context-surfacing (8s)")
        lines.append("  Stop: session-extractor (30s), handoff-generator (30s)")
        lines.append("  SessionStart: pre-warm polling (90s background)")
    else:
        if "mcpServers" not in settings:
            settings["mcpServers"] = {}
        settings["mcpServers"]["mnemon"] = _mcp_config()
        lines.append("  MCP server: mnemon (stdio, local)")
        # Strip any previously-installed mnemon hooks so old configs on
        # a machine being downgraded don't keep erroring.
        existing_hooks = settings.get("hooks")
        if isinstance(existing_hooks, dict):
            _strip_mnemon_hooks(existing_hooks)
            if not existing_hooks:
                del settings["hooks"]
        lines.append("  Hooks:      skipped (local-only setup)")
        lines.append(
            "              Hooks require a remote vault. Run "
            "`mnemon setup claude-code --remote-url <URL>` or "
            "`mnemon upgrade web` to enable them."
        )

    _write_json(settings_path, settings)

    return (
        f"Claude Code configured at {settings_path}\n"
        + "\n".join(lines)
        + "\nRestart Claude Code to activate."
    )


def setup_cursor(*, remote_url: str | None = None, token: str | None = None) -> str:
    """Configure Cursor MCP server.

    When ``remote_url`` is provided, configures Cursor to use the remote
    mnemon server with bearer token auth. Otherwise falls back to stdio.
    """
    cursor_path = Path.home() / ".cursor" / "mcp.json"
    config = _read_json(cursor_path)

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    if remote_url:
        local_token = _ensure_local_token(token)
        _preflight_remote_endpoint(remote_url, local_token)
        _ensure_remote_url(remote_url)
        config["mcpServers"]["mnemon"] = {
            "url": remote_url,
            "headers": {
                "Authorization": f"Bearer {local_token}",
            },
        }
        mode = "remote (preflight OK)"
    else:
        config["mcpServers"]["mnemon"] = _mcp_config()
        mode = "stdio (local)"

    _write_json(cursor_path, config)

    return (
        f"Cursor MCP configured at {cursor_path}\n"
        f"  Mode: {mode}\n"
        "Restart Cursor to activate."
    )


def setup_gemini() -> str:
    """Show Gemini CLI MCP configuration."""
    config = json.dumps({"mnemon": _mcp_config()}, indent=2)
    return (
        "Add this to your Gemini CLI MCP config:\n\n"
        f"{config}"
    )


def setup_hooks(*, remote_url: str | None = None, token: str | None = None) -> str:
    """Configure Claude Code hooks only (no MCP server).

    Hooks require a reachable remote vault — the hook code path speaks
    HTTP exclusively. Calling this without ``--remote-url`` raises
    :class:`SetupError` rather than writing broken config.
    """
    if not remote_url:
        raise SetupError(
            "`mnemon setup hooks` requires --remote-url. The hook code "
            "path is HTTP-only; without a remote endpoint hooks would "
            "emit a config error on every Claude Code prompt. Use "
            "`mnemon setup claude-code` for a local-only (MCP-only) "
            "install, or pass --remote-url to enable hooks."
        )

    settings_path = Path.home() / ".claude" / "settings.json"
    settings = _read_json(settings_path)

    local_token = _ensure_local_token(token)
    _preflight_remote_endpoint(remote_url, local_token)
    _ensure_remote_url(remote_url)

    hooks = _hooks_config(remote_url=remote_url)
    if "hooks" not in settings:
        settings["hooks"] = {}
    settings["hooks"]["UserPromptSubmit"] = hooks["UserPromptSubmit"]
    settings["hooks"]["Stop"] = hooks["Stop"]
    settings["hooks"]["SessionStart"] = hooks["SessionStart"]

    _write_json(settings_path, settings)

    return "\n".join(
        [
            f"Hooks configured at {settings_path}",
            f"  Remote URL: {remote_url}",
            f"  Preflight:  OK (memory_status round-trip < {REMOTE_PREFLIGHT_TIMEOUT_SEC:.0f}s)",
            "  UserPromptSubmit: context-surfacing (8s)",
            "  Stop: session-extractor (30s), handoff-generator (30s)",
            "  SessionStart: pre-warm polling (90s background)",
            "Restart Claude Code to activate.",
        ]
    )


TARGETS = {
    "claude-code": setup_claude_code,
    "cursor": setup_cursor,
    "gemini": setup_gemini,
    "hooks": setup_hooks,
}


def _parse_setup_args(args: list[str]) -> dict:
    """Parse --remote-url, --token, and --skip-doctor flags from CLI args."""
    result: dict[str, str | bool | None] = {
        "remote_url": None,
        "token": None,
        "skip_doctor": False,
    }
    i = 0
    while i < len(args):
        if args[i] == "--remote-url" and i + 1 < len(args):
            result["remote_url"] = args[i + 1]
            i += 2
        elif args[i] == "--token" and i + 1 < len(args):
            result["token"] = args[i + 1]
            i += 2
        elif args[i] == "--skip-doctor":
            result["skip_doctor"] = True
            i += 1
        else:
            i += 1
    return result


def _next_steps_block(target: str, remote_url: str | None) -> str:
    """Human-readable next-steps footer for setup output."""
    lines = ["", "Next steps:"]
    if target == "claude-code":
        lines.append("  • Restart Claude Code to activate the MCP tools.")
    elif target == "cursor":
        lines.append("  • Restart Cursor to activate the MCP tools.")
    elif target == "hooks":
        lines.append("  • Restart Claude Code to activate the hooks.")

    if remote_url is None and target == "claude-code":
        lines.append(
            "  • Want mobile / claude.ai / cross-device memory? "
            "Run `mnemon upgrade web` (coming soon) or pass "
            "--remote-url <URL> to enable hooks now."
        )
    lines.append("  • Run `mnemon doctor` any time to re-verify the setup.")
    return "\n".join(lines)


def run_setup(target: str, args: list[str] | None = None) -> str:
    """Run setup for the given target, auto-run doctor, return status.

    After a successful setup, ``mnemon doctor`` runs against the
    newly-configured vault (local or remote, auto-detected). A failing
    doctor run is surfaced in the returned message but does not raise —
    the setup itself already succeeded, and the doctor output is what
    the user needs to read to act on the gap.

    Pass ``--skip-doctor`` in ``args`` to suppress the automatic run
    (useful for CI or scripted setups where the caller runs doctor
    separately).
    """
    if target not in TARGETS:
        valid = ", ".join(TARGETS.keys())
        return f"Unknown target: {target}\nValid targets: {valid}"

    parsed = _parse_setup_args(args or [])
    func = TARGETS[target]

    try:
        if target == "gemini":
            primary = func()
        else:
            primary = func(
                remote_url=parsed["remote_url"], token=parsed["token"]
            )
    except SetupError as exc:
        return f"setup failed: {exc}"

    footer = _next_steps_block(target, parsed["remote_url"])

    if parsed["skip_doctor"] or target == "gemini":
        return primary + footer

    # Capture doctor output so setup returns a single cohesive string
    # (tests + CLI wrapper both consume this as one blob).
    import io

    from .doctor import run_doctor

    buf = io.StringIO()
    print("", file=buf)
    print("Running mnemon doctor to verify...", file=buf)
    try:
        doctor_rc = run_doctor(out=buf)
    except Exception as exc:  # noqa: BLE001
        buf.write(
            f"\n(doctor invocation crashed: {type(exc).__name__}: {exc})\n"
        )
        doctor_rc = 1

    if doctor_rc != 0:
        buf.write(
            "\nNOTE: doctor reported issues. Setup files were written, "
            "but the environment is not fully ready — fix the failing "
            "check(s) above before using mnemon.\n"
        )

    return primary + footer + "\n" + buf.getvalue()
