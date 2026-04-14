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
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path


MNEMON_DIR = Path.home() / ".mnemon"
LOCAL_TOKEN_FILE = MNEMON_DIR / "local_token"
REMOTE_URL_FILE = MNEMON_DIR / "remote_url"


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


def setup_claude_code(*, remote_url: str | None = None, token: str | None = None) -> str:
    """Configure Claude Code hooks (and optionally MCP server).

    When ``remote_url`` is provided, writes the URL and token to
    ``~/.mnemon/`` config files and adds a SessionStart pre-warm hook.
    The hooks themselves read these files at runtime via ``_remote_client``.

    When ``remote_url`` is not provided, configures a local stdio MCP server
    for development/testing. Hooks will attempt to use remote if
    ``~/.mnemon/remote_url`` exists, otherwise fall back to local.
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    settings = _read_json(settings_path)

    lines = []

    if remote_url:
        _ensure_remote_url(remote_url)
        # Call for side-effect only (writes ~/.mnemon/local_token); the
        # Claude-Code path doesn't need the token string back — hooks
        # read the file directly at invocation time.
        _ensure_local_token(token)
        lines.append(f"  Remote URL: {remote_url}")
        lines.append(f"  Token file: {LOCAL_TOKEN_FILE} (chmod 600)")

        # Remove stdio MCP server — hooks use remote now
        if "mcpServers" in settings and "mnemon" in settings["mcpServers"]:
            del settings["mcpServers"]["mnemon"]
            if not settings["mcpServers"]:
                del settings["mcpServers"]
    else:
        # Local-only mode: add stdio MCP server
        if "mcpServers" not in settings:
            settings["mcpServers"] = {}
        settings["mcpServers"]["mnemon"] = _mcp_config()
        lines.append("  MCP server: mnemon (stdio, local)")

    # Hooks (always written — they detect remote/local from config files)
    hooks = _hooks_config(remote_url=remote_url)
    if "hooks" not in settings:
        settings["hooks"] = {}
    settings["hooks"]["UserPromptSubmit"] = hooks["UserPromptSubmit"]
    settings["hooks"]["Stop"] = hooks["Stop"]
    if "SessionStart" in hooks:
        settings["hooks"]["SessionStart"] = hooks["SessionStart"]
        lines.append("  SessionStart: pre-warm polling (90s background)")

    lines.append("  UserPromptSubmit: context-surfacing (8s)")
    lines.append("  Stop: session-extractor (30s), handoff-generator (30s)")

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
        _ensure_remote_url(remote_url)
        local_token = _ensure_local_token(token)
        config["mcpServers"]["mnemon"] = {
            "url": remote_url,
            "headers": {
                "Authorization": f"Bearer {local_token}",
            },
        }
        mode = "remote"
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
    """Configure Claude Code hooks only (no MCP server)."""
    settings_path = Path.home() / ".claude" / "settings.json"
    settings = _read_json(settings_path)

    if remote_url:
        _ensure_remote_url(remote_url)
        _ensure_local_token(token)

    hooks = _hooks_config(remote_url=remote_url)
    if "hooks" not in settings:
        settings["hooks"] = {}
    settings["hooks"]["UserPromptSubmit"] = hooks["UserPromptSubmit"]
    settings["hooks"]["Stop"] = hooks["Stop"]
    if "SessionStart" in hooks:
        settings["hooks"]["SessionStart"] = hooks["SessionStart"]

    _write_json(settings_path, settings)

    lines = [
        f"Hooks configured at {settings_path}",
        "  UserPromptSubmit: context-surfacing (8s)",
        "  Stop: session-extractor (30s), handoff-generator (30s)",
    ]
    if "SessionStart" in hooks:
        lines.append("  SessionStart: pre-warm polling (90s background)")
    lines.append("Restart Claude Code to activate.")
    return "\n".join(lines)


TARGETS = {
    "claude-code": setup_claude_code,
    "cursor": setup_cursor,
    "gemini": setup_gemini,
    "hooks": setup_hooks,
}


def _parse_setup_args(args: list[str]) -> dict:
    """Parse --remote-url and --token flags from CLI args."""
    result: dict[str, str | None] = {"remote_url": None, "token": None}
    i = 0
    while i < len(args):
        if args[i] == "--remote-url" and i + 1 < len(args):
            result["remote_url"] = args[i + 1]
            i += 2
        elif args[i] == "--token" and i + 1 < len(args):
            result["token"] = args[i + 1]
            i += 2
        else:
            i += 1
    return result


def run_setup(target: str, args: list[str] | None = None) -> str:
    """Run setup for the given target. Returns status message."""
    if target not in TARGETS:
        valid = ", ".join(TARGETS.keys())
        return f"Unknown target: {target}\nValid targets: {valid}"

    parsed = _parse_setup_args(args or [])
    func = TARGETS[target]

    # Gemini doesn't accept remote_url/token kwargs
    if target == "gemini":
        return func()

    return func(remote_url=parsed["remote_url"], token=parsed["token"])
