"""Setup integrations — configure Claude Code, Cursor, Gemini CLI, and hooks.

Auto-detects the Python interpreter and generates MCP server configs
that work whether mnemon is installed globally or in a virtualenv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _python_path() -> str:
    """Return the path to the Python interpreter running mnemon."""
    return sys.executable


def _mcp_config() -> dict:
    """Generate MCP server config for the current Python environment."""
    return {
        "command": _python_path(),
        "args": ["-m", "mnemon", "serve"],
    }


def _hooks_config() -> dict:
    """Generate Claude Code hooks config."""
    py = _python_path()
    return {
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


def setup_claude_code() -> str:
    """Configure Claude Code MCP server + hooks."""
    settings_path = Path.home() / ".claude" / "settings.json"
    settings = _read_json(settings_path)

    # MCP server
    if "mcpServers" not in settings:
        settings["mcpServers"] = {}
    settings["mcpServers"]["mnemon"] = _mcp_config()

    # Hooks
    hooks = _hooks_config()
    if "hooks" not in settings:
        settings["hooks"] = {}
    settings["hooks"]["UserPromptSubmit"] = hooks["UserPromptSubmit"]
    settings["hooks"]["Stop"] = hooks["Stop"]

    _write_json(settings_path, settings)

    return (
        f"Claude Code configured at {settings_path}\n"
        "  MCP server: mnemon (stdio)\n"
        "  Hooks: context-surfacing (8s), session-extractor (30s), handoff-generator (30s)\n"
        "Restart Claude Code to activate."
    )


def setup_cursor() -> str:
    """Configure Cursor MCP server."""
    cursor_path = Path.home() / ".cursor" / "mcp.json"
    config = _read_json(cursor_path)

    if "mcpServers" not in config:
        config["mcpServers"] = {}
    config["mcpServers"]["mnemon"] = _mcp_config()

    _write_json(cursor_path, config)

    return (
        f"Cursor MCP configured at {cursor_path}\n"
        "Restart Cursor to activate."
    )


def setup_gemini() -> str:
    """Show Gemini CLI MCP configuration."""
    config = json.dumps({"mnemon": _mcp_config()}, indent=2)
    return (
        "Add this to your Gemini CLI MCP config:\n\n"
        f"{config}"
    )


def setup_hooks() -> str:
    """Configure Claude Code hooks only (no MCP server)."""
    settings_path = Path.home() / ".claude" / "settings.json"
    settings = _read_json(settings_path)

    hooks = _hooks_config()
    if "hooks" not in settings:
        settings["hooks"] = {}
    settings["hooks"]["UserPromptSubmit"] = hooks["UserPromptSubmit"]
    settings["hooks"]["Stop"] = hooks["Stop"]

    _write_json(settings_path, settings)

    return (
        f"Hooks configured at {settings_path}\n"
        "  UserPromptSubmit → context-surfacing (8s)\n"
        "  Stop → session-extractor (30s)\n"
        "  Stop → handoff-generator (30s)\n"
        "Restart Claude Code to activate."
    )


TARGETS = {
    "claude-code": setup_claude_code,
    "cursor": setup_cursor,
    "gemini": setup_gemini,
    "hooks": setup_hooks,
}


def run_setup(target: str) -> str:
    """Run setup for the given target. Returns status message."""
    if target not in TARGETS:
        valid = ", ".join(TARGETS.keys())
        return f"Unknown target: {target}\nValid targets: {valid}"
    return TARGETS[target]()
