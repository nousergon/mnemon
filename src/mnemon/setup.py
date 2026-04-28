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

    When ``remote_url`` is provided:

    - Adds a SessionStart polling hook that pre-warms the Fly machine so
      subsequent hooks don't pay cold-start latency. Runs up to 60s in
      the background, exits 0 in all cases.
    - Prepends a UserPromptSubmit ``/health`` warm-keeper that resets
      Fly's idle timer on every prompt. Runs *before* context_surfacing
      so Fly is awake before the MCP call. Uses ``curl ... || true`` so
      a slow wake never blocks the prompt. Independent of MCP session
      state — it works even after a cold-stop has invalidated the
      Mcp-Session-Id, which context_surfacing's MCP call cannot.
    """
    py = _python_path()
    user_prompt_hooks: list[dict] = []

    if remote_url:
        # Extract base URL (strip /mcp suffix for health endpoint)
        base_url = remote_url.rstrip("/")
        if base_url.endswith("/mcp"):
            base_url = base_url[:-4]
        health_url = f"{base_url}/health"

        # Warm-keeper: ping /health on every prompt. Resets Fly idle timer
        # so the machine doesn't autostop mid-session, and wakes it on
        # first prompt after idle. `|| true` ensures a slow wake never
        # blocks the prompt.
        user_prompt_hooks.append(
            {
                "type": "command",
                "command": f"curl -fs --max-time 35 {health_url} > /dev/null || true",
                "timeout": 40,
            }
        )

    user_prompt_hooks.append(
        {
            "type": "command",
            "command": f"{py} -m mnemon.hooks.context_surfacing",
            "timeout": 8,
        }
    )

    hooks: dict = {
        "UserPromptSubmit": [
            {
                "matcher": "",
                "hooks": user_prompt_hooks,
            },
        ],
        # PostToolUse auto-mirror — added 2026-04-28 (mnemon 0.6.0rc7)
        # to close the gap surfaced when Claude wrote handoff files to
        # local auto-memory but failed to mirror them to mnemon. Hook
        # fires on Write/Edit/MultiEdit, no-ops unless the touched file
        # matches an auto-memory directory pattern. The hook itself
        # absorbs all errors (exit 0) and surfaces failures via stderr
        # so Claude sees them per feedback_surface_mnemon_unreachable.
        # 12s timeout: the dispatch path is local-fast or remote-fast
        # via the same MemoryClient abstraction; hookd doesn't queue.
        "PostToolUse": [
            {
                "matcher": "Write|Edit|MultiEdit",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{py} -m mnemon.hooks.auto_mirror",
                        "timeout": 12,
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


def _dual_config_info_lines(target_label: str) -> list[str]:
    """Return a list of info lines when a claude.ai-synced mnemon
    registration coexists with the stdio one we just wrote.

    This is **not** a problem state — it's a legitimate dual config.
    When a user has added mnemon via claude.ai (Settings → Connected
    Apps), that registration syncs to every Anthropic-first-party
    client on the same account: Claude Code, Claude Desktop, Claude
    mobile. ``mnemon setup`` writing a stdio entry on top of that just
    means both are configured. In Claude Code + Claude Desktop, the
    web (claude.ai-synced) one takes precedence — that's the default.
    The stdio entry stays on disk and activates automatically if the
    user later removes the claude.ai entry, with no re-run needed.

    Cursor + Gemini CLI don't sync MCP from claude.ai, so this dual
    state can't arise there.

    ``target_label`` is the user-facing client name (e.g.
    ``"Claude Code"``, ``"Claude Desktop"``) used in the message.
    """
    from .uninstall import detect_claude_ai_mnemon

    if not detect_claude_ai_mnemon():
        return []

    return [
        "",
        f"  ℹ Both local + web mnemon configured. {target_label} will "
        "use the web version (claude.ai-synced) by default.",
        "    To switch to local: open claude.ai → Settings → "
        "Connected Apps and remove the mnemon entry. The local stdio "
        "config stays on disk and activates automatically.",
    ]


def _refuse_if_remote_configured(target_label: str) -> None:
    """Raise :class:`SetupError` if ``~/.mnemon/remote_url`` exists.

    Called by local-mode setup paths. Running local setup while a remote
    is still configured leaves the machine in a split state: the stdio
    MCP registration lands, but hooks keep reading the remote URL file
    and routing through ``RemoteMemoryClient``. The user sees inconsistent
    behavior ("Cursor went local but Claude Code still shows Fly").

    Refusing here is the arch-correct "no silent split-brain" answer.
    Users have two clean exits: ``mnemon downgrade local`` (preserves
    web data by pulling from S3) or ``mnemon uninstall`` (wipes all
    mnemon state).
    """
    # Check env var first (highest priority, matches _remote_client
    # resolution order); fall through to the file.
    env_url = os.environ.get("MNEMON_REMOTE_URL", "").strip()
    file_url: str | None = None
    if REMOTE_URL_FILE.exists():
        try:
            file_url = REMOTE_URL_FILE.read_text().strip() or None
        except OSError:
            file_url = None

    if not env_url and not file_url:
        return

    source = (
        f"MNEMON_REMOTE_URL={env_url}"
        if env_url
        else f"{REMOTE_URL_FILE} ({file_url})"
    )
    raise SetupError(
        f"Refusing local-mode `mnemon setup {target_label}` while a "
        f"remote is configured ({source}). "
        "Running local setup now would leave this machine in a split "
        "state — MCP goes local but hooks keep talking to the remote. "
        "Pick one of:\n"
        "  • `mnemon downgrade local` — pull your remote vault back to "
        "local, reconfigure clients, optionally destroy the Fly app.\n"
        "  • `mnemon uninstall` — remove ALL mnemon state from this "
        "machine so you can reinstall from scratch.\n"
        "  • `mnemon setup claude-code --remote-url <URL>` — "
        "reconfigure this client to use the existing remote."
    )


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
    # Import late: the hook client pulls in the MCP SDK transitively,
    # which we do not want to load for users running local-only setup.
    from .hooks._client import RemoteClientConfigError, RemoteMemoryClient

    prior_url = os.environ.get("MNEMON_REMOTE_URL")
    prior_token = os.environ.get("MNEMON_LOCAL_TOKEN")
    os.environ["MNEMON_REMOTE_URL"] = remote_url
    os.environ["MNEMON_LOCAL_TOKEN"] = local_token
    try:
        RemoteMemoryClient().call_tool(
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


def _register_claude_code_mcp_remote(remote_url: str, local_token: str) -> None:
    """Register the mnemon HTTP MCP server with Claude Code.

    Claude Code loads MCP server registrations from its own config
    (managed by ``claude mcp add``), not from ``settings.json.mcpServers``.
    Writing to ``settings.json`` has no effect — the CLI silently ignores
    mnemon entries there. Shelling out to ``claude mcp add`` is the only
    supported registration path for BOTH HTTP and stdio transports.
    """
    _run_claude_mcp_add(
        [
            "--transport", "http",
            "mnemon",
            remote_url,
            "--header", f"Authorization: Bearer {local_token}",
        ]
    )


def _register_claude_code_mcp_stdio() -> None:
    """Register the mnemon stdio MCP server with Claude Code.

    Local-mode counterpart to :func:`_register_claude_code_mcp_remote`.
    Same rationale: settings.json entries are ignored by Claude Code, so
    the only way to make mnemon tools visible to Claude Code locally is
    via ``claude mcp add --transport stdio``.

    This was the missing piece that made local-mode setup silently fail
    for Claude Code users while appearing to succeed (settings.json was
    written, but Claude Code never read it).
    """
    _run_claude_mcp_add(
        [
            "--transport", "stdio",
            "mnemon",
            "--",
            _python_path(),
            "-m",
            "mnemon",
            "serve",
        ]
    )


def _run_claude_mcp_add(add_args: list[str]) -> None:
    """Shared helper: remove any existing mnemon registration, then add
    with the supplied args. Both remote and local paths use this."""
    try:
        subprocess.run(
            ["claude", "mcp", "remove", "--scope", "user", "mnemon"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["claude", "mcp", "add", "--scope", "user"] + add_args,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The `claude` CLI was not found on PATH. Install Claude Code "
            "before running `mnemon setup claude-code`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode(errors="replace").strip()
        raise RuntimeError(
            f"`claude mcp add` failed (exit {exc.returncode}): {stderr}"
        ) from exc


def setup_claude_code(*, remote_url: str | None = None, token: str | None = None) -> str:
    """Configure Claude Code (MCP server + hooks).

    When ``remote_url`` is provided:
      - Preflight the endpoint (aborts setup if unreachable).
      - Write URL + token to ``~/.mnemon/`` config files.
      - Register the HTTP MCP server via ``claude mcp add --transport http``.
      - Install UserPromptSubmit / Stop / SessionStart hooks.

    When ``remote_url`` is not provided:
      - Refuses to run if ``~/.mnemon/remote_url`` is present (you're
        currently in web mode; use ``mnemon downgrade local`` or
        ``mnemon uninstall`` to exit first).
      - Registers the stdio MCP server via ``claude mcp add --transport stdio``.
        Writing to ``settings.json.mcpServers`` has no effect — Claude Code
        reads its own registry, not settings.json.
      - Install UserPromptSubmit / Stop hooks (no SessionStart — no
        remote machine to pre-warm). Hooks dispatch in-process via
        :class:`~mnemon.hooks._client.LocalMemoryClient`.
    """
    if remote_url is None:
        _refuse_if_remote_configured("claude-code")

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

        _register_claude_code_mcp_remote(remote_url, local_token)
        lines.append("  MCP server: mnemon (http, remote) — registered via `claude mcp add`")

        # Strip any stdio mnemon entry in settings.json from a prior
        # local setup. Claude Code ignores the key for mnemon, but
        # leaving a stale entry is confusing when users inspect the file.
        mcp_servers = settings.get("mcpServers")
        if isinstance(mcp_servers, dict) and "mnemon" in mcp_servers:
            del mcp_servers["mnemon"]
            if not mcp_servers:
                del settings["mcpServers"]
            lines.append("  Cleaned up stale settings.json mcpServers.mnemon entry")
    else:
        _register_claude_code_mcp_stdio()
        lines.append("  MCP server: mnemon (stdio, local) — registered via `claude mcp add`")

        # Strip any stdio entry in settings.json — leftover from earlier
        # versions of setup that wrote there. Claude Code ignores it,
        # but it's confusing to see the mnemon name in two places.
        mcp_servers = settings.get("mcpServers")
        if isinstance(mcp_servers, dict) and "mnemon" in mcp_servers:
            del mcp_servers["mnemon"]
            if not mcp_servers:
                del settings["mcpServers"]
            lines.append("  Cleaned up stale settings.json mcpServers.mnemon entry")

    # Hooks: install in both modes. _hooks_config adds SessionStart only
    # when remote_url is supplied (local mode has no cold-start to warm).
    hooks = _hooks_config(remote_url=remote_url)
    if "hooks" not in settings:
        settings["hooks"] = {}
    settings["hooks"]["UserPromptSubmit"] = hooks["UserPromptSubmit"]
    settings["hooks"]["PostToolUse"] = hooks["PostToolUse"]
    settings["hooks"]["Stop"] = hooks["Stop"]
    if "SessionStart" in hooks:
        settings["hooks"]["SessionStart"] = hooks["SessionStart"]
    else:
        # Drop any stale SessionStart (set by a previous remote install);
        # it would poll a remote URL that's no longer authoritative.
        stale_session = settings["hooks"].get("SessionStart")
        if isinstance(stale_session, list):
            filtered = [
                entry
                for entry in stale_session
                if not any(
                    "mnemon" in (h.get("command") or "")
                    for h in entry.get("hooks", [])
                )
            ]
            if filtered:
                settings["hooks"]["SessionStart"] = filtered
            else:
                del settings["hooks"]["SessionStart"]

    hook_mode_tag = "remote" if remote_url else "local (in-process)"
    lines.append(f"  UserPromptSubmit: context-surfacing (8s, {hook_mode_tag})")
    lines.append(f"  PostToolUse: auto-mirror (12s, {hook_mode_tag}) [Write|Edit|MultiEdit]")
    lines.append(f"  Stop: session-extractor (30s), handoff-generator (30s) [{hook_mode_tag}]")
    if remote_url:
        lines.append("  SessionStart: pre-warm polling (90s background)")

    # If a claude.ai-synced mnemon entry exists alongside the stdio
    # entry we just wrote, surface the dual-config state. Web wins by
    # default in Claude Code's resolution; the stdio entry stays
    # on-disk as standby (auto-activates if the user later removes the
    # claude.ai entry).
    if remote_url is None:
        lines.extend(_dual_config_info_lines("Claude Code"))

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
    Refuses local mode if a remote is currently configured (split-brain
    guard — see :func:`_refuse_if_remote_configured`).
    """
    if remote_url is None:
        _refuse_if_remote_configured("cursor")

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


def _claude_desktop_config_path() -> Path:
    """Resolve the Claude Desktop MCP config path for this platform.

    - macOS: ``~/Library/Application Support/Claude/claude_desktop_config.json``
    - Windows: ``%APPDATA%\\Claude\\claude_desktop_config.json``
    - Linux: ``~/.config/Claude/claude_desktop_config.json`` (Claude
      Desktop is not officially supported on Linux but users who build
      from source put the config here).
    """
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
        # Fallback if APPDATA is unset for any reason.
        return (
            Path.home()
            / "AppData"
            / "Roaming"
            / "Claude"
            / "claude_desktop_config.json"
        )
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def setup_claude_desktop(
    *, remote_url: str | None = None, token: str | None = None
) -> str:
    """Configure Claude Desktop's MCP server.

    Claude Desktop has no hook system — this only writes the MCP entry.
    Config format mirrors Cursor's ``mcp.json``. When ``remote_url`` is
    provided, writes an HTTP transport with bearer auth and preflights
    the endpoint; otherwise writes a local stdio entry. Refuses local
    mode if a remote is currently configured (split-brain guard).
    """
    if remote_url is None:
        _refuse_if_remote_configured("claude-desktop")

    desktop_path = _claude_desktop_config_path()
    config = _read_json(desktop_path)

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    if remote_url:
        local_token = _ensure_local_token(token)
        _preflight_remote_endpoint(remote_url, local_token)
        _ensure_remote_url(remote_url)
        config["mcpServers"]["mnemon"] = {
            "url": remote_url,
            "headers": {"Authorization": f"Bearer {local_token}"},
        }
        mode = "remote (preflight OK)"
    else:
        config["mcpServers"]["mnemon"] = _mcp_config()
        mode = "stdio (local)"

    _write_json(desktop_path, config)

    summary = (
        f"Claude Desktop MCP configured at {desktop_path}\n"
        f"  Mode: {mode}"
    )
    # Claude Desktop also syncs MCP from claude.ai, so the same
    # dual-config state can arise. Only surface the info in stdio
    # mode — when configuring remote, the user already wants remote.
    if remote_url is None:
        info_lines = _dual_config_info_lines("Claude Desktop")
        if info_lines:
            summary += "\n" + "\n".join(info_lines)
    summary += "\nRestart Claude Desktop to activate."
    return summary


def setup_gemini() -> str:
    """Show Gemini CLI MCP configuration.

    Gemini CLI config format is installation-specific and the CLI doesn't
    commit to a single canonical path (varies across distributions and
    user customizations), so we print the snippet for the user to paste
    rather than auto-write. Keeps us out of the "accidentally wrote to
    the wrong file" category.
    """
    config = json.dumps({"mnemon": _mcp_config()}, indent=2)
    return (
        "Add this to your Gemini CLI MCP config:\n\n"
        f"{config}"
    )


def setup_hooks(*, remote_url: str | None = None, token: str | None = None) -> str:
    """Configure Claude Code hooks only (no MCP server).

    Works in both modes after P1a: local hooks dispatch in-process via
    :class:`~mnemon.hooks._client.LocalMemoryClient`; remote hooks go
    over HTTP. When ``remote_url`` is supplied, the endpoint is
    preflighted and a SessionStart pre-warm hook is added. Refuses
    local mode if a remote is currently configured (split-brain guard).
    """
    if remote_url is None:
        _refuse_if_remote_configured("hooks")

    settings_path = Path.home() / ".claude" / "settings.json"
    settings = _read_json(settings_path)

    if remote_url:
        local_token = _ensure_local_token(token)
        _preflight_remote_endpoint(remote_url, local_token)
        _ensure_remote_url(remote_url)

    hooks = _hooks_config(remote_url=remote_url)
    if "hooks" not in settings:
        settings["hooks"] = {}
    settings["hooks"]["UserPromptSubmit"] = hooks["UserPromptSubmit"]
    settings["hooks"]["PostToolUse"] = hooks["PostToolUse"]
    settings["hooks"]["Stop"] = hooks["Stop"]
    if "SessionStart" in hooks:
        settings["hooks"]["SessionStart"] = hooks["SessionStart"]

    _write_json(settings_path, settings)

    mode_tag = "remote" if remote_url else "local (in-process)"
    out_lines = [
        f"Hooks configured at {settings_path}",
        f"  Mode: {mode_tag}",
    ]
    if remote_url:
        out_lines += [
            f"  Remote URL: {remote_url}",
            f"  Preflight:  OK (memory_status round-trip < {REMOTE_PREFLIGHT_TIMEOUT_SEC:.0f}s)",
        ]
    out_lines += [
        "  UserPromptSubmit: context-surfacing (8s)",
        "  PostToolUse: auto-mirror (12s) [Write|Edit|MultiEdit]",
        "  Stop: session-extractor (30s), handoff-generator (30s)",
    ]
    if remote_url:
        out_lines.append("  SessionStart: pre-warm polling (90s background)")
    out_lines.append("Restart Claude Code to activate.")
    return "\n".join(out_lines)


TARGETS = {
    "claude-code": setup_claude_code,
    "claude-desktop": setup_claude_desktop,
    "cursor": setup_cursor,
    "gemini": setup_gemini,
    "hooks": setup_hooks,
}


# Auto-detect probes — a target is considered "installed" if its config
# directory exists. Probes are lightweight filesystem checks; they do not
# write anything. ``hooks`` is a pseudo-target (hooks-only install); it
# is intentionally excluded from auto-detect since every auto-detected
# ``claude-code`` install already handles hooks.
_AUTODETECT_ORDER = ["claude-code", "claude-desktop", "cursor"]


def _is_installed(target: str) -> bool:
    """True if the machine appears to have the given MCP client installed."""
    if target == "claude-code":
        return (Path.home() / ".claude").is_dir()
    if target == "claude-desktop":
        return _claude_desktop_config_path().parent.is_dir()
    if target == "cursor":
        return (Path.home() / ".cursor").is_dir()
    # Gemini and hooks have no reliable detection signal.
    return False


def detect_installed_clients() -> list[str]:
    """Return the subset of auto-detect targets whose config dir exists.

    Order is stable (``_AUTODETECT_ORDER``) so output is deterministic.
    """
    return [t for t in _AUTODETECT_ORDER if _is_installed(t)]


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
            "Run `mnemon upgrade web --app-name <name>` or pass "
            "--remote-url <URL> to switch this install to remote mode."
        )
    lines.append("  • Run `mnemon doctor` any time to re-verify the setup.")
    return "\n".join(lines)


def _run_single_target(
    target: str, parsed: dict, *, include_footer: bool = True
) -> str:
    """Invoke one setup target and return its user-facing message.

    Shared by :func:`run_setup` (single-target) and
    :func:`_run_autodetect` (one entry per detected client). When
    ``include_footer`` is False, the "Next steps" block is omitted so
    the caller can print a single aggregate footer instead of one per
    target.
    """
    func = TARGETS[target]
    try:
        if target == "gemini":
            primary = func()
        else:
            primary = func(
                remote_url=parsed["remote_url"], token=parsed["token"]
            )
    except SetupError as exc:
        return f"setup failed ({target}): {exc}"

    if include_footer:
        primary += _next_steps_block(target, parsed["remote_url"])
    return primary


def _run_autodetect(parsed: dict) -> str:
    """Configure every detected client, emit an aggregate summary.

    Gemini is tacked onto the end as a manual-step reminder — we print
    the snippet but don't pretend to have configured it. If no clients
    are detected, returns a clear message pointing at `--help`.
    """
    detected = detect_installed_clients()
    if not detected:
        return (
            "No MCP clients detected on this machine.\n"
            "Checked: ~/.claude (Claude Code), ~/.cursor (Cursor), "
            "Claude Desktop config dir.\n"
            "Install one of those, or run `mnemon setup <target>` "
            "explicitly (see `mnemon setup --help`)."
        )

    blocks: list[str] = [
        f"Detected MCP clients: {', '.join(detected)}",
        "",
    ]
    for target in detected:
        blocks.append(f"── {target} ──")
        blocks.append(_run_single_target(target, parsed, include_footer=False))
        blocks.append("")

    # Gemini tail: print snippet as a manual step reminder.
    blocks.append("── gemini (manual) ──")
    blocks.append(setup_gemini())

    # Single footer for the aggregate.
    remote_url = parsed["remote_url"]
    blocks.append("")
    blocks.append("Next steps:")
    blocks.append("  • Restart each client above to activate the MCP tools.")
    if remote_url is None:
        blocks.append(
            "  • Want mobile / claude.ai / cross-device memory? "
            "Run `mnemon upgrade web --app-name <name>` or rerun with "
            "--remote-url <URL>."
        )
    blocks.append("  • Run `mnemon doctor` any time to re-verify the setup.")

    return "\n".join(blocks)


def run_setup(target: str | None, args: list[str] | None = None) -> str:
    """Run setup for the given target (or auto-detect when ``target`` is None).

    Behavior matrix:

    - ``target=None`` → configure every detected client (auto-detect).
      Matches the documented happy-path ``mnemon setup`` invocation.
    - ``target="claude-code" | "cursor" | "claude-desktop" | "hooks"``
      → configure just that client.
    - ``target="gemini"`` → print the config snippet; never auto-writes.

    After a successful setup, ``mnemon doctor`` runs against the
    newly-configured vault with ``fail_on_warn=True`` (per the P1b plan —
    any config gap should surface as a non-zero exit so scripted installs
    propagate the failure). Pass ``--skip-doctor`` in ``args`` to
    suppress the automatic run.
    """
    parsed = _parse_setup_args(args or [])

    if target is None:
        primary = _run_autodetect(parsed)
        # Auto-detect block already has its own footer; doctor still runs.
        footer = ""
    elif target in TARGETS:
        primary = _run_single_target(target, parsed)
        if primary.startswith("setup failed"):
            return primary
        footer = ""  # _run_single_target already appended the next-steps block
    else:
        valid = ", ".join(TARGETS.keys())
        return f"Unknown target: {target}\nValid targets: {valid}"

    if primary.startswith("setup failed"):
        return primary

    # Doctor doesn't make sense for "gemini-only" or pure print targets.
    if parsed["skip_doctor"] or target == "gemini":
        return primary + footer

    import io

    from .doctor import run_doctor

    buf = io.StringIO()
    print("", file=buf)
    print("Running mnemon doctor to verify...", file=buf)
    try:
        doctor_rc = run_doctor(out=buf, fail_on_warn=True)
    except Exception as exc:  # noqa: BLE001
        buf.write(
            f"\n(doctor invocation crashed: {type(exc).__name__}: {exc})\n"
        )
        doctor_rc = 1

    if doctor_rc != 0:
        buf.write(
            "\nNOTE: doctor reported issues (including warnings, which "
            "are now treated as failures). Setup files were written, but "
            "the environment is not fully ready — fix the failing "
            "check(s) above before using mnemon, or rerun with "
            "--skip-doctor to bypass.\n"
        )

    return primary + footer + "\n" + buf.getvalue()
