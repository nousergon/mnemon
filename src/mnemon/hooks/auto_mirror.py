#!/usr/bin/env python3
"""Auto-mirror hook — Claude Code PostToolUse event.

Fires on every Write / Edit / MultiEdit. No-ops unless the touched file
matches an auto-memory directory pattern (per
:func:`mnemon.mirror._is_auto_memory_path`). On a match, dispatches the
file's contents to mnemon via the same path as the
``mnemon mirror --auto`` CLI subcommand.

Closes the 2026-04-28 gap where Claude Code's local auto-memory writes
silently diverged from the central vault. Without this hook, every user
running mnemon has to remember to call ``memory_save`` immediately after
every local memory write — a discipline failure mode that surfaced on
day-one of alpha testing.

Hook input shape (Claude Code PostToolUse JSON):

    {
      "session_id": "...",
      "transcript_path": "...",
      "tool_name": "Write" | "Edit" | "MultiEdit",
      "tool_input": {
        "file_path": "/abs/path/to/file.md",
        ...
      },
      "tool_response": {...}
    }

The hook reads ``tool_input.file_path`` and routes to
:func:`mnemon.mirror.mirror_path` in ``--auto`` mode. Output is JSON on
stdout (Claude Code ignores the body of the response for PostToolUse,
but emits ``stderr`` directly into the operator's terminal — so any
error message becomes visible).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .framework import log_hook_error, read_stdin, write_output

# Tools whose Write events should trigger an auto-mirror check.
# MultiEdit fires once per call regardless of edit count, so a single
# memory file rewritten via MultiEdit produces one mirror attempt.
_TOOLS_TRIGGERING_MIRROR = ("Write", "Edit", "MultiEdit")


def _extract_file_path(hook_input: dict[str, Any]) -> str | None:
    """Pull ``tool_input.file_path`` out of the PostToolUse payload.

    Returns None when the payload is malformed, the tool is not in the
    mirror trigger set, or the file_path field is missing/empty.
    """
    tool_name = hook_input.get("tool_name", "")
    if tool_name not in _TOOLS_TRIGGERING_MIRROR:
        return None

    tool_input = hook_input.get("tool_input")
    if not isinstance(tool_input, dict):
        return None

    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return None

    return file_path


def main() -> int:
    """Hook entry point. Always exits 0 so a mirror failure never blocks
    Claude Code's continued operation; the error is surfaced via stderr
    so the operator sees it (and Claude sees it via the harness, per
    ``feedback_surface_mnemon_unreachable``).
    """
    try:
        hook_input = read_stdin()
    except Exception as exc:  # noqa: BLE001 — top-level surface
        log_hook_error("auto_mirror", "stdin parse", exc)
        return 0

    file_path = _extract_file_path(hook_input)
    if file_path is None:
        # Tool not in trigger set, or payload malformed. Silent no-op.
        write_output({})
        return 0

    # Defer the mirror_path import so the hook stays cheap on the hot
    # path (every Write tool call). PyYAML, hashing, and the dedup
    # state-file IO all live behind this import.
    from ..mirror import MirrorError, mirror_path

    try:
        result = mirror_path(Path(file_path), auto=True)
    except MirrorError as exc:
        # Expected failure modes (missing frontmatter, empty body,
        # missing name). Surface to stderr so the operator + Claude
        # both see the error; never block.
        log_hook_error("auto_mirror", f"mirror {file_path}", exc)
        write_output({})
        return 0
    except Exception as exc:  # noqa: BLE001 — top-level surface
        # Unexpected — could be RemoteMemoryClient HTTP failure,
        # auth problem, dispatch error. Same posture: surface, never
        # block. Exit 0 so the harness prints the stderr line back to
        # Claude per feedback_surface_mnemon_unreachable.
        log_hook_error("auto_mirror", f"mirror {file_path}", exc)
        write_output({})
        return 0

    if result.status == "saved":
        # Brief operator-visible confirmation. PostToolUse hooks emit
        # nothing user-facing by default; the stderr line is what shows
        # up in Claude Code's hook activity log.
        sys.stderr.write(
            f"mnemon auto_mirror: saved {result.title!r}"
            + (f" (#{result.doc_id})" if result.doc_id else "")
            + "\n"
        )
        sys.stderr.flush()

    write_output({})
    return 0


if __name__ == "__main__":
    sys.exit(main())
