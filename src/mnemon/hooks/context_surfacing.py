#!/usr/bin/env python3
"""Context surfacing hook — UserPromptSubmit.

Calls mnemon's remote ``memory_search`` tool over Streamable HTTP and
injects the matching memories as XML context before Claude processes the
user's prompt.

Pipeline:
  1. Skip noise (slash commands, greetings, short prompts, duplicates)
  2. Call ``memory_search`` on the Fly-hosted vault via _remote_client
  3. Wrap the server's formatted response in ``<mnemon-context>`` tags
  4. Write the context to stdout for Claude Code to inject

On any network/auth/timeout/config error the hook degrades gracefully:
logs to stderr and exits 0 without emitting context. It never crashes
Claude Code — the hook is best-effort augmentation, not a load-bearing
data path.

Phase 3 unification: this hook no longer touches the local SQLite vault.
All memory reads flow to the Fly vault via :mod:`mnemon.hooks._remote_client`.
"""

from __future__ import annotations

import sys

from ..config import (
    HOOK_CHAR_BUDGET,
    HOOK_CHARS_PER_TOKEN,
    HOOK_SLOW_THRESHOLD_SEC,
    HOOK_TOKEN_BUDGET,
)

# Re-exported for tests and external callers. See ``config`` for rationale.
TOKEN_BUDGET = HOOK_TOKEN_BUDGET
CHARS_PER_TOKEN = HOOK_CHARS_PER_TOKEN
CHAR_BUDGET = HOOK_CHAR_BUDGET
SLOW_THRESHOLD_SEC = HOOK_SLOW_THRESHOLD_SEC

CLIENT_LABEL = "claude-code-context-surfacing"
SEARCH_LIMIT = 8

# Sentinel string the server returns when memory_search finds nothing.
# Kept in sync with src/mnemon/server.py memory_search().
NO_RESULTS_SENTINEL = "No memories found matching your query."


def build_context(raw_text: str, *, prefix: str = "") -> str:
    """Wrap the ``memory_search`` response in a ``<mnemon-context>`` block.

    The server returns a pre-formatted markdown list of matches, each
    already truncated to a 300-char snippet. We pass it through unchanged
    and wrap it in a containing tag so Claude can recognise it as mnemon
    context. A character budget is enforced as a safety net in case the
    server ever returns an unexpectedly long payload.

    ``prefix`` is prepended inside the block before the memories header —
    used to surface latency warnings on slow-but-successful calls.

    Returns an empty string when there's nothing worth injecting — empty
    input, the server's no-results sentinel, or whitespace-only content.
    """
    if not raw_text:
        return ""
    trimmed = raw_text.strip()
    if not trimmed:
        return ""
    if NO_RESULTS_SENTINEL in trimmed:
        return ""
    if len(trimmed) > CHAR_BUDGET:
        trimmed = trimmed[:CHAR_BUDGET].rstrip() + "\n...[truncated]"
    lines = []
    if prefix:
        lines.append(prefix)
    lines.append("Relevant memories from previous sessions:")
    lines.append(trimmed)
    inner = "\n".join(lines)
    return f"<mnemon-context>\n{inner}\n</mnemon-context>"


def build_warning_context(message: str) -> str:
    """Wrap a health-indicator warning in a ``<mnemon-context>`` block.

    Used when the remote call fails or is misconfigured — emits a visible
    warning into the prompt context rather than silently logging to stderr.
    The user sees it on every affected prompt without having to watch logs.
    """
    return f"<mnemon-context>\n{message}\n</mnemon-context>"


def main() -> None:
    try:
        from .framework import (
            is_duplicate,
            is_noise,
            mark_seen,
            read_stdin,
            write_output,
        )
        from ._remote_client import RemoteClientConfigError, call_tool_sync

        hook_input = read_stdin()
        prompt = hook_input.get("prompt", "")

        if is_noise(prompt):
            return
        if is_duplicate(prompt):
            return

        try:
            raw, elapsed = call_tool_sync(
                "memory_search",
                {"query": prompt, "limit": SEARCH_LIMIT},
                client_label=CLIENT_LABEL,
            )
        except RemoteClientConfigError as e:
            # Configuration problem — URL or token not resolvable. Emit a
            # visible warning block so the user sees it in the prompt context,
            # not just buried in stderr. Do NOT mark_seen — the user can fix
            # the config and retry the same prompt immediately.
            msg = f"⚠ mnemon config error: {e}"
            print(f"mnemon context-surfacing config error: {e}", file=sys.stderr)
            write_output({
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": build_warning_context(msg),
                },
            })
            return
        except Exception as e:
            # Network error, timeout, auth failure, or MCP protocol error.
            # Emit a visible warning block and log to stderr. Do NOT
            # mark_seen so the same prompt can retry after the transient
            # failure clears (e.g., wifi reconnect, Fly cold-start wakes).
            msg = f"⚠ mnemon unavailable: {type(e).__name__}: {e}"
            print(
                f"mnemon context-surfacing remote error: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            write_output({
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": build_warning_context(msg),
                },
            })
            return

        # Remote call succeeded — record the prompt as seen so we don't
        # re-search on an immediate identical resubmit. This is
        # deliberately AFTER the call so failures above leave dedup state
        # clean and retryable.
        mark_seen(prompt)

        slow_prefix = (
            f"⚠ mnemon slow: {elapsed:.1f}s"
            if elapsed > SLOW_THRESHOLD_SEC
            else ""
        )
        context = build_context(raw, prefix=slow_prefix)
        if not context:
            return

        write_output({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": context,
            },
        })
    except Exception as e:
        print(f"mnemon context-surfacing error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
