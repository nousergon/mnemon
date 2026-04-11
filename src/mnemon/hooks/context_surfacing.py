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

# Overall character budget for the injected context. Kept generous
# because the server already truncates each result to a 300-char snippet.
TOKEN_BUDGET = 800
CHARS_PER_TOKEN = 4
CHAR_BUDGET = TOKEN_BUDGET * CHARS_PER_TOKEN

CLIENT_LABEL = "claude-code-context-surfacing"
SEARCH_LIMIT = 8

# Sentinel string the server returns when memory_search finds nothing.
# Kept in sync with src/mnemon/server.py memory_search().
NO_RESULTS_SENTINEL = "No memories found matching your query."


def build_context(raw_text: str) -> str:
    """Wrap the ``memory_search`` response in a ``<mnemon-context>`` block.

    The server returns a pre-formatted markdown list of matches, each
    already truncated to a 300-char snippet. We pass it through unchanged
    and wrap it in a containing tag so Claude can recognise it as mnemon
    context. A character budget is enforced as a safety net in case the
    server ever returns an unexpectedly long payload.

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
    return (
        "<mnemon-context>\n"
        "Relevant memories from previous sessions:\n"
        f"{trimmed}\n"
        "</mnemon-context>"
    )


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
            raw = call_tool_sync(
                "memory_search",
                {"query": prompt, "limit": SEARCH_LIMIT},
                client_label=CLIENT_LABEL,
            )
        except RemoteClientConfigError as e:
            # Configuration problem — URL or token not resolvable. Report
            # once to stderr and continue with no context; there's no point
            # retrying a misconfiguration inside a hook. Do NOT mark_seen —
            # the user can fix the config and retry the same prompt
            # immediately.
            print(
                f"mnemon context-surfacing config error: {e}",
                file=sys.stderr,
            )
            return
        except Exception as e:
            # Network error, timeout, auth failure, or MCP protocol error.
            # Log to stderr and continue — never crash Claude Code. Do NOT
            # mark_seen so the same prompt can retry after the transient
            # failure clears (e.g., wifi reconnect).
            print(
                f"mnemon context-surfacing remote error: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return

        # Remote call succeeded — record the prompt as seen so we don't
        # re-search on an immediate identical resubmit. This is
        # deliberately AFTER the call so failures above leave dedup state
        # clean and retryable.
        mark_seen(prompt)

        context = build_context(raw)
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
