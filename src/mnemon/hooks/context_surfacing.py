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

import json
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

# Snippet size injected per result — matches the pre-0.5.0 server-side
# truncation so context block size stays bounded independent of vault
# content length.
_SNIPPET_CHARS = 300


def _format_results(results: list[dict]) -> str:
    """Render a memory_search JSON response as a markdown list for prompt
    injection. Mirrors the pre-0.5.0 server-side prose format so the
    shape Claude sees in ``<mnemon-context>`` blocks is unchanged."""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        content = r.get("content", "")
        snippet = content[:_SNIPPET_CHARS]
        ellipsis = "..." if len(content) > _SNIPPET_CHARS else ""
        lines.append(
            f"{i}. [{r.get('content_type', 'note')}] **{r.get('title', '')}** "
            f"(score: {r.get('composite_score', 0):.3f}, "
            f"confidence: {r.get('confidence', 0):.2f})\n"
            f"   {snippet}{ellipsis}\n"
            f"   _id: {r.get('doc_id', '?')} | "
            f"created: {r.get('created_at', '')}_"
        )
    return "\n\n".join(lines)


def build_context(raw_text: str, *, prefix: str = "") -> str:
    """Wrap a ``memory_search`` JSON response in a ``<mnemon-context>`` block.

    Post-0.5.0 the server returns JSON instead of pre-formatted prose; we
    parse it and format client-side so the block the LLM sees is stable
    and token-efficient. A character budget caps the rendered output as
    a safety net against unexpectedly large payloads.

    ``prefix`` is prepended inside the block before the memories header —
    used to surface latency warnings on slow-but-successful calls.

    Returns an empty string when there's nothing worth injecting — empty
    input, an empty result list, or unparseable JSON.
    """
    if not raw_text:
        return ""
    trimmed = raw_text.strip()
    if not trimmed:
        return ""
    try:
        results = json.loads(trimmed)
    except json.JSONDecodeError:
        # Server contract violation — don't inject garbage into the prompt.
        return ""
    if not isinstance(results, list) or not results:
        return ""
    rendered = _format_results(results)
    if len(rendered) > CHAR_BUDGET:
        rendered = rendered[:CHAR_BUDGET].rstrip() + "\n...[truncated]"
    lines = []
    if prefix:
        lines.append(prefix)
    lines.append("Relevant memories from previous sessions:")
    lines.append(rendered)
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
            log_hook_error,
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
            log_hook_error("context-surfacing", "config error", e)
            write_output({
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": build_warning_context(
                        f"⚠ mnemon config error: {e}"
                    ),
                },
            })
            return
        except Exception as e:
            # Network error, timeout, auth failure, or MCP protocol error.
            # Emit a visible warning block and log to stderr. Do NOT
            # mark_seen so the same prompt can retry after the transient
            # failure clears (e.g., wifi reconnect, Fly cold-start wakes).
            log_hook_error("context-surfacing", "remote error", e)
            write_output({
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": build_warning_context(
                        f"⚠ mnemon unavailable: {type(e).__name__}: {e}"
                    ),
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
        # log_hook_error may not be importable if the hook crashed before
        # the import block — write directly to sys.stderr in that tail path.
        try:
            from .framework import log_hook_error
            log_hook_error("context-surfacing", "error", e)
        except Exception:
            sys.stderr.write(f"mnemon context-surfacing error: {e}\n")


if __name__ == "__main__":
    main()
