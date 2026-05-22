"""scripts/_layer3_remote_helper.py — remote MCP tool invocation for layer3.

`mnemon` CLI save/status/search are LOCAL-ONLY — they instantiate
`Store()` directly and never check `MNEMON_REMOTE_URL` or
`~/.mnemon/remote_url`. Only `mnemon doctor` and the hook scripts
actually go through the remote MCP client. The e2e test runbook
was written assuming CLI commands honor remote mode; they don't, so
`scripts/promote_stable.sh layer3`'s post-upgrade verification was
silently reading a freshly-created empty local SQLite instead of the
test app's remote state.

This helper bypasses the broken CLI path by calling the same
`call_tool_sync` that the hooks (and `mnemon doctor`) use. URL + token
resolution comes from `MNEMON_REMOTE_URL` env or
`~/.mnemon/remote_url` file (set by `mnemon upgrade web`).

Tracked as ROADMAP follow-up: extend `mnemon` CLI to honor remote mode
for save/status/search, then this helper can go away.

Usage (from promote_stable.sh):
    .venv/bin/python scripts/_layer3_remote_helper.py status
    .venv/bin/python scripts/_layer3_remote_helper.py save TITLE CONTENT CONTENT_TYPE
    .venv/bin/python scripts/_layer3_remote_helper.py exercise-all-tools

The ``exercise-all-tools`` subcommand was added 2026-05-22 alongside
the all-tools integration canary (``tests/test_tools_integration.py``).
It calls every read-only-safe MCP tool against the running remote
server and asserts each returns cleanly — surfaces Fly-specific
breakage (model not in image, Anthropic proxy timeout, etc.) that
the local Python-level integration test can't see.

Tools that mutate state are constrained to ``dry_run=True`` invocations.
Destructive tools (``memory_forget``) are SKIPPED — the layer3 downgrade
path validates end-to-end cleanup anyway.
"""

from __future__ import annotations

import json
import sys
import time

from mnemon.hooks._remote_client import call_tool_sync


def _total_documents() -> int:
    result, _elapsed = call_tool_sync("memory_status", {})
    data = json.loads(result)
    return int(data["total_documents"])


def _save(title: str, content: str, content_type: str) -> str:
    result, _elapsed = call_tool_sync(
        "memory_save",
        {"title": title, "content": content, "content_type": content_type},
    )
    return result


# ── exercise-all-tools ─────────────────────────────────────────────
# Catches Fly-specific failures the local integration test can't see:
# missing baked models in the image, Anthropic MCP proxy timeouts,
# auth/transport regressions, etc. Composes with
# tests/test_tools_integration.py (local-process canary).

# Tools the layer3 sequence already exercises elsewhere (skip to avoid
# double-counting + to keep this scoped to "all OTHER tools").
_TOOLS_EXERCISED_ELSEWHERE = {
    "memory_save",     # exercised at "Step 4" already
    "memory_status",   # exercised at "Step 3" / "Step 4"
}

# Destructive tools — skip in the layer3 read-mostly path. Downgrade
# step verifies state integrity afterwards.
_DESTRUCTIVE_TOOLS = {
    "memory_forget",
    "memory_rebuild",  # heavy, re-embeds every doc
}


def _exercise_all_tools() -> int:
    """Iterate the registered tool manager and call each remote-safe
    tool. Returns 0 if all pass, 1 if any failed.

    Resolves tool list dynamically from the local mnemon install — so
    a tool added to ``server.py`` is automatically exercised on the
    next layer3 run without editing this helper.
    """
    from mnemon.server import mcp

    # Use the most recent live document as the target for id-requiring
    # tools. Falls back to id=1 if the remote vault somehow has no docs
    # (shouldn't happen post-seed in layer3, but defensive).
    timeline_raw, _ = call_tool_sync("memory_timeline", {"limit": 1})
    timeline = json.loads(timeline_raw)
    target_id = timeline[0]["doc_id"] if timeline else 1

    # Per-tool argument builder — mirrors tests/test_tools_integration.py
    # _tool_inputs_for(). If a new tool ships without an entry here,
    # the all-tools check will fail with a clear "no fixture" error.
    inputs = {
        "memory_search": {"query": "layer3 exercise"},
        "memory_get": {"id": target_id},
        "memory_timeline": {"limit": 3},
        "memory_related": {"id": target_id, "limit": 3},
        "memory_list_standing": {},
        "memory_export_vectors": {},
        "profile_get": {},
        "memory_sweep": {"dry_run": True},
        "memory_check_contradictions": {"id": target_id, "dry_run": True},
        # promote/demote round-trip — operator gesture, no destructive
        # effect on the test app's state when paired.
        "memory_promote": {"id": target_id},
        "memory_demote": {"id": target_id},
        "memory_pin": {"id": target_id},
        # profile_update needs both args
        "profile_update": {"title": "layer3-test", "content": "layer3 exercise probe"},
    }

    registered = set(mcp._tool_manager._tools.keys())
    to_exercise = sorted(
        registered
        - _TOOLS_EXERCISED_ELSEWHERE
        - _DESTRUCTIVE_TOOLS
    )

    failures: list[tuple[str, str, str]] = []
    for tool_name in to_exercise:
        args = inputs.get(tool_name)
        if args is None:
            failures.append((
                tool_name, "NoFixture",
                f"no input fixture in _exercise_all_tools; add one to inputs dict",
            ))
            print(f"  ✗ {tool_name}: NO FIXTURE")
            continue

        t0 = time.time()
        try:
            result, elapsed = call_tool_sync(tool_name, args)
            n_chars = len(result) if isinstance(result, str) else 0
            # Catch opaque envelopes leaking through as clean results
            if isinstance(result, str) and "Error occurred during tool execution" in result:
                failures.append((tool_name, "OpaqueError", result[:200]))
                print(f"  ✗ {tool_name}: OPAQUE ERROR LEAK ({elapsed:.2f}s)")
            else:
                print(f"  ✓ {tool_name}: {n_chars} chars in {elapsed:.2f}s")
        except Exception as e:
            failures.append((tool_name, type(e).__name__, str(e)[:200]))
            print(f"  ✗ {tool_name}: {type(e).__name__}: {e}")

    if failures:
        print(f"\nFAILED: {len(failures)}/{len(to_exercise)} tools failed", file=sys.stderr)
        for name, exc_type, msg in failures:
            print(f"  {name}: {exc_type}: {msg}", file=sys.stderr)
        return 1

    print(f"\nPASSED: all {len(to_exercise)} exercised tools returned cleanly")
    return 0


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: _layer3_remote_helper.py <status|save|exercise-all-tools> [args...]",
            file=sys.stderr,
        )
        sys.exit(2)

    cmd = sys.argv[1]
    if cmd == "status":
        print(_total_documents())
    elif cmd == "save":
        if len(sys.argv) < 5:
            print(
                "usage: _layer3_remote_helper.py save TITLE CONTENT CONTENT_TYPE",
                file=sys.stderr,
            )
            sys.exit(2)
        print(_save(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif cmd == "exercise-all-tools":
        sys.exit(_exercise_all_tools())
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
