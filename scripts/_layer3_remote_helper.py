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
"""

from __future__ import annotations

import json
import sys

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


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: _layer3_remote_helper.py <status|save> [args...]", file=sys.stderr)
        sys.exit(2)

    cmd = sys.argv[1]
    if cmd == "status":
        print(_total_documents())
    elif cmd == "save":
        if len(sys.argv) < 5:
            print("usage: _layer3_remote_helper.py save TITLE CONTENT CONTENT_TYPE", file=sys.stderr)
            sys.exit(2)
        print(_save(sys.argv[2], sys.argv[3], sys.argv[4]))
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
