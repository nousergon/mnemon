"""Mirror a local memory file to mnemon (vault or remote).

Reads a memory file written by Claude Code's auto-memory system (or any
file with a YAML frontmatter block + body), parses the frontmatter, and
saves to mnemon via the standard :class:`MemoryClient` abstraction so
the same code path works in local-stdio + remote-Fly modes.

Two entry points:

- :func:`mirror_path` — programmatic. Returns a structured result dict.
- :func:`run_cli` — wraps ``mirror_path`` for the ``mnemon mirror`` CLI
  subcommand and the ``mnemon.hooks.auto_mirror`` PostToolUse hook.

Memory paths recognized in ``--auto`` mode (no-op outside these):

- ``~/.claude/projects/*/memory/*.md`` — Claude Code auto-memory
- ``~/.cursor/.../memory/*.md`` — Cursor auto-memory (when added)
- ``~/.config/mnemon/auto-memory/*.md`` — generic local pattern

The 2026-04-28 incident that motivated this module: Claude wrote a
session handoff to its local auto-memory directory but failed to mirror
it to mnemon — exactly the gap this hook closes when wired through
``mnemon setup`` and the PostToolUse hook.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Regex matching auto-memory directories whose writes should propagate
# to mnemon. The ``--auto`` CLI flag short-circuits when ``file_path``
# does not match — the hook fires on every Write tool call but the bulk
# are unrelated source code edits.
#
# Anchored to ``$HOME`` to avoid surprising matches outside the user's
# tree (e.g. an unrelated checkout that happens to have a ``memory/``
# folder). Resolved at call time, not import, so per-test ``HOME``
# overrides take effect.
_AUTO_MEMORY_PATTERNS = (
    # Claude Code auto-memory:
    #   ~/.claude/projects/<encoded-cwd>/memory/<name>.md
    re.compile(
        r"^(?P<home>.+?)/\.claude/projects/[^/]+/memory/[^/]+\.md$"
    ),
    # Generic mnemon auto-memory location:
    #   ~/.config/mnemon/auto-memory/<name>.md
    re.compile(
        r"^(?P<home>.+?)/\.config/mnemon/auto-memory/[^/]+\.md$"
    ),
)

# Frontmatter delimiter: ``---`` on its own line at the start of the
# file. Matches the format Claude Code's auto-memory uses + standard
# YAML/Markdown frontmatter conventions.
_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(?P<frontmatter>.*?)\n---\s*\n(?P<body>.*)$",
    re.DOTALL,
)

# Identity marker injected when a memory file is itself a sync-down from
# mnemon. The hook MUST skip these — re-mirroring would create an
# infinite write loop. The presence of this key in the frontmatter is
# load-bearing; future ``mnemon sync down`` paths should write it.
_SYNC_SOURCE_KEY = "mnemon_sync_source"


class MirrorError(Exception):
    """Raised on unrecoverable mirror failures (frontmatter missing,
    file unreadable, dispatch failure). The CLI surfaces the message;
    the hook surfaces it to stderr so Claude sees it per
    feedback_surface_mnemon_unreachable."""


@dataclass
class MirrorResult:
    """Structured result from :func:`mirror_path`. ``status`` is one of:

    - ``"saved"`` — successfully dispatched a ``memory_save`` call.
    - ``"skipped_no_match"`` — ``--auto`` mode + path doesn't match
      any auto-memory pattern. Expected for the bulk of Write events.
    - ``"skipped_sync_source"`` — frontmatter has ``mnemon_sync_source``
      key, indicating this file was synced *down* from mnemon. Avoids
      mirror loops.
    - ``"skipped_duplicate"`` — content hash matches a recent save
      within the dedup window.
    """

    status: str
    title: str | None = None
    doc_id: int | None = None
    elapsed_seconds: float | None = None
    detail: str | None = None


def _is_auto_memory_path(path: Path) -> bool:
    """True iff ``path`` matches one of the auto-memory directory
    patterns. Used by ``--auto`` mode to no-op on unrelated writes."""
    s = str(path.expanduser().resolve())
    return any(pat.match(s) for pat in _AUTO_MEMORY_PATTERNS)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML-style frontmatter from a memory file. Returns
    ``(frontmatter_dict, body)``.

    Implements a minimal subset of YAML — key/value pairs separated by
    ``:`` — sufficient for memory files written by Claude Code's
    auto-memory system. Avoids a hard dependency on PyYAML for the
    common case; users with multi-line YAML in frontmatter can install
    PyYAML and we'll opportunistically use it.

    Raises :class:`MirrorError` if the frontmatter block is missing —
    a memory file without a ``name`` is not safely mirrorable.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise MirrorError(
            "Memory file is missing the YAML frontmatter block. "
            "Expected `---\\n...\\n---` at the top of the file."
        )

    fm_raw = match.group("frontmatter")
    body = match.group("body")

    # Try PyYAML if available — supports lists, multi-line strings, etc.
    try:
        import yaml  # type: ignore[import-not-found]

        parsed = yaml.safe_load(fm_raw) or {}
        if not isinstance(parsed, dict):
            raise MirrorError(
                f"Frontmatter parsed to {type(parsed).__name__}, expected dict."
            )
        return parsed, body
    except ImportError:
        pass

    # Minimal fallback parser — handles ``key: value`` lines, ignores
    # blanks + comments. Sufficient for Claude Code's frontmatter format.
    parsed: dict[str, Any] = {}
    for line in fm_raw.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        parsed[key.strip()] = value.strip()
    return parsed, body


def _content_hash(title: str, content: str) -> str:
    """SHA-256 of (title + body), used for idempotent skip on
    repeated saves of the same file."""
    h = hashlib.sha256()
    h.update(title.encode("utf-8"))
    h.update(b"\x00")
    h.update(content.encode("utf-8"))
    return h.hexdigest()


def _dedup_state_path() -> Path:
    """Where mirror dedup state lives. Separate from the hook
    framework's dedup file so a session_extractor save + a mirror save
    of the same content don't collide."""
    return Path.home() / ".mnemon" / "mirror_dedup.json"


def _check_and_record_dedup(content_hash: str, window_sec: int = 600) -> bool:
    """True iff this content_hash was saved within the last
    ``window_sec`` seconds. Always records the new entry on miss so the
    next call within the window is a hit."""
    import time

    path = _dedup_state_path()
    now = time.time()
    entries: list[dict[str, Any]] = []
    if path.exists():
        try:
            entries = json.loads(path.read_text()) or []
        except (json.JSONDecodeError, OSError):
            entries = []

    # Prune expired
    entries = [e for e in entries if now - e.get("ts", 0) < window_sec]

    for entry in entries:
        if entry.get("hash") == content_hash:
            return True

    entries.append({"hash": content_hash, "ts": now})
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(entries))
    except OSError:
        # Read-only home (sandbox, container) — best effort, don't block
        # the save itself. The mirror still happens; dedup just won't
        # remember the hit next time.
        pass
    return False


def mirror_path(
    path: Path,
    *,
    auto: bool = False,
    client: Any = None,
    timeout: float = 10.0,
) -> MirrorResult:
    """Read ``path``, parse frontmatter, save to mnemon.

    Parameters
    ----------
    path
        Memory file. Must exist; must contain a YAML frontmatter block.
    auto
        If True, no-op when ``path`` does not match an auto-memory
        directory pattern. Used by the PostToolUse hook to ignore the
        bulk of unrelated Write events.
    client
        Optional :class:`MemoryClient` override (test seam). Defaults
        to :func:`mnemon.hooks._client.get_client` which picks
        local/remote based on current config.
    timeout
        Per-call timeout passed to the client. Local mode ignores this;
        remote mode uses it.

    Returns
    -------
    :class:`MirrorResult` with ``status`` describing the outcome.
    """
    path = path.expanduser().resolve()

    if auto and not _is_auto_memory_path(path):
        return MirrorResult(
            status="skipped_no_match",
            detail=f"Path {path} does not match any auto-memory pattern.",
        )

    if not path.exists():
        raise MirrorError(f"Memory file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(text)

    if frontmatter.get(_SYNC_SOURCE_KEY):
        return MirrorResult(
            status="skipped_sync_source",
            detail=(
                f"File has '{_SYNC_SOURCE_KEY}' frontmatter — synced down "
                "from mnemon. Skipping to avoid mirror loop."
            ),
        )

    title = (frontmatter.get("name") or "").strip()
    if not title:
        raise MirrorError(
            "Frontmatter is missing a 'name' field; cannot derive a "
            "memory title. Memory files must declare a name."
        )

    content_type = (frontmatter.get("type") or "note").strip() or "note"
    description = (frontmatter.get("description") or "").strip()

    # The mirrored memory's content is the file body. Description
    # (if present) gets prepended as a single italicized line so it's
    # visible in mnemon's search results without losing the body
    # structure.
    if description:
        content = f"_{description}_\n\n{body.strip()}"
    else:
        content = body.strip()

    if not content:
        raise MirrorError(
            f"Memory file body is empty after frontmatter strip: {path}"
        )

    chash = _content_hash(title, content)
    if _check_and_record_dedup(chash):
        return MirrorResult(
            status="skipped_duplicate",
            title=title,
            detail="Identical content was already mirrored within the dedup window.",
        )

    if client is None:
        from .hooks._client import get_client

        client = get_client()

    # Stable upsert identity = the memory file's slug (frontmatter
    # `name`). A memory file's normal lifecycle is draft → refine →
    # finalize-on-merge, often several edits within one session; without
    # this key each edit mirrored a brand-new document. The server
    # upserts on (collection, source_client, source_key) so the slug
    # stays a single live doc across edits.
    arguments: dict[str, Any] = {
        "title": title,
        "content": content,
        "content_type": content_type,
        "source_client": "mnemon-mirror",
        "source_key": title,
    }

    result_text, elapsed = client.call_tool(
        "memory_save",
        arguments,
        timeout=timeout,
        client_label="mnemon-mirror",
    )

    # Best-effort doc_id parse — the memory_save tool returns
    # human-readable text like "Saved memory #281: ...". Surface for
    # the CLI but never fail on parse errors; the save already happened.
    doc_id: int | None = None
    m = re.search(r"#(\d+)", result_text or "")
    if m:
        try:
            doc_id = int(m.group(1))
        except ValueError:
            pass

    return MirrorResult(
        status="saved",
        title=title,
        doc_id=doc_id,
        elapsed_seconds=elapsed,
        detail=(result_text or "").strip()[:200],
    )


def run_cli(argv: list[str]) -> int:
    """``mnemon mirror <path> [--auto]`` entry point.

    Returns a Unix-style exit code: 0 on success or expected skip,
    non-zero on hard failure. The CLI dispatcher in :mod:`mnemon.cli`
    delegates here.
    """
    auto = False
    path_arg: str | None = None
    timeout = 10.0

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--auto":
            auto = True
            i += 1
        elif a == "--timeout" and i + 1 < len(argv):
            try:
                timeout = float(argv[i + 1])
            except ValueError:
                print(
                    f"mnemon mirror: invalid --timeout {argv[i + 1]!r}",
                    file=__import__("sys").stderr,
                )
                return 2
            i += 2
        elif a.startswith("--"):
            print(
                f"mnemon mirror: unknown flag {a!r}",
                file=__import__("sys").stderr,
            )
            return 2
        else:
            if path_arg is not None:
                print(
                    "mnemon mirror: too many positional arguments "
                    "(expected one path)",
                    file=__import__("sys").stderr,
                )
                return 2
            path_arg = a
            i += 1

    if path_arg is None:
        print(
            "Usage: mnemon mirror <path> [--auto] [--timeout SEC]",
            file=__import__("sys").stderr,
        )
        return 2

    path = Path(path_arg)
    try:
        result = mirror_path(path, auto=auto, timeout=timeout)
    except MirrorError as exc:
        print(f"mnemon mirror: {exc}", file=__import__("sys").stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface every error
        print(
            f"mnemon mirror: {type(exc).__name__}: {exc}",
            file=__import__("sys").stderr,
        )
        return 1

    if result.status == "saved":
        suffix = f" (#{result.doc_id})" if result.doc_id else ""
        print(f"Mirrored: {result.title!r}{suffix}")
        return 0

    if result.status.startswith("skipped"):
        # Quiet by default in --auto mode (the hook fires on every
        # Write); explicit invocation gets the detail line.
        if not auto:
            print(f"{result.status}: {result.detail or ''}".rstrip())
        return 0

    print(
        f"mnemon mirror: unexpected status {result.status!r}",
        file=__import__("sys").stderr,
    )
    return 1
