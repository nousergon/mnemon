"""Hook framework — reads Claude Code hook JSON from stdin,
dispatches to the appropriate handler, writes JSON to stdout.

Handles deduplication (SHA-256, 600s window) and noise filtering.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

DEDUP_WINDOW_SEC = 600  # 10 minutes

NOISE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^/\w"),                          # slash commands
    re.compile(r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure|yep|nope)\s*[.!?]?\s*$", re.I),
    re.compile(r"^(good morning|good night|bye|goodbye)\s*[.!?]?\s*$", re.I),
    re.compile(r"^[yn]$", re.I),                  # single letter confirmations
]


def _dedup_path() -> Path:
    return Path.home() / ".mnemon" / "dedup.json"


def _load_and_prune_entries() -> tuple[list[dict], float]:
    """Read dedup entries from disk, prune expired ones, return ``(entries, now)``."""
    import time

    now = time.time()
    dedup_file = _dedup_path()
    entries: list[dict] = []
    if dedup_file.exists():
        try:
            entries = json.loads(dedup_file.read_text())
        except Exception:
            entries = []
    entries = [
        e for e in entries if now - e.get("timestamp", 0) < DEDUP_WINDOW_SEC
    ]
    return entries, now


def is_duplicate(text: str) -> bool:
    """Check if ``text`` was seen within the dedup window.

    **Read-only.** Does not persist the hash. Callers that want to record
    a prompt as seen must call :func:`mark_seen` after successful
    processing. Splitting the check from the write lets hooks avoid
    locking out a prompt when the downstream call (e.g., a remote
    ``memory_search``) fails partway through — failing midway leaves the
    dedup state clean so an immediate retry works instead of silently
    no-opping for 10 minutes.
    """
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    entries, _ = _load_and_prune_entries()
    return any(e["hash"] == text_hash for e in entries)


def mark_seen(text: str) -> None:
    """Persist ``text`` in the dedup window.

    Called by hooks after successful processing to suppress duplicate
    work on the same prompt within :data:`DEDUP_WINDOW_SEC` seconds. If
    another hook already marked this prompt, no new entry is appended —
    the existing timestamp stays in place.
    """
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    entries, now = _load_and_prune_entries()
    if not any(e["hash"] == text_hash for e in entries):
        entries.append({"hash": text_hash, "timestamp": now})
    dedup_file = _dedup_path()
    dedup_file.parent.mkdir(parents=True, exist_ok=True)
    dedup_file.write_text(json.dumps(entries))


def is_noise(prompt: str) -> bool:
    """Check if prompt is noise (greetings, slash commands, etc.)."""
    trimmed = prompt.strip()
    if len(trimmed) < 3:
        return True
    return any(p.search(trimmed) for p in NOISE_PATTERNS)


def read_stdin() -> dict[str, Any]:
    """Read hook input JSON from stdin."""
    raw = sys.stdin.read()
    return json.loads(raw)


def write_output(output: dict[str, Any]) -> None:
    """Write hook output JSON to stdout."""
    sys.stdout.write(json.dumps(output))
    sys.stdout.flush()


def log_hook_error(hook_name: str, context: str, exc: BaseException) -> None:
    """Write a consistent hook-error line to stderr.

    Format: ``mnemon {hook_name} {context}: {ExceptionType}: {message}``.
    Used by all three hooks for RemoteClientConfigError, generic remote
    failures, and outer catch-alls so log lines from different hooks are
    greppable by a single pattern. Callers handle control flow (return /
    continue / fallthrough) themselves — this only formats and emits.
    """
    sys.stderr.write(
        f"mnemon {hook_name} {context}: {type(exc).__name__}: {exc}\n"
    )
    sys.stderr.flush()


def read_transcript(transcript_path: str, max_chars: int = 8000) -> str:
    """Read the last N characters from the transcript JSONL file.

    Supports two wire formats for the per-line message envelope:

    1. **Flat (legacy / synthesized fixtures):** ``{"role": "user",
       "content": "..."}`` — role and content at the top level of the
       JSON object.
    2. **Nested (real Claude Code JSONL):** ``{"type": "user", "message":
       {"role": "user", "content": "..."}, ...}`` — role and content
       under a nested ``message`` object alongside metadata fields like
       ``parentUuid``, ``sessionId``, ``timestamp``, ``cwd``, etc.

    Real Claude Code uses the nested format; before this was supported,
    ``read_transcript`` returned an empty string against every real
    session, silently breaking ``handoff_generator`` and
    ``session_extractor`` (both call into this function).

    ``content`` itself can be either a string or a list of content
    blocks (Anthropic API tool-use shape — ``[{"type": "text", "text":
    "..."}, {"type": "tool_use", ...}]``). Non-text blocks (tool_use,
    tool_result, image, etc.) are skipped; only text blocks contribute
    to the extracted transcript.

    Skipped lines (no meaningful content):

    - Lines without a recognizable user/assistant message envelope
      (e.g. Claude Code's ``file-history-snapshot`` lines).
    - Messages with no text content (e.g. an assistant turn that was
      pure tool calls — common during agent runs).
    """
    if not transcript_path:
        return ""
    path = Path(transcript_path)
    if not path.exists():
        return ""

    try:
        lines = path.read_text().strip().split("\n")
        messages: list[str] = []
        total_chars = 0

        for line in reversed(lines):
            if total_chars >= max_chars:
                break
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Accept both flat and nested wire formats. Nested is what
            # real Claude Code emits; flat is what synthesized test
            # fixtures use historically. ``inner`` is the dict that
            # carries ``role`` + ``content``.
            inner = envelope.get("message")
            if not isinstance(inner, dict):
                inner = envelope

            role = inner.get("role", "unknown")
            if role not in ("user", "assistant"):
                continue

            content = ""
            raw_content = inner.get("content")
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                content = "\n".join(
                    block.get("text", "")
                    for block in raw_content
                    if isinstance(block, dict) and block.get("type") == "text"
                )

            if content:
                messages.insert(0, f"[{role}]: {content}")
                total_chars += len(content)

        return "\n\n".join(messages)
    except Exception:
        return ""
