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


def read_transcript(transcript_path: str, max_chars: int = 8000) -> str:
    """Read the last N characters from the transcript JSONL file."""
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
                msg = json.loads(line)
                role = msg.get("role", "unknown")
                content = ""

                if isinstance(msg.get("content"), str):
                    content = msg["content"]
                elif isinstance(msg.get("content"), list):
                    content = "\n".join(
                        c.get("text", "") for c in msg["content"] if c.get("type") == "text"
                    )

                if content and role in ("user", "assistant"):
                    messages.insert(0, f"[{role}]: {content}")
                    total_chars += len(content)
            except (json.JSONDecodeError, KeyError):
                continue

        return "\n\n".join(messages)
    except Exception:
        return ""
