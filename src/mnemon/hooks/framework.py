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


def is_duplicate(text: str) -> bool:
    """Check if this text was seen within the dedup window."""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    now = __import__("time").time()

    dedup_file = _dedup_path()
    entries: list[dict] = []
    if dedup_file.exists():
        try:
            entries = json.loads(dedup_file.read_text())
        except Exception:
            entries = []

    # Prune expired
    entries = [e for e in entries if now - e["timestamp"] < DEDUP_WINDOW_SEC]

    # Check duplicate
    if any(e["hash"] == text_hash for e in entries):
        return True

    # Add new entry
    entries.append({"hash": text_hash, "timestamp": now})
    dedup_file.parent.mkdir(parents=True, exist_ok=True)
    dedup_file.write_text(json.dumps(entries))
    return False


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
