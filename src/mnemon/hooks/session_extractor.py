#!/usr/bin/env python3
"""Session extractor hook — Stop event.

Extracts observations from the conversation transcript using heuristic
pattern matching. Saves new observations to the vault.

Phase 2: regex/heuristic extraction (no LLM required).
Phase 3 will upgrade to LLM-based extraction.
"""

from __future__ import annotations

import re
import sys

# Patterns that indicate extractable observations
DECISION_PATTERNS = [
    re.compile(r"(?:decided|decision|chose|choosing|going with|went with|settled on)\s+(.{20,200})", re.I),
    re.compile(r"(?:we(?:'ll| will)|let(?:'s| us))\s+(?:go with|use|stick with|keep)\s+(.{10,200})", re.I),
]

PREFERENCE_PATTERNS = [
    re.compile(r"(?:prefer|preference|always|never|don't|do not)\s+(.{10,200})", re.I),
]

OBSERVATION_PATTERNS = [
    re.compile(r"(?:learned|discovered|found out|realized|turns out|it appears)\s+(?:that\s+)?(.{20,200})", re.I),
    re.compile(r"(?:the (?:issue|problem|root cause|reason) (?:is|was))\s+(.{10,200})", re.I),
]

VALID_TYPES = {"decision", "preference", "observation", "antipattern", "research", "project"}


def extract_observations(transcript: str) -> list[dict]:
    """Extract observations from transcript using heuristic patterns."""
    observations: list[dict] = []
    seen_content: set[str] = set()

    for pattern in DECISION_PATTERNS:
        for match in pattern.finditer(transcript):
            content = match.group(1).strip().rstrip(".")
            if content and content not in seen_content:
                seen_content.add(content)
                observations.append({
                    "type": "decision",
                    "title": content[:80],
                    "content": content,
                })

    for pattern in PREFERENCE_PATTERNS:
        for match in pattern.finditer(transcript):
            content = match.group(1).strip().rstrip(".")
            if content and content not in seen_content:
                seen_content.add(content)
                observations.append({
                    "type": "preference",
                    "title": content[:80],
                    "content": content,
                })

    for pattern in OBSERVATION_PATTERNS:
        for match in pattern.finditer(transcript):
            content = match.group(1).strip().rstrip(".")
            if content and content not in seen_content:
                seen_content.add(content)
                observations.append({
                    "type": "observation",
                    "title": content[:80],
                    "content": content,
                })

    return observations[:5]  # Cap at 5


def main() -> None:
    try:
        from .framework import read_stdin, read_transcript

        hook_input = read_stdin()
        transcript = read_transcript(hook_input.get("transcript_path", ""), 6000)
        if not transcript or len(transcript) < 100:
            return

        observations = extract_observations(transcript)
        if not observations:
            return

        from ..store import Store

        store = Store()
        try:
            saved = 0
            for obs in observations:
                content_type = obs["type"] if obs["type"] in VALID_TYPES else "observation"
                doc_id = store.save(
                    title=obs["title"],
                    content=obs["content"],
                    content_type=content_type,
                    source_client="claude-code-hook",
                )

                # Embed if available
                try:
                    from ..embedder import embed_document
                    doc = store.get(doc_id)
                    if doc:
                        embed_document(store, doc.hash, obs["title"], obs["content"])
                except Exception:
                    pass

                saved += 1
                print(f'mnemon: saved [{content_type}] "{obs["title"]}"', file=sys.stderr)

            if saved > 0:
                print(f"mnemon: extracted {saved} observations from session", file=sys.stderr)
        finally:
            store.close()
    except Exception as e:
        print(f"mnemon session-extractor error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
