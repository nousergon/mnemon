#!/usr/bin/env python3
"""Handoff generator hook — Stop event.

Generates a session summary for continuity across sessions.
Saved as a "handoff" memory with 30-day half-life.

Phase 2: template-based extraction (no LLM required).
Phase 3 will upgrade to LLM-based summarization.
"""

from __future__ import annotations

import re
import sys


def generate_handoff(transcript: str) -> dict | None:
    """Generate a handoff summary from transcript using heuristics."""
    lines = transcript.split("\n")
    user_lines = [l for l in lines if l.startswith("[user]:")]
    assistant_lines = [l for l in lines if l.startswith("[assistant]:")]

    if len(user_lines) < 2:
        return None

    # Extract first user message as topic indicator
    first_topic = user_lines[0].replace("[user]: ", "")[:100] if user_lines else "Unknown"

    # Count exchanges
    exchanges = min(len(user_lines), len(assistant_lines))

    # Look for file modifications mentioned
    file_patterns = re.findall(r"(?:created?|modified?|edited?|updated?|wrote)\s+[`'\"]?([/\w.-]+\.\w+)", transcript, re.I)
    files_modified = list(set(file_patterns))[:10]

    # Build summary bullets
    bullets = []
    bullets.append(f"- Topic: {first_topic}")
    bullets.append(f"- Exchanges: {exchanges}")

    if files_modified:
        bullets.append(f"- Files touched: {', '.join(files_modified[:5])}")

    # Look for decisions
    decision_matches = re.findall(r"(?:decided|going with|settled on|chose)\s+(.{10,100})", transcript, re.I)
    for d in decision_matches[:3]:
        bullets.append(f"- Decision: {d.strip().rstrip('.')}")

    title = first_topic[:60]
    summary = "\n".join(bullets)

    return {"title": title, "summary": summary}


def main() -> None:
    try:
        from .framework import read_stdin, read_transcript

        hook_input = read_stdin()
        transcript = read_transcript(hook_input.get("transcript_path", ""), 6000)
        if not transcript or len(transcript) < 200:
            return

        handoff = generate_handoff(transcript)
        if not handoff:
            return

        from ..store import Store

        store = Store()
        try:
            doc_id = store.save(
                title=f"Session: {handoff['title']}",
                content=handoff["summary"],
                content_type="handoff",
                source_client="claude-code-hook",
            )

            # Embed if available
            try:
                from ..embedder import embed_document
                doc = store.get(doc_id)
                if doc:
                    embed_document(store, doc.hash, handoff["title"], handoff["summary"])
            except Exception:
                pass

            print(f'mnemon: saved handoff "{handoff["title"]}"', file=sys.stderr)
        finally:
            store.close()
    except Exception as e:
        print(f"mnemon handoff-generator error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
