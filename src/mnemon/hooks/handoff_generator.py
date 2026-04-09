#!/usr/bin/env python3
"""Handoff generator hook — Stop event.

Generates a session summary for continuity across sessions.
Saved as a "handoff" memory with 30-day half-life.

Phase 3: LLM-based summarization (replaces Phase 2 template heuristics).
Falls back to regex heuristics if LLM is unavailable.
"""

from __future__ import annotations

import re
import sys

# ── LLM Summarization ──────────────────────────────────────────────────────

HANDOFF_SYSTEM_PROMPT = (
    "You are a session summarizer. Given a conversation transcript, "
    "produce a brief handoff summary for the next session.\n\n"
    "Format your response as:\n"
    "<handoff>\n"
    "  <title>Short descriptive title of the session (max 60 chars)</title>\n"
    "  <summary>\n"
    "  2-4 bullet points covering:\n"
    "  - What was accomplished\n"
    "  - Key decisions made\n"
    "  - Open questions or unfinished work\n"
    "  - Files or systems that were modified\n"
    "  </summary>\n"
    "</handoff>\n\n"
    "Rules:\n"
    "- Be concise — this is a handoff note, not a full report.\n"
    "- Focus on what the NEXT session needs to know.\n"
    "- If the session was trivial (just a question, no real work), output: <none/>"
)


def parse_handoff(response: str) -> dict | None:
    """Parse XML-formatted handoff from LLM response."""
    title_match = re.search(r"<title>(.*?)</title>", response, re.DOTALL)
    summary_match = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)

    if not title_match or not summary_match:
        return None

    title = title_match.group(1).strip()
    summary = summary_match.group(1).strip()

    if not title or not summary:
        return None
    return {"title": title, "summary": summary}


def generate_with_llm(transcript: str) -> dict | None:
    """Generate handoff using local LLM. Returns None if LLM unavailable."""
    try:
        from ..llm import generate, is_available
        if not is_available():
            return None
        response = generate(HANDOFF_SYSTEM_PROMPT, transcript, max_tokens=500)
        if "<none/>" in response:
            return {"skip": True}
        return parse_handoff(response)
    except Exception:
        return None


# ── Regex Fallback (Phase 2 heuristics) ────────────────────────────────────

def generate_with_regex(transcript: str) -> dict | None:
    """Fallback: generate handoff using regex heuristics."""
    lines = transcript.split("\n")
    user_lines = [l for l in lines if l.startswith("[user]:")]
    assistant_lines = [l for l in lines if l.startswith("[assistant]:")]

    if len(user_lines) < 2:
        return None

    first_topic = user_lines[0].replace("[user]: ", "")[:100] if user_lines else "Unknown"
    exchanges = min(len(user_lines), len(assistant_lines))

    file_patterns = re.findall(
        r"(?:created?|modified?|edited?|updated?|wrote)\s+[`'\"]?([/\w.-]+\.\w+)",
        transcript, re.I,
    )
    files_modified = list(set(file_patterns))[:10]

    bullets = [f"- Topic: {first_topic}", f"- Exchanges: {exchanges}"]
    if files_modified:
        bullets.append(f"- Files touched: {', '.join(files_modified[:5])}")

    decision_matches = re.findall(
        r"(?:decided|going with|settled on|chose)\s+(.{10,100})", transcript, re.I
    )
    for d in decision_matches[:3]:
        bullets.append(f"- Decision: {d.strip().rstrip('.')}")

    return {"title": first_topic[:60], "summary": "\n".join(bullets)}


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from .framework import read_stdin, read_transcript

        hook_input = read_stdin()
        transcript = read_transcript(hook_input.get("transcript_path", ""), 6000)
        if not transcript or len(transcript) < 200:
            return

        # Try LLM generation first, fall back to regex
        handoff = generate_with_llm(transcript)
        if handoff is None:
            handoff = generate_with_regex(transcript)

        if not handoff or handoff.get("skip"):
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
