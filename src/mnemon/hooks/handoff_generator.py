#!/usr/bin/env python3
"""Handoff generator hook — Stop event.

Generates a session summary for continuity across sessions.
Saved as a "handoff" memory with 30-day half-life.

Phase 3 unification: saves go to the Fly vault via _remote_client
(``memory_save`` tool call). LLM summarization remains local.
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

CLIENT_LABEL = "claude-code-handoff-generator"


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
    from ..llm import try_generate

    response = try_generate(HANDOFF_SYSTEM_PROMPT, transcript, max_tokens=500)
    if response is None:
        return None
    if "<none/>" in response:
        return {"skip": True}
    return parse_handoff(response)


# ── Regex Fallback (when LLM unavailable) ──────────────────────────────────

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
        from .framework import log_hook_error, read_stdin, read_transcript
        from ._remote_client import RemoteClientConfigError, call_tool_sync

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

        try:
            result, elapsed = call_tool_sync(
                "memory_save",
                {
                    "title": f"Session: {handoff['title']}",
                    "content": handoff["summary"],
                    "content_type": "handoff",
                    "source_client": "claude-code-hook",
                },
                client_label=CLIENT_LABEL,
            )
            print(f'mnemon: saved handoff "{handoff["title"]}"', file=sys.stderr)
        except RemoteClientConfigError as e:
            log_hook_error("handoff-generator", "config error", e)
        except Exception as e:
            log_hook_error("handoff-generator", "save error", e)
    except Exception as e:
        log_hook_error("handoff-generator", "error", e)


if __name__ == "__main__":
    main()
