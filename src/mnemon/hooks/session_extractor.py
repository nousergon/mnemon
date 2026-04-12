#!/usr/bin/env python3
"""Session extractor hook — Stop event.

Extracts observations from the conversation transcript using a local
1.7B LLM model. Deduplicates against existing memories via remote
vector similarity search. Saves new observations to the Fly vault.

Phase 3 unification: saves go to the Fly vault via _remote_client
(``memory_save`` tool call). LLM extraction remains local. Dedup
uses a remote ``memory_search`` call to check for high-similarity
existing content. Falls back to regex heuristics if LLM is unavailable.
"""

from __future__ import annotations

import re
import sys

from ..config import HOOK_DEDUP_SIMILARITY_THRESHOLD, HOOK_DEDUP_TIMEOUT_SEC

# ── LLM Extraction ──────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = (
    "You are a memory extraction assistant. Your job is to extract important "
    "observations from a conversation transcript.\n\n"
    "For each observation, output an XML block:\n"
    "<observation>\n"
    "  <type>decision|preference|observation|antipattern|research|project</type>\n"
    "  <title>Short descriptive title (max 80 chars)</title>\n"
    "  <content>2-3 sentences explaining what was learned, decided, or discovered. "
    "Include WHY, not just WHAT.</content>\n"
    "</observation>\n\n"
    "Rules:\n"
    "- Extract 1-5 observations. Only extract what is genuinely worth remembering.\n"
    "- Skip routine coding tasks (fix typo, run tests, read file) — only capture "
    "decisions, insights, preferences, and discoveries.\n"
    "- Each observation should be self-contained — understandable without the "
    "original conversation.\n"
    '- Use "decision" for architectural choices, "preference" for user workflow '
    'habits, "antipattern" for things that failed, "observation" for learned facts, '
    '"research" for investigations, "project" for project status/goals.\n'
    "- If the conversation has nothing worth remembering, output: <none/>"
)

OBSERVATION_RE = re.compile(
    r"<observation>\s*"
    r"<type>(.*?)</type>\s*"
    r"<title>(.*?)</title>\s*"
    r"<content>(.*?)</content>\s*"
    r"</observation>",
    re.DOTALL,
)

VALID_TYPES = {"decision", "preference", "observation", "antipattern", "research", "project"}

CLIENT_LABEL = "claude-code-session-extractor"


def parse_observations(response: str) -> list[dict]:
    """Parse XML-formatted observations from LLM response."""
    observations = []
    for match in OBSERVATION_RE.finditer(response):
        obs_type = match.group(1).strip()
        title = match.group(2).strip()
        content = match.group(3).strip()
        if title and content:
            observations.append({"type": obs_type, "title": title, "content": content})
    return observations


def extract_with_llm(transcript: str) -> list[dict] | None:
    """Extract observations using local LLM. Returns None if LLM unavailable."""
    from ..llm import try_generate

    response = try_generate(EXTRACTION_SYSTEM_PROMPT, transcript, max_tokens=2000)
    if response is None:
        return None
    if "<none/>" in response:
        return []
    return parse_observations(response)


# ── Regex Fallback (when LLM unavailable) ──────────────────────────────────

DECISION_PATTERNS = [
    re.compile(r"(?:decided|decision|chose|choosing|going with|went with|settled on)\s+(.{20,200})", re.I),
    re.compile(r"(?:we(?:'ll| will)|let(?:'s| us))\s+(?:go with|use|stick with|keep)\s+(.{10,200})", re.I),
]

PREFERENCE_PATTERNS = [
    re.compile(r"(?:prefer|preference|always|never|don't|do not)\s+(.{10,200})", re.I),
]

LEARNING_PATTERNS = [
    re.compile(r"(?:learned|discovered|found out|realized|turns out|it appears)\s+(?:that\s+)?(.{20,200})", re.I),
    re.compile(r"(?:the (?:issue|problem|root cause|reason) (?:is|was))\s+(.{10,200})", re.I),
]


def extract_with_regex(transcript: str) -> list[dict]:
    """Fallback: extract observations using regex patterns."""
    observations: list[dict] = []
    seen_content: set[str] = set()

    for pattern in DECISION_PATTERNS:
        for match in pattern.finditer(transcript):
            content = match.group(1).strip().rstrip(".")
            if content and content not in seen_content:
                seen_content.add(content)
                observations.append({"type": "decision", "title": content[:80], "content": content})

    for pattern in PREFERENCE_PATTERNS:
        for match in pattern.finditer(transcript):
            content = match.group(1).strip().rstrip(".")
            if content and content not in seen_content:
                seen_content.add(content)
                observations.append({"type": "preference", "title": content[:80], "content": content})

    for pattern in LEARNING_PATTERNS:
        for match in pattern.finditer(transcript):
            content = match.group(1).strip().rstrip(".")
            if content and content not in seen_content:
                seen_content.add(content)
                observations.append({"type": "observation", "title": content[:80], "content": content})

    return observations[:5]


# ── Remote Deduplication ───────────────────────────────────────────────────

def is_duplicate_remote(title: str, content: str) -> bool:
    """Check if an observation is too similar to existing memories via remote search.

    Searches the Fly vault for content matching the combined title+content
    using the structured ``memory_search_structured`` tool, which returns a
    JSON array with explicit ``composite_score`` fields — no text parsing.
    If any result scores above ``HOOK_DEDUP_SIMILARITY_THRESHOLD``, treat
    it as a duplicate. Returns False on any error (network, timeout,
    JSON parse failure) — prefer saving a possible duplicate over
    silently dropping a novel observation.
    """
    import json

    try:
        from ._remote_client import call_tool_sync

        query = f"{title}: {content}"
        raw, _elapsed = call_tool_sync(
            "memory_search_structured",
            {"query": query, "limit": 3},
            timeout=HOOK_DEDUP_TIMEOUT_SEC,
            client_label=CLIENT_LABEL,
        )
        results = json.loads(raw)
        return any(
            r.get("composite_score", 0.0) > HOOK_DEDUP_SIMILARITY_THRESHOLD
            for r in results
        )
    except Exception:
        return False


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from .framework import read_stdin, read_transcript
        from ._remote_client import RemoteClientConfigError, call_tool_sync

        hook_input = read_stdin()
        transcript = read_transcript(hook_input.get("transcript_path", ""), 6000)
        if not transcript or len(transcript) < 100:
            return

        # Try LLM extraction first, fall back to regex
        observations = extract_with_llm(transcript)
        if observations is None:
            observations = extract_with_regex(transcript)

        if not observations:
            return

        saved = 0
        for obs in observations:
            content_type = obs["type"] if obs["type"] in VALID_TYPES else "observation"

            # Remote vector dedup check
            if is_duplicate_remote(obs["title"], obs["content"]):
                print(f'mnemon: skipping duplicate observation: "{obs["title"]}"', file=sys.stderr)
                continue

            try:
                result, elapsed = call_tool_sync(
                    "memory_save",
                    {
                        "title": obs["title"],
                        "content": obs["content"],
                        "content_type": content_type,
                        "source_client": "claude-code-hook",
                    },
                    client_label=CLIENT_LABEL,
                )
                saved += 1
                print(f'mnemon: saved [{content_type}] "{obs["title"]}"', file=sys.stderr)
            except RemoteClientConfigError as e:
                print(f"mnemon session-extractor config error: {e}", file=sys.stderr)
                return
            except Exception as e:
                print(
                    f"mnemon session-extractor save error: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                continue

        if saved > 0:
            print(f"mnemon: extracted {saved} observations from session", file=sys.stderr)
    except Exception as e:
        print(f"mnemon session-extractor error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
