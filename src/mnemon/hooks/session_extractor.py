#!/usr/bin/env python3
"""Session extractor hook — Stop event.

Extracts observations from the conversation transcript using a local
1.7B LLM model. Deduplicates against existing memories via vector
similarity. Saves new observations to the vault.

Phase 3: LLM-based extraction with vector dedup (replaces Phase 2 regex).
Falls back to regex heuristics if LLM is unavailable.
"""

from __future__ import annotations

import re
import sys

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
    try:
        from ..llm import generate, is_available
        if not is_available():
            return None
        response = generate(EXTRACTION_SYSTEM_PROMPT, transcript, max_tokens=2000)
        if "<none/>" in response:
            return []
        return parse_observations(response)
    except Exception:
        return None


# ── Regex Fallback (Phase 2 heuristics) ────────────────────────────────────

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


# ── Deduplication ───────────────────────────────────────────────────────────

def is_duplicate(store, title: str, content: str) -> bool:
    """Check if an observation is too similar to existing memories (> 0.92 cosine)."""
    try:
        from ..embedder import embed
        query_emb = embed(f"title: {title} | text: {content}")
        results = store.search_vector(query_emb, 3)
        return any(r.score > 0.92 for r in results)
    except Exception:
        return False


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from .framework import read_stdin, read_transcript

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

        from ..store import Store

        store = Store()
        try:
            saved = 0
            for obs in observations:
                content_type = obs["type"] if obs["type"] in VALID_TYPES else "observation"

                # Vector dedup check
                if is_duplicate(store, obs["title"], obs["content"]):
                    print(f'mnemon: skipping duplicate observation: "{obs["title"]}"', file=sys.stderr)
                    continue

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
