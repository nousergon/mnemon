#!/usr/bin/env python3
"""Context surfacing hook — UserPromptSubmit.

Searches the vault for relevant memories and injects them as
XML context before Claude processes the prompt.

Pipeline:
  1. Skip noise (slash commands, greetings, short prompts, duplicates)
  2. BM25 + vector search
  3. Composite scoring (relevance + recency + confidence)
  4. Tiered injection (HOT/WARM/COLD) within 800 token budget
"""

from __future__ import annotations

import sys

TOKEN_BUDGET = 800
CHARS_PER_TOKEN = 4
CHAR_BUDGET = TOKEN_BUDGET * CHARS_PER_TOKEN

HOT_THRESHOLD = 0.15
WARM_THRESHOLD = 0.10
HOT_SNIPPET_LEN = 300
WARM_SNIPPET_LEN = 150


def build_context(results: list) -> str:
    if not results:
        return ""

    lines: list[str] = []
    chars_used = 0

    for r in results:
        if r.composite_score >= HOT_THRESHOLD:
            snippet = r.content[:HOT_SNIPPET_LEN].replace("\n", " ")
            ellipsis = "..." if len(r.content) > HOT_SNIPPET_LEN else ""
            entry = f"[{r.content_type}] {r.title}: {snippet}{ellipsis}"
        elif r.composite_score >= WARM_THRESHOLD:
            snippet = r.content[:WARM_SNIPPET_LEN].replace("\n", " ")
            entry = f"[{r.content_type}] {r.title}: {snippet}..."
        else:
            entry = f"[{r.content_type}] {r.title}"

        if chars_used + len(entry) > CHAR_BUDGET:
            break

        lines.append(entry)
        chars_used += len(entry)

    if not lines:
        return ""

    return (
        "<mnemon-context>\n"
        "Relevant memories from previous sessions:\n"
        + "\n".join(lines)
        + "\n</mnemon-context>"
    )


def main() -> None:
    try:
        from .framework import read_stdin, write_output, is_noise, is_duplicate

        hook_input = read_stdin()
        prompt = hook_input.get("prompt", "")

        if is_noise(prompt):
            return
        if is_duplicate(prompt):
            return

        from ..store import Store
        from ..search import search

        store = Store()
        try:
            results = search(store, prompt, limit=8, use_vector=True)
            if not results:
                return

            context = build_context(results)
            if not context:
                return

            write_output({
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                },
            })
        finally:
            store.close()
    except Exception as e:
        print(f"mnemon context-surfacing error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
