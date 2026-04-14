"""Configuration — content types, vault paths, scoring constants."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class ContentType(str, Enum):
    DECISION = "decision"
    PREFERENCE = "preference"
    ANTIPATTERN = "antipattern"
    OBSERVATION = "observation"
    RESEARCH = "research"
    PROJECT = "project"
    HANDOFF = "handoff"
    NOTE = "note"


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


# Content type → memory type mapping
MEMORY_TYPE_MAP: dict[ContentType, MemoryType] = {
    ContentType.DECISION: MemoryType.SEMANTIC,
    ContentType.PREFERENCE: MemoryType.SEMANTIC,
    ContentType.ANTIPATTERN: MemoryType.SEMANTIC,
    ContentType.OBSERVATION: MemoryType.SEMANTIC,
    ContentType.RESEARCH: MemoryType.SEMANTIC,
    ContentType.PROJECT: MemoryType.SEMANTIC,
    ContentType.HANDOFF: MemoryType.EPISODIC,
    ContentType.NOTE: MemoryType.SEMANTIC,
}

# Half-lives in days (None = never decay)
HALF_LIVES: dict[ContentType, int | None] = {
    ContentType.DECISION: None,
    ContentType.PREFERENCE: None,
    ContentType.ANTIPATTERN: None,
    ContentType.OBSERVATION: 90,
    ContentType.RESEARCH: 90,
    ContentType.PROJECT: 120,
    ContentType.HANDOFF: 30,
    ContentType.NOTE: 60,
}

# Default confidence per content type
DEFAULT_CONFIDENCE: dict[ContentType, float] = {
    ContentType.DECISION: 0.85,
    ContentType.PREFERENCE: 0.80,
    ContentType.ANTIPATTERN: 0.80,
    ContentType.OBSERVATION: 0.70,
    ContentType.RESEARCH: 0.70,
    ContentType.PROJECT: 0.65,
    ContentType.HANDOFF: 0.60,
    ContentType.NOTE: 0.50,
}

# Scoring constants
RRF_K = 60
MMR_THRESHOLD = 0.6                    # bigram Jaccard ≥ this → candidate is "too similar" to a selected result
MMR_DEMOTION_FACTOR = 0.5              # composite-score multiplier applied to MMR-demoted results
COMPOSITE_WEIGHTS = (0.5, 0.25, 0.25)  # (relevance, recency, confidence)
RECENCY_HALF_LIFE_DAYS = 30
PIN_BOOST = 0.3

# Query expansion
QUERY_EXPANSION_MAX_TOKENS = 200       # LLM token cap for alt-query generation

# Contradiction detection
CONTRADICTION_OVERLAP_THRESHOLD = 0.7  # minimum vector similarity to treat candidate as potentially conflicting
CONTRADICTION_CONTEXT_MAX_CHARS = 500  # per-memory content truncation in the LLM classification prompt

# Hook timeouts and budgets (seconds / chars)
#
# HOOK_REMOTE_TIMEOUT_SEC — matches Claude Code's ~/.claude/settings.json
# hook timeout budget. Longer than the 2s the original plan called for
# because Fly cold starts (wake machine + load FastEmbed ONNX) can take
# 15-25s before the machine responds; 8s lets a warm machine always
# succeed while a cold one surfaces a clean timeout.
HOOK_REMOTE_TIMEOUT_SEC = 8.0

# HOOK_DEDUP_TIMEOUT_SEC — tighter budget for the session_extractor
# dedup check (memory_search against the Fly vault). Runs in a Stop
# hook loop so each observation gets its own call; shorter timeout keeps
# the loop from stacking up against the 30s hook ceiling.
HOOK_DEDUP_TIMEOUT_SEC = 5.0

# HOOK_DEDUP_SIMILARITY_THRESHOLD — cosine similarity above which a new
# observation is treated as a duplicate of an existing memory.
HOOK_DEDUP_SIMILARITY_THRESHOLD = 0.92

# Context surfacing budget (context_surfacing hook)
HOOK_TOKEN_BUDGET = 800          # approx tokens injected per prompt
HOOK_CHARS_PER_TOKEN = 4         # rough conversion factor
HOOK_CHAR_BUDGET = HOOK_TOKEN_BUDGET * HOOK_CHARS_PER_TOKEN

# HOOK_SLOW_THRESHOLD_SEC — elapsed time above which a successful
# context_surfacing call prefixes a ⚠ slow warning. Lets the user see
# latency degradation in the prompt itself without watching logs.
HOOK_SLOW_THRESHOLD_SEC = 3.0

# Content type enum values for validation
CONTENT_TYPE_VALUES = [ct.value for ct in ContentType]


def vault_dir() -> Path:
    return Path(os.environ.get("MNEMON_VAULT_DIR", Path.home() / ".mnemon"))


def vault_path() -> Path:
    return vault_dir() / "default.sqlite"
