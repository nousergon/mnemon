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
MMR_THRESHOLD = 0.6
COMPOSITE_WEIGHTS = (0.5, 0.25, 0.25)  # (relevance, recency, confidence)
RECENCY_HALF_LIFE_DAYS = 30
PIN_BOOST = 0.3

# Content type enum values for validation
CONTENT_TYPE_VALUES = [ct.value for ct in ContentType]


def vault_dir() -> Path:
    return Path(os.environ.get("MNEMON_VAULT_DIR", Path.home() / ".mnemon"))


def vault_path() -> Path:
    return vault_dir() / "default.sqlite"
