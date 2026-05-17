"""Output-boundary safety for recalled memory content.

mnemon stores whatever it is told to remember. Session transcripts
routinely contain ``<system-reminder>`` blocks and the deferred-tool
``<functions><function>{...schema...}</function></functions>`` format,
and those land in the vault via the session-extractor regex fallback,
auto-mirrored handoff files, or explicit ``memory_save`` of conversation
text.

When that content is replayed verbatim back into a model's context —
the Claude Code ``<mnemon-context>`` injection, or an MCP
``memory_search`` / ``memory_get`` result a Claude Desktop session
reads — those tokens can be mis-parsed as live host instructions or
tool registrations. That is a stored prompt-injection / context-
poisoning hazard, independent of mnemon's own (legitimate) tool
framework: a recalled memory must never be able to impersonate the
host's control surface or close mnemon's own context wrapper early.

The mitigation neutralizes a fixed allowlist of control-plane tag
*tokens* at every emit boundary by swapping their angle brackets for
the visually-equivalent guillemets ``‹ ›`` (U+2039 / U+203A). The text
stays fully readable as quoted prose — a human or model reading it as
recalled content loses nothing — but it can no longer be parsed as the
host's own control surface. Storage stays lossless (faithful raw text
in SQLite); only the model-facing copy is defanged, so this also
remediates every memory already in the vault without a migration.
"""

from __future__ import annotations

import re

# Tag tokens that carry host / harness control-plane meaning. Scoped to
# an explicit allowlist so ordinary XML or code in memories (``List<T>``,
# ``<observation>`` from the extractor prompt, HTML snippets) is left
# untouched — only tokens that can impersonate the control surface or
# mnemon's own wrapper are neutralized.
_CONTROL_TAGS = (
    "system-reminder",
    "functions",
    "function",
    "mnemon-context",
    "antml:invoke",
    "antml:parameter",
    "antml:function_calls",
    "antml:thinking_mode",
    # Bare (de-namespaced) forms — copy-paste and transcript capture
    # routinely strip the ``antml:`` prefix; the de-namespaced token is
    # still close enough to impersonate a tool call.
    "invoke",
    "parameter",
    "function_calls",
)

_CONTROL_TAG_RE = re.compile(
    r"<(/?)\s*(" + "|".join(re.escape(t) for t in _CONTROL_TAGS) + r")(\s[^>]*)?>",
    re.IGNORECASE,
)

_LANGLE = "‹"  # ‹  SINGLE LEFT-POINTING ANGLE QUOTATION MARK
_RANGLE = "›"  # ›  SINGLE RIGHT-POINTING ANGLE QUOTATION MARK


def defang_control_markup(text: str) -> str:
    """Neutralize host control-plane tags in untrusted recalled text.

    Replaces the angle brackets of recognized control tags with ``‹ ›``
    so the token can no longer be parsed as a live ``system-reminder``,
    tool registration, or as mnemon's own ``mnemon-context`` wrapper.
    Idempotent — the guillemet form does not re-match. Non-string or
    bracket-free input is returned unchanged (cheap hot-path guard).
    """
    if not text or not isinstance(text, str) or "<" not in text:
        return text
    return _CONTROL_TAG_RE.sub(
        lambda m: f"{_LANGLE}{m.group(1)}{m.group(2)}{m.group(3) or ''}{_RANGLE}",
        text,
    )


def defang_doc(doc: dict) -> dict:
    """Return ``doc`` with its model-facing text fields defanged.

    Mutates and returns the same dict (the JSON-serialized projections in
    :mod:`mnemon.server` are throwaway per-call dicts, so in-place is
    safe and avoids a copy on every result row). Only ``title`` and
    ``content`` carry free-form recalled text; other fields are scalars.
    """
    for key in ("title", "content"):
        val = doc.get(key)
        if isinstance(val, str):
            doc[key] = defang_control_markup(val)
    return doc
