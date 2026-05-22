"""End-to-end integration test — invoke every registered MCP tool
against a real seeded vault.

Driver: 2026-05-22 finding that ``memory_check_contradictions``
returned an opaque "Error occurred during tool execution" envelope
to claude.ai despite all unit tests passing. The unit tests in
``test_server.py`` mock external deps (LLM, NLI, embedder, vecstore),
which is appropriate for asserting per-tool behavior in isolation —
but it leaves a gap where a tool's real call path can raise an
uncaught exception (or hang) that no unit test exercises.

This file closes that gap. The test iterates every tool registered
in the FastMCP tool manager, builds minimal-valid inputs from a
seeded vault, invokes each, and asserts:
  1. No unhandled exception (a tool that raises through FastMCP's
     wrapper produces the opaque "Error occurred during tool
     execution" envelope downstream — never acceptable)
  2. Return shape matches the MCP tool contract (str, dict, or list)

Operates on a temp vault — no production state touched. Mutating
tools are constrained to safe inputs (dry_run flags, throwaway IDs)
or are run last so they can't interfere with earlier reads.

Composes with [[feedback_no_silent_fails]] — every tool's failure
mode must be a NAMED exception caught at the tool boundary and
surfaced as a clean error message, never propagated as an
unhandled exception. This test enforces that contract.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import mnemon.server as server_mod
from mnemon.embedder import embed_document
from mnemon.store import Store


# Tools that are too expensive to exercise with real models in
# every CI run. We still call them but stub the heavy external
# dependency to keep test runtime under a few seconds total.
# (The real-model path is separately validated by
# ``scripts/calibrate_capture_threshold.py`` + the operator
# Layer-3 web test.)
_HEAVY_TOOLS = {"memory_check_contradictions", "memory_rebuild"}


@pytest.fixture
def seeded_store(monkeypatch):
    """Real Store with several seeded memories + vectors indexed.

    Replaces the server singleton so all MCP tools route to this
    fixture. Cleans up the temp vault on teardown.
    """
    fd, db_path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    os.unlink(db_path)

    store = Store(db_path=db_path)

    # Seed three memories of distinct types so the tool inputs can
    # exercise different code paths (note / preference / decision).
    ids = []
    for title, content, ctype in [
        ("Tool integration seed A", "First seeded memory for tool integration testing.", "note"),
        ("Tool integration seed B — preference", "A standing preference about test fixtures.", "preference"),
        ("Tool integration seed C — decision", "A decision about how to handle the integration suite.", "decision"),
    ]:
        doc_id = store.save(title=title, content=content, content_type=ctype)
        ids.append(doc_id)
        doc = store.get(doc_id)
        embed_document(store, doc.hash, doc.title, doc.content)

    # Reset + patch the server's _get_store to return this store
    server_mod._store = None
    monkeypatch.setattr(server_mod, "_get_store", lambda: store)

    yield {"store": store, "ids": ids}

    store.close()
    server_mod._store = None
    for ext in ("", "-wal", "-shm"):
        try:
            os.unlink(db_path + ext)
        except FileNotFoundError:
            pass


def _tool_inputs_for(seeded: dict[str, Any]) -> dict[str, dict]:
    """Build minimal-valid inputs per registered tool name.

    Tools that mutate state are given inputs that won't interfere
    with parallel reads (``dry_run=True``, throwaway IDs allocated
    specifically for the mutating call, etc.).
    """
    ids = seeded["ids"]
    return {
        # Read-only tools — exercise against any seeded id
        "memory_search": {"query": "integration"},
        "memory_get": {"id": ids[0]},
        "memory_status": {},
        "memory_timeline": {"limit": 5},
        "memory_related": {"id": ids[0], "limit": 5},
        "memory_list_standing": {},
        "memory_export_vectors": {},
        "profile_get": {},

        # Pure-create — safe, doesn't modify existing rows
        "memory_save": {
            "title": "Integration test write",
            "content": "Content produced by the all-tools integration test.",
        },
        "profile_update": {
            "title": "test-profile",
            "content": "test profile payload",
        },

        # Mutating tools constrained to dry-run / throwaway:
        "memory_sweep": {"dry_run": True},  # default but pass explicitly
        "memory_check_contradictions": {"id": ids[0], "dry_run": True},

        # Tier mutations — promote then demote so the vault state
        # round-trips. Use seed B since it's a preference (operator
        # intent is to elevate this kind of memory).
        "memory_promote": {"id": ids[1]},
        "memory_demote": {"id": ids[1]},  # will run after promote per ordering below

        # Pin + Forget — use a dedicated seed so the read-only tests
        # before them aren't disrupted. Forget is destructive; the
        # fixture tears down anyway.
        "memory_pin": {"id": ids[2]},
        "memory_forget": {"id": ids[2]},  # last to run

        # Rebuild — heavy, may re-embed every doc. Tested via stub.
        "memory_rebuild": {},
    }


# Ordering matters for mutating tools: pin/promote must run before
# forget/demote+forget on the same id. We invoke tools alphabetically
# but override the ordering for known dependent pairs.
def _tool_order(tool_names: list[str]) -> list[str]:
    """Sort tools so mutations to a shared id happen in a sane order:
    promote before demote (id B); pin before forget (id C); reads
    before destructive writes on those ids."""
    priority = {
        "memory_promote": 90,
        "memory_demote": 91,        # after promote
        "memory_pin": 92,           # after reads
        "memory_forget": 99,        # last — destructive
        "memory_rebuild": 95,       # after most reads
    }
    return sorted(tool_names, key=lambda n: (priority.get(n, 0), n))


class TestAllToolsRoundTrip:
    """Every registered MCP tool must complete without an unhandled
    exception when called with minimal valid inputs. This is the
    canary that catches the failure class of the 2026-05-22
    contradiction-check incident."""

    def test_every_tool_invokes_cleanly(self, seeded_store):
        from mnemon.server import mcp

        registered = set(mcp._tool_manager._tools.keys())
        inputs_map = _tool_inputs_for(seeded_store)

        # Coverage check: every registered tool has a fixture input.
        # If a new tool ships without a fixture entry, this fails
        # explicitly — forcing the contributor to consider how to
        # exercise the new tool in CI.
        missing = registered - set(inputs_map.keys())
        assert not missing, (
            f"Registered MCP tools without integration-test inputs: {sorted(missing)}. "
            "Add a minimal-valid input to _tool_inputs_for() in this file."
        )

        # Conversely, ensure we don't have stale fixtures for removed tools.
        stale = set(inputs_map.keys()) - registered
        assert not stale, (
            f"Stale fixture inputs for unregistered tools: {sorted(stale)}. "
            "Remove these from _tool_inputs_for() or restore the tool."
        )

        results: dict[str, tuple[str, Any]] = {}
        for tool_name in _tool_order(list(registered)):
            tool = mcp._tool_manager._tools[tool_name]
            inputs = inputs_map[tool_name]
            try:
                if tool_name in _HEAVY_TOOLS:
                    # Stub the expensive external dependency so we
                    # exercise the wrapping tool plumbing without
                    # paying the full model-inference cost on every
                    # test run. The real path is validated by the
                    # Layer-3 web test ritual.
                    if tool_name == "memory_check_contradictions":
                        # Make the NLI classifier deterministic +
                        # fast: every pair → "unrelated", so no
                        # mutations land regardless of dry_run.
                        from mnemon.nli import BidirectionalResult, NLIResult
                        neutral = NLIResult(
                            label="neutral",
                            probs={"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0},
                        )
                        bidir = BidirectionalResult(
                            mnemon_label="unrelated",
                            a_implies_b=neutral, b_implies_a=neutral,
                        )
                        with patch("mnemon.nli.classify_pair_bidirectional", return_value=bidir):
                            result = tool.fn(**inputs)
                    elif tool_name == "memory_rebuild":
                        # Patch the embedder so rebuild doesn't re-run
                        # the FastEmbed model over every seed memory.
                        with patch("mnemon.embedder.embed_document"):
                            result = tool.fn(**inputs)
                    else:  # pragma: no cover — exhaustive guard
                        result = tool.fn(**inputs)
                else:
                    result = tool.fn(**inputs)
                results[tool_name] = ("ok", result)
            except Exception as exc:
                pytest.fail(
                    f"Tool {tool_name!r} raised unhandled "
                    f"{type(exc).__name__}: {exc}.  Inputs: {inputs}.  "
                    f"This is the failure class that produces opaque "
                    f"'Error occurred during tool execution' envelopes "
                    f"in claude.ai/Desktop. Every tool must catch its "
                    f"failure modes at the boundary and return a clean "
                    f"error string instead.",
                )

        # Every successful call returned a value of the documented
        # MCP-tool contract type. Mostly strings; the JSON-returning
        # tools (memory_status, memory_list_standing,
        # memory_export_vectors) return JSON strings too.
        for name, (_, result) in results.items():
            assert isinstance(result, (str, dict, list)), (
                f"Tool {name!r} returned unexpected type "
                f"{type(result).__name__}: {result!r}"
            )

    def test_no_tool_returns_opaque_error_string(self, seeded_store):
        """A clean-error response must include either the named
        exception type / specific cause OR a human-actionable
        instruction. Opaque envelopes like 'Error occurred during
        tool execution' must never be the tool's own output —
        that envelope comes from the MCP transport layer wrapping
        a Python exception that escaped the tool boundary."""
        from mnemon.server import mcp

        inputs_map = _tool_inputs_for(seeded_store)

        opaque_phrases = {
            "error occurred during tool execution",
            "internal server error",
            # If a tool ever returns just a request_id string, that's
            # the Anthropic-side wrapping leaking through — also opaque.
        }

        for tool_name in _tool_order(list(mcp._tool_manager._tools.keys())):
            tool = mcp._tool_manager._tools[tool_name]
            inputs = inputs_map[tool_name]
            if tool_name in _HEAVY_TOOLS:
                continue  # covered above with stubs
            result = tool.fn(**inputs)
            if not isinstance(result, str):
                continue  # JSON tools handled by the shape test
            lower = result.lower()
            for phrase in opaque_phrases:
                assert phrase not in lower, (
                    f"Tool {tool_name!r} returned an opaque error string "
                    f"({phrase!r} found in output). Replace with a named "
                    f"cause or human-actionable message. Got: {result!r}"
                )

    def test_destructive_tools_respect_dry_run(self, seeded_store):
        """memory_check_contradictions(dry_run=True) and
        memory_sweep(dry_run=True) must NOT mutate the vault. This
        locks the dry_run contract added in PR #157."""
        from mnemon.server import mcp

        store = seeded_store["store"]
        ids = seeded_store["ids"]

        # Snapshot pre-state
        pre_confidences = {
            i: store.get(i).confidence for i in ids
        }
        pre_relations = store.db.execute(
            "SELECT COUNT(*) AS c FROM relations"
        ).fetchone()["c"]

        # Run dry-run mutations with stubbed NLI
        from mnemon.nli import BidirectionalResult, NLIResult
        upd = NLIResult(label="entailment",
                        probs={"contradiction": 0.0, "entailment": 1.0, "neutral": 0.0})
        bidir = BidirectionalResult(
            mnemon_label="update",  # would-decay if not dry-run
            a_implies_b=upd, b_implies_a=upd,
        )
        with patch("mnemon.nli.classify_pair_bidirectional", return_value=bidir):
            mcp._tool_manager._tools["memory_check_contradictions"].fn(
                id=ids[0], dry_run=True,
            )
        mcp._tool_manager._tools["memory_sweep"].fn(dry_run=True)

        # Confidences unchanged, no new relations
        for i in ids:
            assert store.get(i).confidence == pre_confidences[i], (
                f"Confidence on #{i} changed under dry_run — "
                f"{pre_confidences[i]} → {store.get(i).confidence}"
            )
        post_relations = store.db.execute(
            "SELECT COUNT(*) AS c FROM relations"
        ).fetchone()["c"]
        assert post_relations == pre_relations, (
            f"Relations count changed under dry_run — "
            f"{pre_relations} → {post_relations}"
        )
