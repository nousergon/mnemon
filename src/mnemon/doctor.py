"""Self-diagnostic for mnemon remote deployments.

Runs an ordered series of checks against the configured remote vault and
prints pass/fail per check. Designed as a validator for the Phase 2
cutover (``MNEMON_AS_ENABLED=true`` on Fly) and for anyone self-hosting
their own mnemon instance who wants a quick health signal after
``fly deploy``.

Checks run top-down and short-circuit only on hard config failures — the
rest run independently so a single broken piece does not hide others.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .hooks._remote_client import (
    LOCAL_TOKEN_FILE,
    REMOTE_URL_FILE,
    RemoteClientConfigError,
    call_tool_sync,
    get_local_token,
    get_remote_url,
)

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

HEALTH_TIMEOUT_SEC = 5.0
MCP_TIMEOUT_SEC = 15.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    warn: bool = False  # ok=True + warn=True renders as a ⚠ non-fatal note


# ── Individual checks ──────────────────────────────────────────────────────


def check_remote_url() -> CheckResult:
    try:
        url = get_remote_url()
    except RemoteClientConfigError as exc:
        return CheckResult("Remote URL configured", False, str(exc))
    source = "env" if os.environ.get("MNEMON_REMOTE_URL") else f"file {REMOTE_URL_FILE}"
    return CheckResult("Remote URL configured", True, f"{url} (from {source})")


def check_local_token() -> CheckResult:
    try:
        token = get_local_token()
    except RemoteClientConfigError as exc:
        return CheckResult("Local token configured", False, str(exc))

    from_env = bool(os.environ.get("MNEMON_LOCAL_TOKEN"))
    source = "env" if from_env else f"file {LOCAL_TOKEN_FILE}"
    return CheckResult(
        "Local token configured",
        True,
        f"{len(token)} bytes (from {source})",
    )


def check_token_file_perms() -> CheckResult:
    """Warn if the token is coming from a file that is group/world readable."""
    if os.environ.get("MNEMON_LOCAL_TOKEN"):
        return CheckResult(
            "Token file permissions",
            True,
            "skipped (token from env var)",
            warn=False,
        )
    if not LOCAL_TOKEN_FILE.exists():
        return CheckResult(
            "Token file permissions",
            True,
            "skipped (no token file)",
            warn=False,
        )
    mode = LOCAL_TOKEN_FILE.stat().st_mode
    perms = stat.S_IMODE(mode)
    if perms & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH):
        return CheckResult(
            "Token file permissions",
            True,
            f"{LOCAL_TOKEN_FILE} is {oct(perms)} — should be 0600 (chmod 600)",
            warn=True,
        )
    return CheckResult(
        "Token file permissions",
        True,
        f"{oct(perms)} (0600 as expected)",
    )


def check_health_endpoint() -> CheckResult:
    """GET /health on the deployment base. Works without auth."""
    try:
        url = get_remote_url()
    except RemoteClientConfigError as exc:
        return CheckResult("Health endpoint reachable", False, str(exc))

    base = url.rstrip("/")
    if base.endswith("/mcp"):
        base = base[: -len("/mcp")]
    health_url = f"{base}/health"

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(health_url, timeout=HEALTH_TIMEOUT_SEC) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            elapsed_ms = (time.monotonic() - t0) * 1000
            if resp.status != 200:
                return CheckResult(
                    "Health endpoint reachable",
                    False,
                    f"HTTP {resp.status} from {health_url}",
                )
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return CheckResult(
                    "Health endpoint reachable",
                    False,
                    f"non-JSON body from {health_url}: {body[:80]}",
                )
            if payload.get("status") != "ok":
                return CheckResult(
                    "Health endpoint reachable",
                    False,
                    f"status={payload.get('status')!r} (expected 'ok')",
                )
            return CheckResult(
                "Health endpoint reachable",
                True,
                f"{health_url} ({elapsed_ms:.0f}ms)",
            )
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        return CheckResult(
            "Health endpoint reachable",
            False,
            f"{health_url}: {exc}",
        )


def check_auth_and_tool_call() -> CheckResult:
    """Full MCP handshake + memory_search call. Exercises auth end-to-end."""
    try:
        result, elapsed = call_tool_sync(
            "memory_search",
            {"query": "__mnemon_doctor_probe__", "limit": 1},
            timeout=MCP_TIMEOUT_SEC,
            client_label="mnemon-doctor",
        )
    except RemoteClientConfigError as exc:
        return CheckResult("Auth + MCP tool call", False, str(exc))
    except TimeoutError:
        return CheckResult(
            "Auth + MCP tool call",
            False,
            f"timed out after {MCP_TIMEOUT_SEC:.0f}s (cold start? run again)",
        )
    except Exception as exc:  # noqa: BLE001 — surface any SDK-level failure
        return CheckResult("Auth + MCP tool call", False, f"{type(exc).__name__}: {exc}")

    return CheckResult(
        "Auth + MCP tool call",
        True,
        f"memory_search returned {len(result)} bytes ({elapsed * 1000:.0f}ms)",
    )


def check_round_trip() -> CheckResult:
    """Save a throwaway memory, search for it by exact title, then forget it.

    Uses a UUID-suffixed title so concurrent doctor runs cannot collide.
    """
    import uuid

    # Use the UUID in both title AND content. store.save() dedups by SHA-256
    # of content, so identical content across runs would return the original
    # doc_id without actually inserting a new row — and the search for the
    # run-specific title would then miss.
    probe_id = uuid.uuid4().hex[:8]
    title = f"mnemon-doctor-probe-{probe_id}"
    content = f"Probe memory created by `mnemon doctor` (run {probe_id}). Safe to delete."

    # 1. Save
    try:
        save_result, _ = call_tool_sync(
            "memory_save",
            {"title": title, "content": content, "content_type": "note"},
            timeout=MCP_TIMEOUT_SEC,
            client_label="mnemon-doctor",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Round-trip (save + search + forget)",
            False,
            f"save failed: {type(exc).__name__}: {exc}",
        )

    # Server response looks like: 'Saved memory #123: "..."'
    doc_id: int | None = None
    for token in save_result.split():
        if token.startswith("#"):
            try:
                doc_id = int(token.lstrip("#").rstrip(":"))
                break
            except ValueError:
                continue

    if doc_id is None:
        return CheckResult(
            "Round-trip (save + search + forget)",
            False,
            f"could not parse doc_id from save response: {save_result[:120]}",
        )

    # 2. Search
    try:
        search_result, _ = call_tool_sync(
            "memory_search",
            {"query": title, "limit": 5},
            timeout=MCP_TIMEOUT_SEC,
            client_label="mnemon-doctor",
        )
    except Exception as exc:  # noqa: BLE001
        _best_effort_forget(doc_id)
        return CheckResult(
            "Round-trip (save + search + forget)",
            False,
            f"search failed: {type(exc).__name__}: {exc}",
        )

    if title not in search_result:
        _best_effort_forget(doc_id)
        return CheckResult(
            "Round-trip (save + search + forget)",
            False,
            f"saved memory not found by search for its own title (doc_id={doc_id})",
        )

    # 3. Forget
    try:
        call_tool_sync(
            "memory_forget",
            {"document_id": doc_id},
            timeout=MCP_TIMEOUT_SEC,
            client_label="mnemon-doctor",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Round-trip (save + search + forget)",
            True,
            f"save+search ok but forget failed: {exc} (doc_id={doc_id} leaked)",
            warn=True,
        )

    return CheckResult(
        "Round-trip (save + search + forget)",
        True,
        f"doc_id={doc_id} saved, found, and forgotten",
    )


def _best_effort_forget(doc_id: int) -> None:
    """Try to clean up a probe memory; swallow errors — caller already failed."""
    try:
        call_tool_sync(
            "memory_forget",
            {"document_id": doc_id},
            timeout=MCP_TIMEOUT_SEC,
            client_label="mnemon-doctor",
        )
    except Exception:  # noqa: BLE001
        pass


# ── Runner ─────────────────────────────────────────────────────────────────

CHECKS: list[Callable[[], CheckResult]] = [
    check_remote_url,
    check_local_token,
    check_token_file_perms,
    check_health_endpoint,
    check_auth_and_tool_call,
    check_round_trip,
]


def run_doctor(out=sys.stdout) -> int:
    """Run all checks, print results, return exit code (0 = all pass)."""
    print("mnemon doctor — running diagnostics...\n", file=out)

    results: list[CheckResult] = []
    for check in CHECKS:
        result = check()
        results.append(result)
        prefix = PASS if result.ok and not result.warn else (WARN if result.warn else FAIL)
        print(f"  {prefix} {result.name}: {result.detail}", file=out)

    failed = [r for r in results if not r.ok]
    warned = [r for r in results if r.ok and r.warn]

    print("", file=out)
    if failed:
        print(
            f"{FAIL} {len(failed)}/{len(results)} checks failed.",
            file=out,
        )
        return 1
    if warned:
        print(
            f"{WARN} All checks passed with {len(warned)} warning(s).",
            file=out,
        )
        return 0
    print(f"{PASS} All {len(results)} checks passed.", file=out)
    return 0
