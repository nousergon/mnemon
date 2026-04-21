"""Self-diagnostic for mnemon deployments.

Runs an ordered series of checks against the configured vault and prints
pass/fail per check. Works in both modes:

- **Remote mode** (``MNEMON_REMOTE_URL`` env or ``~/.mnemon/remote_url``
  set): validates the Fly-hosted MCP endpoint end-to-end, including
  OAuth AS metadata when ``MNEMON_AS_ENABLED=true``.
- **Local mode** (neither set): validates the on-disk SQLite vault —
  schema, embedder load, and a save/search/forget round-trip via
  ``Store`` directly.

The caller does not pick a mode; ``run_doctor`` detects it from config
and runs the appropriate check list. Output format is identical across
modes so hook/CI integrations don't have to branch.

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
from pathlib import Path
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


def check_oauth_as_metadata() -> CheckResult:
    """GET /.well-known/oauth-authorization-server to verify the
    self-hosted Authorization Server is serving valid RFC 8414 metadata.

    This exercises the browser-client auth path that ``check_auth_and_
    tool_call`` can't reach — claude.ai, Claude Desktop etc. rely on
    this endpoint to discover token/authorize URLs. A 404 means
    ``MNEMON_AS_ENABLED`` is not set (local-token-only deployment —
    legitimate, but browser clients won't work): warn rather than fail.
    """
    try:
        url = get_remote_url()
    except RemoteClientConfigError as exc:
        return CheckResult("OAuth AS metadata", False, str(exc))

    base = url.rstrip("/")
    if base.endswith("/mcp"):
        base = base[: -len("/mcp")]
    metadata_url = f"{base}/.well-known/oauth-authorization-server"

    try:
        with urllib.request.urlopen(metadata_url, timeout=HEALTH_TIMEOUT_SEC) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return CheckResult(
                "OAuth AS metadata",
                True,
                "AS not enabled (MNEMON_AS_ENABLED unset) — browser clients won't work",
                warn=True,
            )
        return CheckResult(
            "OAuth AS metadata", False, f"{metadata_url}: HTTP {exc.code}"
        )
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        return CheckResult("OAuth AS metadata", False, f"{metadata_url}: {exc}")

    if status != 200:
        return CheckResult(
            "OAuth AS metadata", False, f"HTTP {status} from {metadata_url}"
        )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return CheckResult(
            "OAuth AS metadata",
            False,
            f"non-JSON body from {metadata_url}: {body[:80]}",
        )

    required = [
        "issuer",
        "authorization_endpoint",
        "token_endpoint",
        "registration_endpoint",
    ]
    missing = [f for f in required if f not in payload]
    if missing:
        return CheckResult(
            "OAuth AS metadata",
            False,
            f"missing required RFC 8414 fields: {', '.join(missing)}",
        )

    # Issuer must equal the deployment base — mismatches silently break
    # browser clients by pointing them at the wrong AS. Commonly caused
    # by a typo in MNEMON_PUBLIC_URL.
    if payload["issuer"].rstrip("/") != base:
        return CheckResult(
            "OAuth AS metadata",
            False,
            f"issuer {payload['issuer']!r} ≠ deployment base {base!r} "
            "(check MNEMON_PUBLIC_URL)",
        )

    return CheckResult(
        "OAuth AS metadata",
        True,
        f"issuer={payload['issuer']}, "
        "authorize/token/register endpoints present",
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
            {"id": doc_id},
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
            {"id": doc_id},
            timeout=MCP_TIMEOUT_SEC,
            client_label="mnemon-doctor",
        )
    except Exception:  # noqa: BLE001
        pass


# ── Local-mode checks ──────────────────────────────────────────────────────


def check_local_vault() -> CheckResult:
    """Open the local Store and confirm the SQLite schema is in place."""
    try:
        from .store import Store
        store = Store()
    except Exception as exc:  # noqa: BLE001 — schema/permission errors surface here
        return CheckResult("Local vault reachable", False, f"{type(exc).__name__}: {exc}")

    try:
        stats = store.status()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Local vault reachable",
            False,
            f"status() failed: {type(exc).__name__}: {exc}",
        )

    vault_path = stats.get("vault_path", "?")
    total = stats.get("total_documents", 0)
    return CheckResult(
        "Local vault reachable",
        True,
        f"{vault_path} — {total} documents",
    )


def check_local_embedder() -> CheckResult:
    """Force a FastEmbed cold-load so the failure (missing model, no
    network on first run, etc.) surfaces here rather than inside the
    round-trip check — makes the diagnostic clearer."""
    try:
        from .embedder import embed
        vec = embed("__mnemon_doctor_probe__")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Embedder loadable",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    if vec is None or getattr(vec, "shape", ()) != (384,):
        return CheckResult(
            "Embedder loadable",
            False,
            f"unexpected vector shape: {getattr(vec, 'shape', None)}",
        )
    return CheckResult("Embedder loadable", True, "bge-small-en-v1.5 (384d)")


def check_local_round_trip() -> CheckResult:
    """Save a probe memory, search for it by title, then forget it — all
    via ``Store`` directly so we're checking the local path that hooks
    and the CLI actually use, not the MCP stdio layer."""
    import uuid
    from .search import search as hybrid_search
    from .store import Store

    probe_id = uuid.uuid4().hex[:8]
    title = f"mnemon-doctor-probe-{probe_id}"
    content = f"Probe memory created by `mnemon doctor` (run {probe_id}). Safe to delete."

    try:
        store = Store()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Round-trip (save + search + forget)",
            False,
            f"Store() failed: {type(exc).__name__}: {exc}",
        )

    try:
        doc_id = store.save(title=title, content=content, content_type="note")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Round-trip (save + search + forget)",
            False,
            f"save failed: {type(exc).__name__}: {exc}",
        )

    try:
        results = hybrid_search(store, title, limit=5, use_vector=True)
    except Exception as exc:  # noqa: BLE001
        _best_effort_local_forget(store, doc_id)
        return CheckResult(
            "Round-trip (save + search + forget)",
            False,
            f"search failed: {type(exc).__name__}: {exc}",
        )

    if not any(getattr(r, "title", None) == title for r in results):
        _best_effort_local_forget(store, doc_id)
        return CheckResult(
            "Round-trip (save + search + forget)",
            False,
            f"saved memory not found by search for its own title (doc_id={doc_id})",
        )

    try:
        store.forget(doc_id)
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


def _best_effort_local_forget(store, doc_id: int) -> None:
    """Clean up a probe memory from the local Store; swallow errors."""
    try:
        store.forget(doc_id)
    except Exception:  # noqa: BLE001
        pass


# ── Runner ─────────────────────────────────────────────────────────────────

REMOTE_CHECKS: list[Callable[[], CheckResult]] = [
    check_remote_url,
    check_local_token,
    check_token_file_perms,
    check_health_endpoint,
    check_oauth_as_metadata,
    check_auth_and_tool_call,
    check_round_trip,
]

LOCAL_CHECKS: list[Callable[[], CheckResult]] = [
    check_local_vault,
    check_local_embedder,
    check_local_round_trip,
]

# Backwards-compat shim — one external caller (tests) may still import CHECKS.
CHECKS = REMOTE_CHECKS


def _has_remote_config() -> bool:
    """True if the user has pointed mnemon at a remote vault.

    Mirrors ``mnemon.dashboard.loaders._use_remote`` but kept local to
    avoid pulling the dashboard's streamlit import chain into the CLI.
    """
    if os.environ.get("MNEMON_REMOTE_URL", "").strip():
        return True
    remote_url_file = Path.home() / ".mnemon" / "remote_url"
    if remote_url_file.exists() and remote_url_file.read_text().strip():
        return True
    return False


def run_doctor(out=sys.stdout, *, fail_on_warn: bool = False) -> int:
    """Run all checks, print results, return exit code (0 = all pass).

    Auto-detects remote vs. local mode from config. Prints a one-line
    banner so the user knows which flavor ran; everything else is
    format-identical across modes.

    Parameters
    ----------
    out:
        File-like object for the human-readable report.
    fail_on_warn:
        When True (default False), warnings are treated as failures and
        the exit code is 1 if any check emits a warning. Used by
        ``run_setup`` / ``mnemon upgrade web`` / ``mnemon downgrade
        local`` so scripted installs don't silently succeed in a
        partially-broken state. Consistent with the "hard-fail until
        stabilized" preference captured in the simplification plan
        (``private/mnemon-simplification-plan-260421.md``).
    """
    if _has_remote_config():
        mode_label = "remote"
        try:
            mode_detail = get_remote_url()
        except RemoteClientConfigError:
            mode_detail = "(unresolved)"
        checks = REMOTE_CHECKS
    else:
        mode_label = "local"
        try:
            from .config import vault_path
            mode_detail = str(vault_path())
        except Exception:  # noqa: BLE001
            mode_detail = "~/.mnemon/default.sqlite"
        checks = LOCAL_CHECKS

    print(
        f"mnemon doctor — {mode_label} mode ({mode_detail})\n",
        file=out,
    )

    results: list[CheckResult] = []
    for check in checks:
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
        if fail_on_warn:
            print(
                f"{FAIL} {len(warned)}/{len(results)} checks emitted "
                f"warnings (--fail-on-warn treats these as failures).",
                file=out,
            )
            return 1
        print(
            f"{WARN} All checks passed with {len(warned)} warning(s).",
            file=out,
        )
        return 0
    print(f"{PASS} All {len(results)} checks passed.", file=out)
    return 0
