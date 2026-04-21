"""Memory client abstraction for mnemon hooks.

P1a of the mnemon simplification plan (see
``private/mnemon-simplification-plan-260421.md``). Replaces the previous
HTTP-only hook data layer (:mod:`mnemon.hooks._remote_client`) with a
:class:`MemoryClient` protocol that has two implementations:

- :class:`LocalMemoryClient` — dispatches in-process via :mod:`mnemon.api`
  against the on-disk SQLite vault.
- :class:`RemoteMemoryClient` — wraps the existing
  :func:`~mnemon.hooks._remote_client.call_tool_sync` for remote vaults.

The factory :func:`get_client` picks between them by checking whether a
remote URL is configured (env var or ``~/.mnemon/remote_url`` file). If
yes, the user is in web mode — route calls over HTTP. If no, the user is
local-only — dispatch in-process.

Why this matters
----------------
Before P1a, ``mnemon setup claude-code`` without ``--remote-url`` would
happily install hooks whose very first line imported a URL-required
module. The hooks would then emit a ``RemoteClientConfigError`` banner
on every prompt while the MCP tools kept working, making the broken
state easy to miss. With this abstraction, the same hooks work in both
modes and the decision point is a single import-safe config check.

Backwards compatibility
-----------------------
:mod:`mnemon.hooks._remote_client` remains the low-level HTTP
implementation. Tests that patched ``_remote_client.call_tool_sync``
continue to work because :class:`RemoteMemoryClient` delegates there.
New tests should prefer patching :func:`get_client` or injecting a
specific client for clarity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from . import _remote_client
from ._remote_client import (
    DEFAULT_CLIENT_LABEL,
    DEFAULT_TIMEOUT_SEC,
    RemoteClientConfigError,
)

# Re-export for callers that do ``from ._client import RemoteClientConfigError``
__all__ = [
    "DEFAULT_CLIENT_LABEL",
    "DEFAULT_TIMEOUT_SEC",
    "LocalMemoryClient",
    "MemoryClient",
    "RemoteClientConfigError",
    "RemoteMemoryClient",
    "get_client",
    "has_remote_config",
]


MNEMON_DIR = Path.home() / ".mnemon"
REMOTE_URL_FILE = MNEMON_DIR / "remote_url"


def has_remote_config() -> bool:
    """True if the user has pointed mnemon at a remote vault.

    Checks ``MNEMON_REMOTE_URL`` env var first, then
    ``~/.mnemon/remote_url``. Mirrors the resolution order in
    :func:`mnemon.hooks._remote_client.get_remote_url` without raising
    on miss — this helper is the "should we go remote?" decision point,
    not a URL resolver.
    """
    if os.environ.get("MNEMON_REMOTE_URL", "").strip():
        return True
    try:
        if REMOTE_URL_FILE.exists():
            if REMOTE_URL_FILE.read_text().strip():
                return True
    except OSError:
        return False
    return False


@runtime_checkable
class MemoryClient(Protocol):
    """Protocol for a mnemon memory client.

    Implementations: :class:`LocalMemoryClient` (in-process,
    SQLite-backed) and :class:`RemoteMemoryClient` (HTTP, Fly-backed).
    Hooks should depend on this protocol, not either concrete class.
    """

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        client_label: str = DEFAULT_CLIENT_LABEL,
    ) -> tuple[str, float]:
        """Invoke an MCP tool by name.

        Returns ``(result_text, elapsed_seconds)``. The tuple shape is
        intentionally identical to
        :func:`mnemon.hooks._remote_client.call_tool_sync` so callers can
        migrate without touching call sites.

        Raises on failure — never returns a partial/empty result on
        error. Hooks catch and log.
        """
        ...


class RemoteMemoryClient:
    """HTTP-backed mnemon client for web-mode users.

    Thin wrapper around :func:`~mnemon.hooks._remote_client.call_tool_sync`
    so the URL + token resolution and MCP SDK handshake stay in one place.
    """

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        client_label: str = DEFAULT_CLIENT_LABEL,
    ) -> tuple[str, float]:
        # Access via module (not a local binding) so tests patching
        # ``mnemon.hooks._remote_client.call_tool_sync`` still intercept
        # the remote call through this wrapper.
        return _remote_client.call_tool_sync(
            tool_name,
            arguments,
            timeout=timeout,
            client_label=client_label,
        )


class LocalMemoryClient:
    """In-process mnemon client for local-only users.

    Dispatches via :func:`mnemon.api.dispatch` against the on-disk
    SQLite vault. The ``timeout`` and ``client_label`` arguments are
    accepted for protocol parity but unused — there is no network hop
    to time out on and no remote server to attribute requests to.

    Raises :class:`mnemon.api.UnsupportedToolError` if called with a
    tool name the in-process surface doesn't implement.
    """

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = DEFAULT_TIMEOUT_SEC,  # noqa: ARG002 — protocol parity
        client_label: str = DEFAULT_CLIENT_LABEL,  # noqa: ARG002 — protocol parity
    ) -> tuple[str, float]:
        import time

        from ..api import dispatch

        t0 = time.monotonic()
        result = dispatch(tool_name, arguments)
        return result, time.monotonic() - t0


def get_client() -> MemoryClient:
    """Pick the right client based on current config.

    - If ``MNEMON_REMOTE_URL`` (env or file) is set → :class:`RemoteMemoryClient`.
    - Otherwise → :class:`LocalMemoryClient`.

    This is the single decision point for hook/doctor/setup code that
    needs a memory client. Callers should not branch on mode themselves
    — that defeats the whole point of the abstraction and is how we got
    into the P0 mess in the first place.
    """
    if has_remote_config():
        return RemoteMemoryClient()
    return LocalMemoryClient()
