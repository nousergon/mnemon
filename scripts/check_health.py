#!/usr/bin/env python3
"""Health-check script for the public mnemon Fly deployment.

Designed to run from GitHub Actions on a schedule. Exit non-zero on any
condition the alpha-test watch list flags as "investigation-worthy" so
the workflow run shows red and you see it in the GitHub UI.

Usage::

    python scripts/check_health.py [--url https://mnemon-memory.fly.dev/health]

Exit codes:
    0 — all checks passed
    1 — hard failure (status != ok, stale_session_misses > 0, schema drift)
    2 — soft warning (persisted_sessions_total above threshold)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "https://mnemon-memory.fly.dev/health"

# Soft cap. Since 0.6.0rc12, PersistentSessionManager runs a periodic
# expire_old() task on a 6h tick (DEFAULT_EXPIRE_INTERVAL_SECONDS in
# persistent_sessions.py), so unbounded growth is no longer the failure
# mode this guarded against. The remaining drift is steady-state count
# under the 7-day TTL; the threshold trips when actual session volume
# outpaces the prune cadence — investigate by lowering TTL or tightening
# the prune interval rather than assuming pruning is broken.
PERSISTED_SESSIONS_WARN_THRESHOLD = 5_000


def fetch_health(url: str, timeout: float = 30.0) -> dict:
    # 30s tolerates a Fly cold-start (machine wake + Python boot + bge-small
    # ONNX load + SQLite/FastMCP startup). With min_machines_running=0 the
    # overnight idle window auto-stops the machine; a 10s read timeout raced
    # the wake-up on 2026-05-07 (issue #117). /health responds in <200ms once warm.
    req = urllib.request.Request(url, headers={"User-Agent": "mnemon-health-monitor/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"non-200 status from {url}: {resp.status}")
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    try:
        payload = fetch_health(args.url)
    except (urllib.error.URLError, RuntimeError, TimeoutError) as e:
        print(f"FAIL: could not fetch {args.url}: {type(e).__name__}: {e}")
        return 1

    if payload.get("status") != "ok":
        print(f"FAIL: status field is {payload.get('status')!r}, expected 'ok'")
        print(f"  full payload: {json.dumps(payload)}")
        return 1

    metrics = payload.get("metrics")
    if metrics is None:
        # Pre-PR-#85 deployments don't expose metrics. Treat as warning, not
        # failure — the deploy might be a temporary rollback.
        print("WARN: /health did not return a 'metrics' field — server may be on pre-rc5")
        print(f"  full payload: {json.dumps(payload)}")
        return 2

    failures: list[str] = []
    warnings: list[str] = []

    stale = metrics.get("stale_session_misses")
    if stale is None:
        failures.append("metrics.stale_session_misses missing — schema drift")
    elif stale > 0:
        failures.append(
            f"metrics.stale_session_misses = {stale} (>0 means an unrecoverable "
            f"'Session not found' was returned to a client; investigate "
            f"persistent_sessions.py registration path)"
        )

    persisted = metrics.get("persisted_sessions_total")
    if persisted is None:
        failures.append("metrics.persisted_sessions_total missing — schema drift")
    elif persisted >= PERSISTED_SESSIONS_WARN_THRESHOLD:
        warnings.append(
            f"metrics.persisted_sessions_total = {persisted} "
            f"(>= {PERSISTED_SESSIONS_WARN_THRESHOLD}). Periodic prune runs "
            f"every 6h since rc12, so this is steady-state count under the "
            f"7-day TTL — investigate by lowering TTL or tightening the prune "
            f"interval if the warning fires repeatedly."
        )

    print(f"OK: status={payload['status']}, metrics={json.dumps(metrics)}")

    if failures:
        print()
        for line in failures:
            print(f"FAIL: {line}")
        return 1

    if warnings:
        print()
        for line in warnings:
            print(f"WARN: {line}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
