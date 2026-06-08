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
    2 — soft warning (session prune appears stalled — oldest session far
        past the TTL)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "https://mnemon-memory.fly.dev/health"

# Prune-health threshold (volume-INDEPENDENT). The previous check warned
# on persisted_sessions_total >= 5000, but that metric only counts
# *non-expired* rows (count() filters WHERE last_active_at > cutoff), so
# it can never reveal a broken prune — it just tracks legitimate session
# volume and fired perpetually for active users (issue #208).
#
# The real failure mode is the periodic expire_old() task stalling, which
# lets rows survive past the 7-day TTL. oldest_session_age_seconds
# (added 0.7.1) surfaces exactly that and is independent of how many
# sessions a user creates. Under a working prune nothing lives past
# TTL (7d) + one prune interval (6h); we warn at TTL + 2 intervals (12h
# grace) so a single skipped cycle doesn't trip it.
_TTL_SECONDS = 7 * 24 * 3600              # DEFAULT_TTL_SECONDS
_PRUNE_INTERVAL_SECONDS = 6 * 3600        # DEFAULT_EXPIRE_INTERVAL_SECONDS
SESSION_AGE_WARN_SECONDS = _TTL_SECONDS + 2 * _PRUNE_INTERVAL_SECONDS


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

    # persisted_sessions_total is kept for observability but NOT
    # thresholded — high legitimate volume is healthy. Its absence still
    # signals schema drift.
    if metrics.get("persisted_sessions_total") is None:
        failures.append("metrics.persisted_sessions_total missing — schema drift")

    # Prune-health: oldest session far past the TTL means expire_old()
    # stalled. Absent on pre-0.7.1 deploys — skip rather than fail so a
    # rollback doesn't false-alarm (mirrors the metrics-absent handling).
    oldest = metrics.get("oldest_session_age_seconds")
    if oldest is not None and oldest > SESSION_AGE_WARN_SECONDS:
        oldest_days = oldest / 86400
        warnings.append(
            f"metrics.oldest_session_age_seconds = {oldest} ({oldest_days:.1f}d) "
            f"> {SESSION_AGE_WARN_SECONDS} ({SESSION_AGE_WARN_SECONDS / 86400:.1f}d). "
            f"A session is surviving well past the 7-day TTL — the periodic "
            f"expire_old() prune has likely stopped running; check the server's "
            f"lifespan task group / logs."
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
