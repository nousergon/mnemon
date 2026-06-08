"""Tests for scripts/check_health.py — the health-monitor checker.

Loaded by path (it's a standalone script, not a package module) and
driven with a stubbed ``fetch_health`` so no network is touched.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_health.py"
_spec = importlib.util.spec_from_file_location("check_health", _PATH)
check_health = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_health)


def _run_with(monkeypatch, payload):
    monkeypatch.setattr(check_health, "fetch_health", lambda url, timeout=30.0: payload)
    monkeypatch.setattr(check_health.sys, "argv", ["check_health.py"])
    return check_health.main()


def _healthy_metrics(**ov):
    m = {
        "in_memory_hits": 100,
        "resume_hits": 0,
        "fresh_inits": 10,
        "stale_session_misses": 0,
        "persisted_sessions_total": 12,
        "in_memory_sessions_current": 8,
        "oldest_session_age_seconds": 3600,  # 1h — well under TTL
    }
    m.update(ov)
    return m


def test_healthy_passes(monkeypatch):
    rc = _run_with(monkeypatch, {"status": "ok", "metrics": _healthy_metrics()})
    assert rc == 0


def test_high_volume_no_longer_warns(monkeypatch, capsys):
    # Issue #208 regression: a large persisted_sessions_total is healthy
    # steady-state and must NOT trip the checker now.
    rc = _run_with(
        monkeypatch,
        {"status": "ok", "metrics": _healthy_metrics(persisted_sessions_total=8327)},
    )
    assert rc == 0
    assert "WARN" not in capsys.readouterr().out


def test_stalled_prune_warns(monkeypatch, capsys):
    # Oldest session well past TTL + grace → prune-stalled WARN (exit 2).
    overdue = check_health.SESSION_AGE_WARN_SECONDS + 86400
    rc = _run_with(
        monkeypatch,
        {"status": "ok", "metrics": _healthy_metrics(oldest_session_age_seconds=overdue)},
    )
    assert rc == 2
    assert "prune" in capsys.readouterr().out.lower()


def test_stale_session_misses_is_hard_failure(monkeypatch):
    rc = _run_with(
        monkeypatch,
        {"status": "ok", "metrics": _healthy_metrics(stale_session_misses=3)},
    )
    assert rc == 1


def test_status_not_ok_fails(monkeypatch):
    rc = _run_with(monkeypatch, {"status": "degraded", "metrics": _healthy_metrics()})
    assert rc == 1


def test_missing_persisted_total_is_schema_drift(monkeypatch):
    m = _healthy_metrics()
    del m["persisted_sessions_total"]
    rc = _run_with(monkeypatch, {"status": "ok", "metrics": m})
    assert rc == 1


def test_missing_oldest_age_is_tolerated(monkeypatch):
    # Pre-0.7.1 deploys lack the new metric — must not false-alarm.
    m = _healthy_metrics()
    del m["oldest_session_age_seconds"]
    rc = _run_with(monkeypatch, {"status": "ok", "metrics": m})
    assert rc == 0


def test_threshold_is_ttl_plus_two_prune_cycles():
    assert check_health.SESSION_AGE_WARN_SECONDS == 7 * 24 * 3600 + 2 * 6 * 3600
