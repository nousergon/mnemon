"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _allow_local_store(monkeypatch):
    """Make the suite hermetic w.r.t. the dev machine's remote config.

    The ``Store`` chokepoint refuses to open the *default* local vault when a
    remote is configured (``~/.mnemon/remote_url`` / ``MNEMON_REMOTE_URL``) —
    the two-vaults-bug guard. On a developer machine that points at a cloud
    vault that would make every bare ``Store()`` in the suite raise. Tests
    operate on local/temp stores by design, so bypass the guard here via the
    documented override env. ``TestRemoteModeGuard`` deletes this env per-test
    to exercise the guard itself.
    """
    monkeypatch.setenv("MNEMON_ALLOW_LOCAL_STORE", "1")
