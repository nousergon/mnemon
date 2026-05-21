"""Downgrade web → local: pull Fly vault back, reconfigure clients, stop.

P1d of the mnemon simplification plan
(``private/mnemon-simplification-plan-260421.md``). Symmetric counterpart
to ``mnemon upgrade web`` — users who evaluate web and decide it isn't
worth the cost get a clean exit that preserves every memory accumulated
on the remote.

Contract
--------
``mnemon downgrade local [--destroy-fly-app] [--yes] [--skip-doctor]``:

1. Require a remote config (env var or ``~/.mnemon/remote_url``).
   Refuses to run against an install that's already local-only — there
   is nothing to downgrade from.
2. ``mnemon sync pull`` → overwrite ``~/.mnemon/default.sqlite`` with
   the current S3 state (which the Fly app has been writing to).
3. Reconfigure every detected MCP client back to stdio. Invokes
   ``setup_*`` functions without ``--remote-url``, which writes local
   stdio MCP + in-process hooks (via the LocalMemoryClient added in P1a).
4. If ``--destroy-fly-app`` was passed: prompt for confirmation
   (unless ``--yes``) and run ``flyctl apps destroy <app>``.
5. Run ``mnemon doctor --fail-on-warn`` against the new local vault.
6. Print manual instructions for claude.ai / mobile (those can't be
   auto-reverted — the user has to remove MCP entries in Anthropic's UI).

Isolation overrides
-------------------
Shares ``MNEMON_CLIENT_CONFIG_ROOT`` and ``MNEMON_S3_ENDPOINT_OVERRIDE``
with :mod:`mnemon.upgrade` so Layer 2 tests can exercise the full
round-trip without touching real Fly/AWS.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


class DowngradeError(Exception):
    """Raised when downgrade cannot proceed. Message is user-facing."""


MNEMON_DIR = Path.home() / ".mnemon"
REMOTE_URL_FILE = MNEMON_DIR / "remote_url"
LOCAL_TOKEN_FILE = MNEMON_DIR / "local_token"


def _resolve_remote_url() -> str:
    """Read the current remote URL from env or ``~/.mnemon/remote_url``.

    Mirrors the resolution in :func:`mnemon.hooks._remote_client.get_remote_url`
    but raises :class:`DowngradeError` instead of
    :class:`RemoteClientConfigError` so the caller gets a downgrade-
    specific message.
    """
    env = os.environ.get("MNEMON_REMOTE_URL", "").strip()
    if env:
        return env
    try:
        if REMOTE_URL_FILE.exists():
            content = REMOTE_URL_FILE.read_text().strip()
            if content:
                return content
    except OSError:
        pass
    raise DowngradeError(
        "No remote configured — nothing to downgrade from. "
        f"Checked MNEMON_REMOTE_URL env var and {REMOTE_URL_FILE}. "
        "If your machine is already local-only, this command is a no-op."
    )


def _extract_app_name(remote_url: str) -> str | None:
    """Best-effort parse of a Fly app name from a remote URL.

    Matches ``https://<app>.fly.dev[/...]``. Returns None for any URL
    that doesn't fit the Fly pattern — the user might be self-hosting
    on a custom domain, in which case we can't safely guess the app.
    """
    m = re.match(r"https?://([a-z0-9][-a-z0-9]{0,61})\.fly\.dev(/.*)?$", remote_url)
    return m.group(1) if m else None


def _client_config_root() -> Path:
    override = os.environ.get("MNEMON_CLIENT_CONFIG_ROOT", "").strip()
    if override:
        return Path(override)
    return Path.home()


def _clear_remote_config() -> None:
    """Remove ``~/.mnemon/remote_url`` so doctor + hooks auto-detect
    local mode. Leaves the token file in place — harmless in local
    mode, and the user might want it for a re-upgrade."""
    if REMOTE_URL_FILE.exists():
        REMOTE_URL_FILE.unlink()


def _reconfigure_clients_local(detected: list[str]) -> list[str]:
    """Rewrite each detected client's MCP config back to stdio mode.

    Calls ``setup_*`` without ``--remote-url`` which installs local
    stdio MCP + in-process hooks (the LocalMemoryClient path).
    """
    from . import setup as setup_mod

    prior_home = setup_mod.Path.home
    config_root = _client_config_root()

    def _fake_home() -> Path:
        return config_root

    reconfigured: list[str] = []
    try:
        if os.environ.get("MNEMON_CLIENT_CONFIG_ROOT"):
            setup_mod.Path.home = _fake_home  # type: ignore[method-assign]
        for target in detected:
            if target == "hooks":
                # `hooks` is a pseudo-target used by setup_hooks
                # explicitly; detect_installed_clients never returns it.
                continue
            func = setup_mod.TARGETS[target]
            func(remote_url=None, token=None)
            reconfigured.append(target)
    finally:
        setup_mod.Path.home = prior_home  # type: ignore[method-assign]

    return reconfigured


def _fly_dump_vault(app_name: str) -> None:
    """SSH into the Fly machine and run ``mnemon sync push``.

    Mirror of ``upgrade._fly_seed_vault`` in the opposite direction —
    dumps the *current* Fly vault to S3 so the subsequent local
    ``mnemon sync pull`` gets up-to-date state, not whatever stale
    snapshot was last pushed at upgrade time.

    Without this step, any memory added via remote after the upgrade
    is lost on downgrade — only data that was on S3 at upgrade time
    survives the round-trip. Surfaced 2026-05-21 during the 0.6.0
    Layer-3 test as "expected 4 docs after downgrade, got '3'".
    """
    subprocess.run(
        [
            "flyctl",
            "ssh",
            "console",
            "--app",
            app_name,
            "-C",
            "mnemon sync push",
        ],
        check=True,
    )


def _fly_destroy_app(app_name: str) -> None:
    """Run ``flyctl apps destroy`` unattended (``-y``)."""
    subprocess.run(
        ["flyctl", "apps", "destroy", app_name, "-y"],
        check=True,
    )


def _confirm(prompt: str) -> bool:
    """Interactive y/n confirmation. Returns False for anything that
    isn't an explicit yes. If stdin isn't a TTY (scripted context), we
    default to False — scripted callers should pass ``--yes`` or skip
    destroy entirely."""
    if not sys.stdin.isatty():
        return False
    try:
        reply = input(prompt).strip().lower()
    except EOFError:
        return False
    return reply in {"y", "yes"}


def downgrade_local(
    *,
    destroy_fly_app: bool = False,
    yes: bool = False,
    skip_doctor: bool = False,
    app_name_override: str | None = None,
    skip_fly_push: bool = False,
) -> str:
    """Downgrade a web install back to local-only.

    See module docstring for the step-by-step contract. Returns a
    user-facing summary; raises :class:`DowngradeError` on fatal
    failures (no remote config, sync pull fails, reconfig fails).
    """
    remote_url = _resolve_remote_url()

    # Step 1: ensure S3 has the *current* Fly vault before pulling.
    # `mnemon upgrade web` pushes local→S3→Fly at upgrade time, then
    # the Fly vault evolves independently. Without a Fly→S3 dump here,
    # the subsequent pull would seed the local vault from a stale
    # snapshot (the one from upgrade time) and silently lose any
    # memory the user added via remote post-upgrade. Skipping with
    # --skip-fly-push is an operator escape hatch for the case where
    # SSH isn't reachable (machine off, etc.) and the operator
    # explicitly accepts the stale-pull data loss.
    bucket = os.environ.get("MNEMON_S3_BUCKET", "").strip()
    if not bucket:
        raise DowngradeError(
            "MNEMON_S3_BUCKET not set. Downgrade pulls from S3 to seed "
            "the restored local vault; without a bucket, the freshly-"
            "downgraded local vault would be empty. Export "
            "MNEMON_S3_BUCKET and retry."
        )

    app_name = app_name_override or _extract_app_name(remote_url)
    if app_name and not skip_fly_push:
        try:
            _fly_dump_vault(app_name)
        except subprocess.CalledProcessError as exc:
            raise DowngradeError(
                f"Fly→S3 dump failed ({exc}). Without this push the "
                f"downgrade would seed local from a stale S3 snapshot "
                f"and lose any post-upgrade remote-added memories. "
                f"Investigate via `flyctl ssh console --app {app_name}` "
                f"and retry. If you accept the data loss, rerun with "
                f"--skip-fly-push."
            ) from exc

    # Step 2: sync pull S3 → local
    from .sync import pull as s3_pull

    pull_result = s3_pull()
    if pull_result["errors"]:
        raise DowngradeError(
            "S3 pull failed: " + "; ".join(pull_result["errors"])
        )

    # Step 3: rewrite each detected client to stdio.
    # Clear remote URL *before* rewriting so setup_* picks local mode.
    _clear_remote_config()

    from .setup import detect_installed_clients

    detected = detect_installed_clients()
    reconfigured = _reconfigure_clients_local(detected)

    # Step 4 (optional): destroy the Fly app. `app_name` was resolved
    # up at the Fly→S3 dump step (Step 1) so we don't re-extract here.
    destroyed: str | None = None
    if destroy_fly_app:
        if not app_name:
            raise DowngradeError(
                f"Could not infer Fly app name from remote URL {remote_url!r}. "
                "Pass --app-name <name> explicitly if you want to destroy a "
                "non-fly.dev deployment."
            )
        if not yes:
            ok = _confirm(
                f"Destroy Fly app {app_name!r}? This is irreversible. [y/N]: "
            )
            if not ok:
                print(
                    f"Skipping destroy: Fly app {app_name!r} still running.",
                    file=sys.stderr,
                )
            else:
                _fly_destroy_app(app_name)
                destroyed = app_name
        else:
            _fly_destroy_app(app_name)
            destroyed = app_name

    summary = [
        "Downgrade to local complete.",
        f"  Restored vault: ~/.mnemon/default.sqlite (pulled from S3 bucket {bucket!r})",
    ]
    if reconfigured:
        summary.append(
            "  Reconfigured:   " + ", ".join(reconfigured) + " (stdio mode)"
        )
    if destroyed:
        summary.append(f"  Fly app:        destroyed ({destroyed})")
    elif app_name:
        summary.append(
            f"  Fly app:        {app_name} is still running. "
            "Rerun with --destroy-fly-app to tear it down."
        )

    summary += [
        "",
        "Next steps:",
        "  • Restart each client above to pick up the local MCP config.",
        "  • Manually remove the mnemon MCP entry from claude.ai and "
        "the Claude mobile app (Settings → Connected Apps).",
    ]
    if not destroyed and app_name:
        summary.append(
            f"  • `flyctl apps destroy {app_name}` when you're ready to "
            "stop paying for it."
        )

    if skip_doctor:
        return "\n".join(summary)

    # Step 5: doctor against local.
    import io

    from .doctor import run_doctor

    # Ensure the env doesn't override local-mode detection.
    prior_url_env = os.environ.pop("MNEMON_REMOTE_URL", None)
    buf = io.StringIO()
    try:
        print("", file=buf)
        print("Running mnemon doctor against the restored local vault...", file=buf)
        try:
            doctor_rc = run_doctor(out=buf, fail_on_warn=True)
        except Exception as exc:  # noqa: BLE001
            buf.write(
                f"\n(doctor invocation crashed: {type(exc).__name__}: {exc})\n"
            )
            doctor_rc = 1
    finally:
        if prior_url_env is not None:
            os.environ["MNEMON_REMOTE_URL"] = prior_url_env

    if doctor_rc != 0:
        buf.write(
            "\nNOTE: doctor reported issues against the local vault. "
            "Configs are rewritten; investigate the failing check(s) "
            "above.\n"
        )

    return "\n".join(summary) + "\n" + buf.getvalue()
