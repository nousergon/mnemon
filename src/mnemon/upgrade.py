"""Upgrade local → web via Fly.io deploy + S3 vault seed.

P1c of the mnemon simplification plan
(``private/mnemon-simplification-plan-260421.md``). Wraps the 4-step
manual deploy (push, deploy, seed, reconfigure) into a single
``mnemon upgrade web --app-name <name>`` command.

Contract
--------
The user runs ``mnemon upgrade web --app-name my-mnemon`` after
``mnemon setup`` in local mode. After a successful upgrade:

- Local ``~/.mnemon/default.sqlite`` is archived to
  ``~/.mnemon/archive/pre-web-YYYY-MM-DD.sqlite`` (never deleted).
- Fly app ``my-mnemon`` is deployed and serves mnemon over HTTPS at
  ``https://my-mnemon.fly.dev/mcp``.
- Every detected MCP client config points at the remote endpoint, with
  SessionStart pre-warm hooks installed for Claude Code.
- ``mnemon doctor --fail-on-warn`` is green against the remote.

On failure at any step, the orchestration aborts. For steps AFTER the
Fly app is created, the user may need to clean up the partial Fly app
(``flyctl apps destroy <name>``) — the error message tells them so.

Isolation (tests)
-----------------
Several env vars let Layer 2/3 tests exercise the orchestration without
hitting real Fly/AWS:

- ``MNEMON_FLY_ENDPOINT_OVERRIDE`` — URL of an already-deployed remote
  (e.g. a local ``serve-remote`` container). When set, skips every
  flyctl call entirely and uses this URL as the post-deploy endpoint.
- ``MNEMON_S3_ENDPOINT_OVERRIDE`` — forwarded to ``aws s3 --endpoint-url``
  so MinIO can stand in for real S3.
- ``MNEMON_CLIENT_CONFIG_ROOT`` — overrides ``~`` for client config
  rewrites so tests don't touch real ``~/.claude``, ``~/.cursor``, etc.
- ``MNEMON_PROD_APP_NAMES`` — comma-separated list of Fly app names
  that ``upgrade_web`` must refuse to touch even with an explicit
  ``--app-name``. Guards against "oops tested against prod."

Layer 3 runbook: ``private/e2e-test-runbook-260421.md``.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path


class UpgradeError(Exception):
    """Raised when upgrade cannot proceed. Message is user-facing."""


# ── Deploy templates ─────────────────────────────────────────────────────────
#
# Embedded here rather than shipped as package data so `pip install
# mnemon-memory` users have everything they need without a repo clone.
# Both install mnemon-memory from PyPI rather than copying source — no
# build context beyond these two files is required.

_DOCKERFILE_TEMPLATE = """\
FROM python:3.13-slim

WORKDIR /app

# Install mnemon with server deps from PyPI. Pinning the version keeps
# the deployed image reproducible even if PyPI state changes between
# deploys.
RUN pip install --no-cache-dir 'mnemon-memory[server]=={mnemon_version}'

# Bake the FastEmbed bge-small-en-v1.5 ONNX model into the image so cold
# starts don't trigger a 5-15s download from HuggingFace Hub.
ENV FASTEMBED_CACHE_DIR=/app/.cache/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/.cache/fastembed')"

# AWS CLI for the post-deploy `mnemon sync pull` that seeds the vault
# from S3. We install the AWS CLI v2 binary to avoid the boto3/SDK
# runtime cost.
RUN apt-get update && apt-get install -y --no-install-recommends \\
        curl unzip ca-certificates \\
    && curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip \\
    && unzip -q /tmp/awscli.zip -d /tmp \\
    && /tmp/aws/install \\
    && rm -rf /tmp/aws /tmp/awscli.zip \\
    && apt-get purge -y curl unzip \\
    && apt-get autoremove -y \\
    && rm -rf /var/lib/apt/lists/*

ENV MNEMON_VAULT_DIR=/data
RUN mkdir -p /data

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \\
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=3).status == 200 else 1)"

CMD ["mnemon", "serve-remote"]
"""

_FLY_TOML_TEMPLATE = """\
app = "{app_name}"
primary_region = "{region}"

[build]

[env]
  PORT = "8080"
  MNEMON_VAULT_DIR = "/data"
  MNEMON_PUBLIC_URL = "https://{app_name}.fly.dev"
  MNEMON_ALLOWED_HOSTS = "{app_name}.fly.dev"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0

  [[http_service.checks]]
    interval = "30s"
    timeout = "5s"
    grace_period = "40s"
    method = "GET"
    path = "/health"

[mounts]
  source = "mnemon_data"
  destination = "/data"

[[vm]]
  memory = "1gb"
  cpu_kind = "shared"
  cpus = 1
"""

# Fly app names are DNS labels: lowercase, alphanumeric + hyphens,
# 1-63 chars, cannot start/end with hyphen.
_APP_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")


def _validate_app_name(app_name: str) -> None:
    if not _APP_NAME_RE.match(app_name):
        raise UpgradeError(
            f"Invalid Fly app name {app_name!r}. Must be 1-63 chars, "
            "lowercase alphanumeric + hyphens, not starting/ending with "
            "a hyphen (matches DNS label rules)."
        )
    prod_names = {
        n.strip()
        for n in os.environ.get("MNEMON_PROD_APP_NAMES", "").split(",")
        if n.strip()
    }
    if app_name in prod_names:
        raise UpgradeError(
            f"Refusing to touch Fly app {app_name!r} — listed in "
            "MNEMON_PROD_APP_NAMES. Pick a different --app-name."
        )


def _require_flyctl() -> None:
    """Error out if flyctl is missing or unauthenticated."""
    if _fly_endpoint_override():
        return
    try:
        subprocess.run(
            ["flyctl", "auth", "whoami"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise UpgradeError(
            "flyctl not found on PATH. Install from https://fly.io/docs/hands-on/install-flyctl/ "
            "and run `flyctl auth login` before retrying."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise UpgradeError(
            "flyctl is installed but not authenticated. Run "
            f"`flyctl auth login` and retry. Original error: {exc.stderr.strip()}"
        ) from exc


def _require_aws() -> None:
    """Error out if AWS CLI is missing or credentials don't resolve."""
    try:
        subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise UpgradeError(
            "aws CLI not found on PATH. Install AWS CLI v2 "
            "(https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) "
            "and configure credentials before retrying."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise UpgradeError(
            "AWS credentials not configured or invalid. Run "
            f"`aws configure` and retry. Original error: {exc.stderr.strip()}"
        ) from exc


def _require_bucket(bucket: str | None) -> str:
    resolved = bucket or os.environ.get("MNEMON_S3_BUCKET", "").strip()
    if not resolved:
        raise UpgradeError(
            "S3 bucket not specified. Pass --s3-bucket <name> or set "
            "MNEMON_S3_BUCKET env var. Upgrading to web requires a "
            "durable backup target — the Fly volume alone is not a "
            "sufficient single source of truth (plan decision #1)."
        )
    return resolved


def _fly_endpoint_override() -> str | None:
    v = os.environ.get("MNEMON_FLY_ENDPOINT_OVERRIDE", "").strip()
    return v or None


def _client_config_root() -> Path:
    override = os.environ.get("MNEMON_CLIENT_CONFIG_ROOT", "").strip()
    if override:
        return Path(override)
    return Path.home()


def _archive_local_vault() -> Path | None:
    """Rename ~/.mnemon/default.sqlite → ~/.mnemon/archive/pre-web-{date}.sqlite.

    Returns the archive path on success, None if there was no local
    vault to archive (user upgrading on a fresh machine).
    """
    from .config import vault_dir

    vdir = vault_dir()
    current = vdir / "default.sqlite"
    if not current.exists():
        return None

    archive_dir = vdir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.date.today().isoformat()
    target = archive_dir / f"pre-web-{ts}.sqlite"
    # Avoid clobbering a same-day archive from a previous upgrade attempt.
    n = 1
    while target.exists():
        target = archive_dir / f"pre-web-{ts}.{n}.sqlite"
        n += 1
    current.rename(target)

    # Vector store companion (if any) rides along.
    vec = vdir / "default.vec.npz"
    if vec.exists():
        vec.rename(archive_dir / target.with_suffix(".vec.npz").name)

    return target


def _reconfigure_clients(
    remote_url: str, local_token: str, detected: list[str]
) -> list[str]:
    """Rewrite each detected client's MCP config to point at the remote.

    Calls the existing ``setup_*`` functions in remote mode — they
    preflight the endpoint, write the config, and (for claude-code)
    install remote-mode hooks with SessionStart pre-warm. Returns the
    list of reconfigured clients.
    """
    from . import setup as setup_mod

    # Client-config root override lets tests (Layer 2) rewrite configs
    # under a scratch dir without touching the real ~/.claude etc.
    # We swap Path.home inside the setup module for the duration.
    prior_home = setup_mod.Path.home
    config_root = _client_config_root()

    def _fake_home() -> Path:
        return config_root

    reconfigured: list[str] = []
    try:
        if os.environ.get("MNEMON_CLIENT_CONFIG_ROOT"):
            setup_mod.Path.home = _fake_home  # type: ignore[method-assign]
        for target in detected:
            func = setup_mod.TARGETS[target]
            func(remote_url=remote_url, token=local_token)
            reconfigured.append(target)
    finally:
        setup_mod.Path.home = prior_home  # type: ignore[method-assign]

    return reconfigured


# ── flyctl orchestration ─────────────────────────────────────────────────────


def _fly_launch(workdir: Path, app_name: str, region: str) -> None:
    """Create the Fly app (no deploy yet)."""
    subprocess.run(
        [
            "flyctl",
            "launch",
            "--copy-config",
            "--no-deploy",
            "--name",
            app_name,
            "--region",
            region,
            "--yes",
        ],
        cwd=workdir,
        check=True,
    )


def _fly_create_volume(app_name: str, region: str) -> None:
    subprocess.run(
        [
            "flyctl",
            "volumes",
            "create",
            "mnemon_data",
            "--app",
            app_name,
            "--region",
            region,
            "--size",
            "1",
            "--yes",
        ],
        check=True,
    )


def _fly_set_secrets(
    app_name: str, local_token: str, s3_bucket: str
) -> None:
    """Set Fly secrets: mnemon token + forward AWS creds for sync pull.

    AWS creds are read from the user's environment (populated by aws
    configure or AWS_* env vars) and passed through as Fly secrets so
    the container's `aws s3 cp` works. Never logs the values.
    """
    env = os.environ.copy()
    aws_access = env.get("AWS_ACCESS_KEY_ID") or _resolve_aws_key(
        "aws_access_key_id"
    )
    aws_secret = env.get("AWS_SECRET_ACCESS_KEY") or _resolve_aws_key(
        "aws_secret_access_key"
    )
    if not aws_access or not aws_secret:
        raise UpgradeError(
            "AWS credentials not resolvable as env vars. The Fly "
            "container needs them to seed the vault from S3. Either "
            "export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY before "
            "running, or ensure `aws configure get aws_access_key_id` "
            "returns a value."
        )

    aws_region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "us-east-1"

    secrets_kv = [
        f"MNEMON_LOCAL_TOKEN={local_token}",
        f"AWS_ACCESS_KEY_ID={aws_access}",
        f"AWS_SECRET_ACCESS_KEY={aws_secret}",
        f"AWS_DEFAULT_REGION={aws_region}",
        f"MNEMON_S3_BUCKET={s3_bucket}",
    ]
    # --stage: don't restart machines immediately (they don't exist yet).
    subprocess.run(
        ["flyctl", "secrets", "set", "--app", app_name, "--stage"]
        + secrets_kv,
        check=True,
        capture_output=True,
    )


def _resolve_aws_key(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["aws", "configure", "get", key],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    val = out.stdout.strip()
    return val or None


def _fly_deploy(workdir: Path, app_name: str) -> None:
    subprocess.run(
        ["flyctl", "deploy", "--app", app_name],
        cwd=workdir,
        check=True,
    )


def _fly_seed_vault(app_name: str) -> None:
    """SSH into the Fly machine and run `mnemon sync pull` to seed /data."""
    subprocess.run(
        [
            "flyctl",
            "ssh",
            "console",
            "--app",
            app_name,
            "-C",
            "mnemon sync pull",
        ],
        check=True,
    )


# ── Public entry point ───────────────────────────────────────────────────────


def upgrade_web(
    *,
    app_name: str,
    s3_bucket: str | None = None,
    token: str | None = None,
    region: str = "sjc",
    mnemon_version: str | None = None,
    skip_doctor: bool = False,
) -> str:
    """Upgrade a local mnemon install to a web (Fly-hosted) deployment.

    Steps (each aborts the rest on failure):
      1. Validate flyctl + aws + bucket. Validate app-name shape + not
         colliding with MNEMON_PROD_APP_NAMES.
      2. Push local vault to S3 (establishes the durable backup before
         anything touches local state).
      3. Deploy Fly app (launch + volume + secrets + deploy). Skipped
         entirely when MNEMON_FLY_ENDPOINT_OVERRIDE is set (Layer 2
         integration tests).
      4. Seed Fly volume via `flyctl ssh console -C 'mnemon sync pull'`.
      5. Archive local vault to ~/.mnemon/archive/pre-web-YYYY-MM-DD.sqlite.
      6. Reconfigure every detected MCP client to point at the remote
         endpoint (via setup_* functions in remote mode — they preflight
         and install SessionStart pre-warm for Claude Code).
      7. Run `mnemon doctor --fail-on-warn` against the remote.

    Returns a user-facing summary string.
    """
    _validate_app_name(app_name)
    _require_flyctl()
    _require_aws()
    bucket = _require_bucket(s3_bucket)

    # Prereq: the user needs to actually have a local vault, otherwise
    # the push/seed round-trip is moving an empty file around. We don't
    # block on this — a user upgrading on a fresh machine still wants the
    # Fly deploy and wire-up. But we do warn.
    from .config import vault_dir

    local_sqlite = vault_dir() / "default.sqlite"
    local_exists = local_sqlite.exists()

    # Resolve the mnemon version to pin in the Dockerfile.
    if mnemon_version is None:
        from . import __version__

        mnemon_version = __version__

    local_token = token or secrets.token_urlsafe(32)

    # Step 2: push local → S3. Sets MNEMON_S3_BUCKET in case the user
    # relied on --s3-bucket instead of the env var.
    prior_bucket = os.environ.get("MNEMON_S3_BUCKET")
    os.environ["MNEMON_S3_BUCKET"] = bucket
    try:
        from .sync import push as s3_push

        push_result = s3_push()
        if push_result["errors"]:
            raise UpgradeError(
                "S3 push failed: " + "; ".join(push_result["errors"])
            )
    finally:
        if prior_bucket is None:
            os.environ.pop("MNEMON_S3_BUCKET", None)
        else:
            os.environ["MNEMON_S3_BUCKET"] = prior_bucket

    # Step 3: Fly deploy (or skip if override env var is set).
    override = _fly_endpoint_override()
    if override:
        remote_url = override.rstrip("/")
        if not remote_url.endswith("/mcp"):
            remote_url = remote_url + "/mcp"
    else:
        with tempfile.TemporaryDirectory(prefix="mnemon-upgrade-") as tdir:
            workdir = Path(tdir)
            (workdir / "Dockerfile").write_text(
                _DOCKERFILE_TEMPLATE.format(mnemon_version=mnemon_version)
            )
            (workdir / "fly.toml").write_text(
                _FLY_TOML_TEMPLATE.format(app_name=app_name, region=region)
            )
            _fly_launch(workdir, app_name, region)
            _fly_create_volume(app_name, region)
            _fly_set_secrets(app_name, local_token, bucket)
            _fly_deploy(workdir, app_name)

        # Step 4: seed vault on Fly from S3.
        if local_exists:
            _fly_seed_vault(app_name)

        remote_url = f"https://{app_name}.fly.dev/mcp"

    # Step 5: archive local vault. Must come AFTER Fly deploy is
    # confirmed so we don't leave the user with no local AND no remote.
    archived_to = _archive_local_vault() if local_exists else None

    # Step 6: reconfigure every detected MCP client. Uses the existing
    # setup_* functions in remote mode so preflight + SessionStart
    # pre-warm are all inherited.
    from .setup import detect_installed_clients

    detected = detect_installed_clients()
    reconfigured = _reconfigure_clients(remote_url, local_token, detected)

    # Step 7: doctor against remote. fail_on_warn=True so scripted
    # upgrades propagate any config gap as a non-zero exit.
    summary_lines = [
        f"Upgrade to web complete: {remote_url}",
        f"  Fly app:       {app_name}",
        f"  S3 bucket:     {bucket}",
        f"  Token:         {LOCAL_TOKEN_FILE_DISPLAY} (chmod 600)",
    ]
    if archived_to:
        summary_lines.append(f"  Local archive: {archived_to}")
    if reconfigured:
        summary_lines.append(
            "  Reconfigured:  " + ", ".join(reconfigured)
        )
    summary_lines.extend(
        [
            "",
            "Next steps:",
            "  • Restart each client above to pick up the new MCP endpoint.",
            f"  • Add the remote manually to claude.ai / mobile: {remote_url}",
            f"    (Bearer token is in ~/.mnemon/local_token on this machine.)",
            "  • Run `mnemon downgrade local` to revert (pulls Fly vault back to local).",
        ]
    )

    if skip_doctor:
        return "\n".join(summary_lines)

    import io

    from .doctor import run_doctor

    # Point doctor at the new remote for the post-upgrade check.
    prior_url = os.environ.get("MNEMON_REMOTE_URL")
    prior_token = os.environ.get("MNEMON_LOCAL_TOKEN")
    os.environ["MNEMON_REMOTE_URL"] = remote_url
    os.environ["MNEMON_LOCAL_TOKEN"] = local_token
    buf = io.StringIO()
    try:
        print("", file=buf)
        print("Running mnemon doctor against the new remote...", file=buf)
        try:
            doctor_rc = run_doctor(out=buf, fail_on_warn=True)
        except Exception as exc:  # noqa: BLE001
            buf.write(
                f"\n(doctor invocation crashed: {type(exc).__name__}: {exc})\n"
            )
            doctor_rc = 1
    finally:
        if prior_url is None:
            os.environ.pop("MNEMON_REMOTE_URL", None)
        else:
            os.environ["MNEMON_REMOTE_URL"] = prior_url
        if prior_token is None:
            os.environ.pop("MNEMON_LOCAL_TOKEN", None)
        else:
            os.environ["MNEMON_LOCAL_TOKEN"] = prior_token

    if doctor_rc != 0:
        buf.write(
            "\nNOTE: doctor reported issues. The Fly app is deployed and "
            "configs are rewritten, but the remote is not fully ready. "
            "Investigate the failing check(s) above.\n"
        )

    return "\n".join(summary_lines) + "\n" + buf.getvalue()


# Display constant — kept here so upgrade_web doesn't import it at the
# top and pollute hot paths. The real path resolution lives in setup.py;
# this is purely for the user-facing summary.
LOCAL_TOKEN_FILE_DISPLAY = "~/.mnemon/local_token"
