"""CLI entry point for mnemon.

Setup commands configure clients to use a remote vault. Local vault
commands (status, search, save, forget, sync) operate on the local
``~/.mnemon/default.sqlite`` and are intended for development or
server-side administration — they do not interact with a remote vault.
"""

from __future__ import annotations

import sys

from . import __version__


def main() -> None:
    args = sys.argv[1:]
    command = args[0] if args else "--help"

    if command in ("--version", "-v"):
        print(f"mnemon v{__version__}")
        return

    if command in ("--help", "-h"):
        _print_usage()
        return

    if command == "serve":
        from .server import run_stdio
        run_stdio()

    elif command == "serve-remote":
        from .server_remote import run_remote
        run_remote()

    elif command == "dashboard":
        import subprocess
        from pathlib import Path
        app_path = Path(__file__).parent / "dashboard" / "app.py"
        port = args[1] if len(args) > 1 else "8503"
        try:
            subprocess.run(["streamlit", "run", str(app_path), f"--server.port={port}", "--theme.base=dark", "--client.toolbarMode=minimal"], check=True)
        except FileNotFoundError:
            print("streamlit not found. Install with: pip install 'mnemon-memory[ui]'", file=sys.stderr)
            sys.exit(1)

    elif command == "status":
        from .store import Store
        store = Store()
        stats = store.status()
        print(f"Vault: {stats['vault_path']}")
        print(f"Total memories: {stats['total_documents']}")
        print(f"Vectors: {stats['total_vectors']}")
        print(f"Pinned: {stats['pinned']}")
        print(f"Invalidated: {stats['invalidated']}")
        print("\nBy type:")
        for t in stats["by_type"]:
            print(f"  {t['content_type']}: {t['count']}")
        store.close()

    elif command == "search":
        query = " ".join(args[1:])
        if not query:
            print("Usage: mnemon search <query>", file=sys.stderr)
            sys.exit(1)
        from .search import search
        from .store import Store
        store = Store()
        results = search(store, query, limit=10)
        if not results:
            print("No memories found.")
        else:
            for r in results:
                snippet = r.content[:200]
                ellipsis = "..." if len(r.content) > 200 else ""
                print(f"[{r.content_type}] {r.title} (score: {r.composite_score:.3f})")
                print(f"  {snippet}{ellipsis}")
                print()
        store.close()

    elif command == "save":
        if len(args) < 3:
            print("Usage: mnemon save <title> <content>", file=sys.stderr)
            sys.exit(1)
        title = args[1]
        content = " ".join(args[2:])
        from .store import Store
        store = Store()
        doc_id = store.save(title=title, content=content, source_client="cli")
        print(f'Saved memory #{doc_id}: "{title}"')
        try:
            from .embedder import embed_document
            doc = store.get(doc_id)
            if doc:
                embed_document(store, doc.hash, title, content)
        except Exception as exc:  # noqa: BLE001
            # Loud on the CLI — an interactive save that skipped embedding
            # means vector search won't find this memory. Exit non-zero so
            # scripts catch it; `mnemon rebuild` is the recovery path.
            print(
                f"Warning: embedding failed ({type(exc).__name__}: {exc}). "
                f"Memory saved to vault but will not surface in vector "
                f"search until `mnemon rebuild` runs.",
                file=sys.stderr,
            )
            store.close()
            sys.exit(2)
        store.close()

    elif command == "rebuild":
        # Re-embed every non-invalidated document. Surfaces per-doc
        # failures so users hit real errors here instead of only seeing
        # them whispered into server logs.
        from .store import Store
        try:
            from .embedder import embed_document
        except ImportError:
            print("FastEmbed not installed. Run: pip install fastembed", file=sys.stderr)
            sys.exit(1)
        store = Store()
        docs = store.timeline(10_000)
        embedded = 0
        failed = 0
        for doc in docs:
            try:
                embed_document(store, doc.hash, doc.title, doc.content)
                embedded += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(
                    f"  embed failed: doc_id={doc.id} ({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )
        print(f"Rebuild complete: {embedded} documents embedded, {failed} failed.")
        store.close()
        if failed:
            sys.exit(1)

    elif command == "forget":
        if len(args) < 2 or not args[1].isdigit():
            print("Usage: mnemon forget <id>", file=sys.stderr)
            sys.exit(1)
        doc_id = int(args[1])
        from .store import Store
        store = Store()
        if store.forget(doc_id):
            print(f"Forgot memory #{doc_id}.")
        else:
            print(f"Memory #{doc_id} not found or already forgotten.", file=sys.stderr)
            sys.exit(1)
        store.close()

    elif command == "sync":
        subcommand = args[1] if len(args) > 1 else ""
        if subcommand == "push":
            from .sync import push
            result = push()
            if result["pushed"]:
                print("Pushed:")
                for p in result["pushed"]:
                    print(f"  {p}")
            if result["errors"]:
                print("Errors:", file=sys.stderr)
                for e in result["errors"]:
                    print(f"  {e}", file=sys.stderr)
                sys.exit(1)
            if not result["pushed"] and not result["errors"]:
                print("No vault files found to push.")
        elif subcommand == "pull":
            from .sync import pull
            result = pull()
            if result["pulled"]:
                print("Pulled:")
                for p in result["pulled"]:
                    print(f"  {p}")
            if result["errors"]:
                print("Errors:", file=sys.stderr)
                for e in result["errors"]:
                    print(f"  {e}", file=sys.stderr)
                sys.exit(1)
            if not result["pulled"] and not result["errors"]:
                print("No vault files found on S3.")
        else:
            print("Usage: mnemon sync <push|pull>", file=sys.stderr)
            print("\nEnv vars:")
            print("  MNEMON_S3_BUCKET   S3 bucket name (required)")
            print("  MNEMON_S3_PREFIX   S3 key prefix (default: mnemon/vaults)")
            print("  MNEMON_VAULT_NAME  vault name (default: default)")
            sys.exit(1)

    elif command == "mirror":
        # ``mnemon mirror <path>`` saves the contents of a memory file
        # to mnemon. The path's frontmatter (``name``, ``description``,
        # ``type``) drives the mnemon record's title/content/type. Used
        # by the PostToolUse hook installed by ``mnemon setup`` so any
        # Claude Code auto-memory write also lands in mnemon — closes
        # the 2026-04-28 gap where local-memory writes silently
        # diverged from the central vault. ``--auto`` short-circuits
        # when the path doesn't match an auto-memory directory pattern.
        from .mirror import run_cli
        sys.exit(run_cli(args[1:]))

    elif command == "setup":
        # `mnemon setup` with no target auto-detects every installed
        # client and configures each. Explicit target names
        # (claude-code, claude-desktop, cursor, gemini, hooks) still
        # work for narrow use cases.
        from .setup import run_setup
        if len(args) > 1 and not args[1].startswith("--"):
            target: str | None = args[1]
            setup_args = args[2:]
        else:
            target = None
            setup_args = args[1:]
        print(run_setup(target, setup_args))

    elif command == "doctor":
        from .doctor import run_doctor
        fail_on_warn = "--fail-on-warn" in args[1:]
        sys.exit(run_doctor(fail_on_warn=fail_on_warn))

    elif command == "upgrade":
        # `mnemon upgrade web --app-name <name> [options]`
        subcommand = args[1] if len(args) > 1 else ""
        if subcommand != "web":
            print(
                "Usage: mnemon upgrade web --app-name <name> "
                "[--s3-bucket NAME] [--token TOKEN] [--region REGION] "
                "[--mnemon-version VER] [--skip-doctor]",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            parsed = _parse_upgrade_args(args[2:])
        except ValueError as exc:
            print(f"upgrade failed: {exc}", file=sys.stderr)
            sys.exit(1)
        if not parsed["app_name"]:
            print("--app-name is required.", file=sys.stderr)
            sys.exit(1)
        from .upgrade import UpgradeError, upgrade_web
        try:
            print(
                upgrade_web(
                    app_name=parsed["app_name"],
                    s3_bucket=parsed["s3_bucket"],
                    token=parsed["token"],
                    region=parsed["region"] or "sjc",
                    mnemon_version=parsed["mnemon_version"],
                    skip_doctor=parsed["skip_doctor"],
                )
            )
        except UpgradeError as exc:
            print(f"upgrade failed: {exc}", file=sys.stderr)
            sys.exit(1)

    elif command == "uninstall":
        # `mnemon uninstall [--yes] [--keep-vault]` — wipe all mnemon
        # state from this machine. Useful for testing the full fresh-
        # install experience or for users who want to exit entirely.
        flags = args[1:]
        yes = "--yes" in flags
        keep_vault = "--keep-vault" in flags
        from .uninstall import UninstallError, uninstall
        try:
            print(uninstall(yes=yes, keep_vault=keep_vault))
        except UninstallError as exc:
            print(f"uninstall failed: {exc}", file=sys.stderr)
            sys.exit(1)

    elif command == "downgrade":
        # `mnemon downgrade local [--destroy-fly-app] [--yes] [--skip-doctor]`
        # Symmetric to `mnemon upgrade web`. Pulls the Fly vault state
        # back to local, reconfigures every MCP client to stdio mode,
        # optionally destroys the Fly app.
        subcommand = args[1] if len(args) > 1 else ""
        if subcommand != "local":
            print(
                "Usage: mnemon downgrade local "
                "[--destroy-fly-app] [--yes] [--app-name NAME] "
                "[--skip-doctor]",
                file=sys.stderr,
            )
            sys.exit(1)
        parsed = _parse_downgrade_args(args[2:])
        from .downgrade import DowngradeError, downgrade_local
        try:
            print(
                downgrade_local(
                    destroy_fly_app=parsed["destroy_fly_app"],
                    yes=parsed["yes"],
                    skip_doctor=parsed["skip_doctor"],
                    app_name_override=parsed["app_name"],
                )
            )
        except DowngradeError as exc:
            print(f"downgrade failed: {exc}", file=sys.stderr)
            sys.exit(1)

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_usage()
        sys.exit(1)


def _parse_upgrade_args(args: list[str]) -> dict:
    """Parse ``mnemon upgrade web`` flags.

    Raises ``ValueError`` if ``--mnemon-version`` is malformed — the
    string is interpolated into a Dockerfile shell context, so we
    refuse anything outside ``[A-Za-z0-9._+-]``.
    """
    import re

    result: dict[str, str | bool | None] = {
        "app_name": None,
        "s3_bucket": None,
        "token": None,
        "region": None,
        "mnemon_version": None,
        "skip_doctor": False,
    }
    i = 0
    while i < len(args):
        flag = args[i]
        if flag == "--app-name" and i + 1 < len(args):
            result["app_name"] = args[i + 1]
            i += 2
        elif flag == "--s3-bucket" and i + 1 < len(args):
            result["s3_bucket"] = args[i + 1]
            i += 2
        elif flag == "--token" and i + 1 < len(args):
            result["token"] = args[i + 1]
            i += 2
        elif flag == "--region" and i + 1 < len(args):
            result["region"] = args[i + 1]
            i += 2
        elif flag == "--mnemon-version" and i + 1 < len(args):
            value = args[i + 1]
            if not re.fullmatch(r"[A-Za-z0-9._+-]+", value):
                raise ValueError(
                    f"--mnemon-version must match [A-Za-z0-9._+-]+; "
                    f"got: {value!r}"
                )
            result["mnemon_version"] = value
            i += 2
        elif flag == "--skip-doctor":
            result["skip_doctor"] = True
            i += 1
        else:
            i += 1
    return result


def _parse_downgrade_args(args: list[str]) -> dict:
    """Parse ``mnemon downgrade local`` flags."""
    result: dict[str, str | bool | None] = {
        "destroy_fly_app": False,
        "yes": False,
        "skip_doctor": False,
        "app_name": None,
    }
    i = 0
    while i < len(args):
        flag = args[i]
        if flag == "--destroy-fly-app":
            result["destroy_fly_app"] = True
            i += 1
        elif flag == "--yes":
            result["yes"] = True
            i += 1
        elif flag == "--skip-doctor":
            result["skip_doctor"] = True
            i += 1
        elif flag == "--app-name" and i + 1 < len(args):
            result["app_name"] = args[i + 1]
            i += 2
        else:
            i += 1
    return result


def _print_usage() -> None:
    print(f"""mnemon v{__version__} — Universal long-term memory for AI agents

Setup (configure MCP clients; use --remote-url for web mode):
  mnemon setup              Auto-detect installed clients and configure each
  mnemon setup <target>     Configure one client explicitly
                            (claude-code | claude-desktop | cursor | gemini | hooks)
                            [--remote-url URL] [--token TOKEN] [--skip-doctor]
  mnemon doctor             Run diagnostics (local or remote, auto-detected)
                            [--fail-on-warn] treat warnings as failures

Upgrade local → web (deploys a Fly.io app + reconfigures every client).
Idempotent: rerun to redeploy an existing app with the current mnemon
version (clients keep their URL + token):
  mnemon upgrade web --app-name <name> [--s3-bucket NAME] [--token TOKEN]
                             [--region REGION] [--mnemon-version VER]
                             [--skip-doctor]
                             First-time deploy requires: flyctl, aws CLI
                             with credentials, and an S3 bucket
                             (MNEMON_S3_BUCKET or --s3-bucket). Redeploy
                             against an existing app only needs flyctl.
                             --mnemon-version pins a specific PyPI version
                             in the deployed Dockerfile (defaults to the
                             locally-installed __version__).

Downgrade web → local (pull remote vault back, reconfigure clients to stdio):
  mnemon downgrade local    [--destroy-fly-app] [--yes] [--app-name NAME]
                            [--skip-doctor]
                            Requires MNEMON_S3_BUCKET and aws CLI creds.

Uninstall (remove all mnemon state from this machine):
  mnemon uninstall          [--yes] [--keep-vault]
                            Removes vault, client configs, claude mcp
                            registration. Does NOT touch Fly / S3 / the
                            pip package. Run `downgrade local` first if
                            you want to preserve a live web deployment.

Server:
  mnemon serve              Start MCP server (stdio, local development)
  mnemon serve-remote       Start HTTP server (Streamable HTTP, production)

Auto-mirror (PostToolUse hook installed by `mnemon setup`):
  mnemon mirror <path>      Save a memory file (frontmatter-aware) to mnemon
                            [--auto] no-op when path is outside auto-memory dirs
                            [--timeout SEC] per-call client timeout (default 10)

Local vault (development/server-side only):
  mnemon status             Show local vault health stats
  mnemon search <query>     Search local vault
  mnemon save <title> <c>   Save to local vault
  mnemon forget <id>        Soft-delete from local vault
  mnemon rebuild            Re-embed every document (run after a model
                            change, or to recover from skipped embeddings)
  mnemon sync push          Push local vault to S3
  mnemon sync pull          Pull vault from S3
  mnemon dashboard [port]   Launch web dashboard (default: 8503)

Env vars:
  MNEMON_REMOTE_URL   Remote server URL (or ~/.mnemon/remote_url file)
  MNEMON_LOCAL_TOKEN  Bearer token for remote auth (or ~/.mnemon/local_token file)
  MNEMON_VAULT_DIR    Local vault directory (default: ~/.mnemon)
  MNEMON_S3_BUCKET    S3 bucket for vault sync
  PORT                Remote server port (default: 8502)

Docs: https://github.com/cipher813/mnemon""")


if __name__ == "__main__":
    main()
