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
        store.close()

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

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_usage()
        sys.exit(1)


def _print_usage() -> None:
    print(f"""mnemon v{__version__} — Universal long-term memory for AI agents

Setup (configure MCP clients; use --remote-url for web mode):
  mnemon setup              Auto-detect installed clients and configure each
  mnemon setup <target>     Configure one client explicitly
                            (claude-code | claude-desktop | cursor | gemini | hooks)
                            [--remote-url URL] [--token TOKEN] [--skip-doctor]
  mnemon doctor             Run diagnostics (local or remote, auto-detected)
                            [--fail-on-warn] treat warnings as failures

Server:
  mnemon serve              Start MCP server (stdio, local development)
  mnemon serve-remote       Start HTTP server (Streamable HTTP, production)

Local vault (development/server-side only):
  mnemon status             Show local vault health stats
  mnemon search <query>     Search local vault
  mnemon save <title> <c>   Save to local vault
  mnemon forget <id>        Soft-delete from local vault
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
