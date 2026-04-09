"""CLI entry point for mnemon.

Usage:
    mnemon serve              Start MCP server (stdio transport)
    mnemon serve-remote       Start HTTP server (Streamable HTTP)
    mnemon status             Show vault health stats
    mnemon search <query>     Search memories
    mnemon save <title> <content>  Save a memory
    mnemon sync push          Push vault to S3
    mnemon sync pull          Pull vault from S3
    mnemon --version          Show version
    mnemon --help             Show this help
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

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_usage()
        sys.exit(1)


def _print_usage() -> None:
    print(f"""mnemon v{__version__} — Universal long-term memory for AI agents

Usage:
  mnemon serve              Start MCP server (stdio transport)
  mnemon serve-remote       Start HTTP server (Streamable HTTP)
  mnemon status             Show vault health stats
  mnemon search <query>     Search memories
  mnemon save <title> <c>   Save a memory
  mnemon sync push          Push vault to S3
  mnemon sync pull          Pull vault from S3
  mnemon --version          Show version
  mnemon --help             Show this help

Env vars:
  MNEMON_VAULT_DIR    Vault directory (default: ~/.mnemon)
  MNEMON_TOKEN        Bearer token for remote server auth
  MNEMON_S3_BUCKET    S3 bucket for vault sync
  PORT                Remote server port (default: 8502)

Docs: https://github.com/cipher813/mnemon""")


if __name__ == "__main__":
    main()
