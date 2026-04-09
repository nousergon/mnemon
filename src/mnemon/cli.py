"""CLI entry point for mnemon.

Usage:
    mnemon serve              Start MCP server (stdio transport)
    mnemon status             Show vault health stats
    mnemon search <query>     Search memories
    mnemon save <title> <content>  Save a memory
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

    elif command == "status":
        from .store import Store
        store = Store()
        stats = store.status()
        print(f"Vault: {stats['vault_path']}")
        print(f"Total memories: {stats['total_documents']}")
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

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_usage()
        sys.exit(1)


def _print_usage() -> None:
    print(f"""mnemon v{__version__} — Universal long-term memory for AI agents

Usage:
  mnemon serve              Start MCP server (stdio transport)
  mnemon status             Show vault health stats
  mnemon search <query>     Search memories
  mnemon save <title> <content>  Save a memory
  mnemon --version          Show version
  mnemon --help             Show this help

Requires: Python >= 3.10
Docs: https://github.com/cipher813/mnemon""")


if __name__ == "__main__":
    main()
