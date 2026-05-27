"""CLI entry point for mnemon.

Setup commands configure clients to use a remote vault. Read/write
commands (``status``, ``search``, ``save``) auto-route to the live
remote vault when ``MNEMON_REMOTE_URL`` is set (env var or
``~/.mnemon/remote_url`` file). Otherwise they hit the local
``~/.mnemon/default.sqlite``. Local-only commands (``sync``,
``rebuild``, ``forget``, ``standing``, ``attention-status``,
``doctor``) intentionally stay on the local path — they're either
server-administration (rebuild/sync) or operator-explicit gestures
on the local vault.
"""

from __future__ import annotations

import os
import sys

from . import __version__


def _remote_mode_active() -> bool:
    """True iff a remote vault is configured.

    Mirrors ``hooks._remote_client.get_remote_url`` resolution order:
    env var first, then ``~/.mnemon/remote_url`` file. Doesn't validate
    the URL — that's the caller's job at first network use.
    """
    if os.environ.get("MNEMON_REMOTE_URL", "").strip():
        return True
    from .hooks._remote_client import REMOTE_URL_FILE
    if REMOTE_URL_FILE.exists():
        try:
            return bool(REMOTE_URL_FILE.read_text().strip())
        except OSError:
            return False
    return False


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
        if _remote_mode_active():
            _status_remote()
        else:
            from .store import Store
            store = Store()
            stats = store.status()
            standing = store.standing_tier_status()
            print(f"Vault: {stats['vault_path']}")
            print(f"Total memories: {stats['total_documents']}")
            print(f"Vectors: {stats['total_vectors']}")
            print(f"Pinned: {stats['pinned']}")
            print(f"Standing tier: {standing['count']}/{standing['cap']} "
                  f"(hard ceiling {standing['hard_ceiling']})")
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
        if _remote_mode_active():
            _search_remote(query)
        else:
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
        if _remote_mode_active():
            _save_remote(title, content)
        else:
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

    elif command == "consolidate":
        # Capture-attention Phase C — operator-reviewed cluster
        # consolidation. Default mode LISTS clusters (read-only).
        # --apply <cluster-idx> keeps the canonical (first member,
        # natural canonical-pick order = highest recurrence then
        # confidence) + supersedes the rest via 'supersedes' relations
        # + forget. Operator-review only per the salience-tier plan
        # invariant — no auto-apply.
        from .store import Store
        apply_idx: int | None = None
        recent_days = 30
        threshold = 0.85
        for i, a in enumerate(args[1:], start=1):
            if a == "--apply" and i + 1 < len(args):
                try:
                    apply_idx = int(args[i + 1])
                except ValueError:
                    print("Error: --apply must be an integer cluster index", file=sys.stderr)
                    sys.exit(2)
            elif a == "--recent-days" and i + 1 < len(args):
                try:
                    recent_days = int(args[i + 1])
                except ValueError:
                    print("Error: --recent-days must be an integer", file=sys.stderr)
                    sys.exit(2)
            elif a == "--threshold" and i + 1 < len(args):
                try:
                    threshold = float(args[i + 1])
                except ValueError:
                    print("Error: --threshold must be a float", file=sys.stderr)
                    sys.exit(2)
        store = Store()
        try:
            clusters = store.find_clusters(
                similarity_threshold=threshold,
                recent_days=recent_days,
            )
            _print_consolidate(store, clusters, apply_idx=apply_idx)
        finally:
            store.close()

    elif command == "sweep-contradictions":
        # Retroactive contradiction sweep — walks the vault, classifies
        # pair-wise NLI on cosine-overlapping candidates, applies decays +
        # relations. Closes the gap for memories that landed before
        # check_contradictions was wired in or that slipped past the
        # save-time vector window.
        from .store import Store
        from .contradiction import sweep_contradictions
        max_pairs = 50
        dry_run = "--dry-run" in args[1:]
        for i, a in enumerate(args[1:], start=1):
            if a == "--max-pairs" and i + 1 < len(args):
                try:
                    max_pairs = int(args[i + 1])
                except ValueError:
                    print("Error: --max-pairs must be an integer", file=sys.stderr)
                    sys.exit(2)
        store = Store()
        try:
            result = sweep_contradictions(
                store, max_pairs=max_pairs, dry_run=dry_run,
            )
        finally:
            store.close()
        print(f"Sweep complete{' (dry-run)' if dry_run else ''}:")
        print(f"  pairs examined  : {result['pairs_examined']}")
        print(f"  pairs classified: {result['pairs_classified']}")
        print(f"  pairs skipped   : {result['pairs_skipped']}  (already had classification relation)")
        print(f"  decayed         : {result['decayed']}  (update + contradiction outcomes)")
        print(f"  relations added : {result['relations_added']}")
        if result["nli_unavailable"]:
            print("\n  ⚠ NLI unavailable — sweep aborted early. See server logs.",
                  file=sys.stderr)
            sys.exit(1)

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
                "[--mnemon-version VER] [--skip-doctor] [--testpypi]",
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
                    use_testpypi=parsed["use_testpypi"],
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
                    skip_fly_push=parsed["skip_fly_push"],
                )
            )
        except DowngradeError as exc:
            print(f"downgrade failed: {exc}", file=sys.stderr)
            sys.exit(1)

    elif command == "salience-report":
        # Salience tier Phase 2 — promotion-signal candidate ranking.
        # Surfaces the situational-tier memories with the highest
        # correction_count + contradiction_win_count so the operator
        # can review and promote the genuinely load-bearing ones.
        from .store import Store
        limit = 20
        for i, a in enumerate(args[1:], start=1):
            if a == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    print("Error: --limit must be an integer", file=sys.stderr)
                    sys.exit(2)
        store = Store()
        try:
            _print_salience_report(store, limit=limit)
        finally:
            store.close()

    elif command == "attention-report":
        # Capture attention Phase B — access-count consolidation feedback.
        # Ranks live memories by access_count × recency so the operator
        # can see which fragments are load-bearing — those are exactly
        # the candidates for standing-tier promotion (composes with
        # Salience Phase 2).
        from .store import Store
        limit = 20
        min_count = 2
        for i, a in enumerate(args[1:], start=1):
            if a == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    print(f"Error: --limit must be an integer", file=sys.stderr)
                    sys.exit(2)
            elif a == "--min-access" and i + 1 < len(args):
                try:
                    min_count = int(args[i + 1])
                except ValueError:
                    print(f"Error: --min-access must be an integer", file=sys.stderr)
                    sys.exit(2)
        store = Store()
        try:
            _print_attention_report(store, limit=limit, min_access_count=min_count)
        finally:
            store.close()

    elif command == "attention-status":
        # Capture attention Phase A observability — soak monitor.
        # private/mnemon-capture-attention-plan-260522.md
        from .store import Store
        strict = "--strict" in args[1:]
        store = Store()
        try:
            rate, ceiling = _print_attention_status(store)
        finally:
            store.close()
        if strict and rate > ceiling:
            sys.exit(1)

    elif command == "standing":
        # Salience tier Phase 1 — operator-facing tier management.
        # private/mnemon-salience-tier-plan-260521.md
        subcommand = args[1] if len(args) > 1 else "list"
        _handle_standing(subcommand, args[2:])

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_usage()
        sys.exit(1)


def _status_remote() -> None:
    """``mnemon status`` against the configured remote vault.

    Reads ``memory_status`` + ``memory_list_standing`` to produce the
    same shape the local-mode path prints. Closes the 2026-05-21 gap
    where the CLI silently fell back to a fresh empty SQLite instead
    of reflecting the live remote.
    """
    import json as _json
    from .hooks._remote_client import call_tool_sync, RemoteClientConfigError

    try:
        raw, _ = call_tool_sync("memory_status", {}, timeout=8.0)
        stats = _json.loads(raw)
        std_raw, _ = call_tool_sync("memory_list_standing", {}, timeout=5.0)
        standing_docs = _json.loads(std_raw)
    except (RemoteClientConfigError, Exception) as exc:
        print(f"remote status failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Vault (remote): {stats.get('vault_path', '<remote>')}")
    print(f"Total memories: {stats.get('total_documents', 0)}")
    print(f"Vectors: {stats.get('total_vectors', 0)}")
    print(f"Pinned: {stats.get('pinned', 0)}")
    print(f"Standing tier: {len(standing_docs) if isinstance(standing_docs, list) else 0}")
    print(f"Invalidated: {stats.get('invalidated', 0)}")
    print("\nBy type:")
    for t in stats.get("by_type", []):
        print(f"  {t['content_type']}: {t['count']}")


def _search_remote(query: str) -> None:
    """``mnemon search <query>`` against the configured remote vault."""
    import json as _json
    from .hooks._remote_client import call_tool_sync, RemoteClientConfigError

    try:
        raw, _ = call_tool_sync(
            "memory_search", {"query": query, "limit": 10}, timeout=8.0,
        )
        results = _json.loads(raw)
    except (RemoteClientConfigError, Exception) as exc:
        print(f"remote search failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No memories found.")
        return
    for r in results:
        content = r.get("content", "")
        snippet = content[:200]
        ellipsis = "..." if len(content) > 200 else ""
        print(f"[{r.get('content_type', 'note')}] {r.get('title', '')} "
              f"(score: {r.get('composite_score', 0):.3f})")
        print(f"  {snippet}{ellipsis}")
        print()


def _save_remote(title: str, content: str) -> None:
    """``mnemon save <title> <content>`` against the configured remote
    vault. Tags ``source_client='cli'`` so the save is treated as an
    explicit user gesture (no HOOK_SOURCE_CONFIDENCE_CEILING cap)."""
    from .hooks._remote_client import call_tool_sync, RemoteClientConfigError

    try:
        raw, _ = call_tool_sync(
            "memory_save",
            {"title": title, "content": content, "source_client": "cli"},
            timeout=10.0,
        )
    except (RemoteClientConfigError, Exception) as exc:
        print(f"remote save failed: {exc}", file=sys.stderr)
        sys.exit(1)
    # Server returns a human-facing confirmation line like
    # `Saved memory #N: "Title" [note]` — pass through.
    print(raw)


def _handle_standing(subcommand: str, rest: list[str]) -> None:
    """Salience tier Phase 1 — list / promote / demote subcommands."""
    from .store import (
        Store,
        StandingTierCapReached,
        StandingTierError,
        StandingTierProvenanceRejected,
    )
    store = Store()
    try:
        if subcommand == "list":
            # Phase 3 observability: standing_tier_aging() returns the
            # per-member age + last-injected + signal columns without
            # bumping last_injected_at (that's list_standing's job).
            aging = store.standing_tier_aging()
            status = store.standing_tier_status()
            print(f"Standing tier: {status['count']}/{status['cap']} "
                  f"(hard ceiling {status['hard_ceiling']})")
            if not aging:
                print("  (empty — promote memories via `mnemon standing promote <id>`)")
                return
            print()
            print(f"  {'id':>5}  {'age':>6}  {'last inj':>9}  "
                  f"{'wins':>4}  {'corr':>4}  type           title")
            stale_threshold_days = 90.0
            for a in aging:
                ct = (a["content_type"] or "")[:12]
                title = (a["title"] or "")[:60]
                if a["days_since_injected"] is None:
                    last_inj = "never"
                else:
                    last_inj = f"{a['days_since_injected']:.0f}d ago"
                stale_marker = ""
                if (
                    a["days_since_injected"] is not None
                    and a["days_since_injected"] >= stale_threshold_days
                ):
                    stale_marker = "  ⚠ stale"
                print(
                    f"  #{a['id']:>4}  {a['age_days']:>5.0f}d  "
                    f"{last_inj:>9}  "
                    f"{a['contradiction_win_count']:>4}  "
                    f"{a['correction_count']:>4}  "
                    f"{ct:<13}  {title}{stale_marker}"
                )
            print(
                f"\n  Stale threshold: {stale_threshold_days:.0f}d since last injection. "
                "Phase 3 doesn't auto-demote — review and run "
                "`mnemon standing demote <id>` if no longer load-bearing."
            )

        elif subcommand == "promote":
            if not rest:
                print("Usage: mnemon standing promote <id>", file=sys.stderr)
                sys.exit(2)
            try:
                doc_id = int(rest[0])
            except ValueError:
                print(f"Error: <id> must be an integer (got {rest[0]!r})",
                      file=sys.stderr)
                sys.exit(2)
            try:
                store.promote_to_standing(doc_id)
                status = store.standing_tier_status()
                print(f"Promoted memory #{doc_id} to standing tier "
                      f"({status['count']}/{status['cap']}).")
            except StandingTierCapReached as e:
                print(f"Cap reached: {e}", file=sys.stderr)
                sys.exit(1)
            except StandingTierProvenanceRejected as e:
                print(f"Provenance rejected: {e}", file=sys.stderr)
                sys.exit(1)
            except StandingTierError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif subcommand == "demote":
            if not rest:
                print("Usage: mnemon standing demote <id>", file=sys.stderr)
                sys.exit(2)
            try:
                doc_id = int(rest[0])
            except ValueError:
                print(f"Error: <id> must be an integer (got {rest[0]!r})",
                      file=sys.stderr)
                sys.exit(2)
            try:
                ok = store.demote_to_situational(doc_id)
                status = store.standing_tier_status()
                if ok:
                    print(f"Demoted memory #{doc_id} to situational "
                          f"({status['count']}/{status['cap']} remain standing).")
                else:
                    print(f"Memory #{doc_id} was not on the standing tier.")
            except StandingTierError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        else:
            print(f"Unknown subcommand: standing {subcommand}", file=sys.stderr)
            print("Usage: mnemon standing list | promote <id> | demote <id>",
                  file=sys.stderr)
            sys.exit(2)
    finally:
        store.close()


def _print_consolidate(
    store, clusters: list, *, apply_idx: int | None,
) -> None:
    """Render the Phase C consolidation surface.

    Default mode: list clusters by index with member details + the
    natural canonical pick (first member, highest recurrence+confidence).
    --apply mode: invoke ``store.consolidate_cluster`` on the named
    index. ROADMAP invariant says "operator-review only, no auto-apply"
    — the --apply gate is the operator's explicit gesture.
    """
    if not clusters:
        print("No near-duplicate clusters found in the recent window.")
        print(
            "  Try widening --recent-days (default 30) or lowering "
            "--threshold (default 0.85)."
        )
        return

    if apply_idx is None:
        print(f"Found {len(clusters)} near-duplicate cluster(s) in the recent window.")
        print(
            "  Default canonical = first member (highest recurrence_count, "
            "then highest confidence)."
        )
        print(
            "  Apply with: `mnemon consolidate --apply <cluster-idx>` "
            "(supersedes the non-canonical members)\n"
        )
        for i, cluster in enumerate(clusters):
            canonical = cluster[0]
            print(f"  ── cluster #{i} ({len(cluster)} members) ──")
            for j, m in enumerate(cluster):
                marker = "★ canonical" if j == 0 else f"  → supersede"
                title = (m["title"] or "")[:55]
                print(
                    f"    {marker:<14}  #{m['id']:>5}  "
                    f"[{m['content_type']:<10}]  "
                    f"rec={m['recurrence_count']}  "
                    f"conf={m['confidence']:.2f}  {title}"
                )
            _ = canonical  # readability anchor
            print()
        return

    # --apply mode
    if apply_idx < 0 or apply_idx >= len(clusters):
        print(
            f"Error: cluster #{apply_idx} out of range (have "
            f"{len(clusters)} cluster(s) 0-{len(clusters) - 1})",
            file=sys.stderr,
        )
        sys.exit(2)
    target = clusters[apply_idx]
    cluster_ids = [m["id"] for m in target]
    canonical = target[0]
    victims = target[1:]
    print(f"Consolidating cluster #{apply_idx}:")
    print(f"  canonical:     #{canonical['id']}  {canonical['title']}")
    for v in victims:
        print(f"  → supersede:   #{v['id']}  {v['title']}")
    print()
    confirm = input(
        f"  Proceed? This will forget {len(victims)} memory(ies) "
        "and record 'supersedes' relations. [y/N]: "
    ).strip().lower()
    if confirm not in ("y", "yes"):
        print("Aborted.")
        return

    result = store.consolidate_cluster(cluster_ids)
    print(
        f"\nConsolidated cluster #{apply_idx}: "
        f"canonical=#{result['canonical_id']}, "
        f"superseded={result['superseded_ids']}"
    )
    if result["errors"]:
        print("Warnings:", file=sys.stderr)
        for err in result["errors"]:
            print(f"  - {err}", file=sys.stderr)


def _print_salience_report(store, *, limit: int) -> None:
    """Render the Salience Phase 2 promotion-signal report. Each row
    shows correction_count + contradiction_win_count + combined score
    so the operator can decide which memories deserve standing-tier
    promotion (via `mnemon standing promote <id>`)."""
    rows = store.salience_report(limit=limit)
    print(f"Salience report — top {limit} situational memories by promotion signal\n")
    if not rows:
        print("  (no candidates — correction_count + contradiction_win_count "
              "are 0 across the situational tier; mechanism is observe-only "
              "until an operator gesture or NLI win fires)")
        return
    print(f"  {'id':>5}  {'score':>5}  {'corr':>4}  {'wins':>4}  "
          f"{'conf':>4}  type           title")
    for r in rows:
        ct = (r["content_type"] or "")[:12]
        title = (r["title"] or "")[:60]
        print(
            f"  #{r['id']:>4}  {r['score']:>5}  "
            f"{r['correction_count']:>4}  {r['contradiction_win_count']:>4}  "
            f"{r['confidence']:>.2f}  {ct:<13}  {title}"
        )
    print(
        "\n  Promote a candidate via `mnemon standing promote <id>` "
        "(operator-approved per salience-tier plan invariant 6)."
    )


def _print_attention_report(store, *, limit: int, min_access_count: int) -> None:
    """Print a ranked list of high-access memories — capture-attention
    Phase B consolidation feedback. Each row shows the memory's
    access_count, age, recency factor, and combined score so the
    operator can spot durable load-bearing memories that would benefit
    from standing-tier promotion."""
    rows = store.attention_report(limit=limit, min_access_count=min_access_count)
    print(f"Attention report — top {limit} live memories by access × recency")
    print(f"  (filtered access_count ≥ {min_access_count})\n")
    if not rows:
        print("  (no memories meet the filter — increase access by using "
              "memory_search / memory_get, or lower --min-access)")
        return
    print(f"  {'id':>5}  {'score':>6}  {'×acc':>5}  {'age':>6}  "
          f"{'rec':>5}  tier         type           title")
    for r in rows:
        tier_label = (r["tier"] or "situational")[:12]
        ct = (r["content_type"] or "")[:12]
        title = (r["title"] or "")[:60]
        print(
            f"  #{r['id']:>4}  {r['score']:>6.2f}  "
            f"{r['access_count']:>4}×  {r['age_days']:>5.1f}d  "
            f"{r['recency']:>5.2f}  {tier_label:<12}  {ct:<13}  {title}"
        )


def _print_attention_status(store) -> tuple[float, float]:
    """Print capture-attention soak metrics for the operator.

    Surfaces the two acceptance criteria from the plan-doc:
      1. boost_rate = (boosts in last 7d) / (saves in last 7d) ≤ 0.25
      2. precision floor (operator-judged via --review, not auto-checked)

    Returns ``(boost_rate, ceiling)`` so a caller can drive --strict
    exit codes without re-running the SQL.
    """
    from .config import (
        CAPTURE_ATTENTION_THRESHOLD,
        CAPTURE_ATTENTION_SOAK_BOOST_RATE_MAX,
    )
    from .store import _capture_attention_enabled

    # Boost rate over 7d (boosts = restates relations created)
    boosts_7d = store.db.execute(
        "SELECT COUNT(*) AS c FROM relations "
        "WHERE relation_type = 'restates' "
        "AND created_at >= datetime('now', '-7 days')"
    ).fetchone()["c"]
    saves_7d = store.db.execute(
        "SELECT COUNT(*) AS c FROM documents "
        "WHERE created_at >= datetime('now', '-7 days')"
    ).fetchone()["c"]
    rate = (boosts_7d / saves_7d) if saves_7d else 0.0
    rate_ok = "✓" if rate <= CAPTURE_ATTENTION_SOAK_BOOST_RATE_MAX else "⚠"

    # Effective flag value reflects MNEMON_CAPTURE_ATTENTION_ENABLED env-var
    # override; a Fly secret flip shows up here without restarting the server.
    print(f"Capture attention — soak status")
    print(f"  Flag enabled       : {_capture_attention_enabled()}")
    print(f"  Threshold (cosine) : {CAPTURE_ATTENTION_THRESHOLD}")
    print(f"  Boost-rate 7d      : {boosts_7d} / {saves_7d} = "
          f"{rate:.3f}  {rate_ok} (ceiling {CAPTURE_ATTENTION_SOAK_BOOST_RATE_MAX})")

    # Recurrence count distribution
    hist = store.db.execute(
        "SELECT recurrence_count, COUNT(*) AS n "
        "FROM documents WHERE invalidated_at IS NULL "
        "GROUP BY recurrence_count "
        "ORDER BY recurrence_count"
    ).fetchall()
    print("\n  Recurrence count distribution (live docs):")
    for r in hist:
        print(f"    count={r['recurrence_count']:>3}: {r['n']} docs")

    # Top-10 canonicals
    top = store.db.execute(
        "SELECT id, title, recurrence_count, confidence "
        "FROM documents "
        "WHERE invalidated_at IS NULL AND recurrence_count > 0 "
        "ORDER BY recurrence_count DESC, confidence DESC "
        "LIMIT 10"
    ).fetchall()
    if top:
        print("\n  Top canonicals by recurrence_count:")
        for r in top:
            title = r["title"][:60]
            print(f"    #{r['id']:>5}  ×{r['recurrence_count']:<3}  "
                  f"conf={r['confidence']:.2f}  {title}")
    else:
        print("\n  No canonicals with recurrence_count > 0 yet.")

    # Recent 'restates' relations (audit trail)
    recent = store.db.execute(
        "SELECT source_id, target_id, weight, created_at "
        "FROM relations WHERE relation_type = 'restates' "
        "ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    if recent:
        print("\n  Last 10 'restates' relations:")
        for r in recent:
            print(f"    {r['created_at']}  "
                  f"#{r['source_id']:>5} → #{r['target_id']:>5}  "
                  f"w={r['weight']:.3f}")

    return rate, CAPTURE_ATTENTION_SOAK_BOOST_RATE_MAX


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
        "use_testpypi": False,
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
        elif flag == "--testpypi":
            result["use_testpypi"] = True
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
        "skip_fly_push": False,
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
        elif flag == "--skip-fly-push":
            result["skip_fly_push"] = True
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
                             [--skip-doctor] [--testpypi]
                             First-time deploy requires: flyctl, aws CLI
                             with credentials, and an S3 bucket
                             (MNEMON_S3_BUCKET or --s3-bucket). Redeploy
                             against an existing app only needs flyctl.
                             --mnemon-version pins a specific PyPI version
                             in the deployed Dockerfile (defaults to the
                             locally-installed __version__).
                             --testpypi resolves mnemon-memory from
                             test.pypi.org (transitive deps stay on
                             prod PyPI); for true pre-publish validation
                             via promote_stable.sh testpublish.

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
  mnemon attention-status   Capture-attention soak monitor — boost rate,
                            recurrence distribution, top canonicals,
                            recent 'restates' relations
                            [--strict: exit 1 if boost-rate > ceiling,
                            for periodic health-check wiring]
  mnemon attention-report   Phase B consolidation feedback — rank live
                            memories by access_count × recency
                            [--limit N] [--min-access N]
  mnemon salience-report    Phase 2 promotion-signal candidates — rank
                            situational memories by correction_count +
                            contradiction_win_count [--limit N]
  mnemon sweep-contradictions
                            Retroactive NLI sweep — classify cosine-
                            overlapping memory pairs that bypassed the
                            save-time check; non-destructive (decay +
                            relations only). [--max-pairs N] [--dry-run]
  mnemon consolidate        Phase C operator-review consolidation —
                            list near-duplicate clusters in the recent
                            window. [--apply <cluster-idx>] supersedes
                            non-canonical members of the named cluster.
                            [--recent-days N] [--threshold F]

Salience tier (standing-context recall):
  mnemon standing list      Show all standing-tier memories + count vs cap
  mnemon standing promote <id>   Promote memory to standing tier (capped)
  mnemon standing demote <id>    Demote back to situational
                            Standing memories are injected into every
                            recall context regardless of query similarity.
                            Cap is the contract — default 15, hard 20.
                            Hook-sourced memories cannot be promoted.
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
