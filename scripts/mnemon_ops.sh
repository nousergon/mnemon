#!/usr/bin/env bash
# scripts/mnemon_ops.sh — general operator helpers for mnemon.
#
# Consolidates recurring operator commands that emerged during the
# 0.6.0 stable cut + subsequent releases — each one was being typed
# by hand 3-4 times per session before this script existed. Each
# subcommand is a few lines of bash wrapping a single intent.
#
# Subcommands:
#   cleanup-test-apps    Destroy every `mnemon-test-*` Fly app (dangling
#                        layer3 leftovers + locally-cached snapshots).
#   recover-token        Pull MNEMON_LOCAL_TOKEN out of the running Fly
#                        app and write it to ~/.mnemon/local_token (0o600).
#                        Used after layer3 auth clobbering.
#   restart-machine      `flyctl machine restart` the prod app's first
#                        machine + run `mnemon doctor`. Used when the SSL
#                        handshake wedges mid-cold-start.
#   vault-stats          Print size + recent-changes diff over the local
#                        vault. Quick snapshot for operator awareness.
#   changelog-extract <version>
#                        Print the CHANGELOG.md section for the given
#                        version (e.g. `0.7.0rc5` or `0.7.0`). Used
#                        standalone for ad-hoc "extract release notes"
#                        needs — also called by promote_stable.sh publish.
#   help                 This help.
#
# Conventions (mirrors scripts/salience_phase0.sh):
#   - Resolves the repo + .venv from the script's own location.
#   - Honors env-var overrides where useful (MNEMON_FLY_APP_NAME etc).
#   - Refuses to touch credentials on stdout (token recovery writes the
#     file with 0o600, never echoes the value).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MNEMON_VENV_BIN="${MNEMON_VENV_BIN:-$REPO_ROOT/.venv/bin}"
APP_NAME="${MNEMON_FLY_APP_NAME:-mnemon-memory}"
VAULT_DIR="${MNEMON_VAULT_DIR:-$HOME/.mnemon}"
LOCAL_TOKEN_FILE="$VAULT_DIR/local_token"
CHANGELOG="$REPO_ROOT/CHANGELOG.md"

usage() {
    sed -n '2,/^$/{s/^# \{0,1\}//;p;}' "${BASH_SOURCE[0]}"
}

# Refuse to dump credentials in stdout — applies to any subcommand
# that pulls env vars from the Fly machine.
_redact_in_stdout() {
    # No-op marker for grep; the actual redaction happens by structuring
    # the recovery flow to write the file directly via SSH instead of
    # roundtripping the value through this shell.
    :
}

cmd_cleanup_test_apps() {
    command -v flyctl >/dev/null || { echo "ERROR: flyctl not on PATH" >&2; exit 2; }
    local apps
    apps=$(flyctl apps list 2>/dev/null | awk '/^mnemon-test-/ {print $1}' || true)
    if [ -z "$apps" ]; then
        echo "no dangling mnemon-test-* apps"
        return 0
    fi
    echo "Destroying:"
    echo "$apps" | sed 's/^/  /'
    while IFS= read -r app; do
        [ -z "$app" ] && continue
        flyctl apps destroy "$app" -y || echo "WARN: destroy failed for $app" >&2
    done <<<"$apps"
    echo "cleanup-test-apps: done"
}

cmd_recover_token() {
    command -v flyctl >/dev/null || { echo "ERROR: flyctl not on PATH" >&2; exit 2; }
    mkdir -p "$VAULT_DIR"
    # Stream the token to the file directly via ssh -C; never let the
    # value touch this script's stdout. `printenv` exits non-zero if
    # the env var is unset, which surfaces here as a hard error.
    flyctl ssh console -a "$APP_NAME" -C 'printenv MNEMON_LOCAL_TOKEN' \
        | tail -n 1 \
        > "$LOCAL_TOKEN_FILE.tmp"
    if [ ! -s "$LOCAL_TOKEN_FILE.tmp" ]; then
        rm -f "$LOCAL_TOKEN_FILE.tmp"
        echo "ERROR: empty token from Fly — is MNEMON_LOCAL_TOKEN set on the app?" >&2
        exit 1
    fi
    mv "$LOCAL_TOKEN_FILE.tmp" "$LOCAL_TOKEN_FILE"
    chmod 600 "$LOCAL_TOKEN_FILE"
    echo "token recovered → $LOCAL_TOKEN_FILE (0o600)"
}

cmd_restart_machine() {
    command -v flyctl >/dev/null || { echo "ERROR: flyctl not on PATH" >&2; exit 2; }
    local mid
    mid=$(flyctl machine list -a "$APP_NAME" --json | "$MNEMON_VENV_BIN/python" -c 'import json,sys; print(json.load(sys.stdin)[0]["id"])')
    echo "restarting machine $mid in $APP_NAME ..."
    flyctl machine restart "$mid" -a "$APP_NAME"
    echo "verifying with mnemon doctor ..."
    "$MNEMON_VENV_BIN/mnemon" doctor
}

cmd_vault_stats() {
    local db="$VAULT_DIR/default.sqlite"
    if [ ! -f "$db" ]; then
        echo "no vault at $db"
        return 0
    fi
    "$MNEMON_VENV_BIN/python" - <<PYEOF
import sqlite3, os
db_path = "$db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
size_mb = os.path.getsize(db_path) / 1_048_576
total = conn.execute("SELECT COUNT(*) FROM documents WHERE invalidated_at IS NULL").fetchone()[0]
by_type = conn.execute(
    "SELECT content_type, COUNT(*) FROM documents "
    "WHERE invalidated_at IS NULL GROUP BY content_type ORDER BY 2 DESC"
).fetchall()
recent = conn.execute(
    "SELECT id, title, content_type, created_at FROM documents "
    "WHERE invalidated_at IS NULL ORDER BY id DESC LIMIT 5"
).fetchall()

print(f"Vault: $db")
print(f"  size:       {size_mb:.2f} MB")
print(f"  live docs:  {total}")
print()
print("  by content_type:")
for ct, n in by_type:
    print(f"    {ct:<14} {n:>5}")
print()
print("  most recent 5:")
for r in recent:
    print(f"    #{r['id']:<5}  [{r['content_type']:<12}]  {r['title'][:60]}  ({r['created_at']})")
PYEOF
}

cmd_changelog_extract() {
    local version="${1:-}"
    if [ -z "$version" ]; then
        echo "ERROR: usage: $0 changelog-extract <version>" >&2
        exit 2
    fi
    if [ ! -f "$CHANGELOG" ]; then
        echo "ERROR: $CHANGELOG not found" >&2
        exit 1
    fi
    # Match `## [VERSION]` header through to (but not including) the next
    # `## [` header. Treats `## [Unreleased]` consistently with versioned
    # headers. Brackets are escaped because they're regex metachars.
    "$MNEMON_VENV_BIN/python" - "$version" "$CHANGELOG" <<'PYEOF'
import re, sys
version = sys.argv[1]
path = sys.argv[2]
content = open(path).read()
pattern = rf"^## \[{re.escape(version)}\].*?(?=^## \[|\Z)"
m = re.search(pattern, content, re.MULTILINE | re.DOTALL)
if not m:
    print(f"ERROR: no `## [{version}]` section found in {path}", file=sys.stderr)
    sys.exit(1)
sys.stdout.write(m.group(0).rstrip() + "\n")
PYEOF
}

main() {
    local sub="${1:-help}"
    shift || true
    case "$sub" in
        cleanup-test-apps)   cmd_cleanup_test_apps "$@" ;;
        recover-token)       cmd_recover_token "$@" ;;
        restart-machine)     cmd_restart_machine "$@" ;;
        vault-stats)         cmd_vault_stats "$@" ;;
        changelog-extract)   cmd_changelog_extract "$@" ;;
        help|-h|--help|"")   usage ;;
        *)
            echo "unknown subcommand: $sub" >&2
            echo >&2
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
