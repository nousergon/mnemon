#!/usr/bin/env bash
# scripts/salience_phase0.sh — Phase 0 of the salience-tier plan.
#
# Drives the operator-side workflow for validating the two-tier
# (standing context + situational recall) hypothesis without
# committing to the Phase 1 schema migration.
#
# Plan: private/mnemon-salience-tier-plan-260521.md
# Scoring: scripts/build_standing_set.py
# Feature flag: src/mnemon/hooks/context_surfacing.py (env-var-gated)
#
# Subcommands:
#   snapshot           Atomic backup of prod Fly vault → local file
#                      (SQLite online-backup API; safe with running server)
#   score [--top N]    Run build_standing_set.py against the snapshot
#                      and print top-N candidates (default 30)
#   select <IDS>       Write ~/.mnemon/standing.json from a comma-separated
#                      ID list (e.g. `select 123,456,789`)
#   status             Show current standing-tier state (file, IDs, env)
#   disable            Remove ~/.mnemon/standing.json + remind unset env
#   help               This help
#
# Typical flow:
#   scripts/salience_phase0.sh snapshot
#   scripts/salience_phase0.sh score
#   # operator inspects candidates, picks ~10 by hand
#   scripts/salience_phase0.sh select 123,456,789,...
#   export MNEMON_STANDING_TIER_FILE=~/.mnemon/standing.json
#   # replay runway conversation with/without the env var
#   # record verdict in private/salience-phase0-results-26MMDD.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MNEMON_VENV_BIN="$REPO_ROOT/.venv/bin"
[ -x "$MNEMON_VENV_BIN/python" ] || { echo "ERROR: $MNEMON_VENV_BIN/python not found — install editable venv first" >&2; exit 1; }

SNAPSHOT_PATH="${SALIENCE_SNAPSHOT_PATH:-/tmp/mnemon-prod-snap.sqlite}"
STANDING_FILE="$HOME/.mnemon/standing.json"
FLY_APP="${MNEMON_FLY_APP:-mnemon-memory}"

echo_step() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
echo_ok()   { printf "  \033[1;32m✓\033[0m %s\n" "$*"; }
echo_warn() { printf "  \033[1;33m⚠\033[0m %s\n" "$*" >&2; }
echo_err()  { printf "  \033[1;31m✗\033[0m %s\n" "$*" >&2; }
die()       { echo_err "$*"; exit 1; }


cmd_snapshot() {
    echo_step "Snapshot prod Fly vault → $SNAPSHOT_PATH"

    flyctl auth whoami >/dev/null 2>&1 || die "flyctl not logged in"

    # Vec store lives alongside the sqlite (FastEmbed embeddings).
    # build_standing_set.py needs BOTH for embedding-based signals
    # (constraint_score, time_penalty) — without the .vec.npz, those
    # signals are zero and the scorer falls back to FTS-breadth-only,
    # which surfaces noise. Surfaced 2026-05-21 when bare-sqlite
    # snapshot produced garbage picks ("halt the run", "propagate").
    local VEC_PATH="${SNAPSHOT_PATH%.sqlite}.vec.npz"

    # 1) Backup-API snapshot on the Fly machine to /tmp/snap.sqlite
    # 2) Copy vec.npz to /tmp/snap.vec.npz (numpy savez is atomic via
    #    temp+rename, so plain cp is safe even with serve-remote
    #    actively writing)
    # 3) sftp both files down
    # 4) clean up remote temp files
    #
    # Why backup-API not raw cp for sqlite: live serve-remote has
    # frames in WAL that raw cp of default.sqlite would miss
    # (verified 2026-05-21). Connection.backup() handles WAL atomically.
    echo_step "  Step 1/4 — sqlite backup on Fly via online-backup API"
    flyctl ssh console -a "$FLY_APP" -C \
        "python -c 'import sqlite3; src=sqlite3.connect(\"/data/default.sqlite\"); dst=sqlite3.connect(\"/tmp/snap.sqlite\"); src.backup(dst); src.close(); dst.close(); print(\"snapshot done\")'" \
        || die "remote sqlite backup failed"
    echo_ok "remote /tmp/snap.sqlite written"

    echo_step "  Step 2/4 — vecstore copy on Fly (numpy savez is atomic)"
    flyctl ssh console -a "$FLY_APP" -C \
        "cp /data/default.vec.npz /tmp/snap.vec.npz && echo 'vec copy done'" \
        || echo_warn "remote vecstore copy failed — embedding signals will be zero"
    echo_ok "remote /tmp/snap.vec.npz written"

    echo_step "  Step 3/4 — sftp download (sqlite + vec.npz)"
    rm -f "$SNAPSHOT_PATH" "$VEC_PATH"
    flyctl ssh sftp get -a "$FLY_APP" /tmp/snap.sqlite "$SNAPSHOT_PATH" \
        || die "sftp sqlite download failed"
    [ -f "$SNAPSHOT_PATH" ] || die "sftp succeeded but $SNAPSHOT_PATH missing"
    flyctl ssh sftp get -a "$FLY_APP" /tmp/snap.vec.npz "$VEC_PATH" \
        || echo_warn "sftp vec.npz download failed — embedding signals will be zero"
    local sqlite_size vec_size
    sqlite_size="$(stat -f '%z' "$SNAPSHOT_PATH" 2>/dev/null || stat -c '%s' "$SNAPSHOT_PATH")"
    if [ -f "$VEC_PATH" ]; then
        vec_size="$(stat -f '%z' "$VEC_PATH" 2>/dev/null || stat -c '%s' "$VEC_PATH")"
        echo_ok "downloaded sqlite=${sqlite_size}B vec.npz=${vec_size}B"
    else
        echo_warn "downloaded sqlite=${sqlite_size}B, vec.npz MISSING (embedding signals will be zero)"
    fi

    echo_step "  Step 4/4 — clean up remote temp files"
    flyctl ssh console -a "$FLY_APP" -C "rm -f /tmp/snap.sqlite /tmp/snap.vec.npz" \
        || echo_warn "remote cleanup failed (harmless — /tmp churns on machine restart)"
    echo_ok "remote /tmp files removed"

    # Quick sanity: count live memories + verify vec.npz is loadable
    local live_count vec_count
    live_count="$("$MNEMON_VENV_BIN/python" -c "
import sqlite3
c = sqlite3.connect('$SNAPSHOT_PATH')
print(c.execute('SELECT COUNT(*) FROM documents WHERE invalidated_at IS NULL').fetchone()[0])
")"
    if [ -f "$VEC_PATH" ]; then
        vec_count="$("$MNEMON_VENV_BIN/python" -c "
import numpy as np
d = np.load('$VEC_PATH', allow_pickle=True)
print(len(d['ids']))
" 2>/dev/null || echo '?')"
    else
        vec_count="(absent)"
    fi
    echo_step "Snapshot ready"
    echo "  Sqlite:  $SNAPSHOT_PATH"
    echo "  Vec:     $VEC_PATH"
    echo "  Live memories: $live_count"
    echo "  Vectors:       $vec_count"
    echo ""
    echo "Next: scripts/salience_phase0.sh score"
}


cmd_score() {
    # --show:  how many candidates to DISPLAY (default 30)
    # --top:   how many to AUTO-SELECT into standing.json (default = python
    #          script's default, currently 10; hard ceiling 20)
    # --print-only: don't write standing.json / standing-rendered.md, just print
    local show=30
    local extra=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --show) show="$2"; shift 2 ;;
            --top) extra+=("--top" "$2"); shift 2 ;;
            --print-only) extra+=("--print-only"); shift ;;
            *) die "unknown arg: $1 (supported: --show N, --top N, --print-only)" ;;
        esac
    done

    [ -f "$SNAPSHOT_PATH" ] || die "no snapshot at $SNAPSHOT_PATH — run \`$0 snapshot\` first"

    echo_step "Score candidates against $SNAPSHOT_PATH (show $show, auto-select per python default)"
    # `"${extra[@]+"${extra[@]}"}"` — bash idiom for "expand the array
    # if non-empty, expand to nothing if empty". Plain `"${extra[@]}"`
    # would trip `set -u` (unbound variable) on bash 3.2 (macOS default)
    # when extra is empty.
    "$MNEMON_VENV_BIN/python" "$REPO_ROOT/scripts/build_standing_set.py" \
        --db "$SNAPSHOT_PATH" --show "$show" "${extra[@]+"${extra[@]}"}"

    echo_step "Next"
    echo "  Score auto-wrote standing.json + standing-rendered.md (top N marked with ★)."
    echo "  To override the auto-selection: scripts/salience_phase0.sh select 123,456,789,..."
    echo "  To activate: export MNEMON_STANDING_TIER_FILE=$STANDING_FILE"
}


cmd_select() {
    [ $# -eq 1 ] || die "usage: $0 select <id1,id2,id3,...>"
    local raw="$1"

    # Parse comma-separated, validate each is a positive integer.
    local ids
    ids="$("$MNEMON_VENV_BIN/python" - "$raw" <<'PY'
import json, sys
raw = sys.argv[1]
try:
    parsed = [int(x.strip()) for x in raw.split(",") if x.strip()]
except ValueError as e:
    print(f"ERROR: non-integer in id list: {e}", file=sys.stderr)
    sys.exit(2)
if not parsed:
    print("ERROR: empty id list", file=sys.stderr)
    sys.exit(2)
if any(i <= 0 for i in parsed):
    print("ERROR: ids must be positive integers", file=sys.stderr)
    sys.exit(2)
if len(parsed) > 20:
    print(f"WARN: {len(parsed)} ids — Phase 0 plan suggests ≤20 to preserve salience", file=sys.stderr)
print(json.dumps({"ids": parsed}, indent=2))
PY
)" || die "failed to parse id list"

    mkdir -p "$(dirname "$STANDING_FILE")"
    printf '%s\n' "$ids" > "$STANDING_FILE"
    echo_step "Wrote $STANDING_FILE"
    cat "$STANDING_FILE"
    echo ""
    echo_step "Next"
    echo "  In your shell:"
    echo "    export MNEMON_STANDING_TIER_FILE=$STANDING_FILE"
    echo ""
    echo "  Or per-session config for Claude Code (so it auto-loads on every prompt)."
    echo "  Then replay the runway conversation and compare model responses."
    echo "  Record the verdict in private/salience-phase0-results-\$(date +%y%m%d).md"
}


cmd_status() {
    echo_step "Salience Phase 0 state"
    echo ""
    if [ -f "$SNAPSHOT_PATH" ]; then
        local size mtime live
        size="$(stat -f '%z' "$SNAPSHOT_PATH" 2>/dev/null || stat -c '%s' "$SNAPSHOT_PATH")"
        mtime="$(stat -f '%Sm' "$SNAPSHOT_PATH" 2>/dev/null || stat -c '%y' "$SNAPSHOT_PATH")"
        live="$("$MNEMON_VENV_BIN/python" -c "
import sqlite3
print(sqlite3.connect('$SNAPSHOT_PATH').execute('SELECT COUNT(*) FROM documents WHERE invalidated_at IS NULL').fetchone()[0])
" 2>/dev/null || echo '?')"
        echo "  Snapshot: $SNAPSHOT_PATH ($size B, $live live memories, modified $mtime)"
    else
        echo "  Snapshot: (none — run \`$0 snapshot\`)"
    fi

    if [ -f "$STANDING_FILE" ]; then
        echo "  Standing file: $STANDING_FILE"
        local n_ids
        n_ids="$("$MNEMON_VENV_BIN/python" -c "
import json
print(len(json.load(open('$STANDING_FILE')).get('ids', [])))
" 2>/dev/null || echo '?')"
        echo "    IDs: $n_ids selected"
        "$MNEMON_VENV_BIN/python" -c "
import json
ids = json.load(open('$STANDING_FILE')).get('ids', [])
print('    contents:', ids)
" 2>/dev/null || true
    else
        echo "  Standing file: (none — run \`$0 select <IDS>\` after scoring)"
    fi

    if [ -n "${MNEMON_STANDING_TIER_FILE:-}" ]; then
        echo "  Env: MNEMON_STANDING_TIER_FILE=$MNEMON_STANDING_TIER_FILE (active)"
    else
        echo "  Env: MNEMON_STANDING_TIER_FILE not set in this shell"
        echo "       (set it to activate: export MNEMON_STANDING_TIER_FILE=$STANDING_FILE)"
    fi
}


cmd_disable() {
    if [ -f "$STANDING_FILE" ]; then
        rm "$STANDING_FILE"
        echo_ok "removed $STANDING_FILE"
    else
        echo_warn "no $STANDING_FILE to remove"
    fi
    echo ""
    echo "  Don't forget to unset the env var in your shell:"
    echo "    unset MNEMON_STANDING_TIER_FILE"
}


cmd_help() {
    sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}


case "${1:-help}" in
    snapshot) shift; cmd_snapshot "$@" ;;
    score)    shift; cmd_score "$@" ;;
    select)   shift; cmd_select "$@" ;;
    status)   shift; cmd_status "$@" ;;
    disable)  shift; cmd_disable "$@" ;;
    help|-h|--help) cmd_help ;;
    *) echo_err "unknown subcommand: $1"; cmd_help; exit 2 ;;
esac
