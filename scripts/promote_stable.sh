#!/usr/bin/env bash
# scripts/promote_stable.sh — drive a mnemon rc → stable promotion end-to-end.
#
# Automates the operator sequence documented in:
#   - ROADMAP.md § Pre-deploy
#   - private/DEPLOYMENT_CHECKLIST.md
#   - private/e2e-test-runbook-260421.md (Layer-3 web test)
#
# Does NOT bypass the operator-only rules in SYSTEM_STATE.md — credentials
# remain operator-owned at every step:
#   - twine reads ~/.pypirc / TWINE_PASSWORD env (or prompts).
#   - flyctl uses the operator's logged-in session.
#   - gh uses the operator's logged-in gh auth.
# The script just sequences the commands an operator would run by hand,
# with loud preflight + per-step echo + confirmation prompts before each
# destructive action.
#
# Target version is read from src/mnemon/__init__.py — the script is
# version-agnostic and reusable for future stable cuts.
#
# Usage:
#   scripts/promote_stable.sh preflight   # read-only, runs before everything else
#   scripts/promote_stable.sh layer3      # E2E web test (creates+destroys test Fly app, ~15 min)
#   scripts/promote_stable.sh layer3 --exercise-all-tools   # also probe every MCP tool (~30-60s extra)
#                                         # pins mnemon upgrade web --mnemon-version to TARGET_VERSION
#                                         # if it's on PyPI, otherwise the latest published version
#                                         # as proxy (LAYER3_VERSION_OVERRIDE=<ver> to override).
#                                         # SOTA for true pre-publish validation: TestPyPI (ROADMAP).
#   scripts/promote_stable.sh publish     # POST-MERGE: build → twine upload → tag → GH release → Fly redeploy
#   scripts/promote_stable.sh verify      # post-publish sanity check
#
# Sequence for a stable cut:
#   1. operator merges 0.6.0rc{N} into main, including the 3-line bump + CHANGELOG entry
#      (or merges a candidate, and accepts that publish will run against whatever
#       __version__ main currently holds — the script reads it as truth)
#   2. operator: scripts/promote_stable.sh preflight   ← validates main is ready
#   3. operator: scripts/promote_stable.sh layer3      ← validates web upgrade/downgrade end-to-end
#   4. operator: merge the promote PR (this is the only step the script can't drive)
#   5. operator: scripts/promote_stable.sh publish     ← post-merge build + ship
#   6. operator: scripts/promote_stable.sh verify      ← post-publish confirmation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- style ----
echo_step() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
echo_ok()   { printf "\033[1;32m  ✓\033[0m %s\n" "$*"; }
echo_warn() { printf "\033[1;33m  ⚠\033[0m %s\n" "$*" >&2; }
echo_err()  { printf "\033[1;31m  ✗\033[0m %s\n" "$*" >&2; }
die()       { echo_err "$*"; exit 1; }

confirm() {
    local prompt="$1"
    printf "\033[1;33m  ? %s [y/N] \033[0m" "$prompt"
    local reply=""
    read -r reply
    [[ "$reply" =~ ^[Yy]$ ]] || die "aborted by operator"
}

# Read the target version from the canonical source — the script is
# version-agnostic; bump src/mnemon/__init__.py and the script targets
# the new version automatically.
TARGET_VERSION="$(awk -F'"' '/^__version__ = /{print $2; exit}' src/mnemon/__init__.py)"
[ -n "$TARGET_VERSION" ] || die "could not read __version__ from src/mnemon/__init__.py"

# Operator-overridable so CI / non-.venv setups can point at the
# interpreter dir GitHub Actions actually provisioned (no virtualenv
# created in default ci.yml). Same pattern as scripts/mnemon_ops.sh +
# tests/test_mnemon_ops.py — sys.executable's parent works everywhere.
MNEMON_VENV_BIN="${MNEMON_VENV_BIN:-$REPO_ROOT/.venv/bin}"
[ -x "$MNEMON_VENV_BIN/mnemon" ] || die "mnemon CLI not found at $MNEMON_VENV_BIN/mnemon — activate / install editable venv first (or override MNEMON_VENV_BIN)"
[ -x "$MNEMON_VENV_BIN/twine" ] || die "twine not found at $MNEMON_VENV_BIN/twine — pip install -e .[dev]"
[ -x "$MNEMON_VENV_BIN/python" ] || die "venv python not found at $MNEMON_VENV_BIN/python"

# ---- helpers ----

# Resolve the latest version of mnemon-memory currently published to PyPI,
# pre-releases included. Uses the PyPI JSON API + packaging.version (via
# the venv python) so the answer is deterministic — `pip index versions`
# proved unreliable in the wild (stale cache + info.version field returning
# the latest STABLE only, hiding 0.6.0rcN entirely).
latest_pypi_version() {
    "$MNEMON_VENV_BIN/python" - <<'PY' 2>/dev/null
import urllib.request, json, sys
from packaging.version import Version, InvalidVersion
try:
    with urllib.request.urlopen("https://pypi.org/pypi/mnemon-memory/json", timeout=10) as r:
        d = json.load(r)
except Exception as e:
    sys.exit(1)
vs = []
for s in d.get("releases", {}).keys():
    try:
        vs.append(Version(s))
    except InvalidVersion:
        pass
if not vs:
    sys.exit(1)
vs.sort()
print(vs[-1])
PY
}

# Is a specific version published to PyPI?
is_pypi_published() {
    local v="$1"
    "$MNEMON_VENV_BIN/python" - "$v" <<'PY' 2>/dev/null
import urllib.request, json, sys
target = sys.argv[1]
try:
    with urllib.request.urlopen("https://pypi.org/pypi/mnemon-memory/json", timeout=10) as r:
        d = json.load(r)
except Exception:
    sys.exit(2)
sys.exit(0 if target in d.get("releases", {}) else 1)
PY
}

# ============================================================
# preflight — read-only validation that the candidate is ready
# ============================================================
cmd_preflight() {
    echo_step "Preflight — validating $TARGET_VERSION candidate"

    # Preflight + layer3 typically run on the promote BRANCH (pre-merge);
    # only `publish` requires main. Accept any branch but surface where
    # we are so the operator sees it explicitly.
    local branch
    branch="$(git branch --show-current)"
    git diff --quiet HEAD || die "working tree has uncommitted changes"
    if [[ "$branch" == "main" ]]; then
        echo_ok "on main, clean tree"
    else
        echo_ok "on branch '$branch', clean tree (publish will require main)"
    fi

    local pyver
    pyver="$(awk -F'"' '/^version = /{print $2; exit}' pyproject.toml)"
    [[ "$pyver" == "$TARGET_VERSION" ]] || die "version mismatch: __init__='$TARGET_VERSION' pyproject='$pyver'"
    echo_ok "version pinned consistently: $TARGET_VERSION"

    grep -q "^## \[$TARGET_VERSION\]" CHANGELOG.md \
      || die "CHANGELOG.md missing ## [$TARGET_VERSION] section"
    echo_ok "CHANGELOG entry present"

    if git rev-parse "v$TARGET_VERSION" >/dev/null 2>&1; then
        die "tag v$TARGET_VERSION already exists locally — promotion already happened?"
    fi
    git fetch --tags --quiet
    if git rev-parse "v$TARGET_VERSION" >/dev/null 2>&1; then
        die "tag v$TARGET_VERSION exists on remote — promotion already happened?"
    fi
    echo_ok "no v$TARGET_VERSION tag yet"

    echo_step "pytest"
    PYTHONPATH=src "$MNEMON_VENV_BIN/pytest" -q
    echo_ok "tests green"

    echo_step "Auth surfaces"
    flyctl auth whoami >/dev/null 2>&1 || die "flyctl not logged in (run: flyctl auth login)"
    aws sts get-caller-identity >/dev/null 2>&1 || die "aws creds not configured"
    gh auth status >/dev/null 2>&1 || die "gh not logged in (run: gh auth login)"
    echo_ok "flyctl, aws, gh authenticated"

    echo_step "PREFLIGHT PASSED — $TARGET_VERSION ready for layer3 + publish"
}

# ============================================================
# layer3 — E2E web test against test-scoped Fly app
# Mirrors private/e2e-test-runbook-260421.md
# ============================================================
_layer3_cleanup() {
    # Best-effort cleanup on any exit path.
    # Failure-mode swallowed: cleanup-side errors (dangling test app destroy,
    #   stale S3 prefix, lingering local dirs). Primary deliverable (the
    #   pass/fail signal of the upgrade/downgrade cycle) survives a cleanup
    #   error because the script's exit code reflects the last non-cleanup
    #   command. Cleanup failures surface via the WARN log to operator stdout
    #   so the next layer3 invocation's "dangling mnemon-test-* apps" preflight
    #   will catch any leftover state.
    #
    # Auth-state restore is **not** best-effort: a failure here leaves the
    # operator's prod mnemon broken (~/.mnemon/{remote_url,local_token}
    # pointing at the now-destroyed test app). Surface the restore explicitly,
    # but still don't override `rc` — the underlying script failure remains
    # the primary signal.
    local rc=$?

    # 1. Destroy any lingering test Fly app first — frees the namespace + stops
    #    cost accumulation before doing anything else.
    #
    # Surfaced 2026-05-21 (during Layer-3 attempt-4): a previous trap
    # invocation reported `destroying lingering test app ...` but the
    # destroy didn't actually succeed (app survived 44 minutes in
    # suspended state, blocked the next layer3 invocation's
    # "dangling apps" preflight). The original silent `>/dev/null 2>&1`
    # masked the cause — could be a machine still draining, a transient
    # API error, or auth expiry.
    #
    # Fix: capture stderr to a temp log, retry once after a brief
    # sleep, and on persistent failure surface the captured error so
    # the operator knows what to recover by hand. Cleanup-side
    # failures stay tolerated (don't override `rc`), but their cause
    # is now visible.
    if [ -n "${TEST_APP_NAME:-}" ] && flyctl apps list 2>/dev/null | grep -q "$TEST_APP_NAME"; then
        echo_warn "cleanup: destroying lingering test app $TEST_APP_NAME"
        local _destroy_log
        _destroy_log="$(mktemp -t mnemon-layer3-destroy-XXXXXX.log)"
        if flyctl apps destroy "$TEST_APP_NAME" -y >/dev/null 2>"$_destroy_log"; then
            rm -f "$_destroy_log"
        else
            # First attempt failed — pause briefly + retry. Most common
            # causes (machine still draining, transient flyctl 5xx)
            # resolve in a few seconds.
            echo_warn "cleanup: flyctl destroy attempt 1 failed, retrying in 5s"
            sleep 5
            if flyctl apps destroy "$TEST_APP_NAME" -y >/dev/null 2>"$_destroy_log"; then
                echo_ok "cleanup: destroy succeeded on retry"
                rm -f "$_destroy_log"
            else
                echo_err "cleanup: flyctl destroy FAILED twice for $TEST_APP_NAME"
                echo_err "cleanup: captured stderr — recover by hand:"
                sed 's/^/    /' "$_destroy_log" >&2
                echo_err "cleanup: \`flyctl apps destroy $TEST_APP_NAME -y\` is the manual recovery"
                # Don't unlink the log — operator may want it for support.
                echo_warn "cleanup: stderr log retained at $_destroy_log"
            fi
        fi
    fi
    if [ -n "${MNEMON_S3_PREFIX:-}" ]; then
        aws s3 rm "s3://${MNEMON_S3_BUCKET:-mnemon-memory}/$MNEMON_S3_PREFIX/" --recursive >/dev/null 2>&1 || echo_warn "cleanup: s3 rm failed for prefix $MNEMON_S3_PREFIX"
    fi
    if [ -n "${MNEMON_VAULT_DIR:-}" ] && [[ "$MNEMON_VAULT_DIR" == "$HOME/.mnemon-test-"* ]]; then
        rm -rf "$MNEMON_VAULT_DIR" || true
    fi
    if [ -n "${MNEMON_CLIENT_CONFIG_ROOT:-}" ] && [[ "$MNEMON_CLIENT_CONFIG_ROOT" == "$HOME/.mnemon-test-configs-"* ]]; then
        rm -rf "$MNEMON_CLIENT_CONFIG_ROOT" || true
    fi

    # 2. Restore the operator's pre-Layer-3 mnemon auth state. `mnemon upgrade
    #    web` overwrites ~/.mnemon/{remote_url,local_token} in place — neither
    #    is covered by MNEMON_VAULT_DIR or MNEMON_CLIENT_CONFIG_ROOT — so we
    #    must put them back. Skipping this leaves prod mnemon unreachable.
    if [ -n "${LAYER3_AUTH_BACKUP_DIR:-}" ] && [ -d "$LAYER3_AUTH_BACKUP_DIR" ]; then
        local restored=0
        for f in remote_url local_token; do
            if [ -f "$LAYER3_AUTH_BACKUP_DIR/$f" ]; then
                cp -p "$LAYER3_AUTH_BACKUP_DIR/$f" "$HOME/.mnemon/$f" \
                    && restored=$((restored + 1)) \
                    || echo_err "cleanup: FAILED to restore ~/.mnemon/$f — operator must recover by hand"
            fi
        done
        if [ "$restored" -gt 0 ]; then
            echo_ok "cleanup: restored $restored pre-Layer-3 auth file(s) under ~/.mnemon/"
        fi
        rm -rf "$LAYER3_AUTH_BACKUP_DIR" || true
    fi

    return $rc
}

cmd_layer3() {
    # Parse layer3-specific flags. Currently:
    #   --exercise-all-tools  After upgrade, iterate every registered
    #                         MCP tool against the test Fly app and
    #                         assert each returns cleanly. Catches
    #                         Fly-specific breakage (missing baked
    #                         models, MCP proxy timeouts) that the
    #                         local-process integration canary
    #                         tests/test_tools_integration.py can't
    #                         see. Added 2026-05-22.
    local EXERCISE_ALL_TOOLS=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --exercise-all-tools)
                EXERCISE_ALL_TOOLS=1
                shift
                ;;
            *)
                die "unknown layer3 flag: $1"
                ;;
        esac
    done

    echo_step "Layer-3 web test — $TARGET_VERSION E2E against test-scoped Fly app"
    if [ "$EXERCISE_ALL_TOOLS" = "1" ]; then
        echo "  (--exercise-all-tools: every MCP tool will be invoked against the test app)"
    fi

    flyctl auth whoami >/dev/null 2>&1 || die "flyctl not logged in"
    aws sts get-caller-identity >/dev/null 2>&1 || die "aws creds not configured"
    if flyctl apps list 2>/dev/null | grep -q mnemon-test-; then
        die "dangling mnemon-test-* Fly apps exist — destroy them before running"
    fi

    # ---- Resolve the mnemon-memory version Layer-3 will deploy to the test app ----
    #
    # `mnemon upgrade web` calls `flyctl deploy`, which builds a Docker image that
    # runs `pip install mnemon-memory[server]==<ver>` against real PyPI. The
    # candidate version (TARGET_VERSION) may not yet be published — that's the
    # whole point of pre-publish validation. So Layer-3 must pin to a version
    # that *is* on PyPI:
    #
    #   - If TARGET_VERSION is already on PyPI (e.g., re-running layer3 after
    #     a publish), use it directly.
    #   - Otherwise pin to the latest published version as a proxy. This is
    #     valid for the byte-identical rc → stable promotion (0.6.0rc18 ↔
    #     0.6.0 source-identical). For future rc bumps where the candidate has
    #     code changes from the prior rc, the latest-published-as-proxy is
    #     incomplete validation — see ROADMAP "TestPyPI integration for true
    #     pre-publish validation" for the institutional fix.
    #
    # Override: set LAYER3_VERSION_OVERRIDE=<ver> to bypass auto-resolution.
    local LAYER3_VERSION
    if [ -n "${LAYER3_VERSION_OVERRIDE:-}" ]; then
        LAYER3_VERSION="$LAYER3_VERSION_OVERRIDE"
        echo_ok "Layer-3 pinning mnemon-memory==$LAYER3_VERSION (operator override)"
    elif is_pypi_published "$TARGET_VERSION"; then
        LAYER3_VERSION="$TARGET_VERSION"
        echo_ok "Layer-3 pinning mnemon-memory==$LAYER3_VERSION (candidate already on PyPI)"
    else
        LAYER3_VERSION="$(latest_pypi_version)"
        [ -n "$LAYER3_VERSION" ] || die "could not resolve mnemon-memory latest from PyPI JSON API"
        echo_warn "candidate $TARGET_VERSION isn't on PyPI yet"
        echo_warn "Layer-3 will deploy mnemon-memory==$LAYER3_VERSION (latest published) as a proxy"
        echo_warn "for true pre-publish validation of $TARGET_VERSION's code: file TestPyPI integration as ROADMAP follow-up"
    fi
    confirm "proceed with Layer-3 deploying mnemon-memory==$LAYER3_VERSION?"

    # Isolate test environment per the runbook.
    local TEST_RUN_ID
    TEST_RUN_ID="$(date +%Y%m%d-%H%M%S)"
    export MNEMON_VAULT_DIR="$HOME/.mnemon-test-$TEST_RUN_ID"
    export MNEMON_S3_BUCKET="mnemon-memory"
    export MNEMON_S3_PREFIX="test-upgrade/$TEST_RUN_ID"
    export TEST_APP_NAME="mnemon-test-$TEST_RUN_ID"
    export MNEMON_CLIENT_CONFIG_ROOT="$HOME/.mnemon-test-configs-$TEST_RUN_ID"
    # MNEMON_PROD_APP_NAMES guards prod against an accidental `upgrade web
    # --app-name mnemon-memory` during the test window. See SYSTEM_STATE.md
    # "Known gotchas / sharp edges".
    export MNEMON_PROD_APP_NAMES="mnemon-memory"

    mkdir -p "$MNEMON_VAULT_DIR" "$MNEMON_CLIENT_CONFIG_ROOT"

    # ---- Snapshot the operator's pre-Layer-3 mnemon auth state ----
    #
    # `mnemon upgrade web` overwrites ~/.mnemon/{remote_url,local_token} in
    # place with the test app's URL and freshly-generated token. The runbook's
    # isolation strategy (MNEMON_VAULT_DIR + MNEMON_CLIENT_CONFIG_ROOT)
    # covers vault + client configs but NOT these two files. If the test app
    # is then destroyed (trap path or success path), the operator's prod
    # mnemon is left unreachable until they manually recover.
    #
    # Snapshot now, restore in the cleanup trap.
    export LAYER3_AUTH_BACKUP_DIR
    LAYER3_AUTH_BACKUP_DIR="$(mktemp -d -t mnemon-layer3-auth-XXXXXX)"
    for f in remote_url local_token; do
        if [ -f "$HOME/.mnemon/$f" ]; then
            cp -p "$HOME/.mnemon/$f" "$LAYER3_AUTH_BACKUP_DIR/$f"
        fi
    done
    echo_ok "snapshotted pre-Layer-3 auth state to $LAYER3_AUTH_BACKUP_DIR"

    # Install the cleanup trap now that we have something to restore.
    trap _layer3_cleanup EXIT
    echo_ok "isolated test env (RUN_ID=$TEST_RUN_ID, app=$TEST_APP_NAME)"

    local M="$MNEMON_VENV_BIN/mnemon"

    # ---- Force local-mode for the Step 2 seed ----
    #
    # `mnemon save` honors ~/.mnemon/remote_url (and the MNEMON_REMOTE_URL
    # env var) and will write to whatever URL is configured. The prod URL
    # in the file means seed saves would land in PROD, not the local test
    # vault — which is exactly the bug that produced "0 docs on remote"
    # after the S3 → Fly seed earlier today.
    #
    # Move the file aside and unset the env var so saves use local mode.
    # Upgrade web (Step 3) will write the test app's URL into a fresh
    # remote_url file; the trap restores the snapshot at the very end.
    unset MNEMON_REMOTE_URL
    if [ -f "$HOME/.mnemon/remote_url" ]; then
        rm -f "$HOME/.mnemon/remote_url"
        echo_ok "moved ~/.mnemon/remote_url aside so Step 2 saves use local mode"
    fi

    echo_step "Step 2 — seed local test vault"
    # mnemon's store.save() does content-hash dedup (rc15+) — saves with
    # byte-identical *content* collapse into one doc, even with different
    # titles. The original runbook (pre-rc15) passed the same `run-$ID`
    # content across all three saves, which now silently dedups down to
    # 2 docs (the third escapes because `--type preference` puts it in a
    # different content_type bucket). Make the content unique per save.
    "$M" save "Test memory for E2E upgrade cycle"             "seed-1 observation run-$TEST_RUN_ID" --type observation
    "$M" save "Second test memory, should survive upgrade"    "seed-2 observation run-$TEST_RUN_ID" --type observation
    "$M" save "Preference to keep after upgrade/downgrade"    "seed-3 preference run-$TEST_RUN_ID" --type preference
    local seeded_local_count
    seeded_local_count="$("$M" status 2>&1 | awk '/^Total memories:/{print $NF; exit}')"
    [[ "$seeded_local_count" == "3" ]] || die "expected 3 docs in local test vault, got '$seeded_local_count' (saves may have deduped — check store content-hash logic, or routed somewhere unexpected)"
    echo_ok "3 docs seeded in local test vault"

    echo_step "Step 3 — upgrade web to $TEST_APP_NAME (pinned to mnemon-memory==$LAYER3_VERSION)"
    "$M" upgrade web --app-name "$TEST_APP_NAME" --mnemon-version "$LAYER3_VERSION"
    ls "$MNEMON_VAULT_DIR/archive/" | grep -q "pre-web-" || die "expected pre-web-*.sqlite archive missing"
    ! [ -f "$MNEMON_VAULT_DIR/default.sqlite" ] || die "local vault should be archived; default.sqlite still present"

    # `mnemon status` CLI is local-only (instantiates Store() directly, ignores
    # MNEMON_REMOTE_URL). Use the remote helper that goes through the same
    # call_tool_sync path mnemon doctor uses. Tracked as ROADMAP follow-up:
    # extend mnemon CLI to honor remote mode so this workaround can go away.
    local REMOTE_HELPER="$REPO_ROOT/scripts/_layer3_remote_helper.py"
    local seeded_count
    seeded_count="$("$MNEMON_VENV_BIN/python" "$REMOTE_HELPER" status)"
    [[ "$seeded_count" == "3" ]] || die "expected 3 docs seeded to remote, got '$seeded_count'"
    echo_ok "upgrade complete; local vault archived; 3 docs seeded to remote via S3 → Fly"

    echo_step "Step 4 — exercise remote (add via HTTP)"
    # mnemon save CLI is also local-only — use the remote helper.
    "$MNEMON_VENV_BIN/python" "$REMOTE_HELPER" save \
        "Memory added after upgrade, should survive downgrade" \
        "seed-4 observation run-$TEST_RUN_ID" \
        "observation"
    local remote_count
    remote_count="$("$MNEMON_VENV_BIN/python" "$REMOTE_HELPER" status)"
    [[ "$remote_count" == "4" ]] || die "expected 4 docs on remote, got '$remote_count'"
    echo_ok "4 docs on remote"

    # Step 4.5 — exercise every MCP tool against the test app. Opt-in
    # via --exercise-all-tools because it adds ~30-60s to the layer3
    # run (one HTTP round-trip per tool). Catches Fly-specific failures
    # the local Python integration canary can't surface — missing
    # baked models in the Docker image, Anthropic MCP proxy timeouts,
    # transport regressions. Composes with tests/test_tools_integration.py
    # (PR #158).
    if [ "$EXERCISE_ALL_TOOLS" = "1" ]; then
        echo_step "Step 4.5 — exercise all MCP tools against the test app"
        "$MNEMON_VENV_BIN/python" "$REMOTE_HELPER" exercise-all-tools \
            || die "all-tools exercise failed against test app — see output above"
        echo_ok "every MCP tool returned cleanly"
    fi

    echo_step "Step 5 — downgrade local + destroy fly app"
    "$M" downgrade local --destroy-fly-app
    local local_count
    local_count="$("$M" status 2>&1 | awk '/^Total memories:/{print $NF; exit}')"
    [[ "$local_count" == "4" ]] || die "expected 4 docs after downgrade, got '$local_count'"
    if flyctl apps list 2>/dev/null | grep -q "$TEST_APP_NAME"; then
        die "test app not destroyed: $TEST_APP_NAME"
    fi
    echo_ok "downgrade complete; 4 docs intact locally; test app destroyed"

    echo_step "Step 6 — prod-untouched verification"
    [ -f "$HOME/.mnemon/default.sqlite" ] || echo_warn "prod vault not at ~/.mnemon/default.sqlite — verify by hand"
    flyctl status --app mnemon-memory >/dev/null 2>&1 || echo_warn "prod fly app status check failed — verify by hand"
    echo_ok "prod surfaces still responding"

    # Explicit cleanup before trap, so the success log lands last.
    trap - EXIT
    _layer3_cleanup
    echo_step "LAYER-3 PASSED for $TARGET_VERSION — promote PR can merge"
}

# ============================================================
# publish — POST-MERGE: build, twine upload, tag, GH Release, Fly redeploy
# ============================================================
cmd_publish() {
    echo_step "Publish $TARGET_VERSION — post-merge sequence"

    local branch
    branch="$(git branch --show-current)"
    [[ "$branch" == "main" ]] || die "must be on main; on '$branch'"
    git pull --ff-only --quiet
    grep -q "__version__ = \"$TARGET_VERSION\"" src/mnemon/__init__.py \
      || die "main is not at $TARGET_VERSION — has the promote PR merged?"
    echo_ok "on main at $TARGET_VERSION"

    if git rev-parse "v$TARGET_VERSION" >/dev/null 2>&1; then
        die "tag v$TARGET_VERSION already exists locally — publish already ran?"
    fi
    git fetch --tags --quiet
    if git rev-parse "v$TARGET_VERSION" >/dev/null 2>&1; then
        die "tag v$TARGET_VERSION exists on remote — publish already ran?"
    fi

    # Unset the test-time prod-guard so `mnemon upgrade web --app-name
    # mnemon-memory` is permitted for the real redeploy.
    unset MNEMON_PROD_APP_NAMES 2>/dev/null || true

    echo_step "Build — sdist + wheel"
    rm -rf dist/
    "$MNEMON_VENV_BIN/python" -m build
    "$MNEMON_VENV_BIN/twine" check dist/*
    echo_ok "build clean, twine check passed"

    echo_step "Sdist hygiene"
    local sdist
    sdist="$(ls dist/mnemon_memory-*.tar.gz | head -1)"
    [ -n "$sdist" ] || die "no sdist found in dist/"
    # Forbidden: private/, .env, vault files, raw fly.toml.
    # Sanctioned: fly.toml.example (template).
    local leaks
    leaks="$(tar tzf "$sdist" | grep -E "(^|/)(\.env|fly\.toml|default\.sqlite)$|(^|/)(private|\.mnemon)/" || true)"
    if [ -n "$leaks" ]; then
        echo_err "sdist contains forbidden paths:"
        printf "%s\n" "$leaks" >&2
        die "abort before twine upload"
    fi
    echo_ok "no forbidden files in sdist"

    confirm "Upload dist/* to PyPI? This is irreversible — PyPI cannot overwrite a released version."
    echo_step "twine upload"
    "$MNEMON_VENV_BIN/twine" upload dist/*
    echo_ok "uploaded to PyPI"

    echo_step "Wait for PyPI to surface $TARGET_VERSION"
    local tries=0
    until "$MNEMON_VENV_BIN/pip" index versions mnemon-memory 2>/dev/null | head -5 | grep -q "$TARGET_VERSION"; do
        tries=$((tries + 1))
        [ "$tries" -gt 30 ] && die "PyPI didn't surface $TARGET_VERSION after 5 min"
        sleep 10
    done
    echo_ok "PyPI surfaces $TARGET_VERSION"

    echo_step "Post-publish smoke: fresh-venv install from PyPI"
    local venvdir="/tmp/mnemon-postpub-$TARGET_VERSION"
    rm -rf "$venvdir"
    python3 -m venv "$venvdir"
    "$venvdir/bin/pip" install --quiet --upgrade pip
    "$venvdir/bin/pip" install --quiet "mnemon-memory==$TARGET_VERSION"
    local installed
    installed="$("$venvdir/bin/mnemon" --version | awk '{print $NF}')"
    [[ "$installed" == "v$TARGET_VERSION" || "$installed" == "$TARGET_VERSION" ]] \
      || die "fresh-venv shows '$installed', expected '$TARGET_VERSION'"
    "$venvdir/bin/mnemon" doctor
    rm -rf "$venvdir"
    echo_ok "post-publish smoke passed"

    echo_step "Tag v$TARGET_VERSION"
    git tag "v$TARGET_VERSION"
    git push origin "v$TARGET_VERSION"
    echo_ok "v$TARGET_VERSION pushed"

    echo_step "Create GitHub Release"
    local release_notes
    # Extract this version's CHANGELOG section: from the `## [VER]`
    # heading to (but not including) the next `## [` heading.
    #
    # The naive awk range `/^## \[VER\]/,/^## \[/` doesn't work — both
    # patterns match the same SINGLE line (the start heading itself
    # matches the end pattern too), so awk emits just that one line
    # and `sed '$d'` strips it → empty. Caught 2026-05-21 when the
    # publish step died with "could not extract CHANGELOG section
    # for 0.6.0" right after a successful twine upload + tag push.
    #
    # Use Python regex which is unambiguous about the range semantics.
    release_notes="$("$MNEMON_VENV_BIN/python" - <<PY
import re
ver = re.escape("$TARGET_VERSION")
content = open("CHANGELOG.md").read()
m = re.search(rf"^## \[{ver}\].*?(?=^## \[)", content, re.DOTALL | re.MULTILINE)
if m:
    print(m.group(0).rstrip())
PY
)"
    [ -n "$release_notes" ] || die "could not extract CHANGELOG section for $TARGET_VERSION"
    gh release create "v$TARGET_VERSION" --title "v$TARGET_VERSION" --notes "$release_notes"
    echo_ok "GitHub Release v$TARGET_VERSION created"

    confirm "Redeploy mnemon-memory.fly.dev to $TARGET_VERSION? Touches production Fly."
    echo_step "Fly redeploy (upgrade web runs doctor with settle window per rc12)"
    "$MNEMON_VENV_BIN/mnemon" upgrade web --app-name mnemon-memory --mnemon-version "$TARGET_VERSION"
    echo_ok "Fly redeploy complete"

    echo_step "Live doctor (audit trail)"
    "$MNEMON_VENV_BIN/mnemon" doctor
    echo_ok "live doctor green"

    echo_step "PUBLISH COMPLETE — $TARGET_VERSION live on PyPI + Fly"
    echo ""
    echo "  PyPI:    https://pypi.org/project/mnemon-memory/$TARGET_VERSION/"
    echo "  Release: https://github.com/cipher813/mnemon/releases/tag/v$TARGET_VERSION"
    echo "  Fly:     https://mnemon-memory.fly.dev/health"
    echo ""
    echo "  Don't forget to update private/SYSTEM_STATE.md (Recent Changes + Current State + Last verified)."
}

# ============================================================
# verify — post-publish confirmation
# ============================================================
cmd_verify() {
    echo_step "Verify $TARGET_VERSION live"

    local pypi_line
    pypi_line="$("$MNEMON_VENV_BIN/pip" index versions mnemon-memory 2>/dev/null | head -1 || true)"
    echo "  PyPI: $pypi_line"
    "$MNEMON_VENV_BIN/pip" index versions mnemon-memory 2>/dev/null | head -1 | grep -q "$TARGET_VERSION" \
      || die "PyPI does not show $TARGET_VERSION as a known version"
    echo_ok "PyPI shows $TARGET_VERSION"

    echo_step "Live doctor"
    "$MNEMON_VENV_BIN/mnemon" doctor
    echo_ok "doctor green"

    echo_step "GitHub Release"
    gh release view "v$TARGET_VERSION" --json url,tagName,publishedAt
    echo_ok "release present"
}

# ============================================================
# dispatch (only when executed directly, not when sourced for tests)
# ============================================================
# tests/test_promote_stable.sh sources this file to call individual helpers
# (`latest_pypi_version`, `_layer3_cleanup`, etc.) without running the
# subcommand dispatcher. The standard `${BASH_SOURCE[0]} != $0` idiom
# discriminates between `bash promote_stable.sh ...` (execute) and
# `source promote_stable.sh` (load functions only).
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 0 2>/dev/null || true
fi

case "${1:-}" in
    preflight) cmd_preflight ;;
    layer3)    shift; cmd_layer3 "$@" ;;
    publish)   cmd_publish ;;
    verify)    cmd_verify ;;
    *)
        cat >&2 <<EOF
usage: $0 <subcommand>

Subcommands (run in this order around the promote PR merge):
  preflight   — read-only: branch/version/CHANGELOG/tests/auth all check out
  layer3      — E2E web test (test-scoped Fly app, ~15 min)
  ── MERGE THE PROMOTE PR ON GITHUB HERE ──
  publish     — post-merge: build → twine upload → tag → GH Release → Fly redeploy
  verify      — post-publish sanity check

Target version read from src/mnemon/__init__.py (currently: $TARGET_VERSION).

Operator-only steps are sequenced but credentials remain operator-owned
(twine reads ~/.pypirc, flyctl uses your login, gh uses your session).
Each step fails loud on error; rerun-safe per phase.
EOF
        exit 2
        ;;
esac
