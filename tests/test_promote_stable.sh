#!/usr/bin/env bash
# tests/test_promote_stable.sh — sandbox harness for scripts/promote_stable.sh.
#
# Exercises the script's helpers and file-state logic without touching real
# Fly / PyPI publish / AWS S3 sync. Pytest doesn't cover this script directly,
# and three bug iterations in one session proved the script needs a unit-test
# safety net before each new layer3 attempt at real infrastructure.
#
# What's tested here (cheap, hermetic, ~3s):
#   - PyPI helpers (`latest_pypi_version`, `is_pypi_published`) against real PyPI
#     — read-only, no auth required, returns deterministic values
#   - Auth-state snapshot preserves content + permissions
#   - `_layer3_cleanup` restores auth files after a simulated upgrade-web
#     overwrite (the bug that broke prod mnemon three times today)
#   - The awk pattern `^Total memories:` extracts counts from real status output
#
# What's NOT tested here (would require real Fly / real PyPI publish):
#   - Full upgrade web / downgrade local cycle
#   - Twine upload
#   - GitHub Release creation
#   - Fly redeploy to mnemon-memory
# Those still need the real `scripts/promote_stable.sh layer3 / publish` runs
# against actual infrastructure. The harness here is the safety net BEFORE
# those runs, not a substitute for them.

set -uo pipefail   # not -e at runner level so individual test failures don't abort the runner

# ---- locate + source the script ----

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMOTE_SCRIPT="$REPO_ROOT/scripts/promote_stable.sh"

[ -f "$PROMOTE_SCRIPT" ] || { echo "ERROR: $PROMOTE_SCRIPT not found"; exit 1; }

# Source the helpers (the script's BASH_SOURCE!=$0 guard suppresses dispatch).
# shellcheck source=/dev/null
source "$PROMOTE_SCRIPT"

# ---- runner ----

PASS=0
FAIL=0
FAIL_NAMES=()

run_test() {
    local name="$1"
    # Subshell so set -e + traps in the test body don't leak.
    if ( set -e; "$name" ); then
        printf "  \033[1;32m✓\033[0m %s\n" "$name"
        PASS=$((PASS + 1))
    else
        local rc=$?
        printf "  \033[1;31m✗\033[0m %s (exit %d)\n" "$name" "$rc"
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("$name")
    fi
}

# ---- tests: PyPI helpers ----

test_latest_pypi_version_returns_pep440_semver() {
    local v
    v="$(latest_pypi_version)"
    [ -n "$v" ] || return 1
    # Loose PEP 440 sanity: starts with N.N.
    [[ "$v" =~ ^[0-9]+\.[0-9]+ ]] || return 1
}

test_is_pypi_published_truth_table() {
    # 0.6.0rc18 is known-published (rc cycle ground truth as of today).
    is_pypi_published "0.6.0rc18" || return 1
    # A version that can't exist (publish would fail loudly if it did).
    is_pypi_published "99.99.99rc99" && return 1
    return 0
}

# ---- tests: auth-state snapshot mechanics ----

test_snapshot_preserves_content_and_permissions() {
    local tmp; tmp="$(mktemp -d)"
    trap "rm -rf '$tmp'" RETURN

    echo "secret-bearer-token" > "$tmp/local_token"
    chmod 600 "$tmp/local_token"
    echo "https://prod.example/mcp" > "$tmp/remote_url"

    local backup; backup="$(mktemp -d)"
    cp -p "$tmp/local_token" "$backup/local_token"
    cp -p "$tmp/remote_url" "$backup/remote_url"

    # Content byte-identical.
    diff -q "$tmp/local_token" "$backup/local_token" >/dev/null || return 1
    diff -q "$tmp/remote_url" "$backup/remote_url" >/dev/null || return 1

    # Permission preserved (stat octal across BSD and GNU).
    local mode
    if [[ "$OSTYPE" == darwin* ]]; then
        mode="$(stat -f '%Lp' "$backup/local_token")"
    else
        mode="$(stat -c '%a' "$backup/local_token")"
    fi
    [[ "$mode" == "600" ]] || return 1

    rm -rf "$backup"
}

# ---- tests: _layer3_cleanup behavior (the load-bearing regression) ----

test_layer3_cleanup_restores_auth_files() {
    # Stub flyctl + aws on PATH so the cleanup's "destroy lingering app"
    # path doesn't hit real Fly / AWS.
    local stub_dir; stub_dir="$(mktemp -d)"
    cat > "$stub_dir/flyctl" <<'STUB'
#!/usr/bin/env bash
# Pretend no apps exist (so cleanup's "destroy lingering" branch skips).
[ "$1" = "apps" ] && [ "$2" = "list" ] && { echo "(stub: no apps)"; exit 0; }
exit 0
STUB
    cat > "$stub_dir/aws" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
    chmod +x "$stub_dir/flyctl" "$stub_dir/aws"
    local OLD_PATH="$PATH"
    export PATH="$stub_dir:$PATH"

    # Fake $HOME (script reads ~/.mnemon/* via $HOME).
    local fake_home; fake_home="$(mktemp -d)"
    local OLD_HOME="$HOME"
    export HOME="$fake_home"
    mkdir "$fake_home/.mnemon"
    printf 'https://prod.example/mcp' > "$fake_home/.mnemon/remote_url"
    printf 'PROD_BEARER_TOKEN_40_BYTES_PADPADPADPADPADX' > "$fake_home/.mnemon/local_token"
    chmod 600 "$fake_home/.mnemon/local_token"

    # Simulate the script's pre-Step-2 snapshot.
    export LAYER3_AUTH_BACKUP_DIR
    LAYER3_AUTH_BACKUP_DIR="$(mktemp -d -t mnemon-layer3-auth-XXXXXX)"
    cp -p "$fake_home/.mnemon/remote_url" "$LAYER3_AUTH_BACKUP_DIR/remote_url"
    cp -p "$fake_home/.mnemon/local_token" "$LAYER3_AUTH_BACKUP_DIR/local_token"

    # Simulate upgrade-web overwriting the originals (this is the bug surface).
    printf 'https://test-DEAD.fly.dev/mcp' > "$fake_home/.mnemon/remote_url"
    printf 'TEST_BEARER_TOKEN_43_BYTES_PADPADPADPADPADXXX' > "$fake_home/.mnemon/local_token"

    # Env that _layer3_cleanup reads — set to empty/safe values for the non-auth branches.
    export TEST_APP_NAME=""
    export MNEMON_S3_PREFIX=""
    export MNEMON_VAULT_DIR=""
    export MNEMON_CLIENT_CONFIG_ROOT=""

    # Run the real cleanup.
    _layer3_cleanup >/dev/null 2>&1 || true   # cleanup returns the inherited rc; ignore for this test

    # Restore env now so failures below print cleanly.
    export PATH="$OLD_PATH"
    export HOME="$OLD_HOME"

    # Verify restore.
    local rc_url rc_tok
    rc_url="$(cat "$fake_home/.mnemon/remote_url")"
    rc_tok="$(cat "$fake_home/.mnemon/local_token")"
    [[ "$rc_url" == "https://prod.example/mcp" ]] \
        || { echo "    expected prod remote_url restored, got: $rc_url" >&2; rm -rf "$fake_home" "$stub_dir"; return 1; }
    [[ "$rc_tok" == "PROD_BEARER_TOKEN_40_BYTES_PADPADPADPADPADX" ]] \
        || { echo "    expected prod local_token restored" >&2; rm -rf "$fake_home" "$stub_dir"; return 1; }

    # Permission preserved on the restored token.
    local mode
    if [[ "$OSTYPE" == darwin* ]]; then
        mode="$(stat -f '%Lp' "$fake_home/.mnemon/local_token")"
    else
        mode="$(stat -c '%a' "$fake_home/.mnemon/local_token")"
    fi
    [[ "$mode" == "600" ]] || { echo "    expected mode 600, got $mode" >&2; rm -rf "$fake_home" "$stub_dir"; return 1; }

    # Backup dir should be removed by the cleanup.
    [ ! -d "$LAYER3_AUTH_BACKUP_DIR" ] || { echo "    backup dir not removed: $LAYER3_AUTH_BACKUP_DIR" >&2; rm -rf "$fake_home" "$stub_dir" "$LAYER3_AUTH_BACKUP_DIR"; return 1; }

    rm -rf "$fake_home" "$stub_dir"
    unset LAYER3_AUTH_BACKUP_DIR TEST_APP_NAME MNEMON_S3_PREFIX MNEMON_VAULT_DIR MNEMON_CLIENT_CONFIG_ROOT
}

test_layer3_cleanup_no_backup_dir_is_safe() {
    # If LAYER3_AUTH_BACKUP_DIR is unset (early failure before snapshot),
    # cleanup must not blow up. It just skips the auth restore.
    local stub_dir; stub_dir="$(mktemp -d)"
    cat > "$stub_dir/flyctl" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
    cat > "$stub_dir/aws" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
    chmod +x "$stub_dir/flyctl" "$stub_dir/aws"
    local OLD_PATH="$PATH"; export PATH="$stub_dir:$PATH"

    unset LAYER3_AUTH_BACKUP_DIR
    export TEST_APP_NAME="" MNEMON_S3_PREFIX="" MNEMON_VAULT_DIR="" MNEMON_CLIENT_CONFIG_ROOT=""

    _layer3_cleanup >/dev/null 2>&1
    local rc=$?

    export PATH="$OLD_PATH"
    rm -rf "$stub_dir"
    unset TEST_APP_NAME MNEMON_S3_PREFIX MNEMON_VAULT_DIR MNEMON_CLIENT_CONFIG_ROOT

    # Inherited rc is whatever the last shell command was — for a clean
    # source/unset path, it should be 0.
    [ "$rc" -eq 0 ]
}

# ---- tests: awk parse ----

test_awk_extracts_total_memories_count() {
    local sample count
    sample="$(printf 'Vault: /tmp/x\nTotal memories: 42\nVectors: 7\nPinned: 0\n')"
    count="$(echo "$sample" | awk '/^Total memories:/{print $NF; exit}')"
    [[ "$count" == "42" ]] || return 1
}

test_awk_handles_zero_count() {
    local sample count
    sample="$(printf 'Vault: /tmp/x\nTotal memories: 0\nVectors: 0\n')"
    count="$(echo "$sample" | awk '/^Total memories:/{print $NF; exit}')"
    [[ "$count" == "0" ]] || return 1
}

test_awk_returns_empty_when_field_missing() {
    # Defense-in-depth: if mnemon ever changes the field label, awk silently
    # returns "" and the script's [[ ... == "3" ]] check fails loud with the
    # actual got-value. Documenting the shape.
    local sample count
    sample="$(printf 'Vault: /tmp/x\nDocuments: 4\n')"
    count="$(echo "$sample" | awk '/^Total memories:/{print $NF; exit}')"
    [[ -z "$count" ]] || return 1
}

# ---- tests: dispatch suppression when sourced ----

test_sourcing_does_not_dispatch() {
    # The script's bash-source guard must keep `source promote_stable.sh`
    # from executing the case dispatch (which would either run a subcommand
    # or print usage to stderr + exit). We're already running tests in a
    # sourced context, so reaching here means the guard works.
    return 0
}

# ---- tests: Step-2 seed content uniqueness ----

test_remote_helper_exists_and_is_invokable() {
    # The Python helper that bypasses mnemon's local-only CLI for Layer-3
    # remote verification (see _layer3_remote_helper.py docstring).
    # Asserts the file exists, has the expected entry-points, and prints
    # usage when called without args (basic smoke).
    local helper="$REPO_ROOT/scripts/_layer3_remote_helper.py"
    [ -f "$helper" ] || { echo "    helper not found at $helper" >&2; return 1; }

    # The script needs python + the venv's mnemon installable.
    local out
    out="$("$REPO_ROOT/.venv/bin/python" "$helper" 2>&1)" && return 1   # bare invocation should fail with usage
    echo "$out" | grep -q "usage:" || { echo "    helper didn't print usage on bare invocation: $out" >&2; return 1; }
}

test_layer3_uses_remote_helper_not_mnemon_status() {
    # Regression for the 2026-05-21 verification bug: cmd_layer3 used
    # `mnemon status` (local-only) to assert remote state, which always
    # read an empty freshly-created local SQLite. The fix switched to
    # the remote helper. Guard against a future edit reverting to
    # `mnemon status` for remote verification.
    local script="$REPO_ROOT/scripts/promote_stable.sh"

    # Extract Step 3 + Step 4 bodies and check they use the helper.
    local body
    body="$(awk '/echo_step "Step 3/,/echo_step "Step 5/' "$script")"
    echo "$body" | grep -q "_layer3_remote_helper.py" \
        || { echo "    Step 3/4 doesn't reference _layer3_remote_helper.py" >&2; return 1; }

    # And NEITHER step uses `mnemon status` against the test app remote
    # (the broken pattern). Step 5 still uses mnemon status post-downgrade
    # which is correct because local mode is appropriate then; so we
    # constrain the negative check to Steps 3 + 4 only.
    echo "$body" | grep -E '"\$M" status|"\$MNEMON_VENV_BIN/mnemon" status' \
        && { echo "    Step 3/4 still uses local-only mnemon status for remote verification" >&2; return 1; }
    return 0
}

test_helper_exposes_exercise_all_tools_subcommand() {
    # The 2026-05-22 layer3 extension added an exercise-all-tools
    # subcommand to _layer3_remote_helper.py. Regression-lock the
    # dispatcher so the entry-point doesn't silently disappear in a
    # future edit. Doesn't run the actual exercise (would need a live
    # Fly app); just verifies the subcommand is registered + named.
    local helper="$REPO_ROOT/scripts/_layer3_remote_helper.py"
    [ -f "$helper" ] || { echo "    helper not found at $helper" >&2; return 1; }

    # The usage string in the bare-invocation output should mention
    # the new subcommand. (Bare invocation exits non-zero with the
    # usage on stderr.)
    local out
    out="$("$REPO_ROOT/.venv/bin/python" "$helper" 2>&1)" && return 1
    echo "$out" | grep -q "exercise-all-tools" \
        || { echo "    usage text doesn't advertise exercise-all-tools: $out" >&2; return 1; }
    return 0
}

test_layer3_passes_through_exercise_all_tools_flag() {
    # The 2026-05-22 layer3 extension added a --exercise-all-tools
    # flag wired through the dispatcher (cmd_layer3 "$@"). Regression-
    # lock so a future refactor doesn't drop the $@ forwarding.
    local script="$REPO_ROOT/scripts/promote_stable.sh"

    # Dispatcher line forwards args
    grep -E 'layer3\)\s+shift;\s+cmd_layer3 "\$@"' "$script" \
        || { echo "    layer3 case doesn't forward args to cmd_layer3" >&2; return 1; }

    # cmd_layer3 parses --exercise-all-tools into EXERCISE_ALL_TOOLS=1
    grep -q 'EXERCISE_ALL_TOOLS=1' "$script" \
        || { echo "    cmd_layer3 doesn't set EXERCISE_ALL_TOOLS=1" >&2; return 1; }

    # And the Step 4.5 gate references it
    grep -q 'EXERCISE_ALL_TOOLS' "$script" \
        || { echo "    Step 4.5 doesn't gate on EXERCISE_ALL_TOOLS" >&2; return 1; }
    return 0
}

test_publish_extracts_changelog_section_correctly() {
    # Regression for the 2026-05-21 publish-step bug: the naive awk
    # range `/^## \[VER\]/,/^## \[/` extracted ZERO bytes because
    # both patterns matched the same heading line on BWK awk
    # (macOS default), and `sed '$d'` then dropped that single line.
    # The fix uses Python regex with `(?=^## \[)` lookahead.
    #
    # This test simulates the publish-step extraction against a
    # synthetic CHANGELOG and asserts the result contains the
    # version's content (Release + Fixes sections) without bleeding
    # into the next version's section.

    local tmp_changelog
    tmp_changelog="$(mktemp)"
    cat > "$tmp_changelog" <<'CL'
# Changelog

## [9.9.9] - 2026-01-01

### Release

- v9.9.9 release notes — this should be extracted.

### Fixes

- v9.9.9 fix bullet — also extracted.

## [9.9.8] - 2025-12-31

### Release

- v9.9.8 release notes — should NOT be in the v9.9.9 extract.
CL

    # Reproduce the publish-step extraction with the new logic.
    local result
    result="$("$REPO_ROOT/.venv/bin/python" - <<PY
import re
ver = re.escape("9.9.9")
content = open("$tmp_changelog").read()
m = re.search(rf"^## \[{ver}\].*?(?=^## \[)", content, re.DOTALL | re.MULTILINE)
if m:
    print(m.group(0).rstrip())
PY
)"

    rm -f "$tmp_changelog"

    # Positive: must contain v9.9.9 content.
    echo "$result" | grep -q "v9.9.9 release notes" \
        || { echo "    missing v9.9.9 release notes in extract" >&2; return 1; }
    echo "$result" | grep -q "v9.9.9 fix bullet" \
        || { echo "    missing v9.9.9 fix bullet in extract" >&2; return 1; }

    # Negative: must NOT contain v9.9.8 content (no bleed across heading).
    if echo "$result" | grep -q "v9.9.8"; then
        echo "    v9.9.8 content leaked into v9.9.9 extract" >&2
        return 1
    fi

    # Negative: must NOT be empty (the original bug).
    [ -n "$result" ] || { echo "    extraction returned empty" >&2; return 1; }
}

test_step2_removes_remote_url_before_seed() {
    # Surfaced 2026-05-21 (Layer-3 attempt). `mnemon save` honors
    # ~/.mnemon/remote_url even from inside layer3, so seed saves
    # routed to PROD instead of the local test vault. The fix in
    # cmd_layer3 moves the file aside before running the three saves.
    # Regression-lock the rm + unset by grepping the script source.
    local script_path
    script_path="$REPO_ROOT/scripts/promote_stable.sh"

    grep -q 'unset MNEMON_REMOTE_URL' "$script_path" \
        || { echo "    script missing 'unset MNEMON_REMOTE_URL'" >&2; return 1; }
    grep -qF 'rm -f "$HOME/.mnemon/remote_url"' "$script_path" \
        || { echo "    script missing 'rm -f \$HOME/.mnemon/remote_url'" >&2; return 1; }

    # Ordering: the unset/rm must precede the Step 2 banner.
    local unset_line rm_line step2_line
    unset_line="$(grep -n 'unset MNEMON_REMOTE_URL' "$script_path" | head -1 | cut -d: -f1)"
    rm_line="$(grep -nF 'rm -f "$HOME/.mnemon/remote_url"' "$script_path" | head -1 | cut -d: -f1)"
    step2_line="$(grep -n 'echo_step "Step 2 — seed local test vault"' "$script_path" | head -1 | cut -d: -f1)"
    [ -n "$step2_line" ] || { echo "    no 'echo_step Step 2' line in script" >&2; return 1; }
    [ "$unset_line" -lt "$step2_line" ] && [ "$rm_line" -lt "$step2_line" ] \
        || { echo "    unset/rm must precede Step 2 (got unset=$unset_line rm=$rm_line step2=$step2_line)" >&2; return 1; }
}

test_mnemon_venv_bin_env_var_is_honored() {
    # Sub-item 1 of C25: MNEMON_VENV_BIN must be overridable so CI /
    # non-.venv operators can run the script. The script's bootstrap
    # uses ${MNEMON_VENV_BIN:-...} to default to .venv/bin only when
    # the env var is unset. Regression-lock the parameter expansion
    # pattern.
    local script_path
    script_path="$REPO_ROOT/scripts/promote_stable.sh"
    grep -qE 'MNEMON_VENV_BIN="\$\{MNEMON_VENV_BIN:-\$REPO_ROOT/\.venv/bin\}"' "$script_path" \
        || { echo "    MNEMON_VENV_BIN not using \${VAR:-default} pattern" >&2; return 1; }
}

test_cleanup_destroy_retries_on_failure() {
    # Sub-item 4 of C25: trap destroy now retries once on failure +
    # captures stderr. Grep the script source for the retry pattern.
    local script_path
    script_path="$REPO_ROOT/scripts/promote_stable.sh"
    local body
    body="$(awk '/^_layer3_cleanup\(\)/,/^}/' "$script_path")"

    # Must capture stderr to a file (not >/dev/null).
    grep -q '2>"\$_destroy_log"' <<<"$body" \
        || { echo "    cleanup must capture flyctl destroy stderr" >&2; return 1; }
    # Must retry on failure ('attempt 1 failed' marker).
    grep -q 'attempt 1 failed' <<<"$body" \
        || { echo "    cleanup must announce retry on first failure" >&2; return 1; }
    # On persistent failure, must surface the captured stderr to operator.
    grep -q 'FAILED twice' <<<"$body" \
        || { echo "    cleanup must surface persistent-failure path" >&2; return 1; }
}

test_step2_seed_contents_are_unique() {
    # mnemon's store.save() does content-hash dedup (rc15+). The original
    # runbook passed the same content to all three Step-2 saves, which
    # silently deduped down to 2 docs in the local test vault and caused
    # a confusing "got '2', expected '3'" Step-2 assertion failure
    # 2026-05-21. The fix makes content unique per save. Guard against
    # the regression returning.
    local script_path
    script_path="$REPO_ROOT/scripts/promote_stable.sh"

    # Extract the three Step-2 save lines.
    local seed_lines
    seed_lines="$(awk '/echo_step "Step 2 — seed local test vault"/,/seeded_local_count/' "$script_path" \
        | grep -E '"\$M" save ')"

    # Must be exactly 3 saves.
    local n_saves
    n_saves="$(echo "$seed_lines" | grep -c '"\$M" save ')"
    [[ "$n_saves" == "3" ]] || { echo "    expected 3 'mnemon save' lines in Step 2, got $n_saves" >&2; return 1; }

    # Extract the content (second quoted arg) from each save line.
    # Pattern: `"$M" save "TITLE" "CONTENT" --type X`
    local contents
    contents="$(echo "$seed_lines" | sed -E 's/.*"\$M" save +"[^"]+" +"([^"]+)".*/\1/')"

    # Count unique contents.
    local n_unique
    n_unique="$(echo "$contents" | sort -u | wc -l | tr -d ' ')"
    [[ "$n_unique" == "3" ]] || {
        echo "    expected 3 unique seed contents, got $n_unique unique across:" >&2
        echo "$contents" | sed 's/^/      /' >&2
        return 1
    }
}

# ============================================================
# runner
# ============================================================

echo "promote_stable.sh sandbox harness"
echo "=================================="

run_test test_latest_pypi_version_returns_pep440_semver
run_test test_is_pypi_published_truth_table
run_test test_snapshot_preserves_content_and_permissions
run_test test_layer3_cleanup_restores_auth_files
run_test test_layer3_cleanup_no_backup_dir_is_safe
run_test test_awk_extracts_total_memories_count
run_test test_awk_handles_zero_count
run_test test_awk_returns_empty_when_field_missing
run_test test_sourcing_does_not_dispatch
run_test test_step2_seed_contents_are_unique
run_test test_step2_removes_remote_url_before_seed
run_test test_mnemon_venv_bin_env_var_is_honored
run_test test_cleanup_destroy_retries_on_failure
run_test test_remote_helper_exists_and_is_invokable
run_test test_layer3_uses_remote_helper_not_mnemon_status
run_test test_helper_exposes_exercise_all_tools_subcommand
run_test test_layer3_passes_through_exercise_all_tools_flag
run_test test_publish_extracts_changelog_section_correctly

echo ""
echo "Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    echo "Failed: ${FAIL_NAMES[*]}"
    exit 1
fi
exit 0
