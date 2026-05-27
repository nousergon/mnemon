#!/usr/bin/env bash
# scripts/phase_a_resoak.sh — drive the Phase A capture-attention re-soak.
#
# The 2026-05-27 morning soak failed (boost-rate 232/325 = 0.714 vs 0.25
# ceiling). The hook-source provenance filter (PR #165) closes the
# defect. This script automates the operator sequence to ship the
# filter to Fly + activate the re-soak + monitor + close.
#
# Subcommands:
#   preflight   — read-only: confirm Fly version matches local
#                 __version__ + hook-source filter is in the deployed
#                 contradiction.py + secret is currently OFF.
#   activate    — confirm + `flyctl secrets set
#                 MNEMON_CAPTURE_ATTENTION_ENABLED=true` + verify the
#                 flip via `mnemon attention-status` on Fly.
#   status      — print current Fly `mnemon attention-status` + soak
#                 elapsed-days from the activation timestamp file.
#   close       — read attention-status; if boost-rate ≤ ceiling +
#                 soak elapsed ≥ MIN_SOAK_DAYS, surface pass + the
#                 follow-up PR template to flip CAPTURE_ATTENTION_ENABLED
#                 default-on.
#   deactivate  — emergency off-switch: confirm + `flyctl secrets unset
#                 MNEMON_CAPTURE_ATTENTION_ENABLED`. Use if the live
#                 boost-rate spikes mid-soak.
#
# Operator workflow (assumes the rc has been merged + twine-uploaded +
# `mnemon upgrade web` deployed against Fly):
#
#   scripts/phase_a_resoak.sh preflight   # confirm ready
#   scripts/phase_a_resoak.sh activate    # start the soak clock
#   # ... wait ≥ MIN_SOAK_DAYS, periodically:
#   scripts/phase_a_resoak.sh status
#   # at soak end:
#   scripts/phase_a_resoak.sh close       # surface pass/fail + next-step PR
#
# `twine upload` + `mnemon upgrade web` are NOT driven from here —
# those are the operator's responsibility (credentials + the existing
# promote_stable.sh / mnemon upgrade web entry points).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Operator-overridable per the C25 pattern.
MNEMON_VENV_BIN="${MNEMON_VENV_BIN:-$REPO_ROOT/.venv/bin}"
[ -x "$MNEMON_VENV_BIN/mnemon" ] || { echo "ERROR: mnemon CLI not found at $MNEMON_VENV_BIN/mnemon" >&2; exit 1; }

APP_NAME="${MNEMON_FLY_APP_NAME:-mnemon-memory}"
STATE_FILE="${MNEMON_RESOAK_STATE_FILE:-$HOME/.mnemon/phase_a_resoak.state}"
MIN_SOAK_DAYS="${MNEMON_RESOAK_MIN_DAYS:-7}"
BOOST_RATE_CEILING="${MNEMON_RESOAK_BOOST_CEILING:-0.25}"

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

# Read local version (the candidate the operator just deployed).
LOCAL_VERSION="$(awk -F'"' '/^__version__ = /{print $2; exit}' src/mnemon/__init__.py)"
[ -n "$LOCAL_VERSION" ] || die "could not read __version__ from src/mnemon/__init__.py"

# ---- helpers ----

fly_version() {
    # Run `mnemon --version` inside the Fly machine. Returns the bare
    # version string (strips "mnemon v" prefix) or empty on failure.
    flyctl ssh console -a "$APP_NAME" -C 'mnemon --version' 2>/dev/null \
        | grep -oE '[0-9]+\.[0-9]+\.[0-9]+[a-z0-9]*' | head -1
}

fly_attention_status() {
    flyctl ssh console -a "$APP_NAME" -C 'mnemon attention-status' 2>/dev/null
}

fly_secret_present() {
    # True iff MNEMON_CAPTURE_ATTENTION_ENABLED is set on the Fly app.
    flyctl secrets list -a "$APP_NAME" 2>/dev/null \
        | awk '{print $1}' | grep -qx "MNEMON_CAPTURE_ATTENTION_ENABLED"
}

extract_flag_enabled() {
    # Pull "Flag enabled: True|False" from attention-status output.
    grep -oE 'Flag enabled\s*:\s*(True|False)' \
        | awk '{print $NF}' | head -1
}

extract_boost_rate() {
    # Pull the decimal boost rate from "Boost-rate 7d: N / N = 0.XYZ".
    grep -oE 'Boost-rate 7d\s*:.*=\s*[0-9]+\.[0-9]+' \
        | grep -oE '[0-9]+\.[0-9]+' | tail -1
}

float_le() {
    # Returns 0 if $1 ≤ $2.
    "$MNEMON_VENV_BIN/python" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$1" "$2"
}

# ---- subcommands ----

cmd_preflight() {
    echo_step "Phase A re-soak preflight"

    command -v flyctl >/dev/null || die "flyctl not on PATH"
    flyctl auth whoami >/dev/null 2>&1 || die "flyctl not logged in"
    echo_ok "flyctl authenticated"

    local fv
    fv="$(fly_version)"
    [ -n "$fv" ] || die "could not read mnemon version from Fly app $APP_NAME"
    echo_ok "Fly is running mnemon-memory==$fv"

    if [[ "$fv" != "$LOCAL_VERSION" ]]; then
        echo_warn "Fly version ($fv) differs from local __version__ ($LOCAL_VERSION)"
        echo_warn "  If you intended to deploy $LOCAL_VERSION, run:"
        echo_warn "    $MNEMON_VENV_BIN/mnemon upgrade web --app-name $APP_NAME --mnemon-version $LOCAL_VERSION"
        echo_warn "  Otherwise this is fine — re-soak runs on whatever version is live."
    else
        echo_ok "Fly version matches local __version__"
    fi

    # The hook-source filter shipped in PR #165 (rc5+). If Fly is on rc4
    # or older, the filter isn't live and re-soak would just reproduce
    # the failed soak.
    if [[ "$fv" < "0.7.0rc5" ]]; then
        die "Fly is on $fv — PR #165 hook-source filter requires rc5+. Deploy a newer rc before activating re-soak."
    fi
    echo_ok "hook-source filter is live (Fly ≥ rc5)"

    if fly_secret_present; then
        echo_warn "MNEMON_CAPTURE_ATTENTION_ENABLED is already set on Fly"
        echo_warn "  Re-soak may already be active. Run \`$0 status\` to check."
    else
        echo_ok "MNEMON_CAPTURE_ATTENTION_ENABLED is NOT set (clean pre-soak state)"
    fi

    echo_step "Preflight PASSED — ready to activate"
}

cmd_activate() {
    echo_step "Activate Phase A capture-attention re-soak"

    cmd_preflight

    confirm "Set MNEMON_CAPTURE_ATTENTION_ENABLED=true on Fly app $APP_NAME?"
    flyctl secrets set MNEMON_CAPTURE_ATTENTION_ENABLED=true -a "$APP_NAME"
    echo_ok "secret set; Fly is rolling the deploy"

    # Record activation time for elapsed-day tracking.
    mkdir -p "$(dirname "$STATE_FILE")"
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATE_FILE"
    echo_ok "activation timestamp recorded at $STATE_FILE"

    echo_step "Verifying flag flip via mnemon attention-status"
    local status_out flag
    status_out="$(fly_attention_status)"
    flag="$(echo "$status_out" | extract_flag_enabled)"
    if [[ "$flag" == "True" ]]; then
        echo_ok "Flag enabled: True (re-soak clock is running)"
    else
        echo_err "Flag is not True (got: ${flag:-<empty>}). Check fly machine status."
        echo "  Raw status output:" >&2
        echo "$status_out" | sed 's/^/    /' >&2
        exit 1
    fi
    echo "$status_out" | sed 's/^/  /'
    echo
    echo_ok "Phase A re-soak ACTIVATED. Run \`$0 status\` for periodic checks."
    echo "  Soak target end: ≥$MIN_SOAK_DAYS days from now."
}

cmd_status() {
    echo_step "Phase A re-soak status"

    local status_out flag rate
    status_out="$(fly_attention_status)" || die "could not read attention-status from Fly"
    flag="$(echo "$status_out" | extract_flag_enabled)"
    rate="$(echo "$status_out" | extract_boost_rate)"

    echo "$status_out" | sed 's/^/  /'
    echo

    if [[ "$flag" != "True" ]]; then
        echo_warn "Flag is not True (got: ${flag:-<empty>}). Re-soak isn't running."
        return 0
    fi

    if [ ! -f "$STATE_FILE" ]; then
        echo_warn "no activation timestamp at $STATE_FILE; can't compute elapsed"
        echo_warn "  (was the soak started via this script? if not, just touch the file with the activation date)"
    else
        local started elapsed
        started="$(cat "$STATE_FILE")"
        elapsed="$(
            "$MNEMON_VENV_BIN/python" -c "
import datetime as dt, sys
started = dt.datetime.fromisoformat(sys.argv[1].replace('Z', '+00:00'))
now = dt.datetime.now(dt.timezone.utc)
print(f'{(now - started).total_seconds() / 86400:.1f}')
" "$started"
        )"
        echo_ok "soak elapsed: $elapsed days (started $started)"
        echo "  min soak target: $MIN_SOAK_DAYS days"
    fi

    if [ -n "$rate" ]; then
        if float_le "$rate" "$BOOST_RATE_CEILING"; then
            echo_ok "boost-rate $rate ≤ ceiling $BOOST_RATE_CEILING (passing)"
        else
            echo_err "boost-rate $rate > ceiling $BOOST_RATE_CEILING (FAILING)"
            echo "  If this persists for >24h post-activation, the hook-source"
            echo "  filter may have a hole. Run \`$0 deactivate\` to halt."
        fi
    fi
}

cmd_close() {
    echo_step "Phase A re-soak close — checking pass criteria"

    local status_out flag rate
    status_out="$(fly_attention_status)" || die "could not read attention-status from Fly"
    flag="$(echo "$status_out" | extract_flag_enabled)"
    rate="$(echo "$status_out" | extract_boost_rate)"

    echo "$status_out" | sed 's/^/  /'
    echo

    [[ "$flag" == "True" ]] || die "flag is not True — re-soak wasn't running"
    [ -n "$rate" ] || die "could not extract boost-rate from status output"

    local elapsed_days="-1"
    if [ -f "$STATE_FILE" ]; then
        local started
        started="$(cat "$STATE_FILE")"
        elapsed_days="$(
            "$MNEMON_VENV_BIN/python" -c "
import datetime as dt, sys
started = dt.datetime.fromisoformat(sys.argv[1].replace('Z', '+00:00'))
now = dt.datetime.now(dt.timezone.utc)
print(f'{(now - started).total_seconds() / 86400:.1f}')
" "$started"
        )"
    fi

    local pass_rate=0 pass_time=0
    float_le "$rate" "$BOOST_RATE_CEILING" && pass_rate=1
    "$MNEMON_VENV_BIN/python" -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" \
        "$elapsed_days" "$MIN_SOAK_DAYS" && pass_time=1

    echo_step "Pass criteria"
    if [ "$pass_rate" = "1" ]; then
        echo_ok "boost-rate $rate ≤ ceiling $BOOST_RATE_CEILING"
    else
        echo_err "boost-rate $rate > ceiling $BOOST_RATE_CEILING"
    fi
    if [ "$pass_time" = "1" ]; then
        echo_ok "elapsed $elapsed_days days ≥ min $MIN_SOAK_DAYS days"
    else
        echo_err "elapsed $elapsed_days days < min $MIN_SOAK_DAYS days"
    fi

    if [ "$pass_rate" = "1" ] && [ "$pass_time" = "1" ]; then
        echo
        echo_ok "Phase A re-soak PASSED. Next step:"
        echo
        echo "  Open a follow-up PR:"
        echo "    git checkout main && git pull && git checkout -b feat/capture-attention-default-on"
        echo "    # Edit src/mnemon/config.py: CAPTURE_ATTENTION_ENABLED = False → True"
        echo "    # Update CHANGELOG.md with the re-soak pass note."
        echo "    git commit + push + gh pr create"
        echo
        echo "  Operator workflow still:"
        echo "    1. Merge the default-on PR"
        echo "    2. Bump rc (3-line pattern) + CHANGELOG"
        echo "    3. twine upload"
        echo "    4. mnemon upgrade web --mnemon-version <new-rc>"
        echo "    5. (Optionally) flyctl secrets unset MNEMON_CAPTURE_ATTENTION_ENABLED"
        echo "       once config default-on is the source of truth"
    else
        echo
        echo_warn "Re-soak NOT passing yet. Wait + retry, or investigate."
    fi
}

cmd_deactivate() {
    echo_step "Phase A re-soak DEACTIVATE — emergency off-switch"

    if ! fly_secret_present; then
        echo_warn "MNEMON_CAPTURE_ATTENTION_ENABLED is not set; nothing to deactivate"
        return 0
    fi

    confirm "Unset MNEMON_CAPTURE_ATTENTION_ENABLED on Fly app $APP_NAME?"
    flyctl secrets unset MNEMON_CAPTURE_ATTENTION_ENABLED -a "$APP_NAME"
    echo_ok "secret unset; Fly is rolling the deploy"

    if [ -f "$STATE_FILE" ]; then
        mv "$STATE_FILE" "$STATE_FILE.deactivated-$(date +%Y%m%d%H%M%S)"
        echo_ok "moved state file aside (preserves the run for analysis)"
    fi
}

usage() {
    sed -n '2,/^$/{s/^# \{0,1\}//;p;}' "${BASH_SOURCE[0]}"
}

main() {
    local sub="${1:-help}"
    shift || true
    case "$sub" in
        preflight)   cmd_preflight ;;
        activate)    cmd_activate ;;
        status)      cmd_status ;;
        close)       cmd_close ;;
        deactivate)  cmd_deactivate ;;
        help|-h|--help|"") usage ;;
        *)
            echo "unknown subcommand: $sub" >&2
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
