#!/usr/bin/env bash
#
# validate_cross_device.sh — live-validate mnemon's cross-device (web) path.
#
# Deploys a THROWAWAY Fly app from the current mnemon version, asserts the
# auto-provisioned OAuth Authorization Server serves correct RFC-8414
# metadata (the server-side half of the claude.ai / Desktop login path),
# and surfaces the generated passphrase + connector URL so you can do the
# one step no script can — adding the connector in claude.ai's browser UI.
#
# Safety (mirrors scripts/promote_stable.sh layer3):
#   * MNEMON_PROD_APP_NAMES guard refuses to touch the prod app.
#   * All state is isolated via env overrides (test vault dir, test S3
#     prefix, fake client-config root, unique app name).
#   * An EXIT trap restores ~/.mnemon/{remote_url,local_token,as_passphrase}
#     — which `mnemon upgrade web` overwrites and the env overrides do NOT
#     protect — even on Ctrl-C or error, then destroys the test app and
#     cleans the test S3 prefix + scratch dirs.
#
# Usage:
#   scripts/validate_cross_device.sh                 # full cycle (deploy → assert → pause → teardown)
#   scripts/validate_cross_device.sh --keep          # leave the app up for unhurried browser testing
#   scripts/validate_cross_device.sh --app-name NAME # override the throwaway app name
#   scripts/validate_cross_device.sh --version VER   # pin a specific mnemon-memory version (must be on PyPI)
#   scripts/validate_cross_device.sh --dry-run       # print the plan, no side effects
#   scripts/validate_cross_device.sh --help
#
set -euo pipefail

PROD_APP="mnemon-memory"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MNEMON_BIN="${MNEMON_BIN:-$REPO_ROOT/.venv/bin/mnemon}"
S3_BUCKET="${MNEMON_S3_BUCKET:-mnemon-memory}"
DRY_RUN=0
KEEP=0
APP_NAME=""
VERSION=""

c_ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
c_info() { printf '  %s\n' "$*"; }
c_warn() { printf '\033[33m⚠\033[0m %s\n' "$*"; }
c_err()  { printf '\033[31m✗\033[0m %s\n' "$*" >&2; }
die()    { c_err "$*"; exit 1; }

usage() { sed -n '2,30p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)  DRY_RUN=1 ;;
    --keep)     KEEP=1 ;;
    --app-name) APP_NAME="${2:-}"; shift ;;
    --version)  VERSION="${2:-}"; shift ;;
    --help|-h)  usage ;;
    *) die "unknown arg: $1 (try --help)" ;;
  esac
  shift
done

TEST_RUN_ID="$(date +%Y%m%d-%H%M%S)"
TEST_APP_NAME="${APP_NAME:-mnemon-test-$TEST_RUN_ID}"

# Resolve the version to deploy from the source of truth (single-sourced
# __version__) unless the caller pinned one. It MUST already be on PyPI —
# the Fly Docker build runs `pip install mnemon-memory[server]==<ver>`.
if [ -z "$VERSION" ]; then
  VERSION="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.]*' "$REPO_ROOT/src/mnemon/__init__.py" | head -1)"
fi

# Hard guard: never let the throwaway name collide with prod.
[ "$TEST_APP_NAME" = "$PROD_APP" ] && die "refusing to run against the prod app ($PROD_APP)"

if [ "$DRY_RUN" -eq 1 ]; then
  cat <<EOF
DRY RUN — no side effects. Would:
  1. preflight: flyctl auth whoami; aws sts get-caller-identity; '$MNEMON_BIN' present
  2. snapshot ~/.mnemon/{remote_url,local_token,as_passphrase} + install EXIT-trap restore
  3. isolate: test MNEMON_VAULT_DIR / MNEMON_S3_PREFIX / MNEMON_CLIENT_CONFIG_ROOT
     + MNEMON_PROD_APP_NAMES=$PROD_APP guard
  4. $MNEMON_BIN upgrade web --app-name $TEST_APP_NAME --mnemon-version $VERSION
  5. assert: 'mnemon doctor --fail-on-warn' → 'OAuth AS metadata' ✓
     (issuer=https://$TEST_APP_NAME.fly.dev)
  6. print the AS passphrase + connector URL + claude.ai browser steps
  7. $([ "$KEEP" -eq 1 ] && echo "leave the app running (--keep); print manual teardown" || echo "pause for Enter, then teardown")
  8. teardown (trap): $([ "$KEEP" -eq 1 ] && echo "skip destroy;" ) restore ~/.mnemon snapshot; $([ "$KEEP" -eq 0 ] && echo "destroy $TEST_APP_NAME; rm test S3 prefix;" ) rm scratch dirs
EOF
  exit 0
fi

# ── preflight ────────────────────────────────────────────────────────────────
[ -x "$MNEMON_BIN" ] || die "mnemon CLI not found/executable at $MNEMON_BIN (set MNEMON_BIN=...)"
command -v flyctl >/dev/null 2>&1 || die "flyctl not found"
command -v aws    >/dev/null 2>&1 || die "aws CLI not found"
flyctl auth whoami >/dev/null 2>&1 || die "flyctl not authenticated (flyctl auth login)"
aws sts get-caller-identity >/dev/null 2>&1 || die "AWS credentials not resolvable"
c_ok "preflight: flyctl + aws authenticated; mnemon at $MNEMON_BIN; deploying version $VERSION"

# ── snapshot prod pointers (env overrides do NOT cover these) ─────────────────
SNAPSHOT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mnemon-prod-snapshot.XXXXXX")"
for f in remote_url local_token as_passphrase; do
  [ -f "$HOME/.mnemon/$f" ] && cp -p "$HOME/.mnemon/$f" "$SNAPSHOT_DIR/$f"
done
c_ok "snapshotted prod ~/.mnemon pointers to $SNAPSHOT_DIR"

cleanup() {
  local rc=$?
  echo ""
  echo "── teardown ──────────────────────────────────────────────────────────"
  if [ "$KEEP" -eq 0 ]; then
    if flyctl apps destroy "$TEST_APP_NAME" -y >/dev/null 2>&1; then
      c_ok "destroyed test app $TEST_APP_NAME"
    else
      c_warn "could not destroy $TEST_APP_NAME — recover by hand: flyctl apps destroy $TEST_APP_NAME -y"
    fi
    if [ -n "${MNEMON_S3_PREFIX:-}" ]; then
      aws s3 rm "s3://$S3_BUCKET/$MNEMON_S3_PREFIX/" --recursive >/dev/null 2>&1 \
        || c_warn "s3 cleanup failed for $MNEMON_S3_PREFIX (remove by hand)"
    fi
  else
    c_warn "--keep: test app LEFT RUNNING at https://$TEST_APP_NAME.fly.dev"
    c_info "tear it down later: flyctl apps destroy $TEST_APP_NAME -y"
  fi
  # Auth restore is NOT best-effort — a miss leaves prod mnemon broken.
  local restored=0 had as_path="$HOME/.mnemon/as_passphrase"
  for f in remote_url local_token as_passphrase; do
    had=0; [ -f "$HOME/.mnemon/$f" ] && had=1
    if [ -f "$SNAPSHOT_DIR/$f" ]; then
      cp -p "$SNAPSHOT_DIR/$f" "$HOME/.mnemon/$f" && restored=$((restored + 1)) \
        || c_err "FAILED to restore ~/.mnemon/$f — recover by hand"
    elif [ "$had" -eq 1 ]; then
      # File didn't exist before this run (e.g. as_passphrase the deploy created) → remove it.
      rm -f "$HOME/.mnemon/$f"
    fi
  done
  [ "$restored" -gt 0 ] && c_ok "restored $restored prod auth file(s) under ~/.mnemon/"
  # Remove local scratch (only ever our own test-scoped dirs).
  [ -n "${MNEMON_VAULT_DIR:-}" ] && [[ "$MNEMON_VAULT_DIR" == "$HOME/.mnemon-test-"* ]] && rm -rf "$MNEMON_VAULT_DIR"
  [ -n "${MNEMON_CLIENT_CONFIG_ROOT:-}" ] && [[ "$MNEMON_CLIENT_CONFIG_ROOT" == "$HOME/.mnemon-test-configs-"* ]] && rm -rf "$MNEMON_CLIENT_CONFIG_ROOT"
  rm -rf "$SNAPSHOT_DIR"
  [ "$rc" -ne 0 ] && c_err "validation aborted (exit $rc) — see above; prod state restored"
  return 0
}
trap cleanup EXIT

# ── isolate ──────────────────────────────────────────────────────────────────
export MNEMON_VAULT_DIR="$HOME/.mnemon-test-$TEST_RUN_ID"
export MNEMON_S3_BUCKET="$S3_BUCKET"
export MNEMON_S3_PREFIX="test-cross-device/$TEST_RUN_ID"
export MNEMON_CLIENT_CONFIG_ROOT="$HOME/.mnemon-test-configs-$TEST_RUN_ID"
export MNEMON_PROD_APP_NAMES="$PROD_APP"
mkdir -p "$MNEMON_VAULT_DIR" "$MNEMON_CLIENT_CONFIG_ROOT"
c_ok "isolated test env (vault/$MNEMON_S3_PREFIX/configs scoped to $TEST_RUN_ID; prod guard armed)"

# ── deploy (auto-provisions the OAuth AS on a first-time deploy) ──────────────
echo ""
echo "── deploying $TEST_APP_NAME (this is the slow part) ──────────────────"
"$MNEMON_BIN" upgrade web --app-name "$TEST_APP_NAME" --mnemon-version "$VERSION"

# ── assert: the auto-provisioned AS serves correct metadata ──────────────────
echo ""
echo "── asserting OAuth AS (server-side) ──────────────────────────────────"
# `mnemon doctor` reads the test app's URL+token (upgrade web just wrote them
# to ~/.mnemon/{remote_url,local_token}); --fail-on-warn turns an absent AS
# (404 metadata) into a hard failure, and the check also verifies issuer ==
# the deployment base. Exit 0 here == the AS auto-provision worked.
doctor_out="$("$MNEMON_BIN" doctor --fail-on-warn 2>&1)" || { echo "$doctor_out"; die "doctor failed against $TEST_APP_NAME — AS not validated"; }
echo "$doctor_out" | grep -qiE "OAuth AS metadata.*(present|issuer=)" \
  || { echo "$doctor_out"; die "doctor did not report a passing 'OAuth AS metadata' check"; }
c_ok "OAuth AS metadata validated (RFC 8414 + issuer match) on $TEST_APP_NAME"

# ── surface the manual step ──────────────────────────────────────────────────
PASSPHRASE="$(cat "$HOME/.mnemon/as_passphrase" 2>/dev/null || echo "<see deploy output above>")"
echo ""
echo "══════════════════════════════════════════════════════════════════════"
c_ok "Server side PASSED. Now the one manual step — the browser connector:"
c_info "1. In claude.ai (or Claude Desktop) → Settings → Connectors → add custom connector"
c_info "   URL:        https://$TEST_APP_NAME.fly.dev/mcp"
c_info "   Passphrase: $PASSPHRASE"
c_info "2. Then ask Claude there: \"save a memory: cross-device validation OK\""
c_info "   then \"search your memory for cross-device validation\" — if it round-trips, PASS."
echo "══════════════════════════════════════════════════════════════════════"

if [ "$KEEP" -eq 1 ]; then
  c_warn "--keep set: leaving the app up. Re-run without --keep, or destroy by hand, when done."
  exit 0
fi
echo ""
read -r -p "Press Enter once you've tested the connector (or Ctrl-C) to tear everything down... " _
# trap cleanup runs on exit
