#!/usr/bin/env bash
# Regression test for the cmd_safe monitor deadlock
# (metavacua/larql-to-sparql#246, https://github.com/metavacua/larql-to-sparql/issues/246).
#
# start_monitor() is invoked via command substitution:
#     monitor_pid=$(start_monitor "$logfile")
# If the backgrounded monitor subshell inherits the substitution's stdout pipe,
# $() waits for EOF until the (infinite) monitor loop exits — cmd_safe deadlocks
# before the wrapped command ever runs. The subshell's stdout must be redirected.
#
# This test extracts the real start_monitor definition from the script under
# test and asserts the command substitution returns promptly with a live PID.
# It needs no sudo, no systemd, no model — safe for CI.
#
# Usage: test-larql-probe-monitor.sh [path-to-larql-probe]
set -u

PROBE="${1:-$(dirname "$0")/../larql-probe}"
TIMEOUT_S=5

fail() { echo "FAIL: $*" >&2; exit 1; }

[ -f "$PROBE" ] || fail "script under test not found: $PROBE"

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT

# Extract the start_monitor function body verbatim from the script under test,
# plus a mem_avail_kb stub (its real definition needs /proc specifics).
{
    echo 'mem_avail_kb() { echo 1024; }'
    echo 'POLL_INTERVAL=1'
    sed -n '/^start_monitor()/,/^}/p' "$PROBE"
} > "$workdir/fn.sh"

grep -q '^start_monitor()' "$workdir/fn.sh" || fail "could not extract start_monitor from $PROBE"

# The deadlock manifests as the command substitution never returning: run it
# under timeout. Exit 124 = substitution held open = regression.
timeout "$TIMEOUT_S" bash -c '
    source "$1/fn.sh"
    pid=$(start_monitor "$1/monitor.log")
    [ -n "$pid" ] || exit 3
    kill -0 "$pid" 2>/dev/null || exit 4
    kill "$pid" 2>/dev/null
    exit 0
' _ "$workdir"
rc=$?

case $rc in
    0)   echo "PASS: start_monitor command substitution returns promptly with a live PID" ;;
    124) fail "start_monitor deadlocks under command substitution (monitor subshell holds \$()'s stdout pipe open) — see issue #246" ;;
    3)   fail "start_monitor returned no PID" ;;
    4)   fail "start_monitor returned a PID that is not alive" ;;
    *)   fail "unexpected exit code $rc" ;;
esac
