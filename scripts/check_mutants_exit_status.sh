#!/usr/bin/env bash
# Interpret a `cargo mutants` exit status and report the outcome.
#
# Usage: check_mutants_exit_status.sh <exit-status>
#
# Exit codes (https://mutants.rs/exit-codes.html):
#   0  - every viable mutant was caught
#   2  - some mutants were not covered by tests
#   3  - some tests timed out
#   4  - the baseline itself already fails/hangs — NO mutants were tested
#   5/6 - the --in-diff diff didn't apply / wasn't a valid diff
#   70 - internal error
#
# Called from the `mutants` job's "Mutation-test the diff" step in
# quality.yml, once per matrix leg (ubuntu-latest, macos-14) — this is the
# one place both legs interpret cargo-mutants' outcome, so the two never
# drift out of sync the way inlined copies would.
set -euo pipefail

status="$1"

echo "--- mutation summary ---"
for f in caught missed timeout unviable; do
  path="mutants.out/$f.txt"
  if [ -f "$path" ]; then
    printf '%5d  %s\n' "$(wc -l < "$path")" "$f"
  fi
done

if [ -s mutants.out/missed.txt ]; then
  echo
  echo "Mutants that survived — the tests did not notice these edits:"
  cat mutants.out/missed.txt
fi

# Exit 4 means the UNMUTATED baseline itself failed to build or test,
# before cargo-mutants ever tried a mutation. That is a fundamentally
# different, more serious result than "0 caught/0 missed", and the two
# must not be allowed to look identical: this exact job hit exit 4 (841,
# then 1317 candidate mutants, zero ever run) across two pushes while
# moe_zero_copy.rs broke the ubuntu build (chrishayuk/larql#244), and both
# runs still showed a plain green "cargo-mutants (informational) success"
# in the PR checks tab — a blanket `exit 0` swallowed a real workspace
# build break, not an absence of mutants.
if [ "$status" -eq 4 ]; then
  echo "::warning::cargo-mutants baseline build/test failed before any mutant ran (exit 4) — this is NOT a clean mutation-testing pass, the workspace baseline itself is broken. See the build log above."
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      echo "### :warning: cargo-mutants baseline failed (exit 4)"
      echo
      echo "The unmutated baseline failed to build or test — **zero mutants were tested**. Do not read this job's green check as \"no mutants in this diff\"; the run is incomplete, not clean."
    } >> "$GITHUB_STEP_SUMMARY"
  fi
elif [ "$status" -ge 3 ]; then
  echo "::warning::cargo-mutants exited $status (timeout / diff-parse / internal error, see mutants.rs/exit-codes.html) — treat this run as incomplete, not clean."
fi
