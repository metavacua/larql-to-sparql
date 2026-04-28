#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# Deterministic changelog consistency check.
#
# Computes the canonical Keep a Changelog `[Unreleased]` block from the
# validated Conventional Commits in the range (LAST_TAG..HEAD] using
# git-cliff, then compares it byte-for-byte against the `[Unreleased]` block
# currently committed to CHANGELOG.md. Exits non-zero with a unified diff if
# they differ.
#
# Inputs:
#   - cliff.toml          (transformer rules)
#   - CHANGELOG.md        (source-of-truth file in the working tree)
#   - git history         (validated commits)
#
# This script is pure: same inputs => same output. Do not introduce any
# probabilistic logic, network calls, or LLM-generated content.
#
# Exit codes:
#   0  CHANGELOG.md `[Unreleased]` matches git-cliff output exactly
#   1  Mismatch (a unified diff is printed to stderr)
#   2  Toolchain misconfiguration (git-cliff missing, etc.)

set -euo pipefail

CHANGELOG_FILE="${CHANGELOG_FILE:-CHANGELOG.md}"
CLIFF_CONFIG="${CLIFF_CONFIG:-cliff.toml}"

if ! command -v git-cliff >/dev/null 2>&1; then
  echo "check_changelog: git-cliff is not installed or not on PATH" >&2
  exit 2
fi

if [[ ! -f "$CLIFF_CONFIG" ]]; then
  echo "check_changelog: missing transformer config: $CLIFF_CONFIG" >&2
  exit 2
fi

if [[ ! -f "$CHANGELOG_FILE" ]]; then
  echo "check_changelog: missing changelog file: $CHANGELOG_FILE" >&2
  exit 2
fi

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

expected="$workdir/expected.md"
actual="$workdir/actual.md"

# Render the unreleased section deterministically from validated commits.
# `--unreleased` restricts to commits past the last tag.
# `--strip header` keeps only the body so we can compare to the [Unreleased]
# block of CHANGELOG.md without preamble noise.
git-cliff \
  --config "$CLIFF_CONFIG" \
  --unreleased \
  --strip header \
  --output "$expected"

# Extract the existing [Unreleased] block from the committed CHANGELOG.md.
# Inclusive of the heading; terminates at the next `## [` heading or EOF.
awk '
  /^## \[Unreleased\]/                 { in_block = 1; print; next }
  in_block && (/^## \[/ || /^\[.*\]:/) { in_block = 0 }
  in_block                             { print }
' "$CHANGELOG_FILE" > "$actual"

if ! diff -u "$expected" "$actual" >/dev/null; then
  {
    echo "check_changelog: CHANGELOG.md [Unreleased] does not match the"
    echo "                 deterministic projection of validated commits."
    echo "                 Expected vs. actual diff:"
    echo
    diff -u "$expected" "$actual" || true
    echo
    echo "Remediation: regenerate with"
    echo "  git-cliff --config $CLIFF_CONFIG --unreleased --strip header \\"
    echo "    --prepend $CHANGELOG_FILE"
    echo "or amend the offending commit so its Conventional Commits header"
    echo "matches the rules in cog.toml / cliff.toml."
  } >&2
  exit 1
fi

echo "check_changelog: OK"
