#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# Deterministic SemVer preflight.
#
# Computes the next SemVer string for the repository as a pure function of:
#   - the current tag (last `v[0-9]+.[0-9]+.[0-9]+`, or v0.0.0 if none)
#   - the validated Conventional Commits in the range (last_tag..HEAD]
#
# Bump rules (matching cog.toml policy):
#   any commit with `!` header or `BREAKING CHANGE:` footer  -> major
#   else any `feat:`                                         -> minor
#   else any `fix:` / `perf:` / `refactor:` / `revert:`      -> patch
#   else                                                     -> no bump
#
# This script is informational by default. It writes the computed version
# to stdout and a structured summary to ${GITHUB_OUTPUT:-/dev/null} as
# `next_version=...` and `bump_kind=...`. It exits 0 even on a no-bump
# result; it only exits non-zero on a malformed commit it cannot classify.
#
# Exit codes:
#   0  preflight succeeded (next version printed)
#   1  encountered an unparseable commit in range
#   2  toolchain misconfiguration

set -euo pipefail

last_tag="$(git tag --list 'v[0-9]*' --sort=-v:refname | head -n1 || true)"
if [[ -z "$last_tag" ]]; then
  base_version="0.0.0"
  range="HEAD"
else
  base_version="${last_tag#v}"
  range="${last_tag}..HEAD"
fi

IFS='.' read -r major minor patch <<<"$base_version"
if [[ -z "${major:-}" || -z "${minor:-}" || -z "${patch:-}" ]]; then
  echo "version_preflight: cannot parse base version '$base_version'" >&2
  exit 2
fi

bump="none"

# `-z` makes git emit a NUL between commits (NUL embedded in --format= would
# be truncated at exec()). `%B` gives the full message including footers, so
# multi-line bodies do not break the loop.
while IFS= read -r -d '' commit; do
  header="$(printf '%s\n' "$commit" | head -n1)"

  # Skip merge commits per cog.toml policy.
  if [[ "$header" =~ ^Merge\  ]]; then
    continue
  fi

  # Strict Conventional Commits header grammar:
  #   type(scope)?!?: subject
  if [[ ! "$header" =~ ^(feat|fix|perf|refactor|revert|docs|chore|build|ci|test|style)(\([^\)]+\))?(!)?:[[:space:]].+ ]]; then
    echo "version_preflight: unparseable commit header: $header" >&2
    exit 1
  fi

  type="${BASH_REMATCH[1]}"
  bang="${BASH_REMATCH[3]}"

  if [[ -n "$bang" ]] || printf '%s\n' "$commit" | grep -qE '^BREAKING CHANGE:'; then
    bump="major"
    break  # major dominates; no need to keep scanning
  fi

  case "$type" in
    feat)
      [[ "$bump" == "none" || "$bump" == "patch" ]] && bump="minor"
      ;;
    fix|perf|refactor|revert)
      [[ "$bump" == "none" ]] && bump="patch"
      ;;
  esac
done < <(git log --reverse -z --format='%B' "$range")

case "$bump" in
  major) major=$((major + 1)); minor=0; patch=0 ;;
  minor) minor=$((minor + 1)); patch=0 ;;
  patch) patch=$((patch + 1)) ;;
  none)  : ;;
esac

next_version="${major}.${minor}.${patch}"

echo "version_preflight: base=$base_version bump=$bump next=$next_version"
echo "$next_version"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "next_version=${next_version}"
    echo "bump_kind=${bump}"
    echo "base_version=${base_version}"
  } >> "$GITHUB_OUTPUT"
fi
