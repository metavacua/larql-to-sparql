#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: Apache-2.0
#
# REUSE.toml provenance guard.
#
# Detects attempts to strip copyright attribution or relicense the project by
# bulk modification of REUSE.toml. Prevents:
#   1. Reduction in SPDX-FileCopyrightText line count (copyright laundering)
#   2. Reduction in [[annotations]] block count without file deletion evidence
#   3. Removal or alteration of the `version = 1` field
#
# This guard is a defense-in-depth measure: REUSE.toml is a single point of
# control for all license metadata. An LLM or malicious contributor modifying
# it can relicense the entire repository in bulk. This script fails the PR if
# such tampering is detected.
#
# Exit codes:
#   0  REUSE.toml is consistent with HEAD
#   1  Copyright line count decreased, or annotations removed, or version altered
#   2  Toolchain/repository misconfiguration (git not available, etc.)

set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
  echo "::error::reuse-toml-guard: git is not installed or not on PATH" >&2
  exit 2
fi

if [[ ! -f REUSE.toml ]]; then
  echo "::error file=REUSE.toml::reuse-toml-guard: REUSE.toml missing" >&2
  exit 2
fi

# Determine the source of truth (upstream main or HEAD if not on a PR)
source_ref="${GITHUB_BASE_REF:-main}"
if ! git rev-parse "$source_ref" >/dev/null 2>&1; then
  source_ref="HEAD~1"
fi

# Ensure we have both the current version and the base version
if ! git rev-parse "$source_ref:REUSE.toml" >/dev/null 2>&1; then
  echo "::warning::reuse-toml-guard: $source_ref:REUSE.toml not found; skipping guard (new file or history unavailable)" >&2
  exit 0
fi

before_copyright_count=$(git show "$source_ref:REUSE.toml" | grep -c 'SPDX-FileCopyrightText' || true)
after_copyright_count=$(grep -c 'SPDX-FileCopyrightText' REUSE.toml || true)

if (( after_copyright_count < before_copyright_count )); then
  echo "::error file=REUSE.toml::reuse-toml-guard: SPDX-FileCopyrightText line count decreased" >&2
  echo "  Before: $before_copyright_count lines" >&2
  echo "  After:  $after_copyright_count lines" >&2
  echo "  Action: Restore all copyright attributions. Do not strip provenance." >&2
  exit 1
fi

before_annotations=$(git show "$source_ref:REUSE.toml" | grep -c '^\[\[annotations\]\]' || true)
after_annotations=$(grep -c '^\[\[annotations\]\]' REUSE.toml || true)

if (( after_annotations < before_annotations )); then
  echo "::error file=REUSE.toml::reuse-toml-guard: [[annotations]] block count decreased" >&2
  echo "  Before: $before_annotations blocks" >&2
  echo "  After:  $after_annotations blocks" >&2
  echo "  Action: If you deleted files, update REUSE.toml to remove their annotations." >&2
  echo "          If no files were deleted, restore the annotation blocks." >&2
  exit 1
fi

if ! grep -q '^version = 1' REUSE.toml; then
  echo "::error file=REUSE.toml::reuse-toml-guard: version field missing or altered" >&2
  echo "  REUSE.toml must begin with 'version = 1'" >&2
  exit 1
fi

echo "reuse-toml-guard: OK (copyright lines: $after_copyright_count, annotations: $after_annotations)"
