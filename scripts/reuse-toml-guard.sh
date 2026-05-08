#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: Apache-2.0
#
# REUSE.toml provenance guard.
#
# Detects attempts to strip copyright attribution or relicense the project by
# bulk modification of REUSE.toml. Prevents:
#   1. Net reduction in SPDX-FileCopyrightText line count NOT explained by
#      file deletions in the same PR (copyright laundering)
#   2. Net reduction in [[annotations]] block count NOT explained by file
#      deletions in the same PR (silently dropping provenance)
#   3. Removal or alteration of the `version = 1` field
#
# This guard is a defense-in-depth measure: REUSE.toml is a single point of
# control for all license metadata. An LLM or malicious contributor modifying
# it can relicense the entire repository in bulk. This script fails the PR if
# such tampering is detected, but it explicitly accommodates legitimate
# refactors: deleting files in the same PR may legitimately reduce annotation
# blocks, and consolidating multiple [[annotations]] paths into a single glob
# may legitimately reduce block count without affecting copyright integrity.
#
# Heuristic policy:
#   - Allow up to N deletions where N = files deleted in the PR diff
#   - Allow refactors that consolidate paths (annotation count down, but
#     SPDX-FileCopyrightText count unchanged or up)
#   - Block any net loss of copyright attributions beyond accounted-for deletions
#
# Exit codes:
#   0  REUSE.toml is consistent (or new); changes are accounted for
#   1  Copyright stripped, annotations dropped beyond deletions, or version altered
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

# Determine the source of truth (PR base ref, or HEAD~1 as a fallback)
source_ref="${GITHUB_BASE_REF:-main}"
if ! git rev-parse "$source_ref" >/dev/null 2>&1; then
  source_ref="HEAD~1"
fi

# Skip if there is no base version to compare against (new repo, shallow clone, etc.)
if ! git rev-parse "$source_ref:REUSE.toml" >/dev/null 2>&1; then
  echo "::warning::reuse-toml-guard: $source_ref:REUSE.toml not found; skipping guard (new file or history unavailable)" >&2
  exit 0
fi

# Count files deleted in the PR range (used to allow proportional reduction
# in copyright lines and annotation blocks). `merge-base ... HEAD` works for
# both PR events (where source_ref is the base branch tip) and pushes (where
# source_ref is HEAD~1).
deleted_files_count=0
if merge_base="$(git merge-base "$source_ref" HEAD 2>/dev/null)"; then
  deleted_files_count="$(git diff --name-only --diff-filter=D "$merge_base"...HEAD | wc -l | tr -d '[:space:]')"
fi

before_copyright_count=$(git show "$source_ref:REUSE.toml" | grep -c 'SPDX-FileCopyrightText' || true)
after_copyright_count=$(grep -c 'SPDX-FileCopyrightText' REUSE.toml || true)

# A copyright line typically corresponds to one [[annotations]] block. We
# allow the count to drop by up to `deleted_files_count` (i.e., one per
# deleted file). Beyond that threshold, we treat it as suspicious stripping.
allowed_drop="$deleted_files_count"
copyright_drop=$(( before_copyright_count - after_copyright_count ))

if (( copyright_drop > allowed_drop )); then
  echo "::error file=REUSE.toml::reuse-toml-guard: SPDX-FileCopyrightText line count decreased beyond the deletion budget" >&2
  echo "  Before: $before_copyright_count lines" >&2
  echo "  After:  $after_copyright_count lines (drop: $copyright_drop)" >&2
  echo "  Files deleted in PR: $deleted_files_count" >&2
  echo "  Allowed drop: $allowed_drop" >&2
  echo "  Action: Restore the missing copyright attributions, or document the refactor" >&2
  echo "          (e.g., consolidating multiple annotations into a single glob is fine" >&2
  echo "           as long as no SPDX-FileCopyrightText line is dropped)." >&2
  exit 1
fi

before_annotations=$(git show "$source_ref:REUSE.toml" | grep -c '^\[\[annotations\]\]' || true)
after_annotations=$(grep -c '^\[\[annotations\]\]' REUSE.toml || true)
annotation_drop=$(( before_annotations - after_annotations ))

# Annotation blocks may also be consolidated (multiple paths merged into a
# single block). Block drops are only suspicious if they exceed the deletion
# budget AND copyright lines also dropped (proving lost provenance, not
# refactor).
if (( annotation_drop > allowed_drop )) && (( copyright_drop > 0 )); then
  echo "::error file=REUSE.toml::reuse-toml-guard: [[annotations]] block count decreased beyond the deletion budget AND copyright lines also dropped" >&2
  echo "  Annotation blocks: $before_annotations -> $after_annotations (drop: $annotation_drop)" >&2
  echo "  Copyright lines:   $before_copyright_count -> $after_copyright_count (drop: $copyright_drop)" >&2
  echo "  Files deleted in PR: $deleted_files_count" >&2
  echo "  Action: Annotation consolidation must preserve every SPDX-FileCopyrightText line." >&2
  exit 1
fi

if ! grep -q '^version = 1' REUSE.toml; then
  echo "::error file=REUSE.toml::reuse-toml-guard: version field missing or altered" >&2
  echo "  REUSE.toml must begin with 'version = 1'" >&2
  exit 1
fi

echo "reuse-toml-guard: OK (copyright lines: $before_copyright_count -> $after_copyright_count, annotations: $before_annotations -> $after_annotations, files deleted: $deleted_files_count)"
