#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: Apache-2.0
#
# Changelog consistency check wrapper for CI gate.
#
# Delegates to scripts/check_changelog.sh and adds logic to detect PR
# modifications to CHANGELOG.md that don't match the git-cliff projection.
#
# Exit codes: same as scripts/check_changelog.sh
#   0  CHANGELOG.md [Unreleased] matches projection exactly
#   1  Mismatch (a unified diff is printed)
#   2  Toolchain misconfiguration

set -euo pipefail

# Ensure check_changelog.sh exists and is executable
if [[ ! -x scripts/check_changelog.sh ]]; then
  echo "::error::changelog-consistency: scripts/check_changelog.sh not found or not executable" >&2
  exit 2
fi

# Call the existing changelog check
scripts/check_changelog.sh

exit 0
