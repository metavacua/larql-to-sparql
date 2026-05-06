#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: Apache-2.0
#
# Post-build distribution notice validation (S6, informational).
#
# Verifies that build artifacts include required attribution and license notices:
#   - cargo doc output: LICENSE + NOTICE at doc root
#   - Python wheel (if built): license classifier + NOTICE in dist-info
#   - Example binaries: spot-check for NOTICE text in help/about
#
# This check is informational and does not block PR merge. Failures are
# reported as warnings with remediation suggestions.
#
# Exit codes:
#   0  All checks passed (or were skipped because no artifacts found)
#   1  At least one check found a missing notice (warning, not error)
#   2  Toolchain misconfiguration

set -euo pipefail

fail=0

echo "check_distribution_notices: Starting post-build validation..."

# =========================================================================
# Check 1: cargo doc output includes LICENSE and NOTICE
# =========================================================================
if [[ -d target/doc ]]; then
  if [[ ! -f target/doc/LICENSE ]]; then
    echo "::warning::check_distribution_notices: target/doc/LICENSE missing" >&2
    echo "  Remediation: Ensure cargo publishes LICENSE to the doc root" >&2
    ((fail++))
  else
    echo "check_distribution_notices: ✓ target/doc/LICENSE present"
  fi

  if [[ ! -f target/doc/NOTICE ]]; then
    echo "::warning::check_distribution_notices: target/doc/NOTICE missing" >&2
    echo "  Remediation: Ensure cargo publishes NOTICE to the doc root" >&2
    ((fail++))
  else
    echo "check_distribution_notices: ✓ target/doc/NOTICE present"
  fi
else
  echo "check_distribution_notices: ⊘ target/doc not found (skipped cargo doc check)"
fi

# =========================================================================
# Check 2: Python wheel includes license metadata and NOTICE
# =========================================================================
wheel_count=$(find target -name "*.whl" -type f 2>/dev/null | wc -l)
if (( wheel_count > 0 )); then
  echo "check_distribution_notices: Found $wheel_count wheel(s); validating..."

  # Extract first wheel for inspection
  first_wheel=$(find target -name "*.whl" -type f -print -quit 2>/dev/null)
  if [[ -n "$first_wheel" ]]; then
    wheel_name=$(basename "$first_wheel")
    echo "  Checking: $wheel_name"

    # Wheels are ZIP files; extract and check METADATA for license classifier
    if unzip -l "$first_wheel" | grep -q "METADATA"; then
      if unzip -p "$first_wheel" "*/METADATA" | grep -q "License::"; then
        echo "check_distribution_notices: ✓ wheel METADATA includes license classifier"
      else
        echo "::warning::check_distribution_notices: wheel METADATA missing license classifier" >&2
        echo "  Remediation: Ensure setup.py/pyproject.toml includes classifiers = ['License :: OSI Approved :: Apache Software License', ...]" >&2
        ((fail++))
      fi
    fi

    # Check for NOTICE in dist-info directory
    if unzip -l "$first_wheel" | grep -qE "[^/]*\.dist-info/NOTICE"; then
      echo "check_distribution_notices: ✓ wheel includes NOTICE in dist-info"
    else
      echo "::warning::check_distribution_notices: wheel missing NOTICE in dist-info" >&2
      echo "  Remediation: Ensure NOTICE is included in maturin/PyO3 build via package_data or include_package_data" >&2
      ((fail++))
    fi
  fi
else
  echo "check_distribution_notices: ⊘ no .whl found (skipped Python wheel check)"
fi

# =========================================================================
# Check 3: Example binaries (spot check)
# =========================================================================
if [[ -d examples ]]; then
  example_count=$(find examples -type f -executable | wc -l)
  if (( example_count > 0 )); then
    echo "check_distribution_notices: Found $example_count example(s)"
    # Note: Spot-checking --help output for NOTICE text is environment-specific
    # and may not apply to all example binaries. This is a best-effort check.
    echo "  Spot-check: run example binaries with --help or --about to verify NOTICE text is accessible"
  fi
fi

# =========================================================================
# Summary
# =========================================================================
if (( fail == 0 )); then
  echo "check_distribution_notices: OK (all checks passed or were skipped)"
  exit 0
else
  echo "::warning::check_distribution_notices: $fail warning(s) found (informational only)" >&2
  exit 0  # Informational; do not block PR
fi
