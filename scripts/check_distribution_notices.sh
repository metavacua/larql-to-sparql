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
# This check is informational and does not block PR merge. Missing notices
# are reported as `::warning::` annotations with remediation suggestions, but
# the script always exits 0 so it can run with `continue-on-error: true`
# semantics without flipping the build status to red.
#
# Exit codes:
#   0  Always (this check is informational; warnings surface in the run summary)
#
# Dependencies:
#   unzip               (preferred: inspects wheel ZIP archives)
#   python3 -m zipfile  (fallback if unzip is unavailable)
#   If neither is present, the wheel check emits a warning and is skipped.

set -euo pipefail

fail=0

echo "check_distribution_notices: Starting post-build validation..."

# -------------------------------------------------------------------------
# Detect ZIP-inspection backend up front so the wheel check has a clear
# decision point. We prefer `unzip` (faster, ubiquitous on Linux) but fall
# back to `python3 -m zipfile`. Both are read-only and side-effect-free.
# -------------------------------------------------------------------------
zip_backend=""
if command -v unzip >/dev/null 2>&1; then
  zip_backend="unzip"
elif command -v python3 >/dev/null 2>&1; then
  zip_backend="python3"
else
  echo "::warning::check_distribution_notices: neither 'unzip' nor 'python3' is available; wheel inspection will be skipped" >&2
fi

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
if (( wheel_count > 0 )) && [[ -n "$zip_backend" ]]; then
  echo "check_distribution_notices: Found $wheel_count wheel(s); validating with $zip_backend..."

  # Extract first wheel for inspection
  first_wheel=$(find target -name "*.whl" -type f -print -quit 2>/dev/null)
  if [[ -n "$first_wheel" ]]; then
    wheel_name=$(basename "$first_wheel")
    echo "  Checking: $wheel_name"

    # Helper: list files in the wheel (one per line)
    list_wheel() {
      if [[ "$zip_backend" == "unzip" ]]; then
        unzip -l "$1" | awk 'NR>3 {print $NF}'
      else
        python3 -m zipfile -l "$1" | awk 'NR>1 {print $1}'
      fi
    }

    # Helper: dump the contents of a single file inside the wheel to stdout
    cat_wheel_member() {
      local wheel="$1" member="$2"
      if [[ "$zip_backend" == "unzip" ]]; then
        unzip -p "$wheel" "$member"
      else
        python3 -c "import sys, zipfile; print(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]).decode('utf-8', 'replace'))" "$wheel" "$member"
      fi
    }

    # Wheels are ZIP files; check METADATA for license classifier
    metadata_path="$(list_wheel "$first_wheel" | grep -E '\.dist-info/METADATA$' | head -n1)"
    if [[ -n "$metadata_path" ]]; then
      if cat_wheel_member "$first_wheel" "$metadata_path" | grep -q "License::"; then
        echo "check_distribution_notices: ✓ wheel METADATA includes license classifier"
      else
        echo "::warning::check_distribution_notices: wheel METADATA missing license classifier" >&2
        echo "  Remediation: Ensure pyproject.toml includes classifiers = ['License :: OSI Approved :: Apache Software License', ...]" >&2
        ((fail++))
      fi
    fi

    # Check for NOTICE in dist-info directory
    if list_wheel "$first_wheel" | grep -qE '\.dist-info/NOTICE$'; then
      echo "check_distribution_notices: ✓ wheel includes NOTICE in dist-info"
    else
      echo "::warning::check_distribution_notices: wheel missing NOTICE in dist-info" >&2
      echo "  Remediation: Ensure NOTICE is included in maturin/PyO3 build via package_data or include_package_data" >&2
      ((fail++))
    fi
  fi
elif (( wheel_count > 0 )); then
  echo "::warning::check_distribution_notices: $wheel_count wheel(s) found but no ZIP-inspection backend (unzip/python3) is available; skipped"
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
