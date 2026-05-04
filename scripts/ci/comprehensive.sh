#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# Cross-platform CI orchestrator.
#
# Auto-detects the current platform and invokes the appropriate platform-specific
# build and test script. Aggregates results and exits with appropriate status code.
#
# Usage:
#   ./scripts/ci/comprehensive.sh          # Auto-detect and run
#   PLATFORM=ubuntu ./scripts/ci/comprehensive.sh  # Force platform
#
# Supported platforms: ubuntu, android, chromeos, macos
# Environment variables:
#   PLATFORM: Override auto-detected platform (optional)
#   VERBOSE: Enable verbose output (set to 1)

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Color output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Detect platform if not explicitly set
detect_platform() {
  local os_name
  os_name="$(uname -s)"

  case "${os_name}" in
    Linux)
      # Check if running in ChromeOS/Crostini (Linux container on ChromeOS)
      if [[ -f /etc/lsb-release ]] && grep -qi "CHROMEOS" /etc/lsb-release 2>/dev/null; then
        echo "chromeos"
      # Check for Android (unlikely in CI, but useful for local Android dev)
      elif [[ -f /system/build.prop ]] 2>/dev/null; then
        echo "android"
      else
        echo "ubuntu"
      fi
      ;;
    Darwin)
      echo "macos"
      ;;
    *)
      echo "unsupported" >&2
      exit 1
      ;;
  esac
}

# Run platform-specific script
run_platform_script() {
  local platform="$1"
  local script="${SCRIPT_DIR}/build-${platform}.sh"

  if [[ ! -f "${script}" ]]; then
    echo -e "${RED}✗ Platform script not found: ${script}${NC}" >&2
    exit 1
  fi

  echo -e "${YELLOW}→ Running: ${platform}${NC}"

  if bash "${script}"; then
    echo -e "${GREEN}✓ ${platform} build and test passed${NC}"
    return 0
  else
    echo -e "${RED}✗ ${platform} build and test failed${NC}"
    return 1
  fi
}

main() {
  local platform="${PLATFORM:-}"

  # Auto-detect if not set
  if [[ -z "${platform}" ]]; then
    platform="$(detect_platform)"
  fi

  if [[ "${VERBOSE:-0}" == "1" ]]; then
    echo "Platform detected: ${platform}"
    echo "Repository root: ${REPO_ROOT}"
  fi

  # Validate platform
  case "${platform}" in
    ubuntu|android|chromeos|macos)
      ;;
    *)
      echo -e "${RED}✗ Unsupported platform: ${platform}${NC}" >&2
      exit 1
      ;;
  esac

  # Run the platform-specific script
  cd "${REPO_ROOT}"
  run_platform_script "${platform}"
}

main "$@"
